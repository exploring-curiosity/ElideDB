#include <algorithm>
#include <cstring>

#include "streetdex/core/binary_io.hpp"
#include "streetdex/format/sdx.hpp"
#include "streetdex/metrics/metrics.hpp"

namespace sdx {

namespace {

// Row-level binary search inside one chunk's ts column (mmapped, aligned).
// Only ever used on the first/last surviving chunk; interior chunks are fully
// inside the window by sortedness — that is the point of the invariant.
const int64_t* ts_ptr(const MmapFile& f, const ColSpan& span) {
  return reinterpret_cast<const int64_t*>(f.bytes().data() + span.data_offset);
}

}  // namespace

Result<SdxReader> SdxReader::open(const std::string& path) {
  auto mf = MmapFile::open(path);
  if (!mf) return tl::unexpected(mf.error());
  SdxReader r;
  r.file_ = std::move(*mf);
  const auto bytes = r.file_.bytes();
  const size_t n = bytes.size();
  if (n < 16 || std::memcmp(bytes.data(), kSdxMagic, 4) != 0)
    return fail(Errc::bad_format, "not an SDX file: " + path);
  // Trailing magic present <=> file complete. This is the torn-write check.
  if (std::memcmp(bytes.data() + n - 4, kSdxMagic, 4) != 0)
    return fail(Errc::bad_format, "torn/truncated SDX file: " + path);
  uint32_t footer_len = 0;
  std::memcpy(&footer_len, bytes.data() + n - 8, 4);
  metrics::count(metrics::Cat::sdx_footer, 8);  // logical I/O #1: tail
  if (footer_len + 8ull + 8ull > n)
    return fail(Errc::bad_format, "footer length out of range: " + path);
  const size_t footer_off = n - 8 - footer_len;
  metrics::count(metrics::Cat::sdx_footer, footer_len);  // logical I/O #2
  ByteReader br(bytes.subspan(footer_off, footer_len));

  uint16_t version = 0, flags = 0;
  if (!br.get(version) || !br.get(flags) || version != kSdxVersion)
    return fail(Errc::bad_format, "unsupported SDX version in " + path);
  auto& m = r.meta_;
  uint32_t chunk_target = 0;
  uint16_t ncol = 0;
  int64_t clock_off = 0;
  if (!br.get_string(m.stream_id) || !br.get_string(m.units) ||
      !br.get(clock_off) || !br.get(chunk_target) || !br.get(ncol) || ncol == 0)
    return fail(Errc::bad_format, "bad SDX footer header: " + path);
  m.clock_offset_ns = clock_off;
  m.chunk_target_rows = chunk_target;
  for (uint16_t c = 0; c < ncol; ++c) {
    ColumnSpec spec;
    uint8_t type = 0, pad = 0;
    if (!br.get_string(spec.name) || !br.get(type) || !br.get(pad))
      return fail(Errc::bad_format, "bad SDX column desc: " + path);
    spec.type = static_cast<ColType>(type);
    if (col_type_width(spec.type) == 0)
      return fail(Errc::bad_format, "unknown column type in " + path);
    m.columns.push_back(std::move(spec));
  }
  if (m.columns[0].type != ColType::TS_NS)
    return fail(Errc::bad_format, "column 0 must be TS_NS: " + path);
  uint32_t nchunks = 0;
  if (!br.get(nchunks))
    return fail(Errc::bad_format, "bad SDX chunk count: " + path);
  r.chunks_.reserve(nchunks);
  const size_t data_end = footer_off;
  TimeNs prev_max = std::numeric_limits<TimeNs>::min();
  for (uint32_t k = 0; k < nchunks; ++k) {
    ChunkEntry ch;
    uint32_t pad = 0;
    if (!br.get(ch.row_count) || !br.get(pad))
      return fail(Errc::bad_format, "bad SDX chunk entry: " + path);
    ch.cols.resize(ncol);
    for (auto& cs : ch.cols) {
      if (!br.get(cs.data_offset) || !br.get(cs.data_length) ||
          !br.get(cs.min_raw) || !br.get(cs.max_raw))
        return fail(Errc::bad_format, "bad SDX colspan: " + path);
      if (cs.data_offset + cs.data_length > data_end)
        return fail(Errc::bad_format, "colspan outside data region: " + path);
    }
    // Directory-level invariant: chunk time ranges must be ordered — this is
    // what licenses binary search instead of a directory scan.
    if (ch.ts_min() < prev_max)
      return fail(Errc::bad_format, "chunk ts ranges out of order: " + path);
    prev_max = ch.ts_max();
    m.row_count += ch.row_count;
    r.chunks_.push_back(std::move(ch));
  }
  if (!r.chunks_.empty()) {
    m.min_ts = r.chunks_.front().ts_min();
    m.max_ts = r.chunks_.back().ts_max();
  }
  m.file_size = n;
  m.footer_bytes = footer_len + 8;
  return r;
}

Result<RangeScan> SdxReader::scan(TimeNs t0, TimeNs t1,
                                  std::span<const std::string> columns) const {
  if (t0 > t1) return fail(Errc::bad_argument, "t0 > t1");
  RangeScan out;
  out.chunks_total = static_cast<uint32_t>(chunks_.size());

  // Resolve requested columns to schema indices (empty = all).
  if (columns.empty()) {
    for (uint32_t c = 0; c < meta_.columns.size(); ++c)
      out.column_ids.push_back(c);
  } else {
    for (const auto& name : columns) {
      auto it = std::find_if(meta_.columns.begin(), meta_.columns.end(),
                             [&](const ColumnSpec& s) { return s.name == name; });
      if (it == meta_.columns.end())
        return fail(Errc::bad_argument,
                    "unknown column '" + name + "' in " + meta_.stream_id);
      out.column_ids.push_back(
          static_cast<uint32_t>(it - meta_.columns.begin()));
    }
  }
  out.spans.resize(out.column_ids.size());
  if (chunks_.empty() || t1 < meta_.min_ts || t0 > meta_.max_ts) return out;

  // Zone-map pruning: binary search the directory for the overlap range.
  // Everything outside [first, last) is elided — and, per the metrics
  // contract, elided means those bytes are never charged because they are
  // never touched.
  const auto first = std::partition_point(
      chunks_.begin(), chunks_.end(),
      [&](const ChunkEntry& ch) { return ch.ts_max() < t0; });
  const auto last = std::partition_point(
      first, chunks_.end(),
      [&](const ChunkEntry& ch) { return ch.ts_min() <= t1; });
  for (auto it = first; it != last; ++it) {
    const ChunkEntry& ch = *it;
    // Row trim only at the boundary chunks; interior chunks are whole.
    size_t row_a = 0, row_b = ch.row_count;
    const bool is_first = (it == first);
    const bool is_last = (it == last - 1);
    if (is_first || is_last) {
      const int64_t* ts = ts_ptr(file_, ch.cols[0]);
      if (is_first)
        row_a = std::lower_bound(ts, ts + ch.row_count, t0) - ts;
      if (is_last)
        row_b = std::upper_bound(ts + row_a, ts + ch.row_count, t1) - ts;
      // The trim itself reads the ts column of this chunk; charge it unless
      // ts is among the requested columns (then the span charge covers it).
      const bool ts_requested =
          std::find(out.column_ids.begin(), out.column_ids.end(), 0u) !=
          out.column_ids.end();
      if (!ts_requested)
        metrics::count(metrics::Cat::sdx_data, ch.cols[0].data_length);
    }
    if (row_a >= row_b) continue;
    out.chunks_scanned += 1;
    out.total_rows += row_b - row_a;
    for (size_t c = 0; c < out.column_ids.size(); ++c) {
      const ColSpan& cs = ch.cols[out.column_ids[c]];
      const size_t w = col_type_width(meta_.columns[out.column_ids[c]].type);
      // Normative accounting (FORMAT.md 1.4): a surviving chunk costs its
      // full column span — the chunk is the atomic I/O unit even though the
      // returned slice is row-trimmed in memory.
      metrics::count(metrics::Cat::sdx_data, cs.data_length);
      out.spans[c].push_back(
          {file_.bytes().data() + cs.data_offset + row_a * w, row_b - row_a});
    }
  }
  return out;
}

std::vector<int64_t> SdxReader::gather_i64(const RangeScan& s, size_t col_idx) {
  std::vector<int64_t> out;
  out.reserve(s.total_rows);
  for (const auto& slice : s.spans[col_idx]) {
    const auto* p = reinterpret_cast<const int64_t*>(slice.data);
    out.insert(out.end(), p, p + slice.rows);
  }
  return out;
}

std::vector<double> SdxReader::gather_f64(const RangeScan& s, size_t col_idx,
                                          ColType type) {
  std::vector<double> out;
  out.reserve(s.total_rows);
  for (const auto& slice : s.spans[col_idx]) {
    switch (type) {
      case ColType::F64: {
        const auto* p = reinterpret_cast<const double*>(slice.data);
        out.insert(out.end(), p, p + slice.rows);
        break;
      }
      case ColType::F32: {
        const auto* p = reinterpret_cast<const float*>(slice.data);
        for (size_t i = 0; i < slice.rows; ++i) out.push_back(p[i]);
        break;
      }
      case ColType::TS_NS:
      case ColType::I64: {
        const auto* p = reinterpret_cast<const int64_t*>(slice.data);
        for (size_t i = 0; i < slice.rows; ++i)
          out.push_back(static_cast<double>(p[i]));
        break;
      }
      case ColType::I16: {
        const auto* p = reinterpret_cast<const int16_t*>(slice.data);
        for (size_t i = 0; i < slice.rows; ++i) out.push_back(p[i]);
        break;
      }
    }
  }
  return out;
}

}  // namespace sdx
