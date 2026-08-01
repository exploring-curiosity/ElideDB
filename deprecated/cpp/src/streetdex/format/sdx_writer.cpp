#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

#include "streetdex/core/binary_io.hpp"
#include "streetdex/format/sdx.hpp"

namespace sdx {

namespace {

// Raw 8-byte zone-map encoding: value's bit pattern in the low bytes.
template <typename T>
uint64_t raw8(T v) {
  uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(T));
  return out;
}

template <typename T>
void min_max(const uint8_t* data, size_t rows, uint64_t& mn, uint64_t& mx) {
  T lo = std::numeric_limits<T>::max();
  T hi = std::numeric_limits<T>::lowest();
  for (size_t i = 0; i < rows; ++i) {
    T v;
    std::memcpy(&v, data + i * sizeof(T), sizeof(T));
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  mn = raw8(lo);
  mx = raw8(hi);
}

void compute_zone_map(ColType t, const uint8_t* data, size_t rows,
                      uint64_t& mn, uint64_t& mx) {
  switch (t) {
    case ColType::TS_NS:
    case ColType::I64: min_max<int64_t>(data, rows, mn, mx); return;
    case ColType::F64: min_max<double>(data, rows, mn, mx); return;
    case ColType::F32: min_max<float>(data, rows, mn, mx); return;
    case ColType::I16: min_max<int16_t>(data, rows, mn, mx); return;
  }
}

}  // namespace

struct SdxWriter::Impl {
  std::FILE* f = nullptr;
  std::string path;
  SdxMeta meta;                     // columns[0] is the ts column
  std::vector<std::vector<uint8_t>> chunk_cols;  // buffered current chunk
  size_t chunk_rows = 0;
  uint64_t offset = 0;              // current absolute file offset
  std::vector<ChunkEntry> chunks;
  TimeNs last_ts = std::numeric_limits<TimeNs>::min();
  bool finished = false;

  ~Impl() {
    if (f != nullptr) std::fclose(f);
  }

  Result<void> write_bytes(const void* p, size_t n) {
    if (std::fwrite(p, 1, n, f) != n)
      return fail(Errc::io, "write failed: " + path);
    offset += n;
    return {};
  }
  Result<void> pad_to_8() {
    static const uint8_t zeros[8] = {};
    if (offset % 8 != 0) return write_bytes(zeros, 8 - offset % 8);
    return {};
  }
};

Result<SdxWriter> SdxWriter::create(const std::string& path,
                                    std::string stream_id, std::string units,
                                    TimeNs clock_offset_ns,
                                    std::vector<ColumnSpec> value_columns,
                                    uint32_t chunk_target_rows) {
  if (chunk_target_rows == 0)
    return fail(Errc::bad_argument, "chunk_target_rows must be > 0");
  for (const auto& c : value_columns)
    if (c.type == ColType::TS_NS)
      return fail(Errc::bad_argument,
                  "TS_NS is reserved for column 0 (implicit)");
  SdxWriter w;
  w.impl_ = new Impl();
  auto& im = *w.impl_;
  im.path = path;
  im.f = std::fopen(path.c_str(), "wb");
  if (im.f == nullptr) return fail(Errc::io, "cannot create " + path);
  im.meta.stream_id = std::move(stream_id);
  im.meta.units = std::move(units);
  im.meta.clock_offset_ns = clock_offset_ns;
  im.meta.chunk_target_rows = chunk_target_rows;
  im.meta.columns.push_back({"ts", ColType::TS_NS});
  for (auto& c : value_columns) im.meta.columns.push_back(std::move(c));
  im.chunk_cols.resize(im.meta.columns.size());
  if (auto r = im.write_bytes(kSdxMagic, 4); !r) return tl::unexpected(r.error());
  // Pad so chunk 0 / column 0 starts 8-aligned (arrays must be castable).
  if (auto r = im.pad_to_8(); !r) return tl::unexpected(r.error());
  return w;
}

SdxWriter::~SdxWriter() { delete impl_; }
SdxWriter::SdxWriter(SdxWriter&& o) noexcept : impl_(o.impl_) {
  o.impl_ = nullptr;
}

Result<void> SdxWriter::flush_chunk() {
  auto& im = *impl_;
  if (im.chunk_rows == 0) return {};
  ChunkEntry entry;
  entry.row_count = static_cast<uint32_t>(im.chunk_rows);
  for (size_t c = 0; c < im.meta.columns.size(); ++c) {
    if (auto r = im.pad_to_8(); !r) return r;
    ColSpan span;
    span.data_offset = im.offset;
    span.data_length = im.chunk_cols[c].size();
    compute_zone_map(im.meta.columns[c].type, im.chunk_cols[c].data(),
                     im.chunk_rows, span.min_raw, span.max_raw);
    if (auto r = im.write_bytes(im.chunk_cols[c].data(), im.chunk_cols[c].size());
        !r)
      return r;
    entry.cols.push_back(span);
    im.chunk_cols[c].clear();
  }
  im.chunks.push_back(std::move(entry));
  im.chunk_rows = 0;
  return {};
}

Result<void> SdxWriter::append(size_t rows, const int64_t* ts,
                               const void* const* value_cols) {
  auto& im = *impl_;
  if (im.finished) return fail(Errc::bad_argument, "writer already finished");
  size_t done = 0;
  while (done < rows) {
    const size_t take = std::min(rows - done,
                                 static_cast<size_t>(im.meta.chunk_target_rows) -
                                     im.chunk_rows);
    // ts monotonicity — the invariant zone-map pruning depends on. Reject at
    // write time; a sorted file is what makes chunk ranges disjoint.
    for (size_t i = done; i < done + take; ++i) {
      if (ts[i] < im.last_ts)
        return fail(Errc::bad_argument,
                    "timestamps must be non-decreasing (stream " +
                        im.meta.stream_id + ")");
      im.last_ts = ts[i];
    }
    {
      const auto* p = reinterpret_cast<const uint8_t*>(ts + done);
      im.chunk_cols[0].insert(im.chunk_cols[0].end(), p, p + take * 8);
    }
    for (size_t c = 1; c < im.meta.columns.size(); ++c) {
      const size_t w = col_type_width(im.meta.columns[c].type);
      const auto* p =
          reinterpret_cast<const uint8_t*>(value_cols[c - 1]) + done * w;
      im.chunk_cols[c].insert(im.chunk_cols[c].end(), p, p + take * w);
    }
    im.chunk_rows += take;
    im.meta.row_count += take;
    done += take;
    if (im.chunk_rows == im.meta.chunk_target_rows)
      if (auto r = flush_chunk(); !r) return r;
  }
  return {};
}

Result<void> SdxWriter::finish() {
  auto& im = *impl_;
  if (im.finished) return {};
  if (auto r = flush_chunk(); !r) return r;

  ByteWriter fw;
  fw.put<uint16_t>(kSdxVersion);
  fw.put<uint16_t>(0);  // flags, reserved
  fw.put_string(im.meta.stream_id);
  fw.put_string(im.meta.units);
  fw.put<int64_t>(im.meta.clock_offset_ns);
  fw.put<uint32_t>(im.meta.chunk_target_rows);
  fw.put<uint16_t>(static_cast<uint16_t>(im.meta.columns.size()));
  for (const auto& c : im.meta.columns) {
    fw.put_string(c.name);
    fw.put<uint8_t>(static_cast<uint8_t>(c.type));
    fw.put<uint8_t>(0);  // pad
  }
  fw.put<uint32_t>(static_cast<uint32_t>(im.chunks.size()));
  for (const auto& ch : im.chunks) {
    fw.put<uint32_t>(ch.row_count);
    fw.put<uint32_t>(0);  // pad -> fixed-width entries, binary-searchable
    for (const auto& cs : ch.cols) {
      fw.put<uint64_t>(cs.data_offset);
      fw.put<uint64_t>(cs.data_length);
      fw.put<uint64_t>(cs.min_raw);
      fw.put<uint64_t>(cs.max_raw);
    }
  }
  if (auto r = im.write_bytes(fw.bytes().data(), fw.size()); !r) return r;
  const uint32_t footer_len = static_cast<uint32_t>(fw.size());
  if (auto r = im.write_bytes(&footer_len, 4); !r) return r;
  // Trailing magic: a torn write cannot end in "SDX1" — reader detects it.
  if (auto r = im.write_bytes(kSdxMagic, 4); !r) return r;
  if (std::fflush(im.f) != 0) return fail(Errc::io, "flush failed: " + im.path);
  im.finished = true;
  return {};
}

}  // namespace sdx
