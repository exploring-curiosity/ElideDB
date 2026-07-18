#include <algorithm>
#include <cstring>

#include "streetdex/core/binary_io.hpp"
#include "streetdex/metrics/metrics.hpp"
#include "streetdex/video/sfi.hpp"

namespace sdx {

Result<SfiReader> SfiReader::open(const std::string& path) {
  auto mf = MmapFile::open(path);
  if (!mf) return tl::unexpected(mf.error());
  SfiReader r;
  r.file_ = std::move(*mf);
  const auto b = r.file_.bytes();
  const size_t n = b.size();
  if (n < 4 + 20 || std::memcmp(b.data(), kSfiMagic, 4) != 0)
    return fail(Errc::bad_format, "not an SFI file: " + path);
  if (std::memcmp(b.data() + n - 4, kSfiMagic, 4) != 0)
    return fail(Errc::bad_format, "torn/truncated SFI file: " + path);
  // The whole index is the price of admission for a video file's random
  // access; charge it once at open. It is O(frames), ~0.005% of source size.
  metrics::count(metrics::Cat::sfi_index, n);

  ByteReader br(b);
  uint32_t magic = 0;
  uint16_t version = 0, flags = 0;
  br.get(magic);
  if (!br.get(version) || !br.get(flags) || version != kSfiVersion)
    return fail(Errc::bad_format, "unsupported SFI version: " + path);
  auto& h = r.header_;
  uint32_t pad = 0;
  if (!br.get(h.codec_id) || !br.get(h.width) || !br.get(h.height) ||
      !br.get(h.timebase_num) || !br.get(h.timebase_den) || !br.get(pad) ||
      !br.get(h.first_pts_ns) || !br.get(h.last_pts_ns) ||
      !br.get(h.clock_offset_ns) || !br.get(h.frame_count) ||
      !br.get(h.gop_count) || !br.get(h.source_size) ||
      !br.get_string(h.source_path))
    return fail(Errc::bad_format, "bad SFI header: " + path);

  // Trailing pointers let the reader jump straight to the tables.
  uint64_t gop_off = 0, frame_off = 0;
  std::memcpy(&gop_off, b.data() + n - 20, 8);
  std::memcpy(&frame_off, b.data() + n - 12, 8);
  const uint64_t gop_bytes = h.gop_count * sizeof(SfiGopEntry);
  const uint64_t frame_bytes = h.frame_count * sizeof(SfiFrameEntry);
  if (gop_off % 8 != 0 || frame_off % 8 != 0 || gop_off + gop_bytes > n ||
      frame_off + frame_bytes > n)
    return fail(Errc::bad_format, "SFI table offsets out of range: " + path);
  r.gops_ = {reinterpret_cast<const SfiGopEntry*>(b.data() + gop_off),
             h.gop_count};
  r.frames_ = {reinterpret_cast<const SfiFrameEntry*>(b.data() + frame_off),
               h.frame_count};
  return r;
}

std::span<const SfiGopEntry> SfiReader::gops_in_range(TimeNs t0,
                                                      TimeNs t1) const {
  // Same shape as SDX chunk pruning: tables are pts-sorted, so overlap is a
  // binary search, and everything outside is elided by never being decoded.
  const auto first = std::partition_point(
      gops_.begin(), gops_.end(),
      [&](const SfiGopEntry& g) { return g.last_pts_ns < t0; });
  const auto last = std::partition_point(
      first, gops_.end(),
      [&](const SfiGopEntry& g) { return g.first_pts_ns <= t1; });
  return {first, last};
}

std::pair<uint64_t, uint64_t> SfiReader::frame_rows_in_range(TimeNs t0,
                                                             TimeNs t1) const {
  const auto lo = std::partition_point(
      frames_.begin(), frames_.end(),
      [&](const SfiFrameEntry& f) { return f.pts_ns < t0; });
  const auto hi = std::partition_point(
      lo, frames_.end(), [&](const SfiFrameEntry& f) { return f.pts_ns <= t1; });
  return {static_cast<uint64_t>(lo - frames_.begin()),
          static_cast<uint64_t>(hi - frames_.begin())};
}

}  // namespace sdx
