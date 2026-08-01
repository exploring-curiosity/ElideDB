#pragma once
// SFI v1 — StreetDex Frame Index (docs/FORMAT.md §2). One SFI per source
// video file; the video is NEVER modified. Built once at ingest by a
// packet-level scan (no decode), it answers: "decoded frames for [t0,t1]
// while reading the fewest possible bytes of the video".
//
// GOP granularity is the point: compressed video is only decodable from a
// keyframe, so the minimal correct read unit is the GOP byte range. The GOP
// table IS a zone map over (pts range -> byte range). For the lab MJPEG
// captures every frame is a keyframe, so GOP == 1 frame and reads are
// frame-exact — the degenerate best case of the same design.
//
// pts provenance: for REIP captures, container pts is a lie (frame_number /
// fps on the sensor's unsynced clock). SFI stores *canonical-timeline* ns
// joined from the time/ JSONs at build time (packet order <-> frame_id order
// per file_index). For plain videos without sidecar timing, container pts
// converted to ns is used as-is.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/mmap_file.hpp"
#include "streetdex/core/time.hpp"

namespace sdx {

inline constexpr char kSfiMagic[4] = {'S', 'F', 'I', '1'};
inline constexpr uint16_t kSfiVersion = 1;

// On-disk table entries. Field order gives natural alignment; sizes are
// asserted so the mmap reader can cast in place.
struct SfiGopEntry {
  uint32_t gop_id;
  uint32_t frame_count;
  uint64_t start_byte;  // byte offset of keyframe packet in source file
  uint64_t end_byte;    // one past last packet byte of this GOP
  int64_t first_pts_ns;
  int64_t last_pts_ns;
  uint32_t first_frame_row;  // row into the frame table
  uint32_t pad;
};
static_assert(sizeof(SfiGopEntry) == 48);

struct SfiFrameEntry {
  int64_t pts_ns;
  int64_t dts_ns;
  uint64_t byte_offset;
  uint32_t packet_size;
  uint32_t gop_id;
  uint32_t flags;  // bit0 = keyframe
  uint32_t pad;
};
static_assert(sizeof(SfiFrameEntry) == 40);

struct SfiHeader {
  uint32_t codec_id = 0;  // AVCodecID, informational
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t timebase_num = 0;
  uint32_t timebase_den = 0;
  TimeNs first_pts_ns = 0;
  TimeNs last_pts_ns = 0;
  TimeNs clock_offset_ns = 0;
  uint64_t frame_count = 0;
  uint64_t gop_count = 0;
  uint64_t source_size = 0;  // the elision denominator for this file
  std::string source_path;
};

// ---- Builder ---------------------------------------------------------------
struct SfiBuildInput {
  std::string video_path;
  // Canonical pts per expected frame, in packet order. Empty => trust the
  // container's own pts (converted to ns).
  std::vector<TimeNs> frame_pts_ns;
  TimeNs clock_offset_ns = 0;
};

struct SfiBuildStats {
  uint64_t frames_indexed = 0;
  uint64_t gops = 0;
  uint64_t bytes_scanned = 0;
  // Non-fatal: timing sidecar had more/fewer frames than the container
  // (truncated tail segment, or timing bundles outliving the copied video).
  int64_t timing_minus_container_frames = 0;
};

Result<SfiBuildStats> build_sfi(const SfiBuildInput& in,
                                const std::string& out_path);

// ---- Reader ----------------------------------------------------------------
class SfiReader {
 public:
  static Result<SfiReader> open(const std::string& path);

  const SfiHeader& header() const { return header_; }
  std::span<const SfiGopEntry> gops() const { return gops_; }
  std::span<const SfiFrameEntry> frames() const { return frames_; }

  // Binary-search GOPs overlapping [t0, t1] (closed window).
  std::span<const SfiGopEntry> gops_in_range(TimeNs t0, TimeNs t1) const;
  // Frame rows overlapping the window: [first_row, last_row) half-open.
  std::pair<uint64_t, uint64_t> frame_rows_in_range(TimeNs t0, TimeNs t1) const;

 private:
  MmapFile file_;
  SfiHeader header_;
  std::span<const SfiGopEntry> gops_;
  std::span<const SfiFrameEntry> frames_;
};

}  // namespace sdx
