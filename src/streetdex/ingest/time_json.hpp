#pragma once
// REIP time/ directory parser. Each `{cam}_{session}_{bundle}.json` is a
// 450-frame log bundle of records:
//   { python_timestamp: local wall clock (sec, float),
//     global_timestamp: shared cross-sensor clock ticks (~1200 Hz),
//     frame_id:        per-camera global frame counter,
//     file_index:      which `{cam}_{session}_{file_index}.avi` holds it,
//     resolution, fps, ... }
//
// The bundle suffix is NOT the video segment index — file_index inside each
// record is. Bundles can also reference AVI segments that are absent on disk
// (recording outlived the copy); the SFI builder joins by (file_index, order)
// and tolerates missing/truncated segments.

#include <cstdint>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"

namespace sdx {

struct FrameTimeRecord {
  int64_t frame_id = 0;
  int32_t file_index = 0;
  double python_ts = 0;    // seconds, sensor-local wall clock
  int64_t global_ticks = 0;
};

struct CameraTimeline {
  std::string cam;      // "0" or "2"
  std::string session;  // e.g. "1678894150" — capture-session id in filenames
  int fps = 0;
  int width = 0;
  int height = 0;
  std::vector<FrameTimeRecord> frames;  // sorted by frame_id, gap-free-checked
};

// Streams present in a time/ directory, as (cam, session) pairs.
Result<std::vector<std::pair<std::string, std::string>>> discover_camera_streams(
    const std::string& time_dir);

// Load and concatenate every bundle of one camera stream, sorted by frame_id.
Result<CameraTimeline> load_camera_timeline(const std::string& time_dir,
                                            const std::string& cam,
                                            const std::string& session);

}  // namespace sdx
