#pragma once
// get_window: the end-to-end query path.
//
//   catalog prune (manifest time ranges — skip whole FILES)
//     -> SDX chunk prune (zone maps — skip chunks inside surviving files)
//     -> SFI GOP prune (frame index — skip byte ranges inside video files)
//     -> decode (video pixels produced LAST — late materialization)
//     -> align (resample everything onto the caller's timeline)
//
// Ordering is the C-Store lesson applied to multimodal data: the most
// expensive materialization (video decode) happens after every cheaper
// pruning stage has had its say. Every query reports bytes-read vs corpus.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "streetdex/align/resample.hpp"
#include "streetdex/catalog/manifest.hpp"
#include "streetdex/core/error.hpp"
#include "streetdex/format/sdx.hpp"
#include "streetdex/metrics/metrics.hpp"
#include "streetdex/video/decoder.hpp"
#include "streetdex/video/sfi.hpp"

namespace sdx {

struct WindowQuery {
  TimeNs t0 = 0;
  TimeNs t1 = 0;
  double rate_hz = 30.0;              // output timeline rate
  std::vector<std::string> streams;   // empty = every stream in the snapshot
  Interp interp = Interp::nearest;
  // Sensor scans read [t0-guard, t1+guard]: interpolation AT the window edge
  // needs the neighboring sample just outside it. One second costs at most a
  // couple of extra chunks and makes edge values exact instead of held.
  TimeNs edge_guard_ns = kNsPerSec;
  FrameQueryOptions video;            // scaling / stride for decoded frames
  bool decode_video = true;           // false: plan + sensor data only
};

struct SensorWindow {
  std::string stream_id;
  std::vector<std::string> columns;              // value columns (ts implied)
  std::vector<std::vector<double>> values;       // [column][timeline point]
  size_t raw_rows = 0;                           // rows read pre-resample
};

struct VideoWindow {
  std::string stream_id;
  std::vector<DecodedFrame> frames;      // native capture times within window
  std::vector<int32_t> frame_for_point;  // timeline point -> index in frames
};

struct WindowResult {
  std::vector<TimeNs> timeline;
  std::vector<SensorWindow> sensors;
  std::vector<VideoWindow> video;
  metrics::IoStats io;
  uint64_t corpus_bytes = 0;
  uint64_t files_pruned = 0;  // whole files skipped at the catalog stage
  uint64_t files_touched = 0;
  double wall_ms = 0;

  double elided_pct() const {
    const uint64_t read = io.total_bytes();
    if (corpus_bytes == 0) return 0;
    return 100.0 *
           static_cast<double>(corpus_bytes -
                               std::min(read, corpus_bytes)) /
           static_cast<double>(corpus_bytes);
  }
};

// Engine: an opened snapshot. Readers/decoders are cached per file and
// reused across queries (single decode context per file — hot-path rule).
class Engine {
 public:
  static Result<Engine> open(const std::string& store_dir, int snapshot = -1);

  // Move-only, and say so explicitly: the reader caches hold unique_ptrs, and
  // libc++'s map claims copyability it cannot deliver — generic code (pybind)
  // must see the deleted copy, not a trait lie.
  Engine(Engine&&) noexcept = default;
  Engine& operator=(Engine&&) noexcept = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const Manifest& manifest() const { return manifest_; }
  const std::string& store_dir() const { return store_dir_; }

  Result<WindowResult> get_window(const WindowQuery& q);

  // Cached-open accessors (also used by the semantic layer + bench).
  Result<SdxReader*> sensor_reader(const SensorStreamEntry& e);
  Result<VideoDecoder*> segment_decoder(const VideoSegmentEntry& seg);
  Result<SfiReader*> segment_sfi(const VideoSegmentEntry& seg);

 private:
  Engine() = default;
  std::string store_dir_;
  Manifest manifest_;
  std::map<std::string, std::unique_ptr<SdxReader>> sdx_cache_;
  std::map<std::string, std::unique_ptr<SfiReader>> sfi_cache_;
  std::map<std::string, std::unique_ptr<VideoDecoder>> dec_cache_;
};

}  // namespace sdx
