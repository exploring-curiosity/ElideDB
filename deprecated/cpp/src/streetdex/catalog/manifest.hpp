#pragma once
// Catalog: Iceberg-lite. A store directory holds immutable data/index files
// plus numbered manifest snapshots and a CURRENT pointer:
//
//   store/
//     manifests/manifest-<N>.json   (immutable once written)
//     CURRENT                       (text: the current snapshot number)
//     sfi/*.sfi   sdx/*.sdx   ml/<run>/...
//
// A snapshot answers "what exactly did the corpus contain": every data file,
// every index file, their time ranges (catalog-level pruning), the fitted
// clock model (reproducibility of the timeline itself), and the semantic run
// id. Time travel = load an older manifest; nothing is ever rewritten.

#include <optional>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/time.hpp"
#include "streetdex/ingest/clock_model.hpp"

namespace sdx {

struct VideoSegmentEntry {
  std::string source_path;  // absolute; raw data is immutable, never copied
  std::string sfi_path;     // relative to store dir
  TimeNs first_pts_ns = 0;
  TimeNs last_pts_ns = 0;
  uint64_t source_bytes = 0;
  uint64_t frame_count = 0;
};

struct VideoStreamEntry {
  std::string stream_id;  // e.g. "Sensor 108/cam0"
  TimeNs clock_offset_ns = 0;
  int width = 0, height = 0, fps = 0;
  std::vector<VideoSegmentEntry> segments;  // sorted by first_pts_ns
};

struct SensorStreamEntry {
  std::string stream_id;  // e.g. "Sensor 108/audio"
  std::string sdx_path;   // relative to store dir
  TimeNs min_ts = 0, max_ts = 0;
  uint64_t rows = 0;
  uint64_t bytes = 0;
  std::vector<std::string> columns;
};

struct SemanticRun {
  std::string run_id;      // subdir of store/ml/
  std::string model;       // embedding model id — results are meaningless
                           // without it, so it is part of the snapshot
  int dim = 0;
  uint64_t window_count = 0;
};

struct Manifest {
  int snapshot = 0;
  std::string created_utc;
  std::string dataset_dir;
  ClockModel clock;
  std::vector<VideoStreamEntry> video;
  std::vector<SensorStreamEntry> sensors;
  std::optional<SemanticRun> semantic;

  TimeNs min_ts() const;
  TimeNs max_ts() const;
  uint64_t corpus_bytes() const;  // the elision denominator

  Result<void> save(const std::string& store_dir) const;  // writes manifest-N
                                                          // and updates CURRENT
  static Result<Manifest> load_file(const std::string& path);
  // snapshot < 0 => CURRENT.
  static Result<Manifest> load_store(const std::string& store_dir,
                                     int snapshot = -1);
};

}  // namespace sdx
