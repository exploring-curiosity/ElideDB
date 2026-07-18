#pragma once
// Dataset ingest orchestrator for REIP-style captures:
//
//   <dataset>/Sensor <id>/{video,time,audio,logs}
//
// Produces, WITHOUT touching the raw files:
//   - a fitted clock model (canonical timeline) for the whole dataset
//   - one SFI per present AVI segment (pts joined from time/ JSONs)
//   - one telemetry SDX per camera stream (the time/ data as queryable rows)
//   - one audio SDX per sensor (16ch PCM as a high-rate sensor stream)
//   - manifest snapshot 1
//
// Ingest is restartable-by-rerun (files are deterministic), but v1 keeps it
// simple: rerun into a fresh store.

#include <string>

#include "streetdex/catalog/manifest.hpp"
#include "streetdex/core/error.hpp"
#include "streetdex/format/sdx.hpp"

namespace sdx {

struct IngestOptions {
  std::string dataset_dir;
  std::string store_dir;
  bool ingest_audio = true;
  bool ingest_telemetry = true;
  uint32_t chunk_target_rows = kDefaultChunkTargetRows;
  bool verbose = true;
};

Result<Manifest> ingest_dataset(const IngestOptions& opts);

}  // namespace sdx
