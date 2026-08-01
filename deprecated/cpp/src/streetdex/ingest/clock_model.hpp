#pragma once
// Canonical timeline: the shared global clock, characterized at ingest.
//
// The capture rig stamps every frame with two clocks:
//   - python_timestamp: per-sensor wall clock. Unsynced across sensors (in
//     this dataset they disagree by *years* — one sensor thinks it's 2023,
//     another 2025).
//   - global_timestamp: a hardware-distributed shared counter. Agrees across
//     sensors, but its unit is ticks of an unspecified rate (~1200.2 Hz here).
//
// Storage position: all stored timestamps are ns on the canonical timeline
//   canonical_ns(ticks) = anchor_ns + (ticks - anchor_ticks) * 1e9 / rate_hz
// where rate_hz is FITTED from (python_ts, ticks) pairs (each sensor's python
// clock is a fine short-term frequency reference even though its epoch is
// wrong), and the anchor pins the earliest tick to the dataset's folder
// wall time — the only trustworthy absolute time we have. The fit lives in
// the manifest; per-stream clock_offset_ns (local minus canonical) lives in
// each file's metadata so local clocks are recoverable. Raw files are never
// rewritten to "fix" time — recalibration is a metadata operation.

#include <cstdint>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/time.hpp"

namespace sdx {

struct ClockModel {
  double rate_hz = 0;        // global ticks per second
  int64_t anchor_ticks = 0;  // tick value that maps to anchor_ns
  TimeNs anchor_ns = 0;      // canonical epoch ns at anchor_ticks

  TimeNs to_canonical_ns(int64_t ticks) const {
    return anchor_ns +
           static_cast<TimeNs>(static_cast<double>(ticks - anchor_ticks) *
                               (1e9 / rate_hz));
  }
};

struct ClockPair {
  double python_ts;     // seconds, sensor-local
  int64_t global_ticks;
};

// Least-squares slope d(ticks)/d(python_sec) for one sensor's pairs.
// Returns ticks-per-second; each sensor contributes an independent estimate.
Result<double> fit_tick_rate(const std::vector<ClockPair>& pairs);

// Pool per-sensor estimates (weighted by sample count) into one dataset-wide
// rate, then anchor the earliest tick to `anchor_wall_ns`.
Result<ClockModel> build_clock_model(
    const std::vector<std::vector<ClockPair>>& per_sensor_pairs,
    TimeNs anchor_wall_ns);

// Median of (python_ns - canonical_ns) over a stream's pairs — the per-stream
// clock_offset_ns stored in SDX/SFI metadata.
TimeNs fit_clock_offset(const ClockModel& model,
                        const std::vector<ClockPair>& pairs);

// Parse "YYYYMMDD_HHMMSS" (dataset folder name) as UTC to epoch ns.
Result<TimeNs> parse_dataset_folder_time(const std::string& name);

}  // namespace sdx
