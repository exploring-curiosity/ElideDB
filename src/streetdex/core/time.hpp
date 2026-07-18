#pragma once
// All StreetDex timestamps are i64 nanoseconds since Unix epoch, UTC, on the
// dataset's *canonical* timeline (the fitted global clock — see
// ingest/clock_model.hpp). Per-stream clock_offset_ns maps canonical time back
// to each device's local wall clock; it lives in file metadata, never baked
// into stored timestamps (immutability: recalibration must not rewrite data).

#include <cstdint>

namespace sdx {

using TimeNs = int64_t;

inline constexpr TimeNs kNsPerSec = 1'000'000'000;
inline constexpr TimeNs kNsPerMs = 1'000'000;

struct TimeRange {
  TimeNs t0 = 0;  // inclusive
  TimeNs t1 = 0;  // inclusive (windows are closed; a frame AT t1 belongs)

  bool overlaps(TimeNs lo, TimeNs hi) const { return lo <= t1 && hi >= t0; }
  bool contains(TimeNs t) const { return t >= t0 && t <= t1; }
};

}  // namespace sdx
