#pragma once
// Query-time alignment. Storage never commits to a rate or an interpolation
// policy — raw samples keep their native timestamps forever, and every query
// chooses how to put streams on a common timeline. This is the "late
// (query-time) alignment" design position: alignment is a *view*, not a
// storage transform, so a better policy later never requires re-ingest.

#include <span>
#include <vector>

#include "streetdex/core/time.hpp"

namespace sdx {

enum class Interp {
  nearest,  // sample-and-hold to the closest sample in time
  linear,   // piecewise-linear between neighbors, clamped at the edges
};

// Build the query timeline: t0, t0+dt, ... <= t1 at rate_hz.
std::vector<TimeNs> make_timeline(TimeNs t0, TimeNs t1, double rate_hz);

// Evaluate a sampled signal (ts sorted ascending, one value per ts) at query
// times q (sorted). Outside [ts.front(), ts.back()] the edge value holds —
// NaN would poison downstream math, and "most recent reading" is the honest
// sensor semantic. Empty input yields NaN everywhere (nothing to hold).
std::vector<double> resample(std::span<const TimeNs> ts,
                             std::span<const double> values,
                             std::span<const TimeNs> q, Interp interp);

// Nearest-index lookup used to attach video frames to timeline points:
// result[i] = index into ts of the sample closest to q[i], or -1 if ts empty.
std::vector<int32_t> nearest_indices(std::span<const TimeNs> ts,
                                     std::span<const TimeNs> q);

}  // namespace sdx
