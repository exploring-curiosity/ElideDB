#include "streetdex/align/resample.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sdx {

std::vector<TimeNs> make_timeline(TimeNs t0, TimeNs t1, double rate_hz) {
  std::vector<TimeNs> out;
  if (rate_hz <= 0 || t1 < t0) return out;
  const double dt = 1e9 / rate_hz;
  // Index-based generation, not repeated addition: no drift accumulation
  // over long windows.
  const auto n = static_cast<size_t>(
      std::floor(static_cast<double>(t1 - t0) / dt)) + 1;
  out.reserve(n);
  for (size_t k = 0; k < n; ++k) {
    const TimeNs t = t0 + static_cast<TimeNs>(std::llround(k * dt));
    if (t > t1) break;
    out.push_back(t);
  }
  return out;
}

std::vector<double> resample(std::span<const TimeNs> ts,
                             std::span<const double> values,
                             std::span<const TimeNs> q, Interp interp) {
  std::vector<double> out(q.size(),
                          std::numeric_limits<double>::quiet_NaN());
  if (ts.empty()) return out;
  // Both ts and q are sorted: a single forward cursor makes the whole
  // resample O(n + m) instead of m binary searches.
  size_t hi = 0;  // first sample with ts[hi] >= q[i]
  for (size_t i = 0; i < q.size(); ++i) {
    const TimeNs t = q[i];
    while (hi < ts.size() && ts[hi] < t) ++hi;
    if (hi == 0) {
      out[i] = values[0];  // before first sample: hold the edge
      continue;
    }
    if (hi == ts.size()) {
      out[i] = values[ts.size() - 1];  // after last sample: hold the edge
      continue;
    }
    const size_t lo = hi - 1;
    if (interp == Interp::nearest) {
      out[i] = (t - ts[lo] <= ts[hi] - t) ? values[lo] : values[hi];
    } else {
      const auto span_ns = static_cast<double>(ts[hi] - ts[lo]);
      const double w =
          span_ns == 0 ? 0.0 : static_cast<double>(t - ts[lo]) / span_ns;
      out[i] = values[lo] + w * (values[hi] - values[lo]);
    }
  }
  return out;
}

std::vector<int32_t> nearest_indices(std::span<const TimeNs> ts,
                                     std::span<const TimeNs> q) {
  std::vector<int32_t> out(q.size(), -1);
  if (ts.empty()) return out;
  size_t hi = 0;
  for (size_t i = 0; i < q.size(); ++i) {
    const TimeNs t = q[i];
    while (hi < ts.size() && ts[hi] < t) ++hi;
    if (hi == 0) {
      out[i] = 0;
    } else if (hi == ts.size()) {
      out[i] = static_cast<int32_t>(ts.size() - 1);
    } else {
      out[i] = (t - ts[hi - 1] <= ts[hi] - t) ? static_cast<int32_t>(hi - 1)
                                              : static_cast<int32_t>(hi);
    }
  }
  return out;
}

}  // namespace sdx
