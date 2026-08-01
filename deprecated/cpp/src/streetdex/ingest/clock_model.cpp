#include "streetdex/ingest/clock_model.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>

namespace sdx {

Result<double> fit_tick_rate(const std::vector<ClockPair>& pairs) {
  if (pairs.size() < 2)
    return fail(Errc::bad_argument, "need >= 2 clock pairs to fit a rate");
  // Ordinary least squares of ticks on python seconds, centered for numeric
  // stability (python epochs are ~1.7e9; ticks ~2.8e6).
  double mean_t = 0, mean_g = 0;
  for (const auto& p : pairs) {
    mean_t += p.python_ts;
    mean_g += static_cast<double>(p.global_ticks);
  }
  mean_t /= static_cast<double>(pairs.size());
  mean_g /= static_cast<double>(pairs.size());
  double sxx = 0, sxy = 0;
  for (const auto& p : pairs) {
    const double dt = p.python_ts - mean_t;
    const double dg = static_cast<double>(p.global_ticks) - mean_g;
    sxx += dt * dt;
    sxy += dt * dg;
  }
  if (sxx <= 0)
    return fail(Errc::bad_argument, "degenerate clock pairs (zero time span)");
  return sxy / sxx;
}

Result<ClockModel> build_clock_model(
    const std::vector<std::vector<ClockPair>>& per_sensor_pairs,
    TimeNs anchor_wall_ns) {
  double weighted_rate = 0;
  double total_weight = 0;
  int64_t min_ticks = std::numeric_limits<int64_t>::max();
  for (const auto& pairs : per_sensor_pairs) {
    if (pairs.empty()) continue;
    auto rate = fit_tick_rate(pairs);
    if (!rate) return tl::unexpected(rate.error());
    // Sanity: all sensors observe the SAME physical clock; wildly divergent
    // estimates mean a data problem, not averaging material.
    if (total_weight > 0 &&
        std::abs(*rate - weighted_rate / total_weight) >
            0.01 * (weighted_rate / total_weight))
      return fail(Errc::bad_format,
                  "per-sensor tick-rate estimates disagree by >1% — global "
                  "clock assumption violated");
    const double w = static_cast<double>(pairs.size());
    weighted_rate += *rate * w;
    total_weight += w;
    for (const auto& p : pairs) min_ticks = std::min(min_ticks, p.global_ticks);
  }
  if (total_weight == 0)
    return fail(Errc::bad_argument, "no clock pairs from any sensor");
  ClockModel m;
  m.rate_hz = weighted_rate / total_weight;
  m.anchor_ticks = min_ticks;
  m.anchor_ns = anchor_wall_ns;
  return m;
}

TimeNs fit_clock_offset(const ClockModel& model,
                        const std::vector<ClockPair>& pairs) {
  // Median, not mean: python timestamps carry scheduling-delay outliers
  // (always late, never early), so the median is the honest center.
  std::vector<TimeNs> deltas;
  deltas.reserve(pairs.size());
  for (const auto& p : pairs) {
    const TimeNs py_ns = static_cast<TimeNs>(p.python_ts * 1e9);
    deltas.push_back(py_ns - model.to_canonical_ns(p.global_ticks));
  }
  if (deltas.empty()) return 0;
  const size_t mid = deltas.size() / 2;
  std::nth_element(deltas.begin(), deltas.begin() + mid, deltas.end());
  return deltas[mid];
}

Result<TimeNs> parse_dataset_folder_time(const std::string& name) {
  // "YYYYMMDD_HHMMSS"
  if (name.size() != 15 || name[8] != '_')
    return fail(Errc::bad_argument, "dataset folder not YYYYMMDD_HHMMSS: " + name);
  std::tm tm{};
  tm.tm_year = std::stoi(name.substr(0, 4)) - 1900;
  tm.tm_mon = std::stoi(name.substr(4, 2)) - 1;
  tm.tm_mday = std::stoi(name.substr(6, 2));
  tm.tm_hour = std::stoi(name.substr(9, 2));
  tm.tm_min = std::stoi(name.substr(11, 2));
  tm.tm_sec = std::stoi(name.substr(13, 2));
  const time_t t = timegm(&tm);  // interpret as UTC; a labeling choice,
                                 // recorded in the manifest for transparency
  if (t == static_cast<time_t>(-1))
    return fail(Errc::bad_argument, "unparseable dataset time: " + name);
  return static_cast<TimeNs>(t) * kNsPerSec;
}

}  // namespace sdx
