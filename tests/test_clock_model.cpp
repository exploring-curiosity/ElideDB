#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

#include "streetdex/ingest/clock_model.hpp"

using namespace sdx;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("tick-rate fit recovers a known rate through jitter", "[clock]") {
  // Simulate the rig: global clock at 1200.2 Hz; two sensors whose python
  // clocks have wildly different epochs (years apart) and scheduling jitter
  // that is always LATE (one-sided), like real dispatch delay.
  std::mt19937_64 rng(7);
  std::exponential_distribution<double> late(1.0 / 0.002);  // ~2ms mean delay
  const double rate = 1200.2;
  auto make_sensor = [&](double epoch, int64_t tick0) {
    std::vector<ClockPair> p;
    for (int i = 0; i < 2000; ++i) {
      const double t = i * 0.04;  // 25 Hz of pairs over 80 s
      p.push_back({epoch + t + late(rng),
                   tick0 + static_cast<int64_t>(t * rate)});
    }
    return p;
  };
  auto s1 = make_sensor(1678894151.0, 2800000);
  auto s2 = make_sensor(1767168354.0, 2802000);  // "wrong year" sensor

  auto r1 = fit_tick_rate(s1);
  REQUIRE(r1.has_value());
  CHECK_THAT(*r1, WithinRel(rate, 1e-3));

  auto model = build_clock_model({s1, s2}, 1'000'000'000'000'000'000LL);
  REQUIRE(model.has_value());
  CHECK_THAT(model->rate_hz, WithinRel(rate, 1e-3));
  CHECK(model->anchor_ticks == 2800000);
  CHECK(model->anchor_ns == 1'000'000'000'000'000'000LL);

  // Canonical times of the same tick agree regardless of which sensor saw it,
  // and offsets recover each sensor's (very different) local epoch.
  const TimeNs off1 = fit_clock_offset(*model, s1);
  const TimeNs off2 = fit_clock_offset(*model, s2);
  const double off_diff_s = static_cast<double>(off2 - off1) / 1e9;
  // s2's epoch is later by (1767168354-1678894151) s minus the tick0 gap
  // (2000 ticks ≈ 1.666 s), plus jitter medians which mostly cancel.
  const double want =
      (1767168354.0 - 1678894151.0) - 2000.0 / rate;
  CHECK_THAT(off_diff_s, WithinAbs(want, 0.01));
}

TEST_CASE("divergent per-sensor rates are refused", "[clock]") {
  std::vector<ClockPair> a, b;
  for (int i = 0; i < 100; ++i) {
    a.push_back({i * 0.1, static_cast<int64_t>(i * 120)});   // 1200 Hz
    b.push_back({i * 0.1, static_cast<int64_t>(i * 150)});   // 1500 Hz!
  }
  auto model = build_clock_model({a, b}, 0);
  REQUIRE_FALSE(model.has_value());
  CHECK(model.error().code == Errc::bad_format);
}

TEST_CASE("dataset folder time parses as UTC", "[clock]") {
  auto t = parse_dataset_folder_time("20260403_181625");
  REQUIRE(t.has_value());
  CHECK(*t == 1775240185LL * 1'000'000'000LL);  // 2026-04-03T18:16:25Z
  CHECK_FALSE(parse_dataset_folder_time("garbage").has_value());
  CHECK_FALSE(parse_dataset_folder_time("2026040_1816251").has_value());
}
