#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "streetdex/align/resample.hpp"

using namespace sdx;

TEST_CASE("timeline generation is drift-free and inclusive", "[align]") {
  auto t = make_timeline(0, 1'000'000'000, 10.0);  // 1s at 10 Hz
  REQUIRE(t.size() == 11);                          // 0..1e9 inclusive
  CHECK(t.front() == 0);
  CHECK(t.back() == 1'000'000'000);
  CHECK(t[3] == 300'000'000);

  CHECK(make_timeline(100, 50, 10).empty());
  CHECK(make_timeline(0, 100, 0).empty());
  // Degenerate window: one point at t0.
  auto one = make_timeline(5, 5, 30);
  REQUIRE(one.size() == 1);
  CHECK(one[0] == 5);
}

TEST_CASE("nearest resample picks closest sample, ties to earlier", "[align]") {
  const std::vector<TimeNs> ts = {0, 100, 200};
  const std::vector<double> v = {1.0, 2.0, 3.0};
  const std::vector<TimeNs> q = {-50, 0, 49, 50, 51, 150, 220, 999};
  auto out = resample(ts, v, q, Interp::nearest);
  // -50 holds edge; 50 is equidistant -> earlier sample (1.0)
  const std::vector<double> want = {1, 1, 1, 1, 2, 2, 3, 3};
  CHECK(out == want);
}

TEST_CASE("linear resample interpolates and clamps edges", "[align]") {
  const std::vector<TimeNs> ts = {0, 100};
  const std::vector<double> v = {0.0, 10.0};
  const std::vector<TimeNs> q = {-10, 0, 25, 50, 100, 150};
  auto out = resample(ts, v, q, Interp::linear);
  CHECK(out[0] == 0.0);   // clamp before
  CHECK(out[1] == 0.0);
  CHECK(out[2] == 2.5);
  CHECK(out[3] == 5.0);
  CHECK(out[4] == 10.0);
  CHECK(out[5] == 10.0);  // clamp after
}

TEST_CASE("resample with synthetic clock skew recovers alignment", "[align]") {
  // Two streams sampling the same physical ramp signal f(t)=t/1e6, one at
  // 100 Hz, one at 33 Hz with phase offset. After resampling both to a
  // common 10 Hz timeline, they must agree to within one sample interval's
  // worth of slope — the unit-level version of the TV-clock sync test.
  std::vector<TimeNs> ts_a, ts_b;
  std::vector<double> va, vb;
  for (int i = 0; i < 500; ++i) {
    const TimeNs t = static_cast<TimeNs>(i) * 10'000'000;  // 100 Hz
    ts_a.push_back(t);
    va.push_back(static_cast<double>(t) / 1e6);
  }
  for (int i = 0; i < 165; ++i) {
    const TimeNs t = 7'000'000 + static_cast<TimeNs>(i) * 30'303'030;  // ~33 Hz
    ts_b.push_back(t);
    vb.push_back(static_cast<double>(t) / 1e6);
  }
  auto q = make_timeline(100'000'000, 4'000'000'000, 10.0);
  auto ra = resample(ts_a, va, q, Interp::linear);
  auto rb = resample(ts_b, vb, q, Interp::linear);
  for (size_t i = 0; i < q.size(); ++i) {
    CHECK(std::abs(ra[i] - rb[i]) < 1.0);  // < 1 ms of signal disagreement
  }
}

TEST_CASE("empty input yields NaN, nearest_indices maps frames", "[align]") {
  auto out = resample({}, {}, std::vector<TimeNs>{1, 2}, Interp::nearest);
  REQUIRE(out.size() == 2);
  CHECK(std::isnan(out[0]));

  const std::vector<TimeNs> fts = {100, 200, 300};
  auto idx = nearest_indices(fts, std::vector<TimeNs>{0, 149, 151, 300, 400});
  CHECK(idx == std::vector<int32_t>{0, 0, 1, 2, 2});
  auto none = nearest_indices({}, std::vector<TimeNs>{1});
  CHECK(none == std::vector<int32_t>{-1});
}
