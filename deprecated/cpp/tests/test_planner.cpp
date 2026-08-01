// End-to-end planner test on a synthetic store: two sensor streams with
// different rates/epochs, manifest, engine, get_window with alignment and
// catalog-level pruning.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>

#include "streetdex/query/planner.hpp"
#include "test_util.hpp"

using namespace sdx;
using Catch::Matchers::WithinAbs;

namespace {

SensorStreamEntry make_stream(const std::string& store, const std::string& id,
                              const std::string& file, TimeNs start,
                              TimeNs step, size_t n, double slope) {
  std::filesystem::create_directories(std::filesystem::path(store) / "sdx");
  const std::string rel = "sdx/" + file;
  auto w = SdxWriter::create((std::filesystem::path(store) / rel).string(), id,
                             "", 0, {{"v", ColType::F64}}, 256);
  REQUIRE(w.has_value());
  std::vector<TimeNs> ts(n);
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i) {
    ts[i] = start + static_cast<TimeNs>(i) * step;
    v[i] = slope * static_cast<double>(ts[i]) / 1e9;  // f(t) = slope * t_sec
  }
  const void* cols[] = {v.data()};
  REQUIRE(w->append(n, ts.data(), cols).has_value());
  REQUIRE(w->finish().has_value());

  SensorStreamEntry e;
  e.stream_id = id;
  e.sdx_path = rel;
  e.min_ts = ts.front();
  e.max_ts = ts.back();
  e.rows = n;
  e.bytes = std::filesystem::file_size(std::filesystem::path(store) / rel);
  e.columns = {"ts", "v"};
  return e;
}

}  // namespace

TEST_CASE("get_window aligns streams and prunes at the catalog stage",
          "[planner]") {
  TempDir td("planner");
  const std::string store = td.str();

  Manifest m;
  m.snapshot = 1;
  m.dataset_dir = "synthetic";
  m.clock = {1200.0, 0, 0};
  // Stream A: 100 Hz over [0, 100 s]. Stream B: 33 Hz over [0, 100 s].
  // Stream C lives at [1000 s, 1100 s] — far outside every query window, so
  // the planner must never even open its footer.
  m.sensors.push_back(make_stream(store, "a", "a.sdx", 0, 10'000'000, 10'000, 2.0));
  m.sensors.push_back(make_stream(store, "b", "b.sdx", 3'000'000, 30'303'030,
                                  3'300, 2.0));
  m.sensors.push_back(make_stream(store, "c", "c.sdx",
                                  1'000'000'000'000LL, 10'000'000, 1'000, 5.0));
  REQUIRE(m.save(store).has_value());

  auto e = Engine::open(store);
  REQUIRE(e.has_value());
  CHECK(e->manifest().sensors.size() == 3);

  WindowQuery q;
  q.t0 = 10'000'000'000LL;  // [10 s, 12 s]
  q.t1 = 12'000'000'000LL;
  q.rate_hz = 10;
  q.interp = Interp::linear;
  auto r = e->get_window(q);
  REQUIRE(r.has_value());

  CHECK(r->timeline.size() == 21);
  CHECK(r->files_pruned == 1);   // stream c skipped via manifest time range
  CHECK(r->files_touched == 2);
  REQUIRE(r->sensors.size() == 2);

  // Both streams sample f(t) = 2t; after linear resampling onto the same
  // timeline they must agree with the analytic value and each other.
  for (size_t i = 0; i < r->timeline.size(); ++i) {
    const double t_sec = static_cast<double>(r->timeline[i]) / 1e9;
    CHECK_THAT(r->sensors[0].values[0][i], WithinAbs(2.0 * t_sec, 1e-6));
    CHECK_THAT(r->sensors[1].values[0][i], WithinAbs(2.0 * t_sec, 1e-6));
  }

  // Elision accounting: bytes read must be far below the corpus and must
  // include zero bytes of stream c.
  CHECK(r->corpus_bytes > 0);
  CHECK(r->io.total_bytes() < r->corpus_bytes / 10);
  CHECK(r->elided_pct() > 90.0);

  // Stream selection narrows work.
  WindowQuery q2 = q;
  q2.streams = {"a"};
  auto r2 = e->get_window(q2);
  REQUIRE(r2.has_value());
  CHECK(r2->sensors.size() == 1);
  CHECK(r2->io.total_bytes() < r->io.total_bytes());

  // Snapshot immutability: saving the same snapshot number again must fail.
  CHECK_FALSE(m.save(store).has_value());
}
