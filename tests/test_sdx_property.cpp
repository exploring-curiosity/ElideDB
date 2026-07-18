// Property test (FORMAT.md §1.6.6): pruned scan ≡ full scan post-filter, on
// randomized data and randomized windows. If zone maps ever skip a chunk they
// shouldn't, this is the test that catches it.
#include <catch2/catch_test_macros.hpp>

#include <random>

#include "streetdex/format/sdx.hpp"
#include "test_util.hpp"

using namespace sdx;

TEST_CASE("pruned range scan equals naive full-scan filter", "[sdx][property]") {
  TempDir td("sdxprop");
  std::mt19937_64 rng(1234);

  // Random non-decreasing ts with duplicates and gaps; values random.
  const size_t n = 20000;
  std::vector<int64_t> ts(n);
  std::vector<double> val(n);
  int64_t t = 1'000'000;
  std::uniform_int_distribution<int64_t> gap(0, 500);  // 0 => duplicate ts
  std::normal_distribution<double> noise(0.0, 10.0);
  for (size_t i = 0; i < n; ++i) {
    t += gap(rng);
    ts[i] = t;
    val[i] = noise(rng);
  }

  const std::string path = td.str("prop.sdx");
  auto w = SdxWriter::create(path, "prop", "", 0, {{"v", ColType::F64}}, 64);
  REQUIRE(w.has_value());
  const void* cols[] = {val.data()};
  // Append in ragged batches to exercise chunk-boundary splitting.
  size_t done = 0;
  std::uniform_int_distribution<size_t> batch(1, 1000);
  while (done < n) {
    const size_t take = std::min(batch(rng), n - done);
    const void* c[] = {val.data() + done};
    REQUIRE(w->append(take, ts.data() + done, c).has_value());
    done += take;
  }
  (void)cols;
  REQUIRE(w->finish().has_value());

  auto r = SdxReader::open(path);
  REQUIRE(r.has_value());
  REQUIRE(r->meta().row_count == n);

  std::uniform_int_distribution<int64_t> pick(ts.front() - 1000,
                                              ts.back() + 1000);
  size_t nonempty = 0, pruned_queries = 0;
  for (int q = 0; q < 300; ++q) {
    int64_t a = pick(rng), b = pick(rng);
    if (a > b) std::swap(a, b);
    auto scan = r->scan(a, b);
    REQUIRE(scan.has_value());
    // Naive reference
    std::vector<int64_t> want_ts;
    std::vector<double> want_v;
    for (size_t i = 0; i < n; ++i)
      if (ts[i] >= a && ts[i] <= b) {
        want_ts.push_back(ts[i]);
        want_v.push_back(val[i]);
      }
    auto got_ts = SdxReader::gather_i64(*scan, 0);
    auto got_v = SdxReader::gather_f64(*scan, 1, ColType::F64);
    REQUIRE(got_ts == want_ts);
    REQUIRE(got_v == want_v);
    if (!want_ts.empty()) ++nonempty;
    if (scan->chunks_scanned < scan->chunks_total) ++pruned_queries;
  }
  CHECK(nonempty > 50);              // the test actually exercised data
  CHECK(pruned_queries == 300);      // every query pruned something
}
