// BPT1: bulk build, descent correctness vs std::lower_bound on the raw
// array (property test), duplicates, float key encoding, torn files.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <fstream>
#include <random>

#include "streetdex/index/bptree.hpp"
#include "test_util.hpp"

using namespace sdx;

TEST_CASE("B+ tree lookup == flat binary search (property)", "[bptree]") {
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int64_t> keys(-1'000'000, 1'000'000);
  std::vector<BptPair> pairs;
  for (uint64_t i = 0; i < 300'000; ++i)
    pairs.push_back({keys(rng), i});
  std::sort(pairs.begin(), pairs.end(),
            [](auto& a, auto& b) { return a.key < b.key; });

  auto img = bpt_build(pairs, 64);  // small order => 3+ levels exercised
  auto r = BptReader::from_bytes(img);
  REQUIRE(r.has_value());
  CHECK(r->size() == pairs.size());
  CHECK(r->levels() >= 2);

  for (int q = 0; q < 5'000; ++q) {
    const int64_t k = keys(rng);
    const auto want = std::lower_bound(
        pairs.begin(), pairs.end(), k,
        [](const BptPair& p, int64_t v) { return p.key < v; });
    REQUIRE(r->lower_bound(k) ==
            static_cast<uint64_t>(want - pairs.begin()));
  }
  // extremes
  CHECK(r->lower_bound(INT64_MIN) == 0);
  CHECK(r->lower_bound(INT64_MAX) == pairs.size());
}

TEST_CASE("range scan returns every duplicate", "[bptree]") {
  std::vector<BptPair> pairs;
  for (uint64_t i = 0; i < 1000; ++i)
    pairs.push_back({static_cast<int64_t>(i / 10), i});  // 10 dups per key
  auto img = bpt_build(pairs, 16);
  auto r = BptReader::from_bytes(img);
  REQUIRE(r.has_value());
  CHECK(r->range(5, 7).size() == 30);
  CHECK(r->range(0, 99).size() == 1000);
  CHECK(r->range(1000, 2000).empty());
  CHECK(r->find_first(42).has_value());
  CHECK_FALSE(r->find_first(-1).has_value());
}

TEST_CASE("duplicates spanning a node boundary are all found", "[bptree]") {
  // Construct the adversarial layout directly: with order 4, key 7 runs
  // across the boundary between two leaf nodes AND seeds the next node's
  // first-key. The old upper_bound descent skipped the earlier node.
  std::vector<BptPair> pairs;
  for (int64_t k : {1, 2, 3, 7, 7, 7, 7, 7, 9, 11, 12, 13})
    pairs.push_back({k, static_cast<uint64_t>(pairs.size())});
  auto img = bpt_build(pairs, 4);
  auto r = BptReader::from_bytes(img);
  REQUIRE(r.has_value());
  CHECK(r->lower_bound(7) == 3);        // FIRST 7, in the first node
  CHECK(r->range(7, 7).size() == 5);    // every duplicate
  CHECK(r->range(0, 100).size() == 12);
}

TEST_CASE("double keys keep their order through encoding", "[bptree]") {
  std::vector<double> vals = {-1e300, -3.5, -0.0, 0.0, 1e-9, 2.5, 7e12};
  for (size_t i = 1; i < vals.size(); ++i)
    REQUIRE(bpt_encode_f64(vals[i - 1]) <= bpt_encode_f64(vals[i]));
}

TEST_CASE("mmap round-trip and corruption rejection", "[bptree]") {
  TempDir td("bpt");
  std::vector<BptPair> pairs;
  for (uint64_t i = 0; i < 10'000; ++i)
    pairs.push_back({static_cast<int64_t>(i * 3), i});
  auto img = bpt_build(pairs);
  const auto path = td.str("ix.bpt");
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(img.data()),
             static_cast<std::streamsize>(img.size()));
  auto r = BptReader::open(path);
  REQUIRE(r.has_value());
  CHECK(r->find_first(29'997).value() == 9'999);

  img[1] = 'X';  // break the magic
  auto bad = BptReader::from_bytes(img);
  REQUIRE_FALSE(bad.has_value());
}
