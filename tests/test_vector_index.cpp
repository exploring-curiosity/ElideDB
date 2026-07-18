// Vector index: exact scan, cluster-pruned scan, recall, clip pooling.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <fstream>
#include <random>

#include "streetdex/index/vector_index.hpp"
#include "test_util.hpp"

using namespace sdx;

namespace {

// Three well-separated Gaussian blobs on the unit sphere (d=8), plus noise
// points, mirroring what HDBSCAN hands us: labeled clusters + label -1.
struct Fixture {
  std::vector<std::vector<float>> vecs;
  std::vector<int32_t> labels;
  std::vector<std::vector<float>> centroids;
};

Fixture make_fixture(int per_cluster, std::mt19937_64& rng) {
  const int d = 8;
  Fixture fx;
  std::normal_distribution<float> jitter(0.0f, 0.05f);
  const float axes[3][8] = {{1, 0, 0, 0, 0, 0, 0, 0},
                            {0, 1, 0, 0, 0, 0, 0, 0},
                            {0, 0, 1, 0, 0, 0, 0, 0}};
  fx.centroids.resize(3);
  for (int c = 0; c < 3; ++c) {
    fx.centroids[c].assign(axes[c], axes[c] + d);
    for (int i = 0; i < per_cluster; ++i) {
      std::vector<float> v(d);
      float norm = 0;
      for (int k = 0; k < d; ++k) {
        v[k] = axes[c][k] + jitter(rng);
        norm += v[k] * v[k];
      }
      norm = std::sqrt(norm);
      for (auto& x : v) x /= norm;
      fx.vecs.push_back(std::move(v));
      fx.labels.push_back(c);
    }
  }
  // a few noise points near a fourth direction
  for (int i = 0; i < 5; ++i) {
    std::vector<float> v(d, 0.0f);
    v[3] = 1.0f;
    v[4] = jitter(rng);
    float norm = std::sqrt(v[3] * v[3] + v[4] * v[4]);
    v[3] /= norm;
    v[4] /= norm;
    fx.vecs.push_back(std::move(v));
    fx.labels.push_back(-1);
  }
  return fx;
}

void write_run(const TempDir& td, const Fixture& fx) {
  const int d = 8;
  std::ofstream emb(td.str("embeddings.f32"), std::ios::binary);
  for (const auto& v : fx.vecs)
    emb.write(reinterpret_cast<const char*>(v.data()), d * 4);
  std::ofstream win(td.str("windows.json"));
  win << "{\"dim\": 8, \"windows\": [";
  for (size_t i = 0; i < fx.vecs.size(); ++i)
    win << (i ? "," : "") << "{\"stream_id\": \"cam0\", \"t0_ns\": "
        << i * 1000 << ", \"t1_ns\": " << (i * 1000 + 999) << "}";
  win << "]}";
  std::ofstream cl(td.str("clusters.json"));
  cl << "{\"labels\": [";
  for (size_t i = 0; i < fx.labels.size(); ++i)
    cl << (i ? "," : "") << fx.labels[i];
  cl << "], \"centroids\": [";
  for (size_t c = 0; c < fx.centroids.size(); ++c) {
    cl << (c ? "," : "") << "[";
    for (int k = 0; k < d; ++k)
      cl << (k ? "," : "") << fx.centroids[c][k];
    cl << "]";
  }
  cl << "]}";
}

}  // namespace

TEST_CASE("pruned search matches exact search on clustered data", "[vindex]") {
  TempDir td("vix");
  std::mt19937_64 rng(99);
  auto fx = make_fixture(50, rng);
  write_run(td, fx);

  auto ix = VectorIndex::open(td.str());
  REQUIRE(ix.has_value());
  CHECK(ix->size() == 155);
  CHECK(ix->dim() == 8);

  // Query near cluster 1's axis.
  std::vector<float> q(8, 0.0f);
  q[1] = 1.0f;
  SearchStats st{};
  auto pruned = ix->search(q.data(), 10, 1, &st);
  REQUIRE(pruned.has_value());
  REQUIRE(pruned->size() == 10);
  CHECK(st.clusters_probed == 1);
  // nprobe=1 scans one cell + noise, far less than everything
  CHECK(st.vectors_scanned == 55);

  auto exact = ix->search_exact(q.data(), 10, nullptr);
  REQUIRE(exact.has_value());
  size_t overlap = 0;
  for (const auto& e : *exact)
    for (const auto& p : *pruned)
      if (e.window_id == p.window_id) ++overlap;
  CHECK(overlap == 10);  // clean blobs: pruning loses nothing

  // Noise windows stay findable (never silently dropped by the prune).
  std::vector<float> qn(8, 0.0f);
  qn[3] = 1.0f;
  auto hits = ix->search(qn.data(), 3, 1, nullptr);
  REQUIRE(hits.has_value());
  CHECK(ix->windows()[(*hits)[0].window_id].t0 >= 150 * 1000);  // a noise row

  // Clip pooling: mean of cluster-2 windows points at cluster 2.
  const TimeNs t0 = 100 * 1000, t1 = 130 * 1000;  // rows 100..130 = cluster 2
  auto pooled = ix->pool_window("cam0", t0, t1);
  REQUIRE(pooled.has_value());
  CHECK((*pooled)[2] > 0.99f);

  CHECK_FALSE(ix->pool_window("nope", 0, 10).has_value());
}

TEST_CASE("missing clusters.json degrades to exact scan", "[vindex]") {
  TempDir td("vixflat");
  std::mt19937_64 rng(5);
  auto fx = make_fixture(10, rng);
  write_run(td, fx);
  std::filesystem::remove(td.path / "clusters.json");

  auto ix = VectorIndex::open(td.str());
  REQUIRE(ix.has_value());
  std::vector<float> q(8, 0.0f);
  q[0] = 1.0f;
  SearchStats st{};
  auto hits = ix->search(q.data(), 5, 3, &st);
  REQUIRE(hits.has_value());
  CHECK(st.vectors_scanned == ix->size());  // full scan
  REQUIRE(hits->size() == 5);
}
