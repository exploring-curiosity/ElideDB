#include "streetdex/index/vector_index.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <queue>

#include <nlohmann/json.hpp>

#include "streetdex/metrics/metrics.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace sdx {

namespace {

float dot(const float* a, const float* b, int d) {
  // Accelerate BLAS: NEON-vectorized single-precision dot product. At d=768
  // and <=100k windows a flat scan is well under 10 ms — exact beats
  // approximate until scale forces the trade (HNSW is the v2 answer).
  return cblas_sdot(d, a, 1, b, 1);
}

void top_k_push(std::vector<SearchHit>& heap, int k, SearchHit h) {
  // min-heap on score, capped at k
  auto cmp = [](const SearchHit& a, const SearchHit& b) {
    return a.score > b.score;
  };
  if (static_cast<int>(heap.size()) < k) {
    heap.push_back(h);
    std::push_heap(heap.begin(), heap.end(), cmp);
  } else if (h.score > heap.front().score) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    heap.back() = h;
    std::push_heap(heap.begin(), heap.end(), cmp);
  }
}

std::vector<SearchHit> finish_heap(std::vector<SearchHit> heap) {
  std::sort(heap.begin(), heap.end(),
            [](const SearchHit& a, const SearchHit& b) {
              return a.score > b.score;
            });
  return heap;
}

}  // namespace

Result<VectorIndex> VectorIndex::open(const std::string& run_dir) {
  VectorIndex ix;
  // windows.json: {"dim": d, "windows": [{"stream_id","t0_ns","t1_ns"},...]}
  {
    std::ifstream in(fs::path(run_dir) / "windows.json");
    if (!in) return fail(Errc::not_found, "no windows.json in " + run_dir);
    json j;
    try {
      in >> j;
      ix.dim_ = j.at("dim").get<int>();
      for (const auto& w : j.at("windows")) {
        WindowMeta m;
        m.stream_id = w.at("stream_id").get<std::string>();
        m.t0 = w.at("t0_ns").get<int64_t>();
        m.t1 = w.at("t1_ns").get<int64_t>();
        ix.windows_.push_back(std::move(m));
      }
    } catch (const json::exception& e) {
      return fail(Errc::bad_format, std::string("bad windows.json: ") + e.what());
    }
  }
  ix.n_ = ix.windows_.size();

  auto emb = MmapFile::open((fs::path(run_dir) / "embeddings.f32").string());
  if (!emb) return tl::unexpected(emb.error());
  ix.emb_ = std::move(*emb);
  if (ix.emb_.size() != ix.n_ * static_cast<size_t>(ix.dim_) * 4)
    return fail(Errc::bad_format,
                "embeddings.f32 size mismatch vs windows.json (n=" +
                    std::to_string(ix.n_) + ", d=" + std::to_string(ix.dim_) +
                    ")");

  // clusters.json: {"labels": [...], "centroids": [[...], ...]} — optional;
  // without it every search degrades to the exact flat scan.
  {
    std::ifstream in(fs::path(run_dir) / "clusters.json");
    if (in) {
      json j;
      try {
        in >> j;
        ix.labels_ = j.at("labels").get<std::vector<int32_t>>();
        for (const auto& c : j.at("centroids"))
          ix.centroids_.push_back(c.get<std::vector<float>>());
      } catch (const json::exception& e) {
        return fail(Errc::bad_format,
                    std::string("bad clusters.json: ") + e.what());
      }
      if (ix.labels_.size() != ix.n_)
        return fail(Errc::bad_format, "clusters.json label count mismatch");
      ix.members_.resize(ix.centroids_.size());
      for (uint32_t i = 0; i < ix.n_; ++i) {
        const int32_t l = ix.labels_[i];
        if (l >= 0 && l < static_cast<int32_t>(ix.centroids_.size()))
          ix.members_[l].push_back(i);
        else
          ix.noise_.push_back(i);
      }
      // Normalize centroids so centroid ranking is also a pure dot product.
      for (auto& c : ix.centroids_) {
        float norm = std::sqrt(dot(c.data(), c.data(), ix.dim_));
        if (norm > 0)
          for (auto& v : c) v /= norm;
      }
    }
  }
  return ix;
}

Result<std::vector<SearchHit>> VectorIndex::search_exact(
    const float* query, int k, SearchStats* stats) const {
  std::vector<SearchHit> heap;
  for (uint32_t i = 0; i < n_; ++i)
    top_k_push(heap, k, {i, dot(query, vector(i), dim_)});
  metrics::count(metrics::Cat::vector_data, n_ * dim_ * 4);
  if (stats != nullptr) {
    stats->vectors_scanned = n_;
    stats->vectors_total = n_;
    stats->clusters_probed = static_cast<uint32_t>(centroids_.size());
    stats->clusters_total = static_cast<uint32_t>(centroids_.size());
  }
  return finish_heap(std::move(heap));
}

Result<std::vector<SearchHit>> VectorIndex::search(const float* query, int k,
                                                   int nprobe,
                                                   SearchStats* stats) const {
  if (centroids_.empty()) return search_exact(query, k, stats);

  // Stage 1: rank cells by centroid similarity, keep nprobe best.
  std::vector<std::pair<float, uint32_t>> ranked;
  ranked.reserve(centroids_.size());
  for (uint32_t c = 0; c < centroids_.size(); ++c)
    ranked.emplace_back(dot(query, centroids_[c].data(), dim_), c);
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  const size_t probe = std::min<size_t>(nprobe < 1 ? 1 : nprobe, ranked.size());

  // Stage 2: exact rank within surviving cells (+ the noise remainder).
  std::vector<SearchHit> heap;
  uint64_t scanned = 0;
  for (size_t r = 0; r < probe; ++r)
    for (uint32_t id : members_[ranked[r].second]) {
      top_k_push(heap, k, {id, dot(query, vector(id), dim_)});
      ++scanned;
    }
  for (uint32_t id : noise_) {
    top_k_push(heap, k, {id, dot(query, vector(id), dim_)});
    ++scanned;
  }
  metrics::count(metrics::Cat::vector_data,
                 (scanned + centroids_.size()) * dim_ * 4);
  if (stats != nullptr) {
    stats->vectors_scanned = scanned;
    stats->vectors_total = n_;
    stats->clusters_probed = static_cast<uint32_t>(probe);
    stats->clusters_total = static_cast<uint32_t>(centroids_.size());
  }
  return finish_heap(std::move(heap));
}

Result<std::vector<float>> VectorIndex::pool_window(
    const std::string& stream_id, TimeNs t0, TimeNs t1) const {
  std::vector<float> acc(dim_, 0.0f);
  size_t hits = 0;
  for (uint32_t i = 0; i < n_; ++i) {
    const auto& w = windows_[i];
    if (!stream_id.empty() && w.stream_id != stream_id) continue;
    if (w.t1 < t0 || w.t0 > t1) continue;
    const float* v = vector(i);
    for (int d = 0; d < dim_; ++d) acc[d] += v[d];
    ++hits;
  }
  if (hits == 0)
    return fail(Errc::not_found, "no embedded windows overlap the query clip");
  float norm = std::sqrt(dot(acc.data(), acc.data(), dim_));
  if (norm > 0)
    for (auto& v : acc) v /= norm;
  return acc;
}

}  // namespace sdx
