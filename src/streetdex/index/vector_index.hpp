#pragma once
// Vector index over per-window embeddings. Two-stage retrieval:
//
//   stage 1 (coarse): rank HDBSCAN cluster centroids, keep members of the
//                     top clusters — IVF with *learned* cells. HDBSCAN cells
//                     follow the data's density instead of a k-means grid;
//                     the mechanism (prune by cell, then scan survivors) is
//                     exactly Faiss IVF's nprobe.
//   stage 2 (fine):   exact cosine over survivors in FULL embedding space
//                     (Accelerate cblas_sdot; rows are L2-normalized at
//                     write time so cosine == dot).
//
// UMAP coordinates are visualization ONLY and never appear here: UMAP
// preserves neighborhoods, not distances, so a "radius" in UMAP space is
// semantically meaningless.
//
// Sidecar contract (store/ml/<run>/): embeddings.f32 (row-major n x d,
// L2-normalized), windows.json (window_id -> stream/time range),
// clusters.json (labels + centroids). Files, not RPC — keep it boring.

#include <cstdint>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/mmap_file.hpp"
#include "streetdex/core/time.hpp"

namespace sdx {

struct WindowMeta {
  std::string stream_id;
  TimeNs t0 = 0;
  TimeNs t1 = 0;
};

struct SearchHit {
  uint32_t window_id = 0;
  float score = 0;  // cosine similarity
};

struct SearchStats {
  uint64_t vectors_scanned = 0;
  uint64_t vectors_total = 0;
  uint32_t clusters_probed = 0;
  uint32_t clusters_total = 0;
};

class VectorIndex {
 public:
  static Result<VectorIndex> open(const std::string& run_dir);

  size_t size() const { return n_; }
  int dim() const { return dim_; }
  const std::vector<WindowMeta>& windows() const { return windows_; }
  const float* vector(uint32_t id) const {
    return reinterpret_cast<const float*>(emb_.bytes().data()) +
           static_cast<size_t>(id) * dim_;
  }

  // Two-stage search. nprobe = how many top clusters to scan; noise windows
  // (HDBSCAN label -1) are kept in a pseudo-cluster that is always scanned —
  // recall must not silently drop points the clustering refused to place.
  Result<std::vector<SearchHit>> search(const float* query, int k, int nprobe,
                                        SearchStats* stats = nullptr) const;

  // Exact flat scan (the recall baseline and the v1 fallback).
  Result<std::vector<SearchHit>> search_exact(const float* query, int k,
                                              SearchStats* stats = nullptr)
      const;

  // Mean of all window vectors overlapping [t0,t1] on `stream_id` (empty id =
  // any stream), L2-normalized — query-by-clip without touching pixels.
  Result<std::vector<float>> pool_window(const std::string& stream_id,
                                         TimeNs t0, TimeNs t1) const;

 private:
  MmapFile emb_;
  size_t n_ = 0;
  int dim_ = 0;
  std::vector<WindowMeta> windows_;
  std::vector<int32_t> labels_;                 // per window; -1 = noise
  std::vector<std::vector<float>> centroids_;   // per cluster, normalized
  std::vector<std::vector<uint32_t>> members_;  // cluster -> window ids
  std::vector<uint32_t> noise_;                 // always-probed remainder
};

}  // namespace sdx
