#pragma once
// Semantic queries: text -> clips and clip -> similar clips. C++ owns
// ranking; the Python sidecar owns embedding (mlx SigLIP). For text queries
// the sidecar embeds the text on demand (subprocess, raw f32 on stdout — the
// same "files/pipes, no RPC" boundary as the rest of the ML contract).
// Every hit is a (stream_id, t0, t1) that feeds straight into get_window —
// retrieval returns *time*, and time is what the storage engine serves.

#include <string>
#include <vector>

#include "streetdex/index/vector_index.hpp"
#include "streetdex/query/planner.hpp"

namespace sdx {

struct SemanticHit {
  WindowMeta window;
  float score = 0;
};

struct SemanticResult {
  std::vector<SemanticHit> hits;
  SearchStats stats;
  double prune_recall_at_k = -1;  // vs exact scan; -1 = not evaluated
};

class SemanticSearch {
 public:
  // Opens the snapshot's semantic run (manifest.semantic must be set).
  static Result<SemanticSearch> open(const Engine& engine);

  Result<SemanticResult> query_text(const std::string& text, int k = 10,
                                    int nprobe = 3, bool eval_recall = false);
  Result<SemanticResult> query_clip(const std::string& stream_id, TimeNs t0,
                                    TimeNs t1, int k = 10, int nprobe = 3,
                                    bool eval_recall = false);

  const VectorIndex& index() const { return index_; }

 private:
  Result<SemanticResult> run(const std::vector<float>& q, int k, int nprobe,
                             bool eval_recall,
                             const WindowMeta* exclude = nullptr);
  std::string run_dir_;
  std::string embed_cmd_;  // from meta.json: command that emits f32 to stdout
  VectorIndex index_;
};

}  // namespace sdx
