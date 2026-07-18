#include "streetdex/query/semantic.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;

namespace sdx {

Result<SemanticSearch> SemanticSearch::open(const Engine& engine) {
  const auto& m = engine.manifest();
  if (!m.semantic)
    return fail(Errc::not_found,
                "snapshot has no semantic run — run ml/embed.py + ml/cluster.py "
                "and re-snapshot");
  SemanticSearch s;
  s.run_dir_ =
      (fs::path(engine.store_dir()) / "ml" / m.semantic->run_id).string();
  auto ix = VectorIndex::open(s.run_dir_);
  if (!ix) return tl::unexpected(ix.error());
  s.index_ = std::move(*ix);
  // meta.json carries the exact text-embedding command for this model —
  // queries against snapshot N must embed with snapshot N's model, so the
  // command is part of the run, not of the CLI.
  std::ifstream in(fs::path(s.run_dir_) / "meta.json");
  if (in) {
    json j;
    try {
      in >> j;
      s.embed_cmd_ = j.value("embed_text_cmd", "");
    } catch (const json::exception&) {
    }
  }
  return s;
}

Result<SemanticResult> SemanticSearch::run(const std::vector<float>& q, int k,
                                           int nprobe, bool eval_recall,
                                           const WindowMeta* exclude) {
  SemanticResult out;
  auto hits = index_.search(q.data(), k + (exclude != nullptr ? 8 : 0), nprobe,
                            &out.stats);
  if (!hits) return tl::unexpected(hits.error());
  for (const auto& h : *hits) {
    const auto& w = index_.windows()[h.window_id];
    // Query-by-example: drop windows overlapping the probe clip itself.
    if (exclude != nullptr && w.stream_id == exclude->stream_id &&
        w.t0 <= exclude->t1 && w.t1 >= exclude->t0)
      continue;
    out.hits.push_back({w, h.score});
    if (static_cast<int>(out.hits.size()) == k) break;
  }
  if (eval_recall) {
    auto exact = index_.search_exact(q.data(), k, nullptr);
    if (exact) {
      size_t overlap = 0;
      auto approx = index_.search(q.data(), k, nprobe, nullptr);
      for (const auto& e : *exact)
        for (const auto& a : *approx)
          if (e.window_id == a.window_id) {
            ++overlap;
            break;
          }
      out.prune_recall_at_k =
          exact->empty() ? 1.0
                         : static_cast<double>(overlap) /
                               static_cast<double>(exact->size());
    }
  }
  return out;
}

Result<SemanticResult> SemanticSearch::query_text(const std::string& text,
                                                  int k, int nprobe,
                                                  bool eval_recall) {
  if (embed_cmd_.empty())
    return fail(Errc::not_found,
                "semantic run has no embed_text_cmd in meta.json");
  // Sidecar boundary: popen the recorded command, read raw f32 from stdout.
  std::string cmd = embed_cmd_;
  std::string quoted = text;
  // conservative shell quoting: single-quote, escaping embedded quotes
  std::string safe = "'";
  for (char c : quoted)
    safe += (c == '\'') ? std::string("'\\''") : std::string(1, c);
  safe += "'";
  cmd += " " + safe;
  std::FILE* p = popen(cmd.c_str(), "r");
  if (p == nullptr) return fail(Errc::io, "cannot run: " + cmd);
  std::vector<float> q(index_.dim());
  const size_t got = std::fread(q.data(), 4, q.size(), p);
  const int rc = pclose(p);
  if (rc != 0 || got != q.size())
    return fail(Errc::io,
                "text embedding failed (cmd exit " + std::to_string(rc) +
                    ", " + std::to_string(got) + "/" +
                    std::to_string(q.size()) + " floats): " + cmd);
  return run(q, k, nprobe, eval_recall);
}

Result<SemanticResult> SemanticSearch::query_clip(const std::string& stream_id,
                                                  TimeNs t0, TimeNs t1, int k,
                                                  int nprobe,
                                                  bool eval_recall) {
  auto q = index_.pool_window(stream_id, t0, t1);
  if (!q) return tl::unexpected(q.error());
  WindowMeta probe{stream_id, t0, t1};
  return run(*q, k, nprobe, eval_recall, &probe);
}

}  // namespace sdx
