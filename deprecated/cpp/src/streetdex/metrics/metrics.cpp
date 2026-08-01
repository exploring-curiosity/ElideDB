#include "streetdex/metrics/metrics.hpp"

#include <cstdio>

namespace sdx::metrics {

namespace {
thread_local IoStats* g_active = nullptr;
IoStats g_lifetime;
}  // namespace

const char* cat_name(Cat c) {
  switch (c) {
    case Cat::sdx_footer: return "sdx_footer";
    case Cat::sdx_data: return "sdx_data";
    case Cat::sfi_index: return "sfi_index";
    case Cat::video_data: return "video_data";
    case Cat::vector_data: return "vector_data";
    case Cat::catalog: return "catalog";
    case Cat::ingest: return "ingest";
    case Cat::kCount: break;
  }
  return "?";
}

uint64_t IoStats::total_bytes() const {
  uint64_t t = 0;
  for (auto b : bytes) t += b;
  return t;
}

void IoStats::add(Cat c, uint64_t n) {
  bytes[static_cast<size_t>(c)] += n;
  ops[static_cast<size_t>(c)] += 1;
}

void IoStats::merge(const IoStats& o) {
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] += o.bytes[i];
    ops[i] += o.ops[i];
  }
}

std::string IoStats::summary(uint64_t corpus_bytes) const {
  std::string out;
  char line[160];
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (bytes[i] == 0) continue;
    std::snprintf(line, sizeof(line), "  %-11s %12llu B in %llu ops\n",
                  cat_name(static_cast<Cat>(i)),
                  static_cast<unsigned long long>(bytes[i]),
                  static_cast<unsigned long long>(ops[i]));
    out += line;
  }
  const uint64_t read = total_bytes();
  if (corpus_bytes > 0) {
    const double elided =
        100.0 * static_cast<double>(corpus_bytes - std::min(read, corpus_bytes)) /
        static_cast<double>(corpus_bytes);
    std::snprintf(line, sizeof(line),
                  "  total read %llu B of %llu B corpus -> %.4f%% elided\n",
                  static_cast<unsigned long long>(read),
                  static_cast<unsigned long long>(corpus_bytes), elided);
    out += line;
  }
  return out;
}

Scope::Scope(IoStats& stats) : prev_(g_active) { g_active = &stats; }
Scope::~Scope() { g_active = prev_; }

void count(Cat c, uint64_t n) {
  if (g_active != nullptr) g_active->add(c, n);
  g_lifetime.add(c, n);
}

const IoStats& lifetime() { return g_lifetime; }

}  // namespace sdx::metrics
