#pragma once
// Bytes-read accounting on EVERY I/O path — non-negotiable (CLAUDE.md §7).
// The headline benchmark is "% of bytes elided"; it is only meaningful if every
// byte that enters the process is counted exactly once, in the right bucket.
//
// Two mechanisms:
//  1. CountingFile — pread-style reads (video packets via libav's AVIO
//     callback). Counts actual bytes returned by the OS.
//  2. Span accounting — SDX/SFI are mmapped; syscalls are invisible, so
//     readers charge the *logical* byte spans they touch (footer bytes, chunk
//     column spans, table entries). The formats' directories make every touch
//     an explicit (offset, length), so this is exact, not an estimate.
//
// A query binds an IoStats via Scope (thread-local); library code calls
// metrics::count() unconditionally — it no-ops when no scope is active.

#include <array>
#include <cstdint>
#include <string>

namespace sdx::metrics {

enum class Cat : uint8_t {
  sdx_footer = 0,  // SDX footer + directory bytes
  sdx_data,        // SDX column-chunk data bytes
  sfi_index,       // SFI header/table bytes
  video_data,      // source video bytes fed to the decoder
  vector_data,     // embedding/centroid bytes scanned
  catalog,         // manifest/snapshot JSON bytes
  ingest,          // ingest-time reads (source scan; not part of query elision)
  kCount
};

const char* cat_name(Cat c);

struct IoStats {
  std::array<uint64_t, static_cast<size_t>(Cat::kCount)> bytes{};
  std::array<uint64_t, static_cast<size_t>(Cat::kCount)> ops{};

  uint64_t total_bytes() const;
  void add(Cat c, uint64_t n);
  void merge(const IoStats& o);
  std::string summary(uint64_t corpus_bytes) const;
};

// RAII: binds `stats` as the active accumulator for this thread.
class Scope {
 public:
  explicit Scope(IoStats& stats);
  ~Scope();
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  IoStats* prev_;
};

// Charge n bytes to the active scope (and the process-lifetime totals).
void count(Cat c, uint64_t n);

// Process-lifetime totals (for `sdx bench` reporting).
const IoStats& lifetime();

}  // namespace sdx::metrics
