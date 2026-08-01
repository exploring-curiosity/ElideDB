#pragma once
// BPT1 — immutable, bulk-loaded, memory-mapped B+ tree.
//
// The database's files are immutable, so the index can be too — and an
// immutable B+ tree wants a very different construction than a textbook
// insert-one-at-a-time tree: BULK LOAD. Sort once, pack leaves to 100%
// fill, build each internal level as the first-keys of the level below.
// No splits, no rebalancing, no free lists — the pathological cases of
// mutable B+ trees simply cannot occur.
//
// Layout (little-endian, docs/FORMAT_BPT.md):
//   header  "BPT1" | u32 order | u64 n | u32 levels | u32 pad
//           per level (root first): u64 offset, u64 count
//   level data: internal levels are arrays of i64 first-keys;
//               leaf level is n × (i64 key, u64 value), key-sorted.
//
// Every node is `order` entries (default 256 → 4 KiB of keys), matching
// one page-cache page per touched node: a lookup costs `levels` page
// touches, which for 100M keys at order 256 is 4. Keys are i64 with an
// order-preserving encoding for doubles (sign-flip trick), values are
// opaque u64 (the store packs file_idx << 40 | row).
//
// The SAME format is written and read by python/elidedb/bptree.py — the
// index is a cross-language contract, not a runtime's private state.

#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/mmap_file.hpp"

namespace sdx {

inline constexpr char kBptMagic[4] = {'B', 'P', 'T', '1'};
inline constexpr uint32_t kBptDefaultOrder = 256;

// Order-preserving i64 encoding of an IEEE-754 double: flip the sign bit
// for positives, complement for negatives. Total order of doubles ==
// integer order of encodings (NaNs excluded at build time).
inline int64_t bpt_encode_f64(double v) {
  uint64_t u = std::bit_cast<uint64_t>(v);
  u = (u & 0x8000000000000000ull) ? ~u : (u | 0x8000000000000000ull);
  return static_cast<int64_t>(u ^ 0x8000000000000000ull);
}

struct BptPair {
  int64_t key;
  uint64_t value;
};
static_assert(sizeof(BptPair) == 16);

// ---- Bulk builder ----------------------------------------------------------
// Input pairs must be key-sorted (duplicates allowed — a range scan returns
// them all). Produces the complete file image in one pass per level.
std::vector<uint8_t> bpt_build(std::span<const BptPair> sorted,
                               uint32_t order = kBptDefaultOrder);

// ---- Reader ----------------------------------------------------------------
class BptReader {
 public:
  static Result<BptReader> open(const std::string& path);
  static Result<BptReader> from_bytes(std::span<const uint8_t> data);

  uint64_t size() const { return n_; }
  uint32_t levels() const { return static_cast<uint32_t>(level_off_.size()); }

  // First position whose key >= k (standard B+ descent, binary search
  // inside each node). Positions index the leaf array [0, n).
  uint64_t lower_bound(int64_t k) const;

  // All values with key in [lo, hi] (inclusive).
  std::vector<uint64_t> range(int64_t lo, int64_t hi) const;

  std::optional<uint64_t> find_first(int64_t k) const;

  const BptPair* leaf(uint64_t i) const { return leaves_ + i; }

 private:
  Result<void> parse(std::span<const uint8_t> data);
  MmapFile file_;                       // keeps the mapping alive
  std::vector<uint8_t> owned_;          // or an in-memory image
  uint32_t order_ = kBptDefaultOrder;
  uint64_t n_ = 0;
  std::vector<const int64_t*> level_keys_;  // internal levels, root first
  std::vector<uint64_t> level_off_;         // entry counts per level
  const BptPair* leaves_ = nullptr;
};

}  // namespace sdx
