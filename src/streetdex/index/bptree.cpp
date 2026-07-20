#include "streetdex/index/bptree.hpp"

#include <algorithm>

#include "streetdex/core/binary_io.hpp"

namespace sdx {

std::vector<uint8_t> bpt_build(std::span<const BptPair> sorted,
                               uint32_t order) {
  // Internal levels bottom-up: level k holds the first key of every node of
  // level k-1. Stop when a level fits in one node — that's the root.
  std::vector<std::vector<int64_t>> levels;  // [0]=just above leaves … root
  uint64_t below = sorted.size();
  while (below > order) {
    const uint64_t nodes = (below + order - 1) / order;
    std::vector<int64_t> keys(nodes);
    for (uint64_t i = 0; i < nodes; ++i) {
      const uint64_t child_first = i * order;
      keys[i] = levels.empty()
                    ? sorted[child_first].key
                    : levels.back()[child_first];
    }
    levels.push_back(std::move(keys));
    below = nodes;
  }

  ByteWriter w;
  w.put_bytes({reinterpret_cast<const uint8_t*>(kBptMagic), 4});
  w.put<uint32_t>(order);
  w.put<uint64_t>(sorted.size());
  const uint32_t nlevels = static_cast<uint32_t>(levels.size());
  w.put<uint32_t>(nlevels);
  w.put<uint32_t>(0);  // pad to 8
  // Directory (root first), offsets patched after layout is known.
  const size_t dir_pos = w.size();
  for (uint32_t i = 0; i < nlevels + 1; ++i) {  // +1 for the leaf level
    w.put<uint64_t>(0);
    w.put<uint64_t>(0);
  }
  std::vector<std::pair<uint64_t, uint64_t>> dir;
  for (auto it = levels.rbegin(); it != levels.rend(); ++it) {  // root first
    w.align8();
    dir.emplace_back(w.size(), it->size());
    w.put_bytes({reinterpret_cast<const uint8_t*>(it->data()),
                 it->size() * sizeof(int64_t)});
  }
  w.align8();
  dir.emplace_back(w.size(), sorted.size());
  w.put_bytes({reinterpret_cast<const uint8_t*>(sorted.data()),
               sorted.size() * sizeof(BptPair)});

  std::vector<uint8_t> out = w.bytes();
  uint8_t* p = out.data() + dir_pos;
  for (auto [off, cnt] : dir) {
    std::memcpy(p, &off, 8);
    std::memcpy(p + 8, &cnt, 8);
    p += 16;
  }
  return out;
}

Result<void> BptReader::parse(std::span<const uint8_t> d) {
  if (d.size() < 24 || std::memcmp(d.data(), kBptMagic, 4) != 0)
    return fail(Errc::bad_format, "not a BPT1 index");
  ByteReader br(d);
  uint32_t magic = 0, pad = 0, nlevels = 0;
  br.get(magic);
  br.get(order_);
  br.get(n_);
  br.get(nlevels);
  br.get(pad);
  std::vector<std::pair<uint64_t, uint64_t>> dir(nlevels + 1);
  for (auto& [off, cnt] : dir) {
    if (!br.get(off) || !br.get(cnt))
      return fail(Errc::bad_format, "BPT1 truncated directory");
  }
  for (uint32_t i = 0; i < nlevels; ++i) {
    auto [off, cnt] = dir[i];
    if (off % 8 != 0 || off + cnt * 8 > d.size())
      return fail(Errc::bad_format, "BPT1 level out of range");
    level_keys_.push_back(reinterpret_cast<const int64_t*>(d.data() + off));
    level_off_.push_back(cnt);
  }
  auto [loff, lcnt] = dir[nlevels];
  if (loff % 8 != 0 || loff + lcnt * sizeof(BptPair) > d.size() || lcnt != n_)
    return fail(Errc::bad_format, "BPT1 leaf level out of range");
  leaves_ = reinterpret_cast<const BptPair*>(d.data() + loff);
  return {};
}

Result<BptReader> BptReader::open(const std::string& path) {
  auto mf = MmapFile::open(path);
  if (!mf) return tl::unexpected(mf.error());
  BptReader r;
  r.file_ = std::move(*mf);
  if (auto ok = r.parse(r.file_.bytes()); !ok)
    return tl::unexpected(ok.error());
  return r;
}

Result<BptReader> BptReader::from_bytes(std::span<const uint8_t> data) {
  BptReader r;
  r.owned_.assign(data.begin(), data.end());
  if (auto ok = r.parse(r.owned_); !ok) return tl::unexpected(ok.error());
  return r;
}

uint64_t BptReader::lower_bound(int64_t k) const {
  // Descend: at each internal level, the child to enter is the last node
  // whose first-key <= k. Within a level we only search the current node's
  // children — `order` keys, one binary search, one page touch.
  uint64_t node = 0;  // index of the current node within its level
  for (size_t lvl = 0; lvl < level_keys_.size(); ++lvl) {
    const uint64_t begin = node * order_;
    const uint64_t end = std::min<uint64_t>(begin + order_, level_off_[lvl]);
    const int64_t* keys = level_keys_[lvl];
    // lower_bound + step-back: bulk loading chops sorted runs every
    // `order`, so duplicates of k may START in the child BEFORE the first
    // child whose first-key equals k — enter that one; if k is beyond its
    // last entry the in-node search falls through to the next node's start.
    const int64_t* it = std::lower_bound(keys + begin, keys + end, k);
    node = static_cast<uint64_t>(it - keys);
    node = node > begin ? node - 1 : begin;
  }
  const uint64_t begin = node * order_;
  const uint64_t end = std::min<uint64_t>(begin + order_, n_);
  uint64_t lo = begin, hi = end;
  while (lo < hi) {
    const uint64_t mid = (lo + hi) / 2;
    if (leaves_[mid].key < k)
      lo = mid + 1;
    else
      hi = mid;
  }
  // Duplicates may span node boundaries backwards is impossible (lower
  // bound within the correct node), but k smaller than everything in this
  // node means the answer is its first entry.
  return lo;
}

std::vector<uint64_t> BptReader::range(int64_t lo, int64_t hi) const {
  std::vector<uint64_t> out;
  for (uint64_t i = lower_bound(lo); i < n_ && leaves_[i].key <= hi; ++i)
    out.push_back(leaves_[i].value);
  return out;
}

std::optional<uint64_t> BptReader::find_first(int64_t k) const {
  const uint64_t i = lower_bound(k);
  if (i < n_ && leaves_[i].key == k) return leaves_[i].value;
  return std::nullopt;
}

}  // namespace sdx
