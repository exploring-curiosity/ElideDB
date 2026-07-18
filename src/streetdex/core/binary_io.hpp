#pragma once
// Little-endian on-disk primitives. The formats are defined little-endian
// (docs/FORMAT.md); we static_assert the host is little-endian and then use
// memcpy-based access (safe w.r.t. alignment, compiles to plain loads).

#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "StreetDex on-disk formats are little-endian; big-endian hosts "
              "would need byte-swapping shims (deliberately out of scope).");

namespace sdx {

// ---- Writer: append-only byte buffer, flushed to disk in one pass ----------
class ByteWriter {
 public:
  template <typename T>
  void put(T v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }
  // Strings on disk: u16 length + UTF-8 bytes, no null terminator.
  void put_string(std::string_view s) {
    put<uint16_t>(static_cast<uint16_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
  void put_bytes(std::span<const uint8_t> b) {
    buf_.insert(buf_.end(), b.begin(), b.end());
  }
  // Zero-pad to an 8-byte boundary relative to `base` (an absolute file
  // offset this buffer will land at). Fixed-width tables must be 8-aligned so
  // a reader can mmap + cast without misaligned loads.
  void align8(size_t base = 0) {
    while ((base + buf_.size()) % 8 != 0) buf_.push_back(0);
  }
  size_t size() const { return buf_.size(); }
  const std::vector<uint8_t>& bytes() const { return buf_; }

 private:
  std::vector<uint8_t> buf_;
};

// ---- Reader: bounds-checked cursor over an mmapped span ---------------------
class ByteReader {
 public:
  explicit ByteReader(std::span<const uint8_t> data) : data_(data) {}

  template <typename T>
  bool get(T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (pos_ + sizeof(T) > data_.size()) return false;
    std::memcpy(&out, data_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return true;
  }
  bool get_string(std::string& out) {
    uint16_t n = 0;
    if (!get(n)) return false;
    if (pos_ + n > data_.size()) return false;
    out.assign(reinterpret_cast<const char*>(data_.data() + pos_), n);
    pos_ += n;
    return true;
  }
  bool skip_align8(size_t base = 0) {
    while ((base + pos_) % 8 != 0) {
      if (pos_ >= data_.size()) return false;
      ++pos_;
    }
    return true;
  }
  size_t pos() const { return pos_; }
  size_t remaining() const { return data_.size() - pos_; }

 private:
  std::span<const uint8_t> data_;
  size_t pos_ = 0;
};

}  // namespace sdx
