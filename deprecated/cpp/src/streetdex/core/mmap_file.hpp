#pragma once
// Read-only mmap with RAII. All format readers (SDX footer/chunks, SFI tables)
// go through this; reads are zero-copy pointer arithmetic over the mapping.
// Byte accounting: mmap hides syscalls, so the metrics layer counts *logical*
// spans touched (see metrics/metrics.hpp) — every access must be a span the
// caller declared, which is exactly what the formats' directories give us.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "streetdex/core/error.hpp"

namespace sdx {

class MmapFile {
 public:
  MmapFile() = default;
  ~MmapFile();
  MmapFile(MmapFile&& other) noexcept;
  MmapFile& operator=(MmapFile&& other) noexcept;
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;

  static Result<MmapFile> open(const std::string& path);

  std::span<const uint8_t> bytes() const {
    return {static_cast<const uint8_t*>(addr_), size_};
  }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

  // madvise hints: directories are read hot and random; data spans sequential.
  void advise_random() const;
  void advise_sequential() const;

 private:
  void* addr_ = nullptr;
  size_t size_ = 0;
  std::string path_;
};

}  // namespace sdx
