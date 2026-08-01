#include "streetdex/core/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

namespace sdx {

MmapFile::~MmapFile() {
  if (addr_ != nullptr && size_ > 0) ::munmap(addr_, size_);
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : addr_(std::exchange(other.addr_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      path_(std::move(other.path_)) {}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
  if (this != &other) {
    if (addr_ != nullptr && size_ > 0) ::munmap(addr_, size_);
    addr_ = std::exchange(other.addr_, nullptr);
    size_ = std::exchange(other.size_, 0);
    path_ = std::move(other.path_);
  }
  return *this;
}

Result<MmapFile> MmapFile::open(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    return fail(Errc::io, "open failed: " + path + ": " + std::strerror(errno));
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return fail(Errc::io, "fstat failed: " + path);
  }
  MmapFile f;
  f.size_ = static_cast<size_t>(st.st_size);
  f.path_ = path;
  if (f.size_ == 0) {
    ::close(fd);
    return fail(Errc::bad_format, "empty file: " + path);
  }
  void* addr = ::mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // mapping keeps its own reference to the file
  if (addr == MAP_FAILED)
    return fail(Errc::io, "mmap failed: " + path + ": " + std::strerror(errno));
  f.addr_ = addr;
  return f;
}

void MmapFile::advise_random() const {
  if (addr_) ::madvise(addr_, size_, MADV_RANDOM);
}
void MmapFile::advise_sequential() const {
  if (addr_) ::madvise(addr_, size_, MADV_SEQUENTIAL);
}

}  // namespace sdx
