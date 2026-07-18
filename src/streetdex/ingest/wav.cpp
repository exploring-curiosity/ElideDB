#include "streetdex/ingest/wav.hpp"

#include <cstring>

#include "streetdex/metrics/metrics.hpp"

namespace sdx {

Result<WavFile> WavFile::open(const std::string& path) {
  auto mf = MmapFile::open(path);
  if (!mf) return tl::unexpected(mf.error());
  WavFile w;
  w.file = std::move(*mf);
  const auto b = w.file.bytes();
  if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) != 0 ||
      std::memcmp(b.data() + 8, "WAVE", 4) != 0)
    return fail(Errc::bad_format, "not a RIFF/WAVE file: " + path);

  // Walk chunks: need fmt (format) then data (payload location).
  size_t pos = 12;
  bool have_fmt = false;
  while (pos + 8 <= b.size()) {
    char id[4];
    uint32_t sz = 0;
    std::memcpy(id, b.data() + pos, 4);
    std::memcpy(&sz, b.data() + pos + 4, 4);
    const size_t body = pos + 8;
    if (std::memcmp(id, "fmt ", 4) == 0 && body + 16 <= b.size()) {
      uint16_t fmt = 0;
      std::memcpy(&fmt, b.data() + body, 2);
      std::memcpy(&w.channels, b.data() + body + 2, 2);
      std::memcpy(&w.sample_rate, b.data() + body + 4, 4);
      std::memcpy(&w.bits_per_sample, b.data() + body + 14, 2);
      if (fmt != 1 || w.bits_per_sample != 16)
        return fail(Errc::bad_format, "only PCM s16 supported: " + path);
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (!have_fmt)
        return fail(Errc::bad_format, "data before fmt chunk: " + path);
      const size_t avail = std::min<size_t>(sz, b.size() - body);
      w.data_offset_ = body;
      w.frame_count = avail / (static_cast<size_t>(w.channels) * 2);
      metrics::count(metrics::Cat::ingest, avail);
      return w;
    }
    pos = body + sz + (sz & 1);  // RIFF chunks are 2-byte aligned
  }
  return fail(Errc::bad_format, "no data chunk in " + path);
}

}  // namespace sdx
