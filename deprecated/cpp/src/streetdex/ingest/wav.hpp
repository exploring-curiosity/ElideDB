#pragma once
// Minimal RIFF/WAVE reader for the capture rig's audio: PCM s16le,
// 16 channels, 48 kHz, one 5-second file per wall-clock chunk, named by the
// sensor-local python epoch second at file start. mmap-backed, zero-copy.

#include <cstdint>
#include <span>
#include <string>

#include "streetdex/core/error.hpp"
#include "streetdex/core/mmap_file.hpp"

namespace sdx {

struct WavFile {
  MmapFile file;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  size_t frame_count = 0;  // frames = samples per channel

  // Interleaved s16 samples: frame f, channel c at samples()[f*channels + c].
  std::span<const int16_t> samples() const {
    return {reinterpret_cast<const int16_t*>(file.bytes().data() + data_offset_),
            frame_count * channels};
  }

  static Result<WavFile> open(const std::string& path);

 private:
  size_t data_offset_ = 0;
};

}  // namespace sdx
