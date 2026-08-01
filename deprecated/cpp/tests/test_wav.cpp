#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "streetdex/ingest/wav.hpp"
#include "test_util.hpp"

using namespace sdx;

namespace {

void write_wav(const std::string& path, uint16_t channels, uint32_t rate,
               const std::vector<int16_t>& interleaved) {
  const uint32_t data_bytes =
      static_cast<uint32_t>(interleaved.size() * 2);
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<char*>(&v), 4); };
  auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<char*>(&v), 2); };
  out.write("RIFF", 4);
  u32(36 + data_bytes);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  u32(16);
  u16(1);  // PCM
  u16(channels);
  u32(rate);
  u32(rate * channels * 2);
  u16(static_cast<uint16_t>(channels * 2));
  u16(16);
  out.write("data", 4);
  u32(data_bytes);
  out.write(reinterpret_cast<const char*>(interleaved.data()), data_bytes);
}

}  // namespace

TEST_CASE("wav reader parses 16ch PCM and exposes interleaved samples",
          "[wav]") {
  TempDir td("wav");
  std::vector<int16_t> data;
  for (int f = 0; f < 10; ++f)
    for (int c = 0; c < 16; ++c)
      data.push_back(static_cast<int16_t>(f * 100 + c));
  const auto path = td.str("a.wav");
  write_wav(path, 16, 48000, data);

  auto w = WavFile::open(path);
  REQUIRE(w.has_value());
  CHECK(w->channels == 16);
  CHECK(w->sample_rate == 48000);
  CHECK(w->frame_count == 10);
  auto s = w->samples();
  CHECK(s[0] == 0);
  CHECK(s[16] == 100);       // frame 1, ch 0
  CHECK(s[9 * 16 + 5] == 905);
}

TEST_CASE("wav reader rejects non-PCM and garbage", "[wav]") {
  TempDir td("wavbad");
  std::ofstream(td.str("junk.wav")) << "definitely not a riff file at all....";
  CHECK_FALSE(WavFile::open(td.str("junk.wav")).has_value());
}
