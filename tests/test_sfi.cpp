// SFI build/read/decode tests against a synthetic MJPEG AVI generated with
// the ffmpeg CLI (same codec/container family as the lab captures).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>

#include "streetdex/metrics/metrics.hpp"
#include "streetdex/video/decoder.hpp"
#include "streetdex/video/sfi.hpp"
#include "test_util.hpp"

using namespace sdx;

namespace {

// 2 s of testsrc at 10 fps, 64x64 MJPEG in AVI => 20 frames, all keyframes.
bool make_test_avi(const std::string& path) {
  const std::string cmd =
      "ffmpeg -v error -y -f lavfi -i testsrc=duration=2:size=64x64:rate=10 "
      "-c:v mjpeg -q:v 5 -f avi '" + path + "' 2>/dev/null";
  return std::system(cmd.c_str()) == 0 && std::filesystem::exists(path);
}

}  // namespace

TEST_CASE("SFI build + invariants + windowed byte-range decode", "[sfi]") {
  TempDir td("sfi");
  const auto avi = td.str("test.avi");
  if (!make_test_avi(avi)) SKIP("ffmpeg CLI not available");

  // Sidecar-style timestamps: 20 frames at 10 fps starting at t=5s.
  SfiBuildInput in;
  in.video_path = avi;
  for (int i = 0; i < 20; ++i)
    in.frame_pts_ns.push_back(5'000'000'000LL + i * 100'000'000LL);
  in.clock_offset_ns = 777;
  const auto sfi_path = td.str("test.sfi");
  auto st = build_sfi(in, sfi_path);
  REQUIRE(st.has_value());
  CHECK(st->frames_indexed == 20);
  CHECK(st->gops == 20);  // MJPEG: every frame is a keyframe

  auto r = SfiReader::open(sfi_path);
  REQUIRE(r.has_value());
  const auto& h = r->header();
  CHECK(h.width == 64);
  CHECK(h.height == 64);
  CHECK(h.frame_count == 20);
  CHECK(h.clock_offset_ns == 777);
  CHECK(h.first_pts_ns == 5'000'000'000LL);
  CHECK(h.last_pts_ns == 6'900'000'000LL);
  CHECK(h.source_size == std::filesystem::file_size(avi));

  // FORMAT.md §2.6 invariants on the tables.
  const auto frames = r->frames();
  const auto gops = r->gops();
  for (size_t i = 1; i < frames.size(); ++i)
    REQUIRE(frames[i].pts_ns > frames[i - 1].pts_ns);
  for (size_t g = 0; g < gops.size(); ++g) {
    REQUIRE(gops[g].frame_count == 1);
    REQUIRE(gops[g].first_frame_row == g);
    REQUIRE(gops[g].start_byte < gops[g].end_byte);
    const auto& f = frames[gops[g].first_frame_row];
    REQUIRE((f.flags & 1) == 1);  // keyframe at GOP head
    REQUIRE(f.gop_id == g);
    if (g > 0) REQUIRE(gops[g].start_byte >= gops[g - 1].end_byte);
  }

  // GOP pruning: [5.55s, 5.85s] covers frames at 5.6, 5.7, 5.8.
  auto sel = r->gops_in_range(5'550'000'000LL, 5'850'000'000LL);
  CHECK(sel.size() == 3);

  auto dec = VideoDecoder::open(*r);
  REQUIRE(dec.has_value());

  // Full decode equivalence: whole-range query yields all 20 frames in order.
  auto all = dec->get_frames_vec(h.first_pts_ns, h.last_pts_ns, {});
  REQUIRE(all.has_value());
  REQUIRE(all->size() == 20);
  for (size_t i = 0; i < 20; ++i) {
    CHECK((*all)[i].pts_ns == in.frame_pts_ns[i]);
    CHECK((*all)[i].width == 64);
    CHECK((*all)[i].height == 64);
  }

  // Windowed decode reads ONLY the selected GOP bytes (plus one-time codec
  // parameter read at decoder open, already spent above).
  metrics::IoStats io;
  {
    metrics::Scope s(io);
    auto win = dec->get_frames_vec(5'550'000'000LL, 5'850'000'000LL, {});
    REQUIRE(win.has_value());
    REQUIRE(win->size() == 3);
    CHECK((*win)[0].pts_ns == 5'600'000'000LL);
    // Pixels identical to the same frame from the full decode: byte-range
    // decode is lossless vs full decode.
    CHECK((*win)[0].rgb == (*all)[6].rgb);
  }
  uint64_t gop_bytes = 0;
  for (const auto& g : sel) gop_bytes += g.end_byte - g.start_byte;
  CHECK(io.bytes[static_cast<size_t>(metrics::Cat::video_data)] == gop_bytes);
  CHECK(gop_bytes < h.source_size / 4);  // way less than the file

  // Stride skips whole GOPs (their bytes are never read).
  metrics::IoStats io2;
  {
    metrics::Scope s(io2);
    FrameQueryOptions o;
    o.stride = 2;
    auto win = dec->get_frames_vec(5'000'000'000LL, 6'900'000'000LL, o);
    REQUIRE(win.has_value());
    CHECK(win->size() == 10);
  }
  CHECK(io2.bytes[static_cast<size_t>(metrics::Cat::video_data)] <
        h.source_size * 6 / 10);

  // Scaling emits the requested width.
  FrameQueryOptions o;
  o.out_width = 32;
  auto small = dec->get_frames_vec(5'000'000'000LL, 5'000'000'000LL, o);
  REQUIRE(small.has_value());
  REQUIRE(small->size() == 1);
  CHECK((*small)[0].width == 32);
  CHECK((*small)[0].rgb.size() == 32u * 32u * 3u);
}

TEST_CASE("SFI container-pts fallback and torn-file detection", "[sfi]") {
  TempDir td("sfi2");
  const auto avi = td.str("test.avi");
  if (!make_test_avi(avi)) SKIP("ffmpeg CLI not available");

  SfiBuildInput in;
  in.video_path = avi;  // no sidecar timing: trust container pts
  const auto sfi_path = td.str("test.sfi");
  REQUIRE(build_sfi(in, sfi_path).has_value());
  auto r = SfiReader::open(sfi_path);
  REQUIRE(r.has_value());
  CHECK(r->header().first_pts_ns == 0);
  CHECK(r->header().last_pts_ns == 1'900'000'000LL);  // 10 fps container time

  std::error_code ec;
  std::filesystem::resize_file(sfi_path,
                               std::filesystem::file_size(sfi_path) - 5, ec);
  CHECK_FALSE(SfiReader::open(sfi_path).has_value());
}
