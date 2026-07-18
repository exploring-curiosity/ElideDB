#include <catch2/catch_test_macros.hpp>

#include <fstream>

#include "streetdex/ingest/time_json.hpp"
#include "test_util.hpp"

using namespace sdx;

namespace {

// Two bundles for cam 0, one for cam 2 — mirrors the REIP layout, including
// the lexicographic-vs-numeric bundle ordering trap (_10 sorts before _2).
void write_fixture(const TempDir& td) {
  auto bundle = [&](const std::string& name, int64_t first_frame, int n,
                    int file_index, double py0, int64_t gl0) {
    std::ofstream out(td.str(name));
    out << "{\n\"bundle_id\": 0, \"file_id\": 0";
    for (int i = 0; i < n; ++i) {
      out << ",\n\"buffer_" << i << "\": {"
          << "\"python_timestamp\": " << (py0 + 0.04 * i) << ","
          << "\"gstreamer_timestamp\": 0.5,"
          << "\"global_timestamp\": " << (gl0 + 48 * i) << ","
          << "\"resolution\": [2160, 3840],"
          << "\"file_template\": \"x_%d.avi\","
          << "\"file_index\": " << file_index << ","
          << "\"fps\": 30, \"pixel_format\": null,"
          << "\"frame_id\": " << (first_frame + i) << "}";
    }
    out << "\n}\n";
  };
  bundle("0_111_0.json", 0, 5, 0, 1000.0, 2000);
  bundle("0_111_1.json", 5, 5, 1, 1000.2, 2240);
  bundle("2_111_0.json", 0, 3, 0, 1000.0, 2000);
}

}  // namespace

TEST_CASE("time JSON discovery and ordered load", "[timejson]") {
  TempDir td("tj");
  write_fixture(td);

  auto streams = discover_camera_streams(td.str());
  REQUIRE(streams.has_value());
  REQUIRE(streams->size() == 2);
  CHECK((*streams)[0].first == "0");
  CHECK((*streams)[1].first == "2");
  CHECK((*streams)[0].second == "111");

  auto tl = load_camera_timeline(td.str(), "0", "111");
  REQUIRE(tl.has_value());
  CHECK(tl->fps == 30);
  CHECK(tl->width == 3840);   // resolution is [rows, cols]
  CHECK(tl->height == 2160);
  REQUIRE(tl->frames.size() == 10);
  CHECK(tl->frames.front().frame_id == 0);
  CHECK(tl->frames.back().frame_id == 9);
  CHECK(tl->frames[4].file_index == 0);
  CHECK(tl->frames[5].file_index == 1);
  CHECK(tl->frames[5].global_ticks == 2240);
}

TEST_CASE("duplicate frame ids are rejected", "[timejson]") {
  TempDir td("tjdup");
  std::ofstream(td.str("0_1_0.json"))
      << R"({"buffer_0": {"python_timestamp": 1.0, "global_timestamp": 10,
             "file_index": 0, "frame_id": 3},
             "buffer_1": {"python_timestamp": 1.1, "global_timestamp": 20,
             "file_index": 0, "frame_id": 3}})";
  auto tl = load_camera_timeline(td.str(), "0", "1");
  REQUIRE_FALSE(tl.has_value());
  CHECK(tl.error().code == Errc::bad_format);
}
