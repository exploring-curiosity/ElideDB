#include "streetdex/ingest/time_json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>

#include <nlohmann/json.hpp>

#include "streetdex/metrics/metrics.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace sdx {

Result<std::vector<std::pair<std::string, std::string>>> discover_camera_streams(
    const std::string& time_dir) {
  if (!fs::is_directory(time_dir))
    return fail(Errc::not_found, "no time dir: " + time_dir);
  static const std::regex pat(R"((\d+)_(\d+)_(\d+)\.json)");
  std::set<std::pair<std::string, std::string>> found;
  for (const auto& e : fs::directory_iterator(time_dir)) {
    std::smatch m;
    const std::string name = e.path().filename().string();
    if (std::regex_match(name, m, pat)) found.insert({m[1], m[2]});
  }
  return std::vector<std::pair<std::string, std::string>>(found.begin(),
                                                          found.end());
}

Result<CameraTimeline> load_camera_timeline(const std::string& time_dir,
                                            const std::string& cam,
                                            const std::string& session) {
  // Bundles must be read in numeric order (the _10 < _2 lexicographic trap).
  std::vector<std::pair<int, fs::path>> bundles;
  static const std::regex pat(R"((\d+)_(\d+)_(\d+)\.json)");
  for (const auto& e : fs::directory_iterator(time_dir)) {
    std::smatch m;
    const std::string name = e.path().filename().string();
    if (std::regex_match(name, m, pat) && m[1] == cam && m[2] == session)
      bundles.emplace_back(std::stoi(m[3]), e.path());
  }
  if (bundles.empty())
    return fail(Errc::not_found,
                "no time bundles for cam " + cam + " session " + session);
  std::sort(bundles.begin(), bundles.end());

  CameraTimeline tl;
  tl.cam = cam;
  tl.session = session;
  for (const auto& [idx, path] : bundles) {
    std::ifstream in(path);
    if (!in) return fail(Errc::io, "cannot read " + path.string());
    json doc;
    try {
      in >> doc;
    } catch (const json::exception& ex) {
      return fail(Errc::bad_format,
                  "bad time JSON " + path.string() + ": " + ex.what());
    }
    metrics::count(metrics::Cat::ingest,
                   static_cast<uint64_t>(fs::file_size(path)));
    for (auto& [key, v] : doc.items()) {
      if (key.rfind("buffer_", 0) != 0) continue;  // bundle_id / file_id keys
      FrameTimeRecord rec;
      rec.frame_id = v.at("frame_id").get<int64_t>();
      rec.file_index = v.at("file_index").get<int32_t>();
      rec.python_ts = v.at("python_timestamp").get<double>();
      rec.global_ticks = v.at("global_timestamp").get<int64_t>();
      if (tl.fps == 0 && v.contains("fps") && !v["fps"].is_null())
        tl.fps = v["fps"].get<int>();
      if (tl.width == 0 && v.contains("resolution") &&
          v["resolution"].is_array() && v["resolution"].size() == 2) {
        // REIP writes [rows, cols] = [height, width].
        tl.height = v["resolution"][0].get<int>();
        tl.width = v["resolution"][1].get<int>();
      }
      tl.frames.push_back(rec);
    }
  }
  std::sort(tl.frames.begin(), tl.frames.end(),
            [](const FrameTimeRecord& a, const FrameTimeRecord& b) {
              return a.frame_id < b.frame_id;
            });
  // Sanity: frame ids unique; file_index non-decreasing (a frame never goes
  // back to an earlier segment). Violations mean our join assumption is wrong
  // and pts provenance would be garbage — fail loudly rather than mis-time.
  for (size_t i = 1; i < tl.frames.size(); ++i) {
    if (tl.frames[i].frame_id == tl.frames[i - 1].frame_id)
      return fail(Errc::bad_format, "duplicate frame_id in time bundles");
    if (tl.frames[i].file_index < tl.frames[i - 1].file_index)
      return fail(Errc::bad_format, "file_index regressed in time bundles");
  }
  return tl;
}

}  // namespace sdx
