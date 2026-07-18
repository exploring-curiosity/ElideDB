#include "streetdex/catalog/manifest.hpp"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "streetdex/metrics/metrics.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace sdx {

TimeNs Manifest::min_ts() const {
  TimeNs t = std::numeric_limits<TimeNs>::max();
  for (const auto& v : video)
    for (const auto& s : v.segments) t = std::min(t, s.first_pts_ns);
  for (const auto& s : sensors) t = std::min(t, s.min_ts);
  return t == std::numeric_limits<TimeNs>::max() ? 0 : t;
}

TimeNs Manifest::max_ts() const {
  TimeNs t = std::numeric_limits<TimeNs>::min();
  for (const auto& v : video)
    for (const auto& s : v.segments) t = std::max(t, s.last_pts_ns);
  for (const auto& s : sensors) t = std::max(t, s.max_ts);
  return t == std::numeric_limits<TimeNs>::min() ? 0 : t;
}

uint64_t Manifest::corpus_bytes() const {
  uint64_t b = 0;
  for (const auto& v : video)
    for (const auto& s : v.segments) b += s.source_bytes;
  for (const auto& s : sensors) b += s.bytes;
  return b;
}

namespace {

json to_json(const Manifest& m) {
  json j;
  j["snapshot"] = m.snapshot;
  j["created_utc"] = m.created_utc;
  j["dataset_dir"] = m.dataset_dir;
  j["clock_model"] = {{"rate_hz", m.clock.rate_hz},
                      {"anchor_ticks", m.clock.anchor_ticks},
                      {"anchor_ns", m.clock.anchor_ns},
                      {"note",
                       "canonical time = fitted shared global clock; anchor = "
                       "dataset folder wall time (UTC label)"}};
  for (const auto& v : m.video) {
    json vs = {{"stream_id", v.stream_id},
               {"clock_offset_ns", v.clock_offset_ns},
               {"width", v.width},
               {"height", v.height},
               {"fps", v.fps},
               {"segments", json::array()}};
    for (const auto& s : v.segments)
      vs["segments"].push_back({{"source_path", s.source_path},
                                {"sfi_path", s.sfi_path},
                                {"first_pts_ns", s.first_pts_ns},
                                {"last_pts_ns", s.last_pts_ns},
                                {"source_bytes", s.source_bytes},
                                {"frame_count", s.frame_count}});
    j["video_streams"].push_back(std::move(vs));
  }
  for (const auto& s : m.sensors)
    j["sensor_streams"].push_back({{"stream_id", s.stream_id},
                                   {"sdx_path", s.sdx_path},
                                   {"min_ts", s.min_ts},
                                   {"max_ts", s.max_ts},
                                   {"rows", s.rows},
                                   {"bytes", s.bytes},
                                   {"columns", s.columns}});
  if (m.semantic)
    j["semantic_run"] = {{"run_id", m.semantic->run_id},
                         {"model", m.semantic->model},
                         {"dim", m.semantic->dim},
                         {"window_count", m.semantic->window_count}};
  return j;
}

Result<Manifest> from_json(const json& j) {
  Manifest m;
  try {
    m.snapshot = j.at("snapshot").get<int>();
    m.created_utc = j.value("created_utc", "");
    m.dataset_dir = j.at("dataset_dir").get<std::string>();
    const auto& c = j.at("clock_model");
    m.clock.rate_hz = c.at("rate_hz").get<double>();
    m.clock.anchor_ticks = c.at("anchor_ticks").get<int64_t>();
    m.clock.anchor_ns = c.at("anchor_ns").get<int64_t>();
    for (const auto& vs : j.value("video_streams", json::array())) {
      VideoStreamEntry v;
      v.stream_id = vs.at("stream_id").get<std::string>();
      v.clock_offset_ns = vs.at("clock_offset_ns").get<int64_t>();
      v.width = vs.value("width", 0);
      v.height = vs.value("height", 0);
      v.fps = vs.value("fps", 0);
      for (const auto& sg : vs.at("segments")) {
        VideoSegmentEntry s;
        s.source_path = sg.at("source_path").get<std::string>();
        s.sfi_path = sg.at("sfi_path").get<std::string>();
        s.first_pts_ns = sg.at("first_pts_ns").get<int64_t>();
        s.last_pts_ns = sg.at("last_pts_ns").get<int64_t>();
        s.source_bytes = sg.at("source_bytes").get<uint64_t>();
        s.frame_count = sg.value("frame_count", 0ull);
        v.segments.push_back(std::move(s));
      }
      m.video.push_back(std::move(v));
    }
    for (const auto& ss : j.value("sensor_streams", json::array())) {
      SensorStreamEntry s;
      s.stream_id = ss.at("stream_id").get<std::string>();
      s.sdx_path = ss.at("sdx_path").get<std::string>();
      s.min_ts = ss.at("min_ts").get<int64_t>();
      s.max_ts = ss.at("max_ts").get<int64_t>();
      s.rows = ss.at("rows").get<uint64_t>();
      s.bytes = ss.at("bytes").get<uint64_t>();
      s.columns = ss.value("columns", std::vector<std::string>{});
      m.sensors.push_back(std::move(s));
    }
    if (j.contains("semantic_run")) {
      SemanticRun r;
      r.run_id = j["semantic_run"].at("run_id").get<std::string>();
      r.model = j["semantic_run"].at("model").get<std::string>();
      r.dim = j["semantic_run"].value("dim", 0);
      r.window_count = j["semantic_run"].value("window_count", 0ull);
      m.semantic = std::move(r);
    }
  } catch (const json::exception& e) {
    return fail(Errc::bad_format, std::string("bad manifest: ") + e.what());
  }
  return m;
}

}  // namespace

Result<void> Manifest::save(const std::string& store_dir) const {
  std::error_code ec;
  fs::create_directories(fs::path(store_dir) / "manifests", ec);
  const auto path = fs::path(store_dir) / "manifests" /
                    ("manifest-" + std::to_string(snapshot) + ".json");
  // Snapshots are immutable: refuse to overwrite an existing one.
  if (fs::exists(path))
    return fail(Errc::bad_argument,
                "snapshot " + std::to_string(snapshot) + " already exists");
  {
    std::ofstream out(path);
    if (!out) return fail(Errc::io, "cannot write " + path.string());
    out << to_json(*this).dump(2) << "\n";
  }
  // CURRENT is the only mutable file in the store — a one-line pointer.
  std::ofstream cur(fs::path(store_dir) / "CURRENT");
  if (!cur) return fail(Errc::io, "cannot write CURRENT");
  cur << snapshot << "\n";
  return {};
}

Result<Manifest> Manifest::load_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) return fail(Errc::not_found, "no manifest at " + path);
  json j;
  try {
    in >> j;
  } catch (const json::exception& e) {
    return fail(Errc::bad_format, path + ": " + e.what());
  }
  std::error_code ec;
  const auto sz = fs::file_size(path, ec);
  if (!ec) metrics::count(metrics::Cat::catalog, sz);
  return from_json(j);
}

Result<Manifest> Manifest::load_store(const std::string& store_dir,
                                      int snapshot) {
  if (snapshot < 0) {
    std::ifstream cur(fs::path(store_dir) / "CURRENT");
    if (!cur || !(cur >> snapshot))
      return fail(Errc::not_found, "no CURRENT in " + store_dir);
  }
  return load_file((fs::path(store_dir) / "manifests" /
                    ("manifest-" + std::to_string(snapshot) + ".json"))
                       .string());
}

}  // namespace sdx
