// pystreetdex — numpy-native bindings. get_window returns analysis-ready
// arrays (frames as one [n,h,w,3] uint8 tensor, sensors as float64 arrays on
// the query timeline); nothing is ever re-muxed into video files.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "streetdex/query/planner.hpp"
#include "streetdex/query/semantic.hpp"

namespace py = pybind11;
using namespace sdx;

namespace {

template <typename T>
T value_or_throw(Result<T>&& r) {
  if (!r) throw std::runtime_error(r.error().message);
  return std::move(*r);
}

py::dict window_to_dict(const WindowResult& r) {
  py::dict out;
  out["timeline_ns"] = py::array_t<int64_t>(
      static_cast<py::ssize_t>(r.timeline.size()), r.timeline.data());
  py::dict sensors;
  for (const auto& s : r.sensors) {
    py::dict cols;
    for (size_t c = 0; c < s.columns.size(); ++c)
      cols[py::str(s.columns[c])] = py::array_t<double>(
          static_cast<py::ssize_t>(s.values[c].size()), s.values[c].data());
    sensors[py::str(s.stream_id)] = cols;
  }
  out["sensors"] = sensors;
  py::dict video;
  for (const auto& v : r.video) {
    py::dict d;
    std::vector<int64_t> pts;
    for (const auto& f : v.frames) pts.push_back(f.pts_ns);
    d["pts_ns"] = py::array_t<int64_t>(
        static_cast<py::ssize_t>(pts.size()), pts.data());
    if (!v.frames.empty()) {
      const int h = v.frames[0].height, w = v.frames[0].width;
      py::array_t<uint8_t> frames(
          {static_cast<py::ssize_t>(v.frames.size()),
           static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w),
           static_cast<py::ssize_t>(3)});
      auto buf = frames.mutable_unchecked<4>();
      for (size_t i = 0; i < v.frames.size(); ++i)
        std::memcpy(buf.mutable_data(static_cast<py::ssize_t>(i), 0, 0, 0),
                    v.frames[i].rgb.data(), v.frames[i].rgb.size());
      d["frames"] = frames;
    }
    d["frame_for_point"] = py::array_t<int32_t>(
        static_cast<py::ssize_t>(v.frame_for_point.size()),
        v.frame_for_point.data());
    video[py::str(v.stream_id)] = d;
  }
  out["video"] = video;
  out["bytes_read"] = r.io.total_bytes();
  out["corpus_bytes"] = r.corpus_bytes;
  out["elided_pct"] = r.elided_pct();
  out["wall_ms"] = r.wall_ms;
  return out;
}

}  // namespace

PYBIND11_MODULE(pystreetdex, m) {
  m.doc() = "StreetDex: content-aware timecode-native storage engine";

  py::class_<Engine>(m, "Engine")
      .def_static(
          "open",
          [](const std::string& store, int snapshot) {
            return value_or_throw(Engine::open(store, snapshot));
          },
          py::arg("store"), py::arg("snapshot") = -1)
      .def_property_readonly("min_ts",
                             [](const Engine& e) { return e.manifest().min_ts(); })
      .def_property_readonly("max_ts",
                             [](const Engine& e) { return e.manifest().max_ts(); })
      .def_property_readonly("snapshot",
                             [](const Engine& e) { return e.manifest().snapshot; })
      .def_property_readonly("streams",
                             [](const Engine& e) {
                               std::vector<std::string> out;
                               for (const auto& v : e.manifest().video)
                                 out.push_back(v.stream_id);
                               for (const auto& s : e.manifest().sensors)
                                 out.push_back(s.stream_id);
                               return out;
                             })
      .def(
          "get_window",
          [](Engine& e, TimeNs t0, TimeNs t1, double rate,
             const std::vector<std::string>& streams, const std::string& interp,
             int width, int stride, bool decode_video) {
            WindowQuery q;
            q.t0 = t0;
            q.t1 = t1;
            q.rate_hz = rate;
            q.streams = streams;
            q.interp = interp == "linear" ? Interp::linear : Interp::nearest;
            q.video.out_width = width;
            q.video.stride = stride;
            q.decode_video = decode_video;
            return window_to_dict(value_or_throw(e.get_window(q)));
          },
          py::arg("t0"), py::arg("t1"), py::arg("rate") = 30.0,
          py::arg("streams") = std::vector<std::string>{},
          py::arg("interp") = "nearest", py::arg("width") = 0,
          py::arg("stride") = 1, py::arg("decode_video") = true);

  py::class_<SemanticSearch>(m, "SemanticSearch")
      .def_static("open",
                  [](const Engine& e) {
                    return value_or_throw(SemanticSearch::open(e));
                  })
      .def("query_text",
           [](SemanticSearch& s, const std::string& text, int k, int nprobe) {
             auto r = value_or_throw(s.query_text(text, k, nprobe));
             py::list hits;
             for (const auto& h : r.hits)
               hits.append(py::make_tuple(h.window.stream_id, h.window.t0,
                                          h.window.t1, h.score));
             return hits;
           },
           py::arg("text"), py::arg("k") = 10, py::arg("nprobe") = 3)
      .def("query_clip",
           [](SemanticSearch& s, const std::string& stream, TimeNs t0,
              TimeNs t1, int k, int nprobe) {
             auto r = value_or_throw(s.query_clip(stream, t0, t1, k, nprobe));
             py::list hits;
             for (const auto& h : r.hits)
               hits.append(py::make_tuple(h.window.stream_id, h.window.t0,
                                          h.window.t1, h.score));
             return hits;
           },
           py::arg("stream"), py::arg("t0"), py::arg("t1"), py::arg("k") = 10,
           py::arg("nprobe") = 3);
}
