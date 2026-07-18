#include "streetdex/query/planner.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace sdx {

namespace {

bool stream_selected(const std::vector<std::string>& wanted,
                     const std::string& id) {
  if (wanted.empty()) return true;
  return std::find(wanted.begin(), wanted.end(), id) != wanted.end();
}

}  // namespace

Result<Engine> Engine::open(const std::string& store_dir, int snapshot) {
  Engine e;
  e.store_dir_ = store_dir;
  auto m = Manifest::load_store(store_dir, snapshot);
  if (!m) return tl::unexpected(m.error());
  e.manifest_ = std::move(*m);
  return e;
}

Result<SdxReader*> Engine::sensor_reader(const SensorStreamEntry& e) {
  auto it = sdx_cache_.find(e.sdx_path);
  if (it != sdx_cache_.end()) return it->second.get();
  auto r = SdxReader::open((fs::path(store_dir_) / e.sdx_path).string());
  if (!r) return tl::unexpected(r.error());
  auto* ptr = new SdxReader(std::move(*r));
  sdx_cache_[e.sdx_path] = std::unique_ptr<SdxReader>(ptr);
  return ptr;
}

Result<SfiReader*> Engine::segment_sfi(const VideoSegmentEntry& seg) {
  auto it = sfi_cache_.find(seg.sfi_path);
  if (it != sfi_cache_.end()) return it->second.get();
  auto r = SfiReader::open((fs::path(store_dir_) / seg.sfi_path).string());
  if (!r) return tl::unexpected(r.error());
  auto* ptr = new SfiReader(std::move(*r));
  sfi_cache_[seg.sfi_path] = std::unique_ptr<SfiReader>(ptr);
  return ptr;
}

Result<VideoDecoder*> Engine::segment_decoder(const VideoSegmentEntry& seg) {
  auto it = dec_cache_.find(seg.sfi_path);
  if (it != dec_cache_.end()) return it->second.get();
  auto sfi = segment_sfi(seg);
  if (!sfi) return tl::unexpected(sfi.error());
  auto d = VideoDecoder::open(**sfi);  // SfiReader outlives: owned by cache
  if (!d) return tl::unexpected(d.error());
  auto* ptr = new VideoDecoder(std::move(*d));
  dec_cache_[seg.sfi_path] = std::unique_ptr<VideoDecoder>(ptr);
  return ptr;
}

Result<WindowResult> Engine::get_window(const WindowQuery& q) {
  if (q.t0 > q.t1) return fail(Errc::bad_argument, "t0 > t1");
  const auto start = std::chrono::steady_clock::now();
  WindowResult out;
  out.corpus_bytes = manifest_.corpus_bytes();
  out.timeline = make_timeline(q.t0, q.t1, q.rate_hz);
  metrics::Scope scope(out.io);  // every byte below lands in out.io

  // ---- Stage 1+2: sensor streams (catalog prune, then zone-map prune) ----
  for (const auto& s : manifest_.sensors) {
    if (!stream_selected(q.streams, s.stream_id)) continue;
    // Catalog-level prune: the manifest knows each file's time range; a file
    // wholly outside the window costs ZERO I/O — not even its footer.
    const TimeNs g0 = q.t0 - q.edge_guard_ns, g1 = q.t1 + q.edge_guard_ns;
    if (s.max_ts < g0 || s.min_ts > g1) {
      out.files_pruned++;
      continue;
    }
    out.files_touched++;
    auto rd = sensor_reader(s);
    if (!rd) return tl::unexpected(rd.error());
    auto scan = (*rd)->scan(g0, g1);
    if (!scan) return tl::unexpected(scan.error());

    SensorWindow w;
    w.stream_id = s.stream_id;
    w.raw_rows = scan->total_rows;
    const auto& cols = (*rd)->meta().columns;
    // Column 0 is ts; resample every value column onto the query timeline.
    auto ts = SdxReader::gather_i64(*scan, 0);
    for (size_t c = 1; c < cols.size(); ++c) {
      auto vals = SdxReader::gather_f64(*scan, c, cols[c].type);
      w.columns.push_back(cols[c].name);
      w.values.push_back(resample(ts, vals, out.timeline, q.interp));
    }
    out.sensors.push_back(std::move(w));
  }

  // ---- Stage 3+4: video streams (catalog -> SFI GOP prune -> decode) -----
  for (const auto& v : manifest_.video) {
    if (!stream_selected(q.streams, v.stream_id)) continue;
    VideoWindow w;
    w.stream_id = v.stream_id;
    for (const auto& seg : v.segments) {
      if (seg.last_pts_ns < q.t0 || seg.first_pts_ns > q.t1) {
        out.files_pruned++;
        continue;  // whole video file skipped: no SFI open, no decode
      }
      out.files_touched++;
      if (!q.decode_video) continue;
      auto dec = segment_decoder(seg);
      if (!dec) return tl::unexpected(dec.error());
      auto r = (*dec)->get_frames(q.t0, q.t1, q.video,
                                  [&](DecodedFrame&& f) {
                                    w.frames.push_back(std::move(f));
                                  });
      if (!r) return tl::unexpected(r.error());
    }
    // Frames arrive per-segment in pts order; segments are time-sorted, but
    // enforce global order anyway (cheap, and required by nearest_indices).
    std::sort(w.frames.begin(), w.frames.end(),
              [](const DecodedFrame& a, const DecodedFrame& b) {
                return a.pts_ns < b.pts_ns;
              });
    std::vector<TimeNs> fts;
    fts.reserve(w.frames.size());
    for (const auto& f : w.frames) fts.push_back(f.pts_ns);
    w.frame_for_point = nearest_indices(fts, out.timeline);
    if (!w.frames.empty() || !q.decode_video)
      out.video.push_back(std::move(w));
  }

  out.wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count();
  return out;
}

}  // namespace sdx
