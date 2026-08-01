#include "streetdex/ingest/dataset.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <regex>

#include "streetdex/ingest/clock_model.hpp"
#include "streetdex/ingest/time_json.hpp"
#include "streetdex/ingest/wav.hpp"
#include "streetdex/video/sfi.hpp"

namespace fs = std::filesystem;

namespace sdx {

namespace {

std::string sanitize(std::string s) {
  for (auto& c : s)
    if (c == '/' || c == ' ') c = '_';
  return s;
}

struct CamStream {
  std::string sensor;  // "Sensor 108"
  CameraTimeline timeline;
  TimeNs clock_offset_ns = 0;
};

// Telemetry SDX: the time/ directory as first-class queryable data. One row
// per captured frame: capture jitter and clock skew become sensor signals.
Result<SensorStreamEntry> write_telemetry_sdx(const CamStream& cs,
                                              const ClockModel& clock,
                                              const std::string& store_dir,
                                              uint32_t chunk_rows) {
  const std::string rel =
      "sdx/" + sanitize(cs.sensor + "_cam" + cs.timeline.cam) + "_telemetry.sdx";
  const std::string path = (fs::path(store_dir) / rel).string();
  auto w = SdxWriter::create(
      path, cs.sensor + "/cam" + cs.timeline.cam + "/telemetry", "",
      cs.clock_offset_ns,
      {{"frame_interval_ms", ColType::F64},
       {"clock_skew_ms", ColType::F64},  // python wall clock minus canonical
       {"frame_id", ColType::I64},
       {"file_index", ColType::I64}},
      chunk_rows);
  if (!w) return tl::unexpected(w.error());

  const auto& fr = cs.timeline.frames;
  std::vector<int64_t> ts(fr.size()), frame_id(fr.size()), file_index(fr.size());
  std::vector<double> interval(fr.size()), skew(fr.size());
  TimeNs prev = 0;
  for (size_t i = 0; i < fr.size(); ++i) {
    const TimeNs t = clock.to_canonical_ns(fr[i].global_ticks);
    ts[i] = t;
    interval[i] = i == 0 ? 0.0 : static_cast<double>(t - prev) / 1e6;
    skew[i] = (fr[i].python_ts * 1e3) -
              (static_cast<double>(t) / 1e6);  // ms, local minus canonical
    frame_id[i] = fr[i].frame_id;
    file_index[i] = fr[i].file_index;
    prev = t;
  }
  const void* cols[] = {interval.data(), skew.data(), frame_id.data(),
                        file_index.data()};
  if (auto r = w->append(fr.size(), ts.data(), cols); !r)
    return tl::unexpected(r.error());
  if (auto r = w->finish(); !r) return tl::unexpected(r.error());

  auto rd = SdxReader::open(path);
  if (!rd) return tl::unexpected(rd.error());
  SensorStreamEntry e;
  e.stream_id = cs.sensor + "/cam" + cs.timeline.cam + "/telemetry";
  e.sdx_path = rel;
  e.min_ts = rd->meta().min_ts;
  e.max_ts = rd->meta().max_ts;
  e.rows = rd->meta().row_count;
  e.bytes = rd->meta().file_size;
  for (const auto& c : rd->meta().columns) e.columns.push_back(c.name);
  return e;
}

// Audio SDX: 16-channel PCM as a 48 kHz sensor stream. Timestamps are placed
// on the canonical timeline via the sensor's fitted clock offset (WAV names
// are sensor-local epoch seconds). The ADC's own drift vs the global clock is
// accepted as-is in v1 — measured, not corrected.
Result<SensorStreamEntry> write_audio_sdx(const std::string& sensor,
                                          const std::string& audio_dir,
                                          TimeNs clock_offset_ns,
                                          const std::string& store_dir,
                                          uint32_t chunk_rows, bool verbose) {
  std::vector<std::pair<int64_t, fs::path>> wavs;
  static const std::regex pat(R"((\d+)\.wav)");
  for (const auto& e : fs::directory_iterator(audio_dir)) {
    std::smatch m;
    const std::string name = e.path().filename().string();
    if (std::regex_match(name, m, pat)) wavs.emplace_back(std::stoll(m[1]), e.path());
  }
  if (wavs.empty()) return fail(Errc::not_found, "no wavs in " + audio_dir);
  std::sort(wavs.begin(), wavs.end());

  const std::string rel = "sdx/" + sanitize(sensor) + "_audio.sdx";
  const std::string path = (fs::path(store_dir) / rel).string();

  std::optional<SdxWriter> writer;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  TimeNs prev_end_ts = std::numeric_limits<TimeNs>::min();
  std::vector<int64_t> ts;
  std::vector<std::vector<int16_t>> ch_data;

  for (const auto& [epoch_sec, wav_path] : wavs) {
    auto wav = WavFile::open(wav_path.string());
    if (!wav) return tl::unexpected(wav.error());
    if (writer.has_value() &&
        (wav->channels != channels || wav->sample_rate != sample_rate))
      return fail(Errc::bad_format, "wav format changed mid-stream in " +
                                        audio_dir);
    if (!writer.has_value()) {
      channels = wav->channels;
      sample_rate = wav->sample_rate;
      std::vector<ColumnSpec> cols;
      for (int c = 0; c < channels; ++c)
        cols.push_back({"ch" + std::to_string(c), ColType::I16});
      auto w = SdxWriter::create(path, sensor + "/audio", "pcm_s16",
                                 clock_offset_ns, std::move(cols), chunk_rows);
      if (!w) return tl::unexpected(w.error());
      writer.emplace(std::move(*w));
      ch_data.resize(channels);
    }
    // File start on the canonical timeline: local wall clock minus offset.
    TimeNs t0 = epoch_sec * kNsPerSec - clock_offset_ns;
    if (t0 < prev_end_ts) {
      // Names are whole seconds while files are exactly 5.000 s of samples;
      // a sub-second collision would violate ts monotonicity. Clamp + warn —
      // the overlap is bounded by naming granularity (1 s).
      if (verbose)
        std::fprintf(stderr, "[audio] %s: start clamps %.3f ms onto previous file\n",
                     wav_path.filename().string().c_str(),
                     static_cast<double>(prev_end_ts - t0) / 1e6);
      t0 = prev_end_ts;
    }
    const size_t n = wav->frame_count;
    ts.resize(n);
    for (auto& v : ch_data) v.resize(n);
    const auto samples = wav->samples();
    for (size_t i = 0; i < n; ++i) {
      ts[i] = t0 + static_cast<TimeNs>(i * 1000000000ull / sample_rate);
      for (int c = 0; c < channels; ++c)
        ch_data[c][i] = samples[i * channels + c];
    }
    prev_end_ts = ts.empty() ? prev_end_ts : ts.back() + 1;
    std::vector<const void*> cols(channels);
    for (int c = 0; c < channels; ++c) cols[c] = ch_data[c].data();
    if (auto r = writer->append(n, ts.data(), cols.data()); !r)
      return tl::unexpected(r.error());
  }
  if (auto r = writer->finish(); !r) return tl::unexpected(r.error());

  auto rd = SdxReader::open(path);
  if (!rd) return tl::unexpected(rd.error());
  SensorStreamEntry e;
  e.stream_id = sensor + "/audio";
  e.sdx_path = rel;
  e.min_ts = rd->meta().min_ts;
  e.max_ts = rd->meta().max_ts;
  e.rows = rd->meta().row_count;
  e.bytes = rd->meta().file_size;
  for (const auto& c : rd->meta().columns) e.columns.push_back(c.name);
  return e;
}

}  // namespace

Result<Manifest> ingest_dataset(const IngestOptions& opts) {
  if (!fs::is_directory(opts.dataset_dir))
    return fail(Errc::not_found, "no dataset at " + opts.dataset_dir);
  std::error_code ec;
  fs::create_directories(fs::path(opts.store_dir) / "sfi", ec);
  fs::create_directories(fs::path(opts.store_dir) / "sdx", ec);

  // ---- 1. Load every camera timeline; pool clock pairs per sensor ---------
  std::vector<CamStream> streams;
  std::map<std::string, std::vector<ClockPair>> sensor_pairs;
  for (const auto& entry : fs::directory_iterator(opts.dataset_dir)) {
    if (!entry.is_directory()) continue;
    const std::string sensor = entry.path().filename().string();
    if (sensor.rfind("Sensor", 0) != 0) continue;
    const std::string time_dir = (entry.path() / "time").string();
    auto cams = discover_camera_streams(time_dir);
    if (!cams) continue;  // sensor dir without time data — nothing to index
    for (const auto& [cam, session] : *cams) {
      auto ctl = load_camera_timeline(time_dir, cam, session);
      if (!ctl) return tl::unexpected(ctl.error());
      auto& pairs = sensor_pairs[sensor];
      for (const auto& f : ctl->frames)
        pairs.push_back({f.python_ts, f.global_ticks});
      streams.push_back({sensor, std::move(*ctl), 0});
    }
  }
  if (streams.empty())
    return fail(Errc::not_found, "no camera streams under " + opts.dataset_dir);

  // ---- 2. Fit the canonical clock ------------------------------------------
  const std::string ds_name = fs::path(opts.dataset_dir).filename().string();
  auto anchor = parse_dataset_folder_time(ds_name);
  if (!anchor) return tl::unexpected(anchor.error());
  std::vector<std::vector<ClockPair>> pair_groups;
  for (auto& [_, p] : sensor_pairs) pair_groups.push_back(std::move(p));
  auto clock = build_clock_model(pair_groups, *anchor);
  if (!clock) return tl::unexpected(clock.error());
  if (opts.verbose)
    std::fprintf(stderr, "[clock] global tick rate %.4f Hz, anchor %lld ticks\n",
                 clock->rate_hz, static_cast<long long>(clock->anchor_ticks));

  Manifest m;
  m.snapshot = 1;
  m.dataset_dir = fs::absolute(opts.dataset_dir).string();
  m.clock = *clock;
  {
    char buf[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    m.created_utc = buf;
  }

  // ---- 3. Per camera stream: clock offset, SFIs, telemetry SDX -------------
  std::map<std::string, TimeNs> sensor_offset;
  for (auto& cs : streams) {
    std::vector<ClockPair> pairs;
    pairs.reserve(cs.timeline.frames.size());
    for (const auto& f : cs.timeline.frames)
      pairs.push_back({f.python_ts, f.global_ticks});
    cs.clock_offset_ns = fit_clock_offset(*clock, pairs);
    sensor_offset[cs.sensor] = cs.clock_offset_ns;  // audio reuses camera fit

    VideoStreamEntry vs;
    vs.stream_id = cs.sensor + "/cam" + cs.timeline.cam;
    vs.clock_offset_ns = cs.clock_offset_ns;
    vs.width = cs.timeline.width;
    vs.height = cs.timeline.height;
    vs.fps = cs.timeline.fps;

    // Group frames by file_index (frames are frame_id-sorted, so groups are
    // contiguous), then index each AVI that actually exists on disk.
    const auto& fr = cs.timeline.frames;
    size_t i = 0;
    while (i < fr.size()) {
      const int32_t fi = fr[i].file_index;
      size_t j = i;
      std::vector<TimeNs> pts;
      while (j < fr.size() && fr[j].file_index == fi) {
        pts.push_back(m.clock.to_canonical_ns(fr[j].global_ticks));
        ++j;
      }
      const auto avi = fs::path(opts.dataset_dir) / cs.sensor / "video" /
                       (cs.timeline.cam + "_" + cs.timeline.session + "_" +
                        std::to_string(fi) + ".avi");
      if (fs::exists(avi)) {
        const std::string rel =
            "sfi/" + sanitize(cs.sensor) + "_cam" + cs.timeline.cam + "_seg" +
            std::to_string(fi) + ".sfi";
        SfiBuildInput in;
        in.video_path = fs::absolute(avi).string();
        in.frame_pts_ns = std::move(pts);
        in.clock_offset_ns = cs.clock_offset_ns;
        auto st = build_sfi(in, (fs::path(opts.store_dir) / rel).string());
        if (!st) return tl::unexpected(st.error());
        auto sfi = SfiReader::open((fs::path(opts.store_dir) / rel).string());
        if (!sfi) return tl::unexpected(sfi.error());
        VideoSegmentEntry seg;
        seg.source_path = in.video_path;
        seg.sfi_path = rel;
        seg.first_pts_ns = sfi->header().first_pts_ns;
        seg.last_pts_ns = sfi->header().last_pts_ns;
        seg.source_bytes = sfi->header().source_size;
        seg.frame_count = sfi->header().frame_count;
        vs.segments.push_back(std::move(seg));
        if (opts.verbose)
          std::fprintf(stderr, "[sfi] %s seg %d: %llu frames, %llu gops\n",
                       vs.stream_id.c_str(), fi,
                       static_cast<unsigned long long>(st->frames_indexed),
                       static_cast<unsigned long long>(st->gops));
      } else if (opts.verbose) {
        std::fprintf(stderr,
                     "[sfi] %s seg %d: AVI absent (timing-only), skipped\n",
                     vs.stream_id.c_str(), fi);
      }
      i = j;
    }
    std::sort(vs.segments.begin(), vs.segments.end(),
              [](const auto& a, const auto& b) {
                return a.first_pts_ns < b.first_pts_ns;
              });
    if (!vs.segments.empty()) m.video.push_back(std::move(vs));

    if (opts.ingest_telemetry) {
      auto t = write_telemetry_sdx(cs, m.clock, opts.store_dir,
                                   opts.chunk_target_rows);
      if (!t) return tl::unexpected(t.error());
      m.sensors.push_back(std::move(*t));
    }
  }

  // ---- 4. Audio per sensor --------------------------------------------------
  if (opts.ingest_audio) {
    for (const auto& [sensor, offset] : sensor_offset) {
      const auto audio_dir = fs::path(opts.dataset_dir) / sensor / "audio";
      if (!fs::is_directory(audio_dir)) continue;
      auto a = write_audio_sdx(sensor, audio_dir.string(), offset,
                               opts.store_dir, opts.chunk_target_rows,
                               opts.verbose);
      if (!a) return tl::unexpected(a.error());
      if (opts.verbose)
        std::fprintf(stderr, "[audio] %s: %llu rows\n", sensor.c_str(),
                     static_cast<unsigned long long>(a->rows));
      m.sensors.push_back(std::move(*a));
    }
  }

  if (auto r = m.save(opts.store_dir); !r) return tl::unexpected(r.error());
  return m;
}

}  // namespace sdx
