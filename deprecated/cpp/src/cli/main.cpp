// sdx — StreetDex CLI. Exceptions and exit codes live here and only here;
// the library speaks Result<T>.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "streetdex/ingest/dataset.hpp"
#include "streetdex/query/planner.hpp"
#include "streetdex/query/semantic.hpp"

namespace fs = std::filesystem;
using namespace sdx;

namespace {

constexpr const char* kVersion = "streetdex 0.1.0";

[[noreturn]] void die(const Error& e) {
  std::fprintf(stderr, "error: %s\n", e.message.c_str());
  std::exit(1);
}

[[noreturn]] void usage() {
  std::fprintf(stderr, R"(usage:
  sdx --version
  sdx init <name> --store <dir>            (create an empty named database)
  sdx ingest <dataset_dir> --store <dir> [--no-audio] [--no-telemetry]
             [--chunk-rows N]              (REIP-layout dataset adapter)
  sdx index-video <video> --store <dir> --stream-id ID [--timestamps FILE]
             [--segment N]                 (any video; FILE = one ns per line;
                                            creates a new snapshot)
  sdx ingest-csv <file.csv> --store <dir> --stream-id ID [--units U]
             [--chunk-rows N]   (creates a new snapshot with the added stream)
  sdx info --store <dir> [--snapshot N]
  sdx window --store <dir> --t0 <t> --t1 <t> [--rate HZ] [--interp nearest|linear]
             [--streams a,b] [--width W] [--stride N] [--no-video]
             [--dump DIR] [--snapshot N]
  sdx query --store <dir> (--text "..." | --clip STREAM,T0,T1) [-k N]
             [--nprobe N] [--eval-recall] [--snapshot N]
  sdx bench --store <dir> [--windows N] [--dur SECS] [--rate HZ] [--seed S]
             [--naive N] [--out FILE] [--snapshot N]

time arguments: raw ns since epoch, or +SECS relative to corpus start.
)");
  std::exit(2);
}

std::string arg_value(std::vector<std::string>& args, const std::string& flag,
                      const std::string& def = "") {
  for (size_t i = 0; i + 1 < args.size(); ++i)
    if (args[i] == flag) {
      std::string v = args[i + 1];
      args.erase(args.begin() + i, args.begin() + i + 2);
      return v;
    }
  return def;
}

bool arg_flag(std::vector<std::string>& args, const std::string& flag) {
  auto it = std::find(args.begin(), args.end(), flag);
  if (it == args.end()) return false;
  args.erase(it);
  return true;
}

TimeNs parse_time(const std::string& s, TimeNs corpus_min) {
  if (!s.empty() && s[0] == '+')
    return corpus_min +
           static_cast<TimeNs>(std::stod(s.substr(1)) * 1e9);
  return std::stoll(s);
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t c = s.find(',', pos);
    if (c == std::string::npos) {
      if (pos < s.size()) out.push_back(s.substr(pos));
      break;
    }
    out.push_back(s.substr(pos, c - pos));
    pos = c + 1;
  }
  return out;
}

double human_sec(TimeNs t, TimeNs base) {
  return static_cast<double>(t - base) / 1e9;
}

void write_ppm(const fs::path& p, const DecodedFrame& f) {
  std::ofstream out(p, std::ios::binary);
  out << "P6\n" << f.width << " " << f.height << "\n255\n";
  out.write(reinterpret_cast<const char*>(f.rgb.data()),
            static_cast<std::streamsize>(f.rgb.size()));
}

// ---- subcommands -----------------------------------------------------------

int cmd_ingest(std::vector<std::string> args) {
  IngestOptions o;
  o.store_dir = arg_value(args, "--store");
  o.ingest_audio = !arg_flag(args, "--no-audio");
  o.ingest_telemetry = !arg_flag(args, "--no-telemetry");
  const std::string rows = arg_value(args, "--chunk-rows");
  if (!rows.empty()) o.chunk_target_rows = std::stoul(rows);
  if (args.size() != 1 || o.store_dir.empty()) usage();
  o.dataset_dir = args[0];

  const auto t0 = std::chrono::steady_clock::now();
  auto m = ingest_dataset(o);
  if (!m) die(m.error());
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  const double gb = static_cast<double>(m->corpus_bytes()) / 1e9;
  std::printf("ingested snapshot %d: %zu video streams, %zu sensor streams\n",
              m->snapshot, m->video.size(), m->sensors.size());
  std::printf("corpus %.2f GB in %.1f s (%.2f GB/min)\n", gb, secs,
              gb / secs * 60.0);
  return 0;
}

// Create an empty named database. StreetDex is not tied to one capture rig:
// anything timestamped enters through index-video (video + ns sidecar) and
// ingest-csv (scalar streams); `sdx ingest` is merely the REIP adapter.
int cmd_init(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  if (args.size() != 1 || store.empty()) usage();
  Manifest m;
  m.snapshot = 1;
  m.dataset_dir = args[0];  // the database's human name / provenance label
  // Generic stores put source timestamps directly on the canonical timeline
  // (identity clock). Rig-specific clock fitting is the adapter's job.
  m.clock = {1e9, 0, 0};
  char buf[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  m.created_utc = buf;
  std::error_code ec;
  fs::create_directories(fs::path(store) / "sfi", ec);
  fs::create_directories(fs::path(store) / "sdx", ec);
  if (auto r = m.save(store); !r) die(r.error());
  std::printf("created database '%s' at %s (snapshot 1)\n", args[0].c_str(),
              store.c_str());
  return 0;
}

// Index ANY video file into the store: SFI build + manifest entry + new
// snapshot. --timestamps supplies canonical ns per frame (one integer per
// line, packet order); without it the container's own pts is trusted.
int cmd_index_video(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  const std::string stream_id = arg_value(args, "--stream-id");
  const std::string ts_file = arg_value(args, "--timestamps");
  const int segment = std::stoi(arg_value(args, "--segment", "0"));
  if (args.size() != 1 || store.empty() || stream_id.empty()) usage();

  auto m = Manifest::load_store(store, -1);
  if (!m) die(m.error());

  SfiBuildInput in;
  in.video_path = fs::absolute(args[0]).string();
  if (!ts_file.empty()) {
    std::ifstream tf(ts_file);
    if (!tf) die({Errc::io, "cannot open " + ts_file});
    int64_t t;
    while (tf >> t) in.frame_pts_ns.push_back(t);
  }
  std::string safe = stream_id;
  for (auto& ch : safe)
    if (ch == '/' || ch == ' ') ch = '_';
  const std::string rel =
      "sfi/" + safe + "_seg" + std::to_string(segment) + ".sfi";
  const auto t_start = std::chrono::steady_clock::now();
  auto st = build_sfi(in, (fs::path(store) / rel).string());
  if (!st) die(st.error());
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t_start)
                          .count();
  auto sfi = SfiReader::open((fs::path(store) / rel).string());
  if (!sfi) die(sfi.error());

  VideoSegmentEntry seg;
  seg.source_path = in.video_path;
  seg.sfi_path = rel;
  seg.first_pts_ns = sfi->header().first_pts_ns;
  seg.last_pts_ns = sfi->header().last_pts_ns;
  seg.source_bytes = sfi->header().source_size;
  seg.frame_count = sfi->header().frame_count;

  auto it = std::find_if(m->video.begin(), m->video.end(),
                         [&](const VideoStreamEntry& v) {
                           return v.stream_id == stream_id;
                         });
  if (it == m->video.end()) {
    VideoStreamEntry vs;
    vs.stream_id = stream_id;
    vs.width = static_cast<int>(sfi->header().width);
    vs.height = static_cast<int>(sfi->header().height);
    const auto span = sfi->header().last_pts_ns - sfi->header().first_pts_ns;
    if (span > 0 && sfi->header().frame_count > 1)
      vs.fps = static_cast<int>(
          std::llround(static_cast<double>(sfi->header().frame_count - 1) *
                       1e9 / static_cast<double>(span)));
    m->video.push_back(std::move(vs));
    it = m->video.end() - 1;
  }
  it->segments.push_back(std::move(seg));
  std::sort(it->segments.begin(), it->segments.end(),
            [](const auto& a, const auto& b) {
              return a.first_pts_ns < b.first_pts_ns;
            });
  m->snapshot += 1;
  char buf[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  m->created_utc = buf;
  if (auto r = m->save(store); !r) die(r.error());
  std::printf("indexed %s: %llu frames, %llu gops, %.2f GB scanned in %.1f s "
              "-> snapshot %d\n",
              stream_id.c_str(),
              static_cast<unsigned long long>(st->frames_indexed),
              static_cast<unsigned long long>(st->gops),
              static_cast<double>(st->bytes_scanned) / 1e9, secs, m->snapshot);
  return 0;
}

// CSV -> SDX -> new snapshot. Immutability in action: the existing snapshot
// keeps meaning exactly what it meant; the added stream exists only from the
// new snapshot onward.
int cmd_ingest_csv(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  const std::string stream_id = arg_value(args, "--stream-id");
  const std::string units = arg_value(args, "--units");
  const uint32_t chunk_rows =
      std::stoul(arg_value(args, "--chunk-rows", "4096"));
  if (args.size() != 1 || store.empty() || stream_id.empty()) usage();
  const std::string csv = args[0];

  auto m = Manifest::load_store(store, -1);
  if (!m) die(m.error());

  std::ifstream in(csv);
  if (!in) die({Errc::io, "cannot open " + csv});
  std::string line;
  if (!std::getline(in, line)) die({Errc::bad_format, "empty csv"});
  auto names = split_csv(line);
  if (names.size() < 2) die({Errc::bad_format, "need ts + >=1 value column"});
  std::vector<ColumnSpec> specs;
  for (size_t c = 1; c < names.size(); ++c)
    specs.push_back({names[c], ColType::F64});

  std::string safe = stream_id;
  for (auto& ch : safe)
    if (ch == '/' || ch == ' ') ch = '_';
  const std::string rel = "sdx/" + safe + ".sdx";
  auto w = SdxWriter::create((fs::path(store) / rel).string(), stream_id,
                             units, 0, specs, chunk_rows);
  if (!w) die(w.error());

  // Batched append: parse into columnar buffers, flush every 64k rows.
  const size_t ncol = names.size() - 1;
  std::vector<int64_t> ts;
  std::vector<std::vector<double>> vals(ncol);
  auto flush = [&]() {
    if (ts.empty()) return;
    std::vector<const void*> cols(ncol);
    for (size_t c = 0; c < ncol; ++c) cols[c] = vals[c].data();
    if (auto r = w->append(ts.size(), ts.data(), cols.data()); !r)
      die(r.error());
    ts.clear();
    for (auto& v : vals) v.clear();
  };
  uint64_t rows = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto parts = split_csv(line);
    if (parts.size() != names.size())
      die({Errc::bad_format, "row " + std::to_string(rows) + " has " +
                                 std::to_string(parts.size()) + " fields"});
    ts.push_back(std::stoll(parts[0]));
    for (size_t c = 0; c < ncol; ++c)
      vals[c].push_back(std::stod(parts[c + 1]));
    if (++rows % 65536 == 0) flush();
  }
  flush();
  if (auto r = w->finish(); !r) die(r.error());

  auto rd = SdxReader::open((fs::path(store) / rel).string());
  if (!rd) die(rd.error());
  SensorStreamEntry entry;
  entry.stream_id = stream_id;
  entry.sdx_path = rel;
  entry.min_ts = rd->meta().min_ts;
  entry.max_ts = rd->meta().max_ts;
  entry.rows = rd->meta().row_count;
  entry.bytes = rd->meta().file_size;
  for (const auto& c : rd->meta().columns) entry.columns.push_back(c.name);
  m->sensors.push_back(std::move(entry));
  m->snapshot += 1;
  {
    char buf[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    m->created_utc = buf;
  }
  if (auto r = m->save(store); !r) die(r.error());
  std::printf("added %s (%llu rows, %zu chunks) as snapshot %d\n",
              stream_id.c_str(), static_cast<unsigned long long>(rd->meta().row_count),
              rd->chunks().size(), m->snapshot);
  return 0;
}

int cmd_info(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  const int snap = std::stoi(arg_value(args, "--snapshot", "-1"));
  if (store.empty()) usage();
  auto e = Engine::open(store, snap);
  if (!e) die(e.error());
  const auto& m = e->manifest();
  const TimeNs base = m.min_ts();
  std::printf("snapshot %d  (created %s)\n", m.snapshot, m.created_utc.c_str());
  std::printf("dataset   %s\n", m.dataset_dir.c_str());
  std::printf("clock     global rate %.4f Hz, anchor_ns %lld\n",
              m.clock.rate_hz, static_cast<long long>(m.clock.anchor_ns));
  std::printf("timeline  [%.3f .. %.3f] s (span %.1f s)\n", 0.0,
              human_sec(m.max_ts(), base), human_sec(m.max_ts(), base));
  std::printf("corpus    %.2f GB\n\n", static_cast<double>(m.corpus_bytes()) / 1e9);
  for (const auto& v : m.video) {
    uint64_t frames = 0, bytes = 0;
    for (const auto& s : v.segments) {
      frames += s.frame_count;
      bytes += s.source_bytes;
    }
    std::printf("video   %-28s %2zu segs %7llu frames %8.2f GB  [%+.1f .. %+.1f]s  offset %+.3fs\n",
                v.stream_id.c_str(), v.segments.size(),
                static_cast<unsigned long long>(frames),
                static_cast<double>(bytes) / 1e9,
                human_sec(v.segments.front().first_pts_ns, base),
                human_sec(v.segments.back().last_pts_ns, base),
                static_cast<double>(v.clock_offset_ns) / 1e9);
  }
  for (const auto& s : m.sensors)
    std::printf("sensor  %-28s %9llu rows %8.2f GB  [%+.1f .. %+.1f]s  %zu cols\n",
                s.stream_id.c_str(), static_cast<unsigned long long>(s.rows),
                static_cast<double>(s.bytes) / 1e9,
                human_sec(s.min_ts, base), human_sec(s.max_ts, base),
                s.columns.size());
  if (m.semantic)
    std::printf("semantic run %s model %s dim %d windows %llu\n",
                m.semantic->run_id.c_str(), m.semantic->model.c_str(),
                m.semantic->dim,
                static_cast<unsigned long long>(m.semantic->window_count));
  return 0;
}

int cmd_window(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  const int snap = std::stoi(arg_value(args, "--snapshot", "-1"));
  if (store.empty()) usage();
  auto e = Engine::open(store, snap);
  if (!e) die(e.error());
  const TimeNs base = e->manifest().min_ts();

  WindowQuery q;
  const std::string t0s = arg_value(args, "--t0"), t1s = arg_value(args, "--t1");
  if (t0s.empty() || t1s.empty()) usage();
  q.t0 = parse_time(t0s, base);
  q.t1 = parse_time(t1s, base);
  q.rate_hz = std::stod(arg_value(args, "--rate", "30"));
  q.interp = arg_value(args, "--interp", "nearest") == "linear"
                 ? Interp::linear
                 : Interp::nearest;
  q.streams = split_csv(arg_value(args, "--streams"));
  q.video.out_width = std::stoi(arg_value(args, "--width", "0"));
  q.video.stride = std::stoi(arg_value(args, "--stride", "1"));
  q.decode_video = !arg_flag(args, "--no-video");
  const std::string dump = arg_value(args, "--dump");

  auto r = e->get_window(q);
  if (!r) die(r.error());

  std::printf("window [%+.3f .. %+.3f]s  rate %.1f Hz  %zu timeline points\n",
              human_sec(q.t0, base), human_sec(q.t1, base), q.rate_hz,
              r->timeline.size());
  for (const auto& s : r->sensors)
    std::printf("  sensor %-28s %zu cols, %zu raw rows read\n",
                s.stream_id.c_str(), s.columns.size(), s.raw_rows);
  size_t total_frames = 0;
  for (const auto& v : r->video) {
    std::printf("  video  %-28s %zu frames decoded\n", v.stream_id.c_str(),
                v.frames.size());
    total_frames += v.frames.size();
    if (!dump.empty()) {
      fs::create_directories(dump);
      for (const auto& f : v.frames) {
        char name[256];
        std::string sid = v.stream_id;
        for (auto& c : sid)
          if (c == '/' || c == ' ') c = '_';
        std::snprintf(name, sizeof(name), "%s_%+012.6fs.ppm", sid.c_str(),
                      human_sec(f.pts_ns, base));
        write_ppm(fs::path(dump) / name, f);
      }
    }
  }
  std::printf("files: %llu touched, %llu pruned at catalog stage\n",
              static_cast<unsigned long long>(r->files_touched),
              static_cast<unsigned long long>(r->files_pruned));
  std::printf("%s", r->io.summary(r->corpus_bytes).c_str());
  std::printf("wall %.2f ms\n", r->wall_ms);
  if (!dump.empty() && total_frames > 0)
    std::printf("dumped %zu frames to %s\n", total_frames, dump.c_str());
  return 0;
}

int cmd_query(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  const int snap = std::stoi(arg_value(args, "--snapshot", "-1"));
  const std::string text = arg_value(args, "--text");
  const std::string clip = arg_value(args, "--clip");
  const int k = std::stoi(arg_value(args, "-k", "10"));
  const int nprobe = std::stoi(arg_value(args, "--nprobe", "3"));
  const bool eval = arg_flag(args, "--eval-recall");
  if (store.empty() || (text.empty() == clip.empty())) usage();

  auto e = Engine::open(store, snap);
  if (!e) die(e.error());
  auto s = SemanticSearch::open(*e);
  if (!s) die(s.error());
  const TimeNs base = e->manifest().min_ts();

  Result<SemanticResult> r = fail(Errc::internal, "unreachable");
  if (!text.empty()) {
    r = s->query_text(text, k, nprobe, eval);
  } else {
    auto parts = split_csv(clip);
    if (parts.size() != 3) usage();
    r = s->query_clip(parts[0], parse_time(parts[1], base),
                      parse_time(parts[2], base), k, nprobe, eval);
  }
  if (!r) die(r.error());

  std::printf("top %zu hits (probed %u/%u clusters, scanned %llu/%llu vectors"
              " = %.1f%% of corpus)\n",
              r->hits.size(), r->stats.clusters_probed,
              r->stats.clusters_total,
              static_cast<unsigned long long>(r->stats.vectors_scanned),
              static_cast<unsigned long long>(r->stats.vectors_total),
              r->stats.vectors_total == 0
                  ? 0.0
                  : 100.0 * static_cast<double>(r->stats.vectors_scanned) /
                        static_cast<double>(r->stats.vectors_total));
  for (const auto& h : r->hits)
    std::printf("  %.4f  %-28s [%+.2f .. %+.2f]s   (sdx window --t0 +%.2f --t1 +%.2f)\n",
                h.score, h.window.stream_id.c_str(),
                human_sec(h.window.t0, base), human_sec(h.window.t1, base),
                human_sec(h.window.t0, base), human_sec(h.window.t1, base));
  if (r->prune_recall_at_k >= 0)
    std::printf("prune recall@%d vs exact scan: %.3f\n", k,
                r->prune_recall_at_k);
  return 0;
}

// Naive baseline (~the promised 50 lines): read EVERY byte of every file that
// overlaps the window, decode all video frames in those files, scan all
// sensor rows, then post-filter by time. This is what "no index, no format"
// costs, and it is the denominator that makes the elision number mean
// something.
struct NaiveCost {
  uint64_t bytes = 0;
  double wall_ms = 0;
};
NaiveCost naive_window(Engine& e, TimeNs t0, TimeNs t1) {
  const auto start = std::chrono::steady_clock::now();
  NaiveCost c;
  std::vector<char> buf(1 << 20);
  auto slurp = [&](const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    while (in.read(buf.data(), static_cast<std::streamsize>(buf.size())) ||
           in.gcount() > 0)
      c.bytes += static_cast<uint64_t>(in.gcount());
  };
  for (const auto& v : e.manifest().video)
    for (const auto& seg : v.segments)
      if (!(seg.last_pts_ns < t0 || seg.first_pts_ns > t1))
        slurp(seg.source_path);  // full-file read stands in for full decode
  for (const auto& s : e.manifest().sensors)
    if (!(s.max_ts < t0 || s.min_ts > t1))
      slurp((fs::path(e.store_dir()) / s.sdx_path).string());
  c.wall_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start)
                  .count();
  return c;
}

int cmd_bench(std::vector<std::string> args) {
  const std::string store = arg_value(args, "--store");
  const int snap = std::stoi(arg_value(args, "--snapshot", "-1"));
  const int nwin = std::stoi(arg_value(args, "--windows", "20"));
  const double dur = std::stod(arg_value(args, "--dur", "2"));
  const double rate = std::stod(arg_value(args, "--rate", "10"));
  const int nnaive = std::stoi(arg_value(args, "--naive", "2"));
  const uint64_t seed = std::stoull(arg_value(args, "--seed", "42"));
  const std::string out_path = arg_value(args, "--out", "bench_results.jsonl");
  if (store.empty()) usage();

  auto e = Engine::open(store, snap);
  if (!e) die(e.error());
  // Sample query windows where data actually lives: pick a stream weighted
  // by its span, then an offset inside it. Corpus [min,max] alone is wrong
  // once streams from different epochs coexist (a synthetic stream at epoch
  // 0 would turn the corpus range into decades of empty time).
  struct Range { TimeNs lo, hi; };
  std::vector<Range> ranges;
  const TimeNs dur_ns = static_cast<TimeNs>(dur * 1e9);
  for (const auto& v : e->manifest().video)
    for (const auto& s : v.segments)
      if (s.last_pts_ns - s.first_pts_ns > dur_ns)
        ranges.push_back({s.first_pts_ns, s.last_pts_ns - dur_ns});
  for (const auto& s : e->manifest().sensors)
    if (s.max_ts - s.min_ts > dur_ns) ranges.push_back({s.min_ts, s.max_ts - dur_ns});
  if (ranges.empty()) die({Errc::bad_argument, "corpus shorter than --dur"});
  std::vector<double> weights;
  for (const auto& r : ranges)
    weights.push_back(static_cast<double>(r.hi - r.lo));

  std::mt19937_64 rng(seed);
  std::discrete_distribution<size_t> pick_range(weights.begin(), weights.end());
  auto dist = [&](std::mt19937_64& g) {
    const Range& r = ranges[pick_range(g)];
    return std::uniform_int_distribution<TimeNs>(r.lo, r.hi)(g);
  };
  std::ofstream out(out_path, std::ios::app);

  std::vector<double> lat_ms;
  std::vector<double> elided;
  uint64_t frames = 0;
  for (int i = 0; i < nwin; ++i) {
    WindowQuery q;
    q.t0 = dist(rng);
    q.t1 = q.t0 + static_cast<TimeNs>(dur * 1e9);
    q.rate_hz = rate;
    q.video.out_width = 640;  // benchmark the analysis-ready path, not 4K blits
    auto r = e->get_window(q);
    if (!r) die(r.error());
    lat_ms.push_back(r->wall_ms);
    elided.push_back(r->elided_pct());
    for (const auto& v : r->video) frames += v.frames.size();
    out << "{\"kind\":\"window\",\"t0\":" << q.t0 << ",\"dur_s\":" << dur
        << ",\"wall_ms\":" << r->wall_ms << ",\"bytes_read\":"
        << r->io.total_bytes() << ",\"corpus_bytes\":" << r->corpus_bytes
        << ",\"elided_pct\":" << r->elided_pct() << "}\n";
  }
  std::sort(lat_ms.begin(), lat_ms.end());
  auto pct = [&](double p) {
    return lat_ms[std::min(lat_ms.size() - 1,
                           static_cast<size_t>(p * lat_ms.size()))];
  };
  const double mean_elided =
      std::accumulate(elided.begin(), elided.end(), 0.0) / elided.size();
  std::printf("get_window (%d windows of %.1fs, %.0f Hz, 640px):\n", nwin, dur,
              rate);
  std::printf("  latency p50 %.1f ms   p99 %.1f ms   (%llu frames decoded)\n",
              pct(0.50), pct(0.99), static_cast<unsigned long long>(frames));
  std::printf("  bytes elided: %.4f%% mean\n", mean_elided);

  for (int i = 0; i < nnaive; ++i) {
    WindowQuery q;
    q.t0 = dist(rng);
    q.t1 = q.t0 + static_cast<TimeNs>(dur * 1e9);
    auto naive = naive_window(*e, q.t0, q.t1);
    q.rate_hz = rate;
    q.video.out_width = 640;
    auto r = e->get_window(q);
    if (!r) die(r.error());
    std::printf("  naive baseline win %d: read %.2f GB in %.0f ms  vs  "
                "streetdex %.2f MB in %.1f ms  (%.0fx fewer bytes)\n",
                i, static_cast<double>(naive.bytes) / 1e9, naive.wall_ms,
                static_cast<double>(r->io.total_bytes()) / 1e6, r->wall_ms,
                static_cast<double>(naive.bytes) /
                    static_cast<double>(std::max<uint64_t>(
                        r->io.total_bytes(), 1)));
    out << "{\"kind\":\"naive\",\"t0\":" << q.t0 << ",\"dur_s\":" << dur
        << ",\"naive_bytes\":" << naive.bytes << ",\"naive_ms\":"
        << naive.wall_ms << ",\"sdx_bytes\":" << r->io.total_bytes()
        << ",\"sdx_ms\":" << r->wall_ms << "}\n";
  }
  std::printf("raw results appended to %s\n", out_path.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) usage();
  const std::string cmd = args[0];
  args.erase(args.begin());
  if (cmd == "--version" || cmd == "version") {
    std::printf("%s\n", kVersion);
    return 0;
  }
  if (cmd == "init") return cmd_init(std::move(args));
  if (cmd == "ingest") return cmd_ingest(std::move(args));
  if (cmd == "index-video") return cmd_index_video(std::move(args));
  if (cmd == "ingest-csv") return cmd_ingest_csv(std::move(args));
  if (cmd == "info") return cmd_info(std::move(args));
  if (cmd == "window") return cmd_window(std::move(args));
  if (cmd == "query") return cmd_query(std::move(args));
  if (cmd == "bench") return cmd_bench(std::move(args));
  usage();
} catch (const std::exception& ex) {
  std::fprintf(stderr, "fatal: %s\n", ex.what());
  return 1;
}
