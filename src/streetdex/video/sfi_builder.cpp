extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <cstdio>
#include <vector>

#include "streetdex/core/binary_io.hpp"
#include "streetdex/video/avio_counting.hpp"
#include "streetdex/video/sfi.hpp"

namespace sdx {

namespace {

TimeNs rescale_to_ns(int64_t pts, AVRational tb) {
  return av_rescale_q(pts, tb, AVRational{1, 1000000000});
}

}  // namespace

Result<SfiBuildStats> build_sfi(const SfiBuildInput& in,
                                const std::string& out_path) {
  // Packet-level scan: byte offsets, sizes, keyframe flags — no decode.
  // This is the once-per-file cost that buys frame-exact random access
  // forever after.
  CountingAvio avio;
  auto fmt_r = avio.open(in.video_path, metrics::Cat::ingest);
  if (!fmt_r) return tl::unexpected(fmt_r.error());
  AVFormatContext* fmt = *fmt_r;

  int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (vstream < 0) {
    avio.close(&fmt);
    return fail(Errc::decode, "no video stream in " + in.video_path);
  }
  const AVStream* st = fmt->streams[vstream];
  const AVRational tb = st->time_base;

  std::vector<SfiFrameEntry> frames;
  std::vector<SfiGopEntry> gops;
  AVPacket* pkt = av_packet_alloc();
  SfiBuildStats stats;
  bool warned_pts = false;

  while (av_read_frame(fmt, pkt) >= 0) {
    if (pkt->stream_index != vstream) {
      av_packet_unref(pkt);
      continue;
    }
    const uint64_t row = frames.size();
    SfiFrameEntry fe{};
    // pts provenance: sidecar timing wins; container pts is the fallback.
    if (row < in.frame_pts_ns.size()) {
      fe.pts_ns = in.frame_pts_ns[row];
      fe.dts_ns = fe.pts_ns;  // MJPEG has no reordering; sidecar has no dts
    } else if (in.frame_pts_ns.empty()) {
      fe.pts_ns = rescale_to_ns(pkt->pts, tb);
      fe.dts_ns =
          pkt->dts == AV_NOPTS_VALUE ? fe.pts_ns : rescale_to_ns(pkt->dts, tb);
    } else {
      // Container has more frames than the timing sidecar knows about;
      // indexing them with invented times would poison alignment. Stop here —
      // the indexed prefix is still fully usable.
      if (!warned_pts) {
        std::fprintf(stderr,
                     "[sfi] %s: container frames exceed timing records "
                     "(%zu known); indexing prefix only\n",
                     in.video_path.c_str(), in.frame_pts_ns.size());
        warned_pts = true;
      }
      av_packet_unref(pkt);
      break;
    }
    if (pkt->pos < 0) {
      av_packet_unref(pkt);
      av_packet_free(&pkt);
      avio.close(&fmt);
      return fail(Errc::decode,
                  "container gives no packet byte positions: " + in.video_path);
    }
    fe.byte_offset = static_cast<uint64_t>(pkt->pos);
    fe.packet_size = static_cast<uint32_t>(pkt->size);
    const bool key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
    fe.flags = key ? 1u : 0u;

    if (key || gops.empty()) {
      // A GOP starts at every keyframe. Frames before the first keyframe
      // (broken stream) get an artificial GOP so invariants hold.
      SfiGopEntry ge{};
      ge.gop_id = static_cast<uint32_t>(gops.size());
      ge.start_byte = fe.byte_offset;
      ge.first_pts_ns = fe.pts_ns;
      ge.first_frame_row = static_cast<uint32_t>(row);
      gops.push_back(ge);
    }
    SfiGopEntry& g = gops.back();
    fe.gop_id = g.gop_id;
    g.frame_count += 1;
    g.end_byte = fe.byte_offset + fe.packet_size;
    g.last_pts_ns = std::max(g.last_pts_ns, fe.pts_ns);
    frames.push_back(fe);
    av_packet_unref(pkt);
  }
  av_packet_free(&pkt);

  SfiHeader h;
  h.codec_id = static_cast<uint32_t>(st->codecpar->codec_id);
  h.width = static_cast<uint32_t>(st->codecpar->width);
  h.height = static_cast<uint32_t>(st->codecpar->height);
  h.timebase_num = static_cast<uint32_t>(tb.num);
  h.timebase_den = static_cast<uint32_t>(tb.den);
  h.clock_offset_ns = in.clock_offset_ns;
  h.frame_count = frames.size();
  h.gop_count = gops.size();
  h.source_size = static_cast<uint64_t>(avio.size);
  h.source_path = in.video_path;
  if (!frames.empty()) {
    h.first_pts_ns = frames.front().pts_ns;
    h.last_pts_ns = frames.back().pts_ns;
  }
  avio.close(&fmt);
  if (frames.empty())
    return fail(Errc::decode, "no video packets in " + in.video_path);

  // ---- Serialize (layout: FORMAT.md §2.1) ----------------------------------
  ByteWriter w;
  w.put_bytes({reinterpret_cast<const uint8_t*>(kSfiMagic), 4});
  w.put<uint16_t>(kSfiVersion);
  w.put<uint16_t>(0);  // flags
  w.put<uint32_t>(h.codec_id);
  w.put<uint32_t>(h.width);
  w.put<uint32_t>(h.height);
  w.put<uint32_t>(h.timebase_num);
  w.put<uint32_t>(h.timebase_den);
  w.put<uint32_t>(0);  // pad
  w.put<int64_t>(h.first_pts_ns);
  w.put<int64_t>(h.last_pts_ns);
  w.put<int64_t>(h.clock_offset_ns);
  w.put<uint64_t>(h.frame_count);
  w.put<uint64_t>(h.gop_count);
  w.put<uint64_t>(h.source_size);
  w.put_string(h.source_path);
  w.align8();  // tables must be castable in place after mmap
  const uint64_t gop_table_offset = w.size();
  w.put_bytes({reinterpret_cast<const uint8_t*>(gops.data()),
               gops.size() * sizeof(SfiGopEntry)});
  const uint64_t frame_table_offset = w.size();
  w.put_bytes({reinterpret_cast<const uint8_t*>(frames.data()),
               frames.size() * sizeof(SfiFrameEntry)});
  w.put<uint64_t>(gop_table_offset);
  w.put<uint64_t>(frame_table_offset);
  w.put_bytes({reinterpret_cast<const uint8_t*>(kSfiMagic), 4});

  std::FILE* out = std::fopen(out_path.c_str(), "wb");
  if (out == nullptr) return fail(Errc::io, "cannot create " + out_path);
  const bool ok =
      std::fwrite(w.bytes().data(), 1, w.size(), out) == w.size() &&
      std::fflush(out) == 0;
  std::fclose(out);
  if (!ok) return fail(Errc::io, "write failed: " + out_path);

  stats.frames_indexed = frames.size();
  stats.gops = gops.size();
  stats.bytes_scanned = h.source_size;
  stats.timing_minus_container_frames =
      static_cast<int64_t>(in.frame_pts_ns.size()) -
      static_cast<int64_t>(frames.size());
  return stats;
}

}  // namespace sdx
