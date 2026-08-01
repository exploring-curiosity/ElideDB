extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cstdio>

#include "streetdex/metrics/metrics.hpp"
#include "streetdex/video/avio_counting.hpp"
#include "streetdex/video/decoder.hpp"

namespace sdx {

struct VideoDecoder::Impl {
  const SfiReader* sfi = nullptr;
  std::FILE* src = nullptr;  // raw source reads: pread of GOP byte ranges only
  AVCodecContext* codec = nullptr;
  SwsContext* sws = nullptr;
  int sws_out_w = -1, sws_out_h = -1;
  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;  // reused across all decodes (hot-path rule)
  std::vector<uint8_t> gop_buf;

  ~Impl() {
    if (src != nullptr) std::fclose(src);
    if (codec != nullptr) avcodec_free_context(&codec);
    if (sws != nullptr) sws_freeContext(sws);
    if (pkt != nullptr) av_packet_free(&pkt);
    if (frame != nullptr) av_frame_free(&frame);
  }

  Result<void> read_span(uint64_t offset, uint64_t length) {
    gop_buf.resize(length);
    if (::fseeko(src, static_cast<off_t>(offset), SEEK_SET) != 0 ||
        std::fread(gop_buf.data(), 1, length, src) != length)
      return fail(Errc::io, "source read failed at offset " +
                                std::to_string(offset));
    metrics::count(metrics::Cat::video_data, length);
    return {};
  }

  Result<void> ensure_sws(int in_w, int in_h, AVPixelFormat in_fmt, int out_w,
                          int out_h) {
    if (sws != nullptr && out_w == sws_out_w && out_h == sws_out_h) return {};
    if (sws != nullptr) sws_freeContext(sws);
    sws = sws_getContext(in_w, in_h, in_fmt, out_w, out_h, AV_PIX_FMT_RGB24,
                         SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws == nullptr) return fail(Errc::decode, "sws_getContext failed");
    sws_out_w = out_w;
    sws_out_h = out_h;
    return {};
  }

  Result<void> emit(AVFrame* f, const FrameQueryOptions& opts,
                    const FrameSink& sink) {
    int out_w = f->width, out_h = f->height;
    if (opts.out_width > 0 && opts.out_width < f->width) {
      out_w = opts.out_width;
      out_h = static_cast<int>(static_cast<int64_t>(f->height) * out_w /
                               f->width);
      out_h &= ~1;  // keep even for chroma subsampled sources
      if (out_h == 0) out_h = 2;
    }
    if (auto r = ensure_sws(f->width, f->height,
                            static_cast<AVPixelFormat>(f->format), out_w, out_h);
        !r)
      return r;
    DecodedFrame out;
    out.pts_ns = f->pts;  // we set pkt->pts = canonical ns; reorder-safe
    out.width = out_w;
    out.height = out_h;
    out.rgb.resize(static_cast<size_t>(out_w) * out_h * 3);
    uint8_t* dst[4] = {out.rgb.data(), nullptr, nullptr, nullptr};
    int dst_stride[4] = {out_w * 3, 0, 0, 0};
    sws_scale(sws, f->data, f->linesize, 0, f->height, dst, dst_stride);
    sink(std::move(out));
    return {};
  }
};

Result<VideoDecoder> VideoDecoder::open(const SfiReader& sfi) {
  VideoDecoder d;
  d.impl_ = new Impl();
  auto& im = *d.impl_;
  im.sfi = &sfi;

  // SFI stores codec_id, and intra-refresh codecs like MJPEG carry all their
  // parameters in-band — so the decoder can be built with ZERO container
  // I/O. Only codecs whose parameter sets live in container extradata
  // (H.264/HEVC in MP4) need the one-time header open below.
  av_log_set_level(AV_LOG_ERROR);
  const AVCodecID codec_id = static_cast<AVCodecID>(sfi.header().codec_id);
  const AVCodec* dec = avcodec_find_decoder(codec_id);
  if (dec == nullptr)
    return fail(Errc::decode, "no decoder for codec id " +
                                  std::to_string(sfi.header().codec_id));
  im.codec = avcodec_alloc_context3(dec);
  const bool needs_extradata =
      codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC ||
      codec_id == AV_CODEC_ID_AV1 || codec_id == AV_CODEC_ID_VP9;
  if (needs_extradata) {
    CountingAvio avio;
    auto fmt_r = avio.open(sfi.header().source_path, metrics::Cat::video_data,
                           /*probe_streams=*/false);
    if (!fmt_r) return tl::unexpected(fmt_r.error());
    AVFormatContext* fmt = *fmt_r;
    int vstream =
        av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vstream < 0) {
      avio.close(&fmt);
      return fail(Errc::decode, "no video stream: " + sfi.header().source_path);
    }
    const int rc =
        avcodec_parameters_to_context(im.codec, fmt->streams[vstream]->codecpar);
    avio.close(&fmt);
    if (rc < 0) return fail(Errc::decode, "cannot copy codec parameters");
  }
  if (avcodec_open2(im.codec, dec, nullptr) < 0)
    return fail(Errc::decode, "cannot open decoder");

  im.src = std::fopen(sfi.header().source_path.c_str(), "rb");
  if (im.src == nullptr)
    return fail(Errc::io, "cannot open source " + sfi.header().source_path);
  im.pkt = av_packet_alloc();
  im.frame = av_frame_alloc();
  return d;
}

VideoDecoder::~VideoDecoder() { delete impl_; }
VideoDecoder::VideoDecoder(VideoDecoder&& o) noexcept : impl_(o.impl_) {
  o.impl_ = nullptr;
}

Result<void> VideoDecoder::get_frames(TimeNs t0, TimeNs t1,
                                      const FrameQueryOptions& opts,
                                      const FrameSink& sink) {
  if (t0 > t1) return fail(Errc::bad_argument, "t0 > t1");
  auto& im = *impl_;
  const auto frames = im.sfi->frames();
  const auto [row_a, row_b] = im.sfi->frame_rows_in_range(t0, t1);
  if (row_a >= row_b) return {};
  const int stride = opts.stride < 1 ? 1 : opts.stride;

  // Walk selected rows; group by GOP so each GOP's bytes are read and its
  // packets sent exactly once. With stride > 1, whole unselected GOPs are
  // skipped entirely — zero bytes read for them (elision via the index).
  uint64_t next_selected = row_a;
  const auto gops = im.sfi->gops();
  bool decoder_dirty = false;  // needs flush when we jump between GOPs

  while (next_selected < row_b) {
    const uint32_t gop_id = frames[next_selected].gop_id;
    const SfiGopEntry& g = gops[gop_id];
    // Last selected row inside this GOP: early-stop point for the decode
    // loop (free elision inside the tail GOP — FORMAT.md §2.4).
    uint64_t last_in_gop = next_selected;
    {
      uint64_t r = next_selected;
      while (r < row_b && frames[r].gop_id == gop_id) {
        last_in_gop = r;
        r += stride;
      }
    }
    if (auto rd = im.read_span(g.start_byte, g.end_byte - g.start_byte); !rd)
      return rd;
    if (decoder_dirty) avcodec_flush_buffers(im.codec);
    decoder_dirty = true;

    // Send packets from GOP start (decode dependency) up to last_in_gop;
    // emit only selected rows with pts inside the window.
    const uint64_t send_end = last_in_gop + 1;
    for (uint64_t row = g.first_frame_row; row < send_end; ++row) {
      const SfiFrameEntry& fe = frames[row];
      im.pkt->data = im.gop_buf.data() + (fe.byte_offset - g.start_byte);
      im.pkt->size = static_cast<int>(fe.packet_size);
      im.pkt->pts = fe.pts_ns;  // canonical ns rides through reordering
      im.pkt->dts = fe.dts_ns;
      if (avcodec_send_packet(im.codec, im.pkt) < 0)
        return fail(Errc::decode, "send_packet failed (row " +
                                      std::to_string(row) + ")");
      while (avcodec_receive_frame(im.codec, im.frame) == 0) {
        const TimeNs pts = im.frame->pts;
        const bool selected =
            pts >= t0 && pts <= t1 &&
            (static_cast<uint64_t>(
                 std::lower_bound(frames.begin() + row_a,
                                  frames.begin() + row_b, pts,
                                  [](const SfiFrameEntry& f, TimeNs v) {
                                    return f.pts_ns < v;
                                  }) -
                 frames.begin() -
                 row_a) %
                 stride ==
             0);
        if (selected) {
          if (auto r = im.emit(im.frame, opts, sink); !r) {
            av_frame_unref(im.frame);
            return r;
          }
        }
        av_frame_unref(im.frame);
      }
    }
    // Drain frames still buffered (B-frame codecs); MJPEG returns 1:1 so
    // this is a no-op there.
    avcodec_send_packet(im.codec, nullptr);
    while (avcodec_receive_frame(im.codec, im.frame) == 0) {
      const TimeNs pts = im.frame->pts;
      if (pts >= t0 && pts <= t1) {
        const uint64_t idx =
            std::lower_bound(frames.begin() + row_a, frames.begin() + row_b,
                             pts,
                             [](const SfiFrameEntry& f, TimeNs v) {
                               return f.pts_ns < v;
                             }) -
            frames.begin() - row_a;
        if (idx % stride == 0)
          if (auto r = im.emit(im.frame, opts, sink); !r) return r;
      }
      av_frame_unref(im.frame);
    }
    avcodec_flush_buffers(im.codec);

    // Advance to the first selected row beyond this GOP.
    uint64_t r = next_selected;
    while (r < row_b && frames[r].gop_id == gop_id) r += stride;
    next_selected = r;
  }
  return {};
}

Result<std::vector<DecodedFrame>> VideoDecoder::get_frames_vec(
    TimeNs t0, TimeNs t1, const FrameQueryOptions& o) {
  std::vector<DecodedFrame> out;
  auto r = get_frames(t0, t1, o, [&](DecodedFrame&& f) {
    out.push_back(std::move(f));
  });
  if (!r) return tl::unexpected(r.error());
  return out;
}

}  // namespace sdx
