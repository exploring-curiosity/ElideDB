#pragma once
// Counting AVIO: every byte libav reads from a source video flows through
// this callback pair, so decode-time reads are counted exactly — including
// container headers, index chunks, and seek resyncs. This is the
// metrics::CountingReader for the video path; shelling out to ffmpeg could
// never give us this number, which is why libav is linked directly.

extern "C" {
#include <libavformat/avformat.h>
}

#include <cstdio>
#include <string>

#include "streetdex/core/error.hpp"
#include "streetdex/metrics/metrics.hpp"

namespace sdx {

struct CountingAvio {
  std::FILE* f = nullptr;
  int64_t size = 0;
  metrics::Cat cat = metrics::Cat::video_data;
  AVIOContext* ctx = nullptr;

  static int read(void* opaque, uint8_t* buf, int buf_size) {
    auto* self = static_cast<CountingAvio*>(opaque);
    const size_t n = std::fread(buf, 1, static_cast<size_t>(buf_size), self->f);
    if (n == 0) return AVERROR_EOF;
    metrics::count(self->cat, n);
    return static_cast<int>(n);
  }

  static int64_t seek(void* opaque, int64_t offset, int whence) {
    auto* self = static_cast<CountingAvio*>(opaque);
    if (whence == AVSEEK_SIZE) return self->size;
    whence &= ~AVSEEK_FORCE;
    if (::fseeko(self->f, offset, whence) != 0) return AVERROR(EIO);
    return ::ftello(self->f);
  }

  // Opens `path` and returns an AVFormatContext whose I/O is counted under
  // `cat`. Caller owns the returned context via close().
  // probe_streams: run avformat_find_stream_info (deep probe, can read tens
  // of MB). The ingest scan wants it for reliable metadata; the decoder does
  // NOT — it only needs codec_id from the header parse, and the codecs we
  // decode (MJPEG/H.26x) carry their own parameters in-band.
  Result<AVFormatContext*> open(const std::string& path, metrics::Cat category,
                                bool probe_streams = true);
  void close(AVFormatContext** fmt);
  ~CountingAvio();
};

inline Result<AVFormatContext*> CountingAvio::open(const std::string& path,
                                                   metrics::Cat category,
                                                   bool probe_streams) {
  av_log_set_level(AV_LOG_ERROR);  // probe-score chatter is not actionable
  cat = category;
  f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return fail(Errc::io, "cannot open " + path);
  ::fseeko(f, 0, SEEK_END);
  size = ::ftello(f);
  ::fseeko(f, 0, SEEK_SET);

  constexpr int kBufSize = 1 << 16;  // 64 KiB read granule
  auto* buf = static_cast<uint8_t*>(av_malloc(kBufSize));
  ctx = avio_alloc_context(buf, kBufSize, 0, this, &read, nullptr, &seek);
  if (ctx == nullptr) return fail(Errc::internal, "avio_alloc_context failed");

  AVFormatContext* fmt = avformat_alloc_context();
  fmt->pb = ctx;
  fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
  if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0)
    return fail(Errc::decode, "avformat_open_input failed: " + path);
  if (probe_streams && avformat_find_stream_info(fmt, nullptr) < 0) {
    avformat_close_input(&fmt);
    return fail(Errc::decode, "avformat_find_stream_info failed: " + path);
  }
  return fmt;
}

inline void CountingAvio::close(AVFormatContext** fmt) {
  if (fmt != nullptr && *fmt != nullptr) avformat_close_input(fmt);
  if (ctx != nullptr) {
    av_freep(&ctx->buffer);
    avio_context_free(&ctx);
  }
  if (f != nullptr) {
    std::fclose(f);
    f = nullptr;
  }
}

inline CountingAvio::~CountingAvio() {
  if (ctx != nullptr) {
    av_freep(&ctx->buffer);
    avio_context_free(&ctx);
  }
  if (f != nullptr) std::fclose(f);
}

}  // namespace sdx
