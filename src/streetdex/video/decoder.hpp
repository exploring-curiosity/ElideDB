#pragma once
// GOP-range decoder: turns SFI byte ranges into decoded RGB frames while
// reading ONLY those byte ranges from the source file.
//
// Design position: at decode time we do NOT demux. The SFI frame table
// already records exact packet framing (byte_offset, packet_size, order), so
// the decoder preads the GOP's byte span and hands packets straight to
// avcodec. The container is opened once at construction purely to fetch
// codec parameters/extradata (a few KB, counted). This is late
// materialization applied to video: all pruning happens on the index; the
// expensive artifact (pixels) is produced last, and only for surviving rows.

#include <functional>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/time.hpp"
#include "streetdex/video/sfi.hpp"

namespace sdx {

struct FrameQueryOptions {
  int out_width = 0;  // 0 = native resolution; else scale (aspect preserved)
  int stride = 1;     // emit every Nth frame of the window
};

struct DecodedFrame {
  TimeNs pts_ns = 0;
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;  // HWC, RGB24, tightly packed
};

using FrameSink = std::function<void(DecodedFrame&&)>;

class VideoDecoder {
 public:
  // `sfi` must outlive the decoder. One decoder per source file, reused
  // across queries (single decode context — hot-path rule).
  static Result<VideoDecoder> open(const SfiReader& sfi);
  ~VideoDecoder();
  VideoDecoder(VideoDecoder&&) noexcept;
  VideoDecoder& operator=(VideoDecoder&&) = delete;
  VideoDecoder(const VideoDecoder&) = delete;

  // Decode frames with pts in [t0, t1] (closed), honoring stride/scale.
  // Reads exactly the byte ranges of the GOPs containing selected frames.
  Result<void> get_frames(TimeNs t0, TimeNs t1, const FrameQueryOptions& opts,
                          const FrameSink& sink);
  Result<std::vector<DecodedFrame>> get_frames_vec(TimeNs t0, TimeNs t1,
                                                   const FrameQueryOptions& o);

 private:
  VideoDecoder() = default;
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace sdx
