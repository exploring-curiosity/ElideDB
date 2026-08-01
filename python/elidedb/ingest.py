"""Ingest primitives: the shared surface the write scripts build on.

WHY THIS MODULE EXISTS
----------------------
`scripts/full_write.py` imported nine names from `scripts/write_once.py`,
which is a SCRIPT with its own `main()`. That is the wrong dependency
direction: a runnable entry point had become a library, so importing it
pulled in argv parsing, module-level side effects and a second CLI, and
neither file could be changed without checking the other.

The same shape appears seven more times across `scripts/` - files
importing `bench_product` for a constant, `extract_events` for two
functions. Entry points should depend on the package; the package should
never depend on an entry point.

So the durable pieces live here, in `elidedb`, and the scripts become
thin: parse arguments, call in, report. Anything with a `main()` is a
program. Anything imported is a module. Not both.

WHAT IS HERE
------------
Corpus geometry (where the bytes are, what the clock is) and the two
loops every writer needs: streaming a file once, and cutting a demo's
frames out of a continuous timeline.

Dataset LAYOUT belongs in `corpus.json` (see corpus.py) rather than in
constants here; the constants below are the Bridge defaults kept so the
existing scripts keep working, and each is overridable.
"""
from __future__ import annotations

import subprocess

import numpy as np

from .fftools import find

# Bridge defaults. A different corpus supplies these through corpus.json
# rather than by editing a constant - see corpus.Corpus.
CAM = "observation.images.image_0"
EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
NGEOM = 12                      # frames per episode handed to geometry
GEOM_W, GEOM_H = 256, 192       # geometry runs on its own small raster
EMBED_W, EMBED_H = 192, 144     # FDNN-V's raster
GAP_S = 60.0                    # silence inserted between demos
CRF = 26                        # per-demo H.264 quality


def gapped(spans, gap_s=GAP_S):
    """Uniform gap between consecutive demos of a stream.

    Without it two adjacent demos are contiguous in time and a window
    query cannot express "this demo and not the next one". Returns
    {(stream, t0): shift_ns}.
    """
    gap = int(gap_s * 1e9)
    prev, shift = {}, {}
    for s in sorted(spans, key=lambda r: (r["stream"], r["t0"])):
        st = s["stream"]
        new = s["t0"] if st not in prev else prev[st] + gap
        shift[(st, s["t0"])] = new - s["t0"]
        prev[st] = new + (s["t1"] - s["t0"])
    return shift


def frame_owner(spans, base, fps=FPS):
    """{frame index -> episode id} for one file.

    An episode owns EXACTLY the n frames the corpus declares. Deriving
    the end by rounding t1 claims one extra frame at an inclusive bound -
    verified against the seek-based writer, which produced 30 frames for
    an episode where rounding produced 31, every other frame identical.
    Trust the declared length, not a rounded timestamp.
    """
    owner = {}
    for e in spans:
        i0 = int(round((e["t0"] - base) / 1e9 * fps))
        for i in range(i0, i0 + int(e["n"])):
            owner[i] = e["episode"]
    return owner


def decode_stream(path, width=None, height=None, fps=FPS, chunk_frames=16):
    """Decode a video ONCE, yielding raw RGB frames in batches.

    The write path decoded the same pixels three times - once to cut
    per-demo segments, once for the embedding raster, once again from
    the store's own segments for region proposal. Measured, the third
    was 41% of the presence stage and the second was the entire embed
    stage. Everything that needs pixels should ride ONE decode, which is
    what this yields.

    `width`/`height` None keeps the source resolution: consumers that
    want a smaller raster resize the frames they are given, rather than
    each opening its own decoder.
    """
    vf = [f"fps={fps}"]
    if width and height:
        vf.append(f"scale={width}:{height}")
    if width is None or height is None:
        w, h = probe_size(path)
        width, height = width or w, height or h
    fb = width * height * 3
    proc = subprocess.Popen(
        [find("ffmpeg"), "-v", "error", "-i", str(path),
         "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24",
         "pipe:1"], stdout=subprocess.PIPE, bufsize=fb * chunk_frames)
    buf = b""
    try:
        while True:
            data = proc.stdout.read(fb * chunk_frames - len(buf))
            if data:
                buf += data
            n = len(buf) // fb
            if n == 0 and not data:
                break
            if n == 0:
                continue
            yield np.frombuffer(buf[:n * fb], np.uint8).reshape(
                n, height, width, 3)
            buf = buf[n * fb:]
    finally:
        proc.stdout.close()
        proc.wait()


def probe_size(path):
    """(width, height) of a video, without decoding it."""
    r = subprocess.run(
        [find("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=True)
    w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
    return w, h


def encoder(path, width, height, fps=FPS, crf=CRF):
    """An ffmpeg process that takes RAW FRAMES on stdin.

    The store needs per-demo H.264 with exactly one IDR so a 2 s read is
    a byte range. That encode is unavoidable; the DECODE inside it is
    not - feeding already-decoded frames removes a whole pass over the
    source.

    -bf 0 keeps packet order equal to presentation order, which the
    frame index depends on; -g/-keyint_min large with -sc_threshold 0
    guarantees the single IDR.
    """
    return subprocess.Popen(
        [find("ffmpeg"), "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-bf", "0", "-g", "10000", "-keyint_min", "10000",
         "-sc_threshold", "0", "-f", "h264", str(path)],
        stdin=subprocess.PIPE)
