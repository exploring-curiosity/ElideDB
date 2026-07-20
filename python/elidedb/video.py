"""Video path: the frame index is a Parquet table; pixels stay in the
source files. A window query filters index ROWS (cheap, columnar), then
decode preads exactly the byte ranges those rows point at — late
materialization, media never copied or re-encoded into the store."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa


def scan_video_packets(video_path, timestamps_ns=None) -> dict:
    """Packet-level scan (no decode) via PyAV if present, else ffprobe.
    Returns columnar dict for the frame_index schema."""
    import json
    import subprocess

    from .fftools import find
    r = subprocess.run(
        [find("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_packets",
         "-show_entries", "packet=pos,size,flags,pts",
         "-show_entries", "stream=codec_name,width,height,time_base",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True)
    doc = json.loads(r.stdout)
    stream = doc["streams"][0]
    pk = doc["packets"]
    n = len(pk)
    if timestamps_ns is not None and len(timestamps_ns) < n:
        n = len(timestamps_ns)  # truncated sidecar: index the known prefix
    if timestamps_ns is not None:
        ts = list(timestamps_ns[:n])
    else:
        # container pts are TIMEBASE TICKS, not seconds — convert via the
        # stream's time_base (e.g. AVI at 1/30: pts 0,1,2 = 0, 33.3, 66.7 ms)
        num, den = (int(x) for x in stream.get("time_base", "1/1000000000").split("/"))
        ts = [int(float(p.get("pts", i)) * num / den * 1e9)
              for i, p in enumerate(pk[:n])]
    return {
        "ts": pa.array(ts, pa.int64()),
        "byte_offset": pa.array([int(p["pos"]) for p in pk[:n]], pa.int64()),
        "packet_size": pa.array([int(p["size"]) for p in pk[:n]], pa.int32()),
        "keyframe": pa.array([p.get("flags", "").startswith("K") for p in pk[:n]]),
        "width": pa.array([stream["width"]] * n, pa.int32()),
        "height": pa.array([stream["height"]] * n, pa.int32()),
        "codec": pa.array([stream["codec_name"]] * n),
        "source": pa.array([str(Path(video_path).resolve())] * n),
    }


class FrameSet:
    """Lazy handle over frame_index rows in a query window."""

    def __init__(self, store, table_name, rows: pa.Table):
        self.store = store
        self.table_name = table_name
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        streams = (set(self.rows.column("stream").to_pylist())
                   if "stream" in self.rows.column_names else set())
        return f"<FrameSet {len(self.rows)} frames, streams={sorted(streams)}>"

    def streams(self):
        if "stream" not in self.rows.column_names:
            return []
        return sorted(set(self.rows.column("stream").to_pylist()))

    def _resolve(self, src: str) -> str:
        # "@media/..." = store-managed media (standalone store); anything
        # else is an external reference-in-place.
        return str(self.store.dir / src[1:]) if src.startswith("@") else src

    def decode(self, stream=None, stride=1, width=None, limit=None,
               workers=8):
        """pread + decode selected frames → list of (ts_ns, np.uint8 HWC).

        Reads EXACTLY the byte ranges the selection requires. Two paths:
        - intra codecs (mjpeg): each packet is a standalone JPEG — read the
          selected packets, decode them in parallel (cv2 releases the GIL).
        - inter codecs (hevc/h264 elementary streams, the compressed
          `transcode=` tier): decode is GOP-granular — read from the
          preceding keyframe through the last selected packet, pipe through
          ffmpeg, keep the selected frames. Keyframe interval = the
          random-access dial chosen at ingest."""
        import pyarrow.compute as pc
        rows = self.rows
        if stream is not None and "stream" in rows.column_names:
            rows = rows.filter(pc.equal(rows.column("stream"), stream))
        if len(rows) == 0:
            self.last_bytes_read = 0
            return []
        sel = list(range(0, len(rows), stride))
        if limit:
            sel = sel[:limit]
        codec = rows.column("codec")[0].as_py() if "codec" in rows.column_names \
            else "mjpeg"
        if codec in ("hevc", "h264"):
            return self._decode_gop(rows, sel, codec, width)

        # ---- intra path: parallel per-packet decode -------------------------
        import cv2
        from concurrent.futures import ThreadPoolExecutor
        offs = rows.column("byte_offset").to_pylist()
        sizes = rows.column("packet_size").to_pylist()
        tss = rows.column("ts").to_pylist()
        srcs = rows.column("source").to_pylist()
        bufs, bytes_read, handles = [], 0, {}
        try:
            for i in sel:
                src = srcs[i]
                if src not in handles:
                    handles[src] = open(self._resolve(src), "rb")
                f = handles[src]
                f.seek(offs[i])
                b = f.read(sizes[i])
                bytes_read += len(b)
                bufs.append((tss[i], b))
        finally:
            for f in handles.values():
                f.close()

        def _one(item):
            ts, b = item
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None  # torn tail packet of a mid-write segment
            if width and img.shape[1] > width:
                h = int(img.shape[0] * width / img.shape[1])
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            return (ts, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = [r for r in ex.map(_one, bufs) if r is not None]
        self.last_bytes_read = bytes_read
        return out

    def _decode_gop(self, rows, sel, codec, width):
        """GOP-range decode for elementary inter-codec streams: pread
        [preceding keyframe .. last selected packet], pipe to ffmpeg, keep
        the selected frames. Bytes read = the GOP span, nothing more."""
        import subprocess

        import cv2
        import pyarrow.compute as pc

        from .fftools import find
        first_ts = rows.column("ts")[sel[0]].as_py()
        last = sel[-1]
        src = rows.column("source")[sel[0]].as_py()
        w = rows.column("width")[0].as_py()
        h = rows.column("height")[0].as_py()

        # The window scan may start mid-GOP; fetch back to the keyframe from
        # the frame-index table (an index read, not a media read).
        full = self.store.table(self.table_name).scan(
            first_ts - 60_000_000_000, first_ts)
        m = pc.and_(pc.equal(full.column("source"), src),
                    pc.equal(full.column("keyframe"), True))
        heads = full.filter(m)
        key_off = int(rows.column("byte_offset")[sel[0]].as_py())
        key_ts = first_ts
        if not rows.column("keyframe")[sel[0]].as_py() and len(heads):
            key_off = int(heads.column("byte_offset")[-1].as_py())
            key_ts = heads.column("ts")[-1].as_py()
        end = (rows.column("byte_offset")[last].as_py() +
               rows.column("packet_size")[last].as_py())
        with open(self._resolve(src), "rb") as f:
            f.seek(key_off)
            payload = f.read(end - key_off)
        self.last_bytes_read = len(payload)

        proc = subprocess.run(
            [find("ffmpeg"), "-v", "error", "-f", codec, "-i", "pipe:0",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            input=payload, capture_output=True)
        frame_bytes = w * h * 3
        n_frames = len(proc.stdout) // frame_bytes

        # Frames come out in packet order from the keyframe; align them with
        # the index rows over [key_ts, last_ts] to recover timestamps.
        span = self.store.table(self.table_name).scan(
            key_ts, rows.column("ts")[last].as_py())
        span = span.filter(pc.equal(span.column("source"), src))
        span_ts = span.column("ts").to_pylist()[:n_frames]
        want = {rows.column("ts")[i].as_py() for i in sel}
        out = []
        for k, ts in enumerate(span_ts):
            if ts not in want:
                continue
            img = np.frombuffer(
                proc.stdout, np.uint8, count=frame_bytes,
                offset=k * frame_bytes).reshape(h, w, 3)
            if width and w > width:
                nh = int(h * width / w)
                img = cv2.resize(img, (width, nh),
                                 interpolation=cv2.INTER_AREA)
            out.append((ts, img.copy()))
        return out
