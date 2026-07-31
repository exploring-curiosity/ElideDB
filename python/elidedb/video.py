"""Video path: the frame index is a Parquet table; pixels stay in the
source files. A window query filters index ROWS (cheap, columnar), then
decode preads exactly the byte ranges those rows point at — late
materialization, media never copied or re-encoded into the store."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa

from .store import _uncached          # media pages are DATABASE pages:
# the pixel path moves far more bytes than every Parquet read combined,
# so leaving it on the OS page cache would have kept the cache the
# engine claims not to use. Same F_NOCACHE policy, same ELIDEDB_CACHE
# switch, scoped to the store's own media files.


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
                    handles[src] = _uncached(self._resolve(src))
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
        """GOP-granular decode for inter-codec elementary streams.

        The selection is partitioned into CONTIGUOUS RUNS and each run is read
        and decoded separately. That partitioning is the whole point: the
        earlier version read one span from the first selected frame's keyframe
        to the last selected packet, so a scattered selection — exactly what
        `stride=N` sampling produces — read and decoded the entire file to
        return a handful of frames. Measured on a 4103 s Bridge stream: 96
        frames spread across the file cost 188 ms/frame, against 2.7 ms/frame
        for a contiguous window, because 20,515 frames were being decoded to
        hand back 96.

        Runs whose byte ranges touch are merged, so a dense selection still
        becomes one large sequential read (one ffmpeg call, not one per GOP)
        while a sparse one becomes many small GOP reads. Both regimes read
        only the bytes they need.
        """
        import subprocess

        import cv2
        import pyarrow.compute as pc

        from .fftools import find
        src = rows.column("source")[sel[0]].as_py()
        w = rows.column("width")[0].as_py()
        h = rows.column("height")[0].as_py()

        # Full frame index for this source: needed to find each selected
        # frame's governing keyframe. An index read, not a media read.
        idx = self.store.table(self.table_name).scan()
        idx = idx.filter(pc.equal(idx.column("source"), src))
        ts_all = idx.column("ts").to_numpy()
        off_all = idx.column("byte_offset").to_numpy()
        size_all = idx.column("packet_size").to_numpy()
        key_all = np.asarray(idx.column("keyframe").to_pylist(), dtype=bool)
        key_pos = np.where(key_all)[0]
        if len(key_pos) == 0:
            key_pos = np.array([0])

        sel_ts = np.array([rows.column("ts")[i].as_py() for i in sel],
                          dtype=np.int64)
        pos = np.searchsorted(ts_all, sel_ts)
        pos = np.clip(pos, 0, len(ts_all) - 1)
        gov = key_pos[np.clip(np.searchsorted(key_pos, pos, side="right") - 1,
                              0, len(key_pos) - 1)]

        # Build merged [byte_start, byte_end) runs, each tagged with the index
        # position its first decoded frame corresponds to.
        runs = []
        for p, g in zip(pos, gov):
            a = int(off_all[g])
            b = int(off_all[p]) + int(size_all[p])
            if runs and a <= runs[-1]["end"]:
                runs[-1]["end"] = max(runs[-1]["end"], b)
                runs[-1]["want"].add(int(p))
            else:
                runs.append({"start": a, "end": b, "first": int(g),
                             "want": {int(p)}})

        frame_bytes = w * h * 3
        ff = find("ffmpeg")
        # Each run begins at a keyframe, so the runs concatenate into one
        # valid elementary stream: N GOP reads still cost only ONE decoder
        # invocation. Process spawn was the dominant cost once the byte ranges
        # were correct (96 runs = 96 ffmpeg starts ~= 20 ms each).
        payloads, spans = [], []
        total = 0
        with _uncached(self._resolve(src)) as f:
            for r in runs:
                f.seek(r["start"])
                buf = f.read(r["end"] - r["start"])
                total += len(buf)
                payloads.append(buf)
                spans.append((r["first"], max(r["want"]) - r["first"] + 1,
                              r["want"]))
        self.last_bytes_read = total

        proc = subprocess.run(
            [ff, "-v", "error", "-f", codec, "-i", "pipe:0",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            input=b"".join(payloads), capture_output=True)
        n_out = len(proc.stdout) // frame_bytes
        expected = sum(n for _, n, _ in spans)

        def _take(buf, k):
            img = np.frombuffer(buf, np.uint8, count=frame_bytes,
                                offset=k * frame_bytes).reshape(h, w, 3)
            if width and w > width:
                nh = int(h * width / w)
                img = cv2.resize(img, (width, nh),
                                 interpolation=cv2.INTER_AREA)
            return img.copy()

        out = []
        if n_out == expected:
            base = 0
            for first, n, want in spans:
                for k in range(n):
                    ip = first + k
                    if ip in want:
                        out.append((int(ts_all[ip]), _take(proc.stdout,
                                                           base + k)))
                base += n
        else:
            # The concatenated decode did not line up (open GOPs, a truncated
            # tail). Fall back to decoding each run on its own rather than
            # returning frames under the wrong timestamps — a silently
            # misaligned frame is worse than a slow one.
            for buf, (first, _n, want) in zip(payloads, spans):
                pr = subprocess.run(
                    [ff, "-v", "error", "-f", codec, "-i", "pipe:0",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                    input=buf, capture_output=True)
                for k in range(len(pr.stdout) // frame_bytes):
                    ip = first + k
                    if ip in want:
                        out.append((int(ts_all[ip]), _take(pr.stdout, k)))
        out.sort(key=lambda x: x[0])
        return out
