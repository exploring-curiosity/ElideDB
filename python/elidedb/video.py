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
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
         "-show_entries", "packet=pos,size,flags,pts",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True)
    doc = json.loads(r.stdout)
    stream = doc["streams"][0]
    pk = doc["packets"]
    n = len(pk)
    if timestamps_ns is not None and len(timestamps_ns) < n:
        n = len(timestamps_ns)  # truncated sidecar: index the known prefix
    ts = (list(timestamps_ns[:n]) if timestamps_ns is not None
          else [int(float(p.get("pts", i)) * 1e9) for i, p in enumerate(pk[:n])])
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

    def decode(self, stream=None, stride=1, width=None, limit=None):
        """pread + decode selected frames → list of (ts_ns, np.uint8 HWC).
        Reads EXACTLY the packet byte ranges of selected rows. MJPEG packets
        are standalone JPEGs (PIL); other codecs would need a codec context
        seeded from the keyframe — MJPEG-family is what the current corpora
        use, and the index schema already carries what a GOP decoder needs."""
        from PIL import Image
        rows = self.rows
        if stream is not None and "stream" in rows.column_names:
            import pyarrow.compute as pc
            rows = rows.filter(pc.equal(rows.column("stream"), stream))
        idx = range(0, len(rows), stride)
        if limit:
            idx = list(idx)[:limit]
        out = []
        bytes_read = 0
        handles = {}
        try:
            for i in idx:
                src = rows.column("source")[i].as_py()
                if src not in handles:
                    handles[src] = open(src, "rb")
                f = handles[src]
                f.seek(rows.column("byte_offset")[i].as_py())
                buf = f.read(rows.column("packet_size")[i].as_py())
                bytes_read += len(buf)
                try:
                    img = Image.open(io.BytesIO(buf))
                    if width and img.width > width:
                        img.draft("RGB", (width, width * 4))
                    img = img.convert("RGB")
                    if width and img.width > width:
                        img.thumbnail((width, width * 4))
                    out.append((rows.column("ts")[i].as_py(),
                                np.asarray(img, dtype=np.uint8)))
                except Exception:
                    pass  # torn tail packet of a mid-write segment
        finally:
            for f in handles.values():
                f.close()
        self.last_bytes_read = bytes_read
        return out
