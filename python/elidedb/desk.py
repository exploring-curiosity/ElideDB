#!/usr/bin/env python3
"""ElideDB Desk — local database browser (the Atlas/Compass role).

Zero-dependency server: stdlib http.server + the elidedb package. All state
lives in the stores themselves; Desk only reads. Thumbnails are decoded on
demand through the same byte-range path queries use — nothing is pre-baked,
so every embedded window can always show its frame.

  elidedb desk [--root lake] [--port 8787] [--open]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path.cwd()

import numpy as np  # noqa: E402
from elidedb import Store  # noqa: E402

STORES: dict[str, Store] = {}
LAKE = ROOT / "lake"
_CACHE: dict = {}


def discover():
    STORES.clear()
    if LAKE.is_dir():
        for p in sorted(LAKE.iterdir()):
            if (p / "_store.json").exists():
                try:
                    STORES[p.name] = Store.open(p)
                except Exception:
                    pass


def store_summary(key: str):
    db = STORES[key]
    desc = db.describe()
    total_rows = sum(d["rows"] for d in desc)
    total_bytes = sum(d["bytes"] for d in desc)
    span_tabs = [d for d in desc if d["rows"] and d["table"] != "centroids"]
    lo = min((d["min_ts"] for d in span_tabs), default=0)
    hi = max((d["max_ts"] for d in span_tabs), default=0)
    emb = next((d for d in desc if d["table"] == "embeddings"), None)
    # source media referenced (not stored) by frame_index tables
    media_bytes = 0
    for d in desc:
        if d["kind"] != "frame_index":
            continue
        t = db.table(d["table"]).scan(columns=["source"])
        for s in set(t.column("source").to_pylist()):
            p = Path(s)
            if p.exists():
                media_bytes += p.stat().st_size
    return {
        "key": key, "name": db.name, "path": str(db.dir),
        "tables": desc, "rows": total_rows, "bytes": total_bytes,
        "media_bytes": media_bytes,
        "min_ts": lo, "max_ts": hi,
        "windows": emb["rows"] if emb else 0,
        "model": (emb or {}).get("meta", {}).get("model", ""),
        "display": db.meta.get("display", {}),
    }


def api_map(key: str):
    """2D layout of the embeddings table. UMAP, cached beside the log
    (a derived view, clearly named; never used for retrieval)."""
    ck = ("map", key)
    if ck in _CACHE:
        return _CACHE[ck]
    db = STORES[key]
    t = db.table("embeddings").scan()
    if len(t) == 0:
        return {"points": []}
    vecs = np.stack([np.asarray(v, np.float32)
                     for v in t.column("vector").to_pylist()])
    cache_file = db.dir / "tables" / "embeddings" / "_desk_umap.json"
    st = db.table("embeddings").state()
    xy = None
    if cache_file.exists():
        c = json.loads(cache_file.read_text())
        if c.get("version") == st.version and len(c["xy"]) == len(t):
            xy = np.array(c["xy"], np.float32)
    if xy is None:
        try:
            import umap
            from sklearn.decomposition import PCA
            red = PCA(n_components=min(50, len(vecs), vecs.shape[1]),
                      random_state=0).fit_transform(vecs)
            xy = umap.UMAP(n_components=2, random_state=0).fit_transform(red)
        except Exception:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=0).fit_transform(vecs)
        cache_file.write_text(json.dumps(
            {"version": st.version, "xy": np.round(xy, 3).tolist()}))
    labels = (t.column("cluster").to_pylist()
              if "cluster" in t.column_names else [0] * len(t))
    out = {"points": [
        {"x": float(xy[i][0]), "y": float(xy[i][1]), "c": int(labels[i]),
         "s": t.column("stream")[i].as_py(),
         "t0": t.column("ts")[i].as_py(), "t1": t.column("t1")[i].as_py()}
        for i in range(len(t))]}
    _CACHE[ck] = out
    return out


def api_geo(key: str):
    """Generic geo panel: any timeseries table with latitude+longitude."""
    db = STORES[key]
    for d in db.describe():
        if d["kind"] != "timeseries":
            continue
        cols = db.table(d["table"]).scan(columns=None)
        if {"latitude", "longitude"} <= set(cols.column_names):
            la = cols.column("latitude").to_numpy()
            lo = cols.column("longitude").to_numpy()
            ts = cols.column("ts").to_numpy()
            step = max(1, len(la) // 2500)
            return {"table": d["table"],
                    "points": [{"la": float(la[i]), "lo": float(lo[i]),
                                "t": int(ts[i])}
                               for i in range(0, len(la), step)]}
    return {"points": []}


def api_thumb(key: str, stream: str, t: int, width: int = 360):
    from PIL import Image
    db = STORES[key]
    win, _ = db.window(t - 2_000_000_000, t + 2_000_000_000, tables=["frames"])
    fs = win.get("frames")
    if fs is None or len(fs) == 0:
        return None
    decoded = fs.decode(stream=stream or None, width=width, limit=1)
    if not decoded:
        return None
    img = Image.fromarray(decoded[0][1])
    rot = db.meta.get("display", {}).get("rotate", 0)
    if rot:
        img = img.rotate(rot, expand=True)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _audio_for_stream(db, stream: str, t0: int, t1: int):
    """Find the sensor's audio table (e.g. 'Sensor 108/cam0' →
    sensor_108_audio), return mono WAV bytes for the window, or None."""
    import struct as _struct
    prefix = stream.split("/")[0].replace("/", "_").replace(" ", "_").lower()
    cand = f"{prefix}_audio"
    if cand not in db.tables():
        return None
    t = db.table(cand).scan(t0, t1, columns=["ts", "ch0"])
    if len(t) < 100:
        return None
    ts = t.column("ts").to_numpy()
    rate = int(round((len(ts) - 1) * 1e9 / max(int(ts[-1] - ts[0]), 1)))
    pcm = t.column("ch0").to_numpy().astype("<i2").tobytes()
    hdr = b"RIFF" + _struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + \
        _struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + \
        b"data" + _struct.pack("<I", len(pcm))
    return hdr + pcm


def api_clip(key: str, stream: str, t0: int, t1: int, width: int = 640):
    """Playable clip: byte-range decode of the window's frames → H.264 MP4
    (plus the sensor's microphone track when the store has one). Cached by
    parameters; the source media is only ever read, never touched."""
    import hashlib
    import subprocess
    import tempfile
    if t1 - t0 > 30_000_000_000:
        raise ValueError("clip window capped at 30 s")
    cache_dir = Path(tempfile.gettempdir()) / "elidedb_clips"
    cache_dir.mkdir(exist_ok=True)
    ck = hashlib.sha1(f"{key}|{stream}|{t0}|{t1}|{width}".encode()).hexdigest()
    out_path = cache_dir / f"{ck}.mp4"
    if out_path.exists():
        return out_path.read_bytes()

    from PIL import Image
    db = STORES[key]
    win, _ = db.window(t0, t1, tables=["frames"])
    fs = win.get("frames")
    if fs is None or len(fs) == 0:
        return None
    decoded = fs.decode(stream=stream or None, width=width)
    if len(decoded) < 2:
        return None
    rot = db.meta.get("display", {}).get("rotate", 0)
    span_s = max((decoded[-1][0] - decoded[0][0]) / 1e9, 0.1)
    fps = max(round((len(decoded) - 1) / span_s, 2), 1)

    wav = _audio_for_stream(db, stream, decoded[0][0], decoded[-1][0])
    wav_path = None
    if wav:
        wav_path = cache_dir / f"{ck}.wav"
        wav_path.write_bytes(wav)
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "image2pipe", "-framerate", str(fps), "-i", "-"]
    if wav_path:
        cmd += ["-i", str(wav_path)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for (_ts, arr) in decoded:
        img = Image.fromarray(arr)
        if rot:
            img = img.rotate(rot, expand=True)
        # libx264 yuv420 needs even dimensions
        if img.width % 2 or img.height % 2:
            img = img.crop((0, 0, img.width & ~1, img.height & ~1))
        img.save(proc.stdin, "JPEG", quality=90)
    proc.stdin.close()
    proc.wait()
    if wav_path:
        wav_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError("ffmpeg mux failed")
    return out_path.read_bytes()


def api_query(key: str, body: dict):
    db = STORES[key]
    kind = body.get("type")
    t_start = time.perf_counter()
    if kind == "sql":
        df = db.sql(body["sql"]).head(200)
        return {"columns": list(df.columns),
                "rows": json.loads(df.to_json(orient="values")),
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "text":
        hits, stats = db.search_text(body["text"], k=int(body.get("k", 8)),
                                     nprobe=int(body.get("nprobe", 3)))
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "clip":
        hits, stats = db.search_clip(body["stream"], int(body["t0"]),
                                     int(body["t1"]), k=int(body.get("k", 8)))
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "window":
        w, stats = db.window(int(body["t0"]), int(body["t1"]),
                             tables=body.get("tables") or None)
        out = {"tables": {}, "ms": round(stats.wall_ms, 1),
               "stats": {"files": f"{stats.files_touched}/{stats.files_total}",
                         "bytes_touched": stats.bytes_touched,
                         "corpus_bytes": stats.corpus_bytes,
                         "elided_pct": round(stats.elided_pct, 3),
                         "rows": stats.rows_returned}}
        from elidedb.video import FrameSet
        for name, v in w.items():
            if isinstance(v, FrameSet):
                out["tables"][name] = {"kind": "frames", "count": len(v),
                                       "streams": v.streams()}
            else:
                df = v.to_pandas().head(6)
                out["tables"][name] = {
                    "kind": "rows", "count": len(v),
                    "columns": list(df.columns),
                    "head": json.loads(df.to_json(orient="values"))}
        return out
    raise ValueError(f"unknown query type {kind!r}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, body, ctype):
        """Range-aware send: <video> elements (Safari especially) seek via
        byte-range requests; a byte-range database ought to honor them."""
        rng = self.headers.get("Range")
        total = len(body)
        if rng and rng.startswith("bytes="):
            spec = rng[6:].split("-")
            a = int(spec[0]) if spec[0] else 0
            b = int(spec[1]) if len(spec) > 1 and spec[1] else total - 1
            b = min(b, total - 1)
            chunk = body[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {a}-{b}/{total}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/" or u.path == "/index.html":
                html = (Path(__file__).parent / "desk_ui.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/api/stores":
                discover()
                return self._json([store_summary(k) for k in STORES])
            if u.path == "/api/map":
                return self._json(api_map(q["store"]))
            if u.path == "/api/geo":
                return self._json(api_geo(q["store"]))
            if u.path == "/api/history":
                db = STORES[q["store"]]
                return self._json(db.table(q["table"]).history())
            if u.path == "/api/thumb":
                jpg = api_thumb(q["store"], q.get("stream", ""),
                                int(q["t"]), int(q.get("w", "360")))
                if jpg is None:
                    return self._json({"error": "no frame"}, 404)
                return self._send(200, jpg, "image/jpeg", cache=True)
            if u.path == "/api/clip":
                mp4 = api_clip(q["store"], q.get("stream", ""),
                               int(q["t0"]), int(q["t1"]),
                               int(q.get("w", "640")))
                if mp4 is None:
                    return self._json({"error": "no frames in window"}, 404)
                return self._send_media(mp4, "video/mp4")
            return self._json({"error": "not found"}, 404)
        except Exception as e:  # surface, don't die
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/query":
                return self._json(api_query(body["store"], body))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global LAKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LAKE))
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    LAKE = Path(args.root).resolve()
    discover()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ElideDB Desk: http://localhost:{args.port}  "
          f"({len(STORES)} stores under {LAKE})")
    if args.open:
        import subprocess
        threading.Timer(0.4, lambda: subprocess.run(
            ["open", f"http://localhost:{args.port}"])).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
