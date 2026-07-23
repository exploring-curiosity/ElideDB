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
    # The database size is the STORE DIRECTORY: managed media + parquet
    # tables + logs. A standalone store carries everything; only stores with
    # reference-in-place media (copy=False ingest) have external bytes, and
    # those are flagged separately, never mixed into the database size.
    media_bytes = sum(p.stat().st_size
                      for p in (db.dir / "media").glob("*")
                      if p.is_file()) if (db.dir / "media").is_dir() else 0
    db_bytes = sum(p.stat().st_size for p in db.dir.rglob("*") if p.is_file())
    external_bytes = 0
    for d in desc:
        if d["kind"] != "frame_index":
            continue
        t = db.table(d["table"]).scan(columns=["source"])
        for s in set(t.column("source").to_pylist()):
            if not s.startswith("@") and Path(s).exists():
                external_bytes += Path(s).stat().st_size
    return {
        "key": key, "name": db.name, "path": str(db.dir),
        "tables": desc, "rows": total_rows, "bytes": total_bytes,
        "db_bytes": db_bytes, "media_bytes": media_bytes,
        "emb_bytes": emb["bytes"] if emb else 0,
        "external_bytes": external_bytes,
        "min_ts": lo, "max_ts": hi,
        "windows": emb["rows"] if emb else 0,
        "model": (emb or {}).get("meta", {}).get("model", ""),
        "display": db.meta.get("display", {}),
    }


def api_storage(key: str, table: str):
    """The Parquet format, made visible: this table's commit log plus every
    active file's row-group layout straight from the Parquet footers."""
    import pyarrow.parquet as pq
    db = STORES[key]
    tab = db.table(table)
    st = tab.state()
    files = []
    for f in st.files:
        pf = pq.ParquetFile(db.dir / "tables" / table / f.path)
        md = pf.metadata
        names = md.schema.names
        ts_i = names.index("ts") if "ts" in names else 0
        rgs = []
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            s = rg.column(ts_i).statistics
            rgs.append({"rows": rg.num_rows,
                        "bytes": sum(rg.column(c).total_compressed_size
                                     for c in range(rg.num_columns)),
                        "min_ts": s.min if s else None,
                        "max_ts": s.max if s else None})
        files.append({"name": f.path, "bytes": f.bytes, "rows": f.rows,
                      "min_ts": f.min_ts, "max_ts": f.max_ts,
                      "footer_bytes": md.serialized_size,
                      "columns": names, "row_groups": rgs})
    return {"table": table, "kind": st.kind, "version": st.version,
            "schema": st.schema, "history": tab.history(), "files": files}


MAP_MAX_POINTS = 6000


def api_map(key: str):
    """2D layout of the embeddings table. UMAP over a bounded SAMPLE,
    cached beside the log (a derived view; never used for retrieval).

    The unsampled version killed the app at pilot scale, three ways at
    once (measured on the 100 h store): to_pylist() over 180k x 1152
    vectors is ~25 GB of Python floats (4 -> 28 GB RSS), UMAP over 180k
    points runs for minutes, and a 180k-point JSON payload crushes the
    WebView. A map is an OVERVIEW: an even time-stride sample of a few
    thousand windows shows the same structure, reads from the mmap
    sidecar, and stays bounded no matter how large the corpus grows."""
    ck = ("map", key)
    if ck in _CACHE:
        return _CACHE[ck]
    db = STORES[key]
    from elidedb.embeddings import _vec_table
    try:
        t, vecs = _vec_table(db, "embeddings")
    except Exception:
        return {"points": []}
    n = len(t)
    if n == 0:
        return {"points": []}
    stride = max(1, n // MAP_MAX_POINTS)
    idx = np.arange(0, n, stride)
    sample = np.asarray(vecs[idx], np.float32)

    cache_file = db.dir / "tables" / "embeddings" / "_desk_umap.json"
    st = db.table("embeddings").state()
    xy = None
    if cache_file.exists():
        c = json.loads(cache_file.read_text())
        if c.get("version") == st.version and len(c["xy"]) == len(idx):
            xy = np.array(c["xy"], np.float32)
    if xy is None:
        try:
            import umap
            from sklearn.decomposition import PCA
            red = PCA(n_components=min(50, len(sample), sample.shape[1]),
                      random_state=0).fit_transform(sample)
            xy = umap.UMAP(n_components=2, random_state=0,
                           low_memory=True).fit_transform(red)
        except Exception:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=0).fit_transform(sample)
        cache_file.write_text(json.dumps(
            {"version": st.version, "xy": np.round(xy, 3).tolist()}))
    labels = (t.column("cluster").to_pylist()
              if "cluster" in t.column_names else None)
    ss = t.column("stream").to_pylist()
    ta = t.column("ts").to_pylist()
    tb = t.column("t1").to_pylist()
    out = {"points": [
        {"x": float(xy[j][0]), "y": float(xy[j][1]),
         "c": int(labels[i]) if labels else 0,
         "s": ss[i], "t0": ta[i], "t1": tb[i]}
        for j, i in enumerate(idx)],
        "sampled_of": n, "stride": int(stride)}
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
    # Merged segments can be minutes long; the player previews the first 30 s
    # rather than refusing (full-range export belongs to the Python API).
    t1 = min(t1, t0 + 30_000_000_000)
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
    # Every failure below names itself. A <video> element cannot render an
    # error body, so the player fetches the clip and shows these strings —
    # "could not build a clip" with no reason is not a diagnosis.
    if fs is None or len(fs) == 0:
        return {"error": "no frames indexed in this window",
                "detail": f"{stream or 'all streams'} "
                          f"{(t1 - t0) / 1e9:.2f}s window"}
    have = fs.streams()
    if stream and stream not in have:
        return {"error": f"stream '{stream}' has no frames here",
                "detail": f"streams present in this window: "
                          f"{', '.join(have) or 'none'}"}
    try:
        decoded = fs.decode(stream=stream or None, width=width)
    except Exception as e:
        return {"error": f"decode failed: {type(e).__name__}", "detail": str(e)}
    if len(decoded) < 2:
        return {"error": "not enough decodable frames for a clip",
                "detail": f"{len(decoded)} frame(s) decoded from "
                          f"{len(fs)} indexed"}
    rot = db.meta.get("display", {}).get("rotate", 0)
    span_s = max((decoded[-1][0] - decoded[0][0]) / 1e9, 0.1)
    fps = max(round((len(decoded) - 1) / span_s, 2), 1)

    wav = _audio_for_stream(db, stream, decoded[0][0], decoded[-1][0])
    wav_path = None
    if wav:
        wav_path = cache_dir / f"{ck}.wav"
        wav_path.write_bytes(wav)
    from .fftools import find
    cmd = [find("ffmpeg"), "-v", "error", "-y",
           "-f", "image2pipe", "-framerate", str(fps), "-i", "-"]
    if wav_path:
        cmd += ["-i", str(wav_path)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        for (_ts, arr) in decoded:
            img = Image.fromarray(arr)
            if rot:
                img = img.rotate(rot, expand=True)
            # libx264 yuv420 needs even dimensions
            if img.width % 2 or img.height % 2:
                img = img.crop((0, 0, img.width & ~1, img.height & ~1))
            img.save(proc.stdin, "JPEG", quality=90)
        proc.stdin.close()
    except BrokenPipeError:
        pass  # ffmpeg died early; the stderr below explains why
    err = proc.stderr.read().decode(errors="replace").strip()
    proc.wait()
    if wav_path:
        wav_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_path.exists():
        return {"error": "ffmpeg could not mux the clip",
                "detail": err or "no output produced"}
    return out_path.read_bytes()


def build_id() -> str:
    """Identity of the code this server is actually running.

    The launcher reuses whatever already listens on the port, so a server
    started from an older checkout keeps serving stale code forever — Python
    caches modules at import, so editing files changes nothing until restart.
    That is invisible from the browser and produces bug reports about
    behaviour that no longer exists in the source. The launcher now compares
    this against the on-disk files and restarts on a mismatch.
    """
    import hashlib
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")) + \
            [Path(__file__).parent / "desk_ui.html"]:
        if f.exists():
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def api_schema(key: str, table: str | None = None):
    """What the data actually IS: columns, types, and real sample rows.
    The first thing anyone opening a database wants to see."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    db = STORES[key]
    names = [table] if table else db.tables()
    out = []
    for name in names:
        st = db.table(name).state()
        if not st.files:
            continue
        pf = pq.ParquetFile(db.dir / "tables" / name / st.files[0].path)
        schema = pf.schema_arrow
        cols = []
        for f in schema:
            t = str(f.type)
            if pa.types.is_fixed_size_list(f.type):
                t = f"vector[{f.type.list_size}]"
            cols.append({"name": f.name, "type": t})
        sample = []
        if table:  # only materialise rows for the focused table
            head = db.table(name).scan(columns=[c["name"] for c in cols
                                                if not c["type"].startswith("vector")]
                                       ).slice(0, 8).to_pylist()
            for row in head:
                sample.append({k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in row.items()})
        out.append({"table": name, "kind": st.kind, "rows": st.rows,
                    "bytes": st.bytes, "version": st.version,
                    "min_ts": st.min_ts, "max_ts": st.max_ts,
                    "columns": cols, "sample": sample})
    return out


def api_indexes(key: str):
    """Every index in the store: B+ trees per table + ANN tiers on
    embeddings, with size and the data version each was built for."""
    db = STORES[key]
    out = {"bptree": [], "ann": []}
    for name in db.tables():
        ixdir = db.dir / "tables" / name / "_index"
        if not ixdir.is_dir():
            continue
        for p in ixdir.glob("*.bpt"):
            col, ver = p.stem.rsplit(".v", 1)
            out["bptree"].append({"table": name, "column": col,
                                  "version": int(ver),
                                  "bytes": p.stat().st_size})
        for p in list(ixdir.glob("hnsw.v*.bin")) + list(ixdir.glob("ivfpq.v*.npz")):
            kind = "hnsw" if p.name.startswith("hnsw") else "ivfpq"
            ver = int(p.name.split(".v")[1].split(".")[0])
            cur = db.table("embeddings").state().version
            out["ann"].append({"kind": kind, "version": ver,
                               "current": cur, "stale": ver != cur,
                               "bytes": p.stat().st_size})
    # candidate numeric columns for B+ indexing
    out["indexable"] = {}
    for name in db.tables():
        st = db.table(name).state()
        if st.kind != "timeseries" or not st.files:
            continue
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(db.dir / "tables" / name / st.files[0].path) \
            .schema_arrow
        import pyarrow as pa
        cols = [f.name for f in schema
                if f.name != "ts" and (pa.types.is_integer(f.type)
                                       or pa.types.is_floating(f.type))]
        if cols:
            out["indexable"][name] = cols
    out["has_embeddings"] = "embeddings" in db.tables() and \
        db.table("embeddings").state().rows > 0
    return out


def api_build_index(key: str, body: dict):
    db = STORES[key]
    t0 = time.perf_counter()
    if body.get("ann"):
        from . import ann
        r = (ann.build_hnsw(db) if body["ann"] == "hnsw"
             else ann.build_ivfpq(db))
        r["ms"] = round((time.perf_counter() - t0) * 1e3)
        return r
    r = db.table(body["table"]).create_index(body["column"])
    r["ms"] = round((time.perf_counter() - t0) * 1e3)
    return r


def api_maintenance(key: str, body: dict):
    db = STORES[key]
    op = body["op"]
    t0 = time.perf_counter()
    if op == "compact":
        r = db.table(body["table"]).compact()
    elif op == "vacuum":
        r = db.vacuum(retain_versions=int(body.get("retain", 3)),
                      dry_run=bool(body.get("dry_run", False)))
    elif op == "delete_range":
        r = db.table(body["table"]).delete_range(int(body["t0"]),
                                                 int(body["t1"]))
    else:
        raise ValueError(f"unknown maintenance op {op!r}")
    r["ms"] = round((time.perf_counter() - t0) * 1e3)
    return r


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
        floor = body.get("floor")
        kw = {}
        if body.get("floor_mode") == "percentile" and floor is not None:
            kw["percentile"] = float(floor)
        elif body.get("floor_mode") == "min_score" and floor is not None:
            kw["min_score"] = float(floor)
        hits, stats = db.search(
            body["text"], k=int(body.get("k", 8)),
            nprobe=int(body.get("nprobe", 3)),
            method=body.get("method", "auto"),
            neg_weight=float(body.get("neg_weight", 0.5)),
            t0=body.get("t0"), t1=body.get("t1"),
            streams=body.get("streams") or None,
            rerank=bool(body.get("rerank")),
            rerank_top=int(body.get("rerank_top", 10)), **kw)
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "context":
        hits, stats = db.search_context(
            body["text"], k=int(body.get("k", 8)),
            pool=int(body.get("pool", 48)),
            deep=6 if body.get("rerank") else 0,
            t0=body.get("t0"), t1=body.get("t1"),
            streams=body.get("streams") or None)
        # a result you can check: attach the clip's caption when the store
        # has captions
        try:
            for h in hits:
                ex = db.explain(h["t0"], h["t1"], h["stream"])
                if ex:
                    h["caption"] = ex[0]["caption"]
        except Exception:
            pass
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "predicate":
        from elidedb.store import QueryStats
        qs = QueryStats()
        tab = db.table(body["table"])
        vals = [float(body["value"])] if body.get("value2") in (None, "") \
            else [float(body["value"]), float(body["value2"])]
        out = tab.where(body["column"], body["op"], *vals, stats=qs).to_pandas()
        return {"columns": list(out.columns[:8]),
                "rows": json.loads(out.head(50).iloc[:, :8].to_json(
                    orient="values")),
                "count": len(out),
                "stats": {"files": f"{qs.files_touched}/{qs.files_total}",
                          "bytes_touched": qs.bytes_touched,
                          "corpus_bytes": qs.corpus_bytes,
                          "elided_pct": round(qs.elided_pct, 3)},
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "clip":
        hits, stats = db.search_clip(body["stream"], int(body["t0"]),
                                     int(body["t1"]), k=int(body.get("k", 8)),
                                     method=body.get("method", "auto"))
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
            if u.path == "/api/storage":
                return self._json(api_storage(q["store"], q["table"]))
            if u.path == "/api/indexes":
                return self._json(api_indexes(q["store"]))
            if u.path == "/api/schema":
                return self._json(api_schema(q["store"], q.get("table")))
            if u.path == "/api/thumb":
                jpg = api_thumb(q["store"], q.get("stream", ""),
                                int(q["t"]), int(q.get("w", "360")))
                if jpg is None:
                    return self._json({"error": "no frame"}, 404)
                return self._send(200, jpg, "image/jpeg", cache=True)
            if u.path == "/api/version":
                return self._json({"build": build_id()})
            if u.path == "/api/clip":
                mp4 = api_clip(q["store"], q.get("stream", ""),
                               int(q["t0"]), int(q["t1"]),
                               int(q.get("w", "640")))
                if isinstance(mp4, dict):        # structured failure
                    return self._json(mp4, 422)
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
            if u.path == "/api/build_index":
                return self._json(api_build_index(body["store"], body))
            if u.path == "/api/maintenance":
                return self._json(api_maintenance(body["store"], body))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def _warm():
    """Pre-build the per-store matrix caches AND load the text tower so the
    FIRST query of a session is as fast as the hundredth. Off the request
    path; measured: an unwarmed first query pays ~1 s of model load."""
    try:
        from elidedb.embeddings import embed_text
        embed_text("warmup")
    except Exception:
        pass
    for key, db in list(STORES.items()):
        try:
            from elidedb.embeddings import _vec_table
            _vec_table(db, "embeddings")
            from elidedb.verified import _recording_spans, _verdict_map
            _verdict_map(db)
            _recording_spans(db)
        except Exception:
            pass
        try:
            _vec_table(db, "motion_vectors")   # builds the mmap sidecar
        except Exception:
            pass


def main():
    global LAKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LAKE))
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    LAKE = Path(args.root).resolve()
    discover()
    threading.Thread(target=_warm, daemon=True).start()
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
