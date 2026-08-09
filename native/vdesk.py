#!/usr/bin/env python3
"""ElideDB Desk — the QbE surface for the vision-native stores.

    "I give you a sample, I want all samples like that."

Desk's original half serves the Parquet lake through `elidedb.Store`. That
object cannot open these stores: a vision store is a different thing -
window rows (media, t0, t1, scale, energy) beside a memory-mapped channel
column, coarse cells and a hubness column - and it has no tables, streams
or schema for the old UI to show. So this is its own server rather than a
branch inside the other one, sharing the console's visual language and
nothing of its store abstraction.

Two ways to ask, both of them the SAME query path:

  FROM THE STORE   pick a media, drag out a span. No upload, no ingest.
                   The example is decoded live from the source file
                   through the identical byte-range cut the write path
                   used, so what you hand the system is what it saw.
  A NEW VIDEO      drop a file in. It is decoded, encoded and searched -
                   and never written to a store. A query is not an
                   ingest, and raw data is immutable.

Everything the read path decided is shown, because a retrieval box that
only prints answers cannot be checked: the abstention threshold and how
many rows cleared it, the candidate pool the coarse cells opened, the
per-channel weights this query earned, and the bytes actually read
against the corpus size. That last pair is the whole thesis of the
project - the best read is the read elided - so it is on screen, not in
a log.

    python native/vdesk.py --open            # http://127.0.0.1:8788
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import vsrc                                               # noqa: E402
import vstore                                             # noqa: E402
from vstore import Store                                  # noqa: E402

# Thumbnails and preview clips are derived, disposable and binary, so
# they live under stores/ which is gitignored.
CACHE = ROOT / "stores/_deskcache"
UPLOADS = CACHE / "uploads"

# One encoder process-wide, and one lock around it. The model is a single
# torch module on one MPS device; two concurrent requests through it is
# not a throughput win on a compute-bound GPU (batch 32 was measured as
# the plateau), it is just a way to interleave two slow queries into two
# slower ones.
_ENC_LOCK = threading.Lock()
_MODEL = {"state": "cold", "err": None, "t": None}

_STORES: dict[str, Store] = {}
_SRCS: dict[str, dict] = {}          # store -> {media_id: Source}
_SRC_LOCK = threading.Lock()


# ----------------------------------------------------------------- stores

def discover():
    """Every vision store under stores/vision, by manifest."""
    _STORES.clear()
    base = ROOT / os.environ.get("SDX_STORES", "stores/vision")
    if not base.is_dir():
        return
    for p in sorted(base.iterdir()):
        if (p / "manifest.json").exists():
            try:
                _STORES[p.name] = Store(p)
            except Exception:                            # noqa: BLE001
                traceback.print_exc()


def sources(key):
    """Media sources for a store, opened once and kept.

    Opening a source ffprobes the file, so 150 of them is 150 process
    spawns; doing that on every request would make the media list feel
    broken. The store is immutable, so caching them is safe.
    """
    with _SRC_LOCK:
        if key not in _SRCS:
            _SRCS[key] = {s.id: s for s in vsrc.sources(key)}
        return _SRCS[key]


def store_summary(key):
    st = _STORES[key]
    m = st.man
    sz = sum(f.stat().st_size for f in st.path.rglob("*") if f.is_file())
    return dict(
        key=key, corpus=m["corpus"], media=m["media"], windows=m["windows"],
        minutes=round(m["seconds"] / 60.0, 1),
        source_bytes=int(m.get("source_bytes", 0)), store_bytes=int(sz),
        encoder=m["encoder"].split("/")[-1], pool=m["pool"], res=m["res"],
        fps=m["fps"], width=m["width"], scales=m["scales"],
        dims=m["dims"], channels=list(vstore.CHANNELS),
        cells={c: int(len(v)) for c, v in st.centroids.items()},
    )


def api_media(key):
    """Media in a store, each with the energy track already in the store.

    The sparkline costs no decode: `energy` is the per-window distance
    from its own local surroundings, written at ingest. It is the
    vision-native "something happened here" signal, so the timeline can
    show where a media is eventful before a single frame is read.
    """
    st = _STORES[key]
    srcs = sources(key)
    fine = float(min(st.man["scales"]))
    out = []
    for mid in sorted(set(st.media.tolist())):
        rows = np.flatnonzero((st.media == mid) & (st.scale == fine))
        if not len(rows):
            rows = np.flatnonzero(st.media == mid)
        o = np.argsort(st.t0[rows])
        rows = rows[o]
        e = st.energy[rows].astype(float)
        if len(e) > 96:                       # downsample for the wire
            idx = np.linspace(0, len(e) - 1, 96).round().astype(int)
            e = e[idx]
        lo, hi = float(e.min()), float(e.max()) if len(e) else (0.0, 1.0)
        spark = ([round((v - lo) / (hi - lo), 3) for v in e]
                 if hi > lo else [0.0] * len(e))
        src = srcs.get(mid)
        # THE TIMELINE IS THE STORE'S COVERAGE, NOT THE FILE'S LENGTH.
        # Several corpora were ingested with a per-media cap (bridge at
        # 60 s of a 40-minute packed file), so the source is far longer
        # than what was indexed. Showing the file length would offer the
        # user 2456 seconds to drag through, of which the store knows 60
        # - every span past the cap would search for something that was
        # never written, and the exclusion of the example's own moment
        # would silently stop meaning anything.
        covered = float(st.t1[np.flatnonzero(st.media == mid)].max())
        file_dur = float(src.dur) if src else covered
        out.append(dict(
            id=mid, dur=round(covered, 2), file_dur=round(file_dur, 2),
            capped=bool(file_dur - covered > 1.0),
            windows=int((st.media == mid).sum()),
            kind=src.kind if src else "missing",
            playable=bool(src), spark=spark))
    return out


# ------------------------------------------------------------ media bytes

def _jpeg(arr, quality=82):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=quality)
    return b.getvalue()


def _src_for(key, mid, upload=None):
    if upload:
        return _upload_source(upload)
    s = sources(key).get(mid)
    if s is None:
        raise KeyError(f"no media {mid} in {key}")
    return s


def api_frame(key, mid, t, width, upload=None):
    """One frame, decoded through the same byte-range cut a query uses.

    Nothing is pre-baked. A thumbnail is a 0.3 s decode at an offset,
    which is the project's own claim about random access being exercised
    by its own UI rather than asserted in a README.
    """
    kid = hashlib.sha1(
        f"{key}|{mid}|{upload}|{t:.2f}|{width}".encode()).hexdigest()[:20]
    fp = CACHE / "frames" / f"{kid}.jpg"
    if fp.exists():
        return fp.read_bytes()
    src = _src_for(key, mid, upload)
    t = max(0.0, min(float(t), max(src.dur - 0.3, 0.0)))
    F = src.cut(t, t + 0.3)
    if not len(F):
        raise ValueError("no frame at that offset")
    im = F[0]
    if width and width < im.shape[1]:
        from PIL import Image
        h = max(int(round(im.shape[0] * width / im.shape[1])), 8)
        im = np.asarray(Image.fromarray(im).resize((width, h)))
    b = _jpeg(im)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_bytes(b)
    return b


def api_clip(key, mid, t0, t1, upload=None, width=480):
    """A span as an mp4, built from the frames the STORE actually sees.

    Deliberately not a re-encode of the original at native frame rate:
    the system reads this media at vsrc.FPS, and showing a smooth 30 Hz
    clip would show something the retrieval never had. The UI labels the
    rate for the same reason.
    """
    kid = hashlib.sha1(
        f"{key}|{mid}|{upload}|{t0:.2f}|{t1:.2f}|{width}".encode()
    ).hexdigest()[:20]
    fp = CACHE / "clips" / f"{kid}.mp4"
    if fp.exists():
        return fp.read_bytes()
    src = _src_for(key, mid, upload)
    t0 = max(0.0, float(t0))
    t1 = min(float(t1), src.dur)
    F = src.cut(t0, t1)
    if not len(F):
        raise ValueError("empty span")
    if width and width < F.shape[2]:
        from PIL import Image
        h = max(int(round(F.shape[1] * width / F.shape[2] / 2) * 2), 8)
        F = np.stack([np.asarray(Image.fromarray(f).resize((width, h)))
                      for f in F])
    n, h, w, _ = F.shape
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", f"{vsrc.FPS:g}", "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         # fragmented mp4: a normal moov must be written at the end after
         # a seek, and a pipe cannot seek. Kept to the two flags every
         # ffmpeg build accepts - default_base_is_moof is rejected
         # outright by the Homebrew build here.
         "-movflags", "frag_keyframe+empty_moov",
         "-f", "mp4", "pipe:1"],
        input=F.tobytes(), capture_output=True)
    if not p.stdout:
        raise RuntimeError(p.stderr.decode()[:400] or "ffmpeg produced nothing")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_bytes(p.stdout)
    return p.stdout


# --------------------------------------------------------------- uploads

def _upload_path(uid):
    p = UPLOADS / uid
    if not p.exists():
        raise KeyError("unknown upload")
    return p


def _upload_source(uid):
    p = _upload_path(uid)
    return vsrc.VideoSource(f"upload/{uid}", p)


VERDICTS = ROOT / "native/verdicts.jsonl"


def api_verdict(body):
    """Record a human judgement on one (query, hit) pair.

    THE ONLY LEGAL JUDGE. Under single-view and no-labels there is no
    admissible automatic answer key for "is this the same KIND of thing"
    - that is written into vgrade's own docstring - so the benchmark
    measures a necessary condition (a distorted clip finds its original)
    and nothing else. Reporting that number as retrieval quality is how
    two weeks went by on a system that returns open roads for a query
    about a tight one.

    A verdict here is the missing quantity: a person looked at the source
    and the result and said yes or no. Appended, never rewritten, with
    the exact span and the build that produced it, so a later run can be
    scored against judgements made before it existed.
    """
    q, h = body["query"], body["hit"]
    rec = dict(when=time.strftime("%Y-%m-%dT%H:%M:%S"), build=build_id(),
               store=body["store"], verdict=body["verdict"],
               query=dict(media=q.get("media"), upload=q.get("upload"),
                          t0=round(float(q["t0"]), 2),
                          t1=round(float(q["t1"]), 2)),
               hit=dict(media=h["media"], t0=round(float(h["t0"]), 2),
                        t1=round(float(h["t1"]), 2),
                        score=float(h.get("score", 0))))
    with open(VERDICTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return dict(ok=True, total=api_verdicts()["total"])


def api_verdicts():
    if not VERDICTS.exists():
        return dict(total=0, good=0, bad=0, queries=0)
    good = bad = 0
    qs = set()
    for line in VERDICTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:                                # noqa: BLE001
            continue
        good += r["verdict"] == "good"
        bad += r["verdict"] == "bad"
        q = r["query"]
        qs.add((r["store"], q.get("media") or q.get("upload"),
                q["t0"], q["t1"]))
    return dict(total=good + bad, good=good, bad=bad, queries=len(qs))


def api_upload(name, body):
    """Store the bytes, probe them, hand back a handle.

    The file lands in a scratch directory and is NEVER written into a
    store. Ingest is a separate, deliberate act; a query that quietly
    grew the corpus would make every elision number afterwards a
    statement about a different corpus than the one on disk.
    """
    UPLOADS.mkdir(parents=True, exist_ok=True)
    uid = hashlib.sha1(body[:1 << 20] + str(len(body)).encode()).hexdigest()[:16]
    ext = Path(name or "clip.mp4").suffix or ".mp4"
    p = UPLOADS / f"{uid}{ext}"
    if not p.exists():
        p.write_bytes(body)
    dur, w, h = vsrc._probe(p)
    if dur <= 0:
        p.unlink(missing_ok=True)
        raise ValueError("not a decodable video (ffprobe found no stream)")
    return dict(upload_id=p.name, name=name, dur=round(float(dur), 2),
                width=w, height=h, bytes=len(body))


# ------------------------------------------------------------------ model

def warm():
    """Load the encoder up front, in the background, and say so.

    A lazy 1.1 GB load inside the first query makes that query take
    ~25 s and look hung. The status endpoint reports cold/loading/ready
    so the UI can disable Search honestly instead of spinning.
    """
    if _MODEL["state"] != "cold":
        return
    _MODEL["state"] = "loading"
    t = time.time()
    try:
        import vcore
        vcore.feature_dim()                  # forces the real load
        _MODEL["t"] = round(time.time() - t, 1)
        _MODEL["state"] = "ready"
    except Exception as e:                                # noqa: BLE001
        _MODEL["err"] = f"{type(e).__name__}: {e}"
        _MODEL["state"] = "error"
        traceback.print_exc()


def api_status():
    import vcore
    return dict(model=_MODEL["state"], load_s=_MODEL["t"],
                error=_MODEL["err"], encoder=vcore.ENCODER,
                pool=vcore.POOL, res=list(vcore.RES),
                channels=list(vstore.CHANNELS),
                stores=sorted(_STORES))


# ------------------------------------------------------------------- qbe

def api_qbe(body):
    """One example in, ranked spans out. The product entry point.

    The example is either a span of a media already in the store or a
    span of an uploaded file; both arrive here as raw frames, and the
    read path cannot tell which is which. That is the point - there is
    one query path, and the Desk is not a second implementation of it.
    """
    import vcore
    import vqbe

    key = body["store"]
    if key not in _STORES:
        raise KeyError(f"no store {key}")
    st = _STORES[key]
    upload = body.get("upload_id")
    mid = body.get("media")
    t0, t1 = float(body["t0"]), float(body["t1"])
    if t1 - t0 < min(st.man["scales"]) + 0.5:
        raise ValueError(
            f"an example must be at least {min(st.man['scales']) + 0.5:.1f}s "
            f"— shorter than one window scale cannot form a unit")

    src = _src_for(key, mid, upload)
    t_dec = time.perf_counter()
    F = src.cut(t0, min(t1, src.dur))
    dec_ms = (time.perf_counter() - t_dec) * 1e3
    if len(F) < 8:
        raise ValueError("decoded fewer than 8 frames from that span")

    if body.get("mode") == "wm":
        # WORLD-MODEL PATH: the representation is the predictor's state
        # trajectory; similarity is sequence matching over stored
        # trajectories. One operator for write and read (vwm_qbe).
        if key != "sim":
            raise ValueError("world-model states cover the sim store "
                             "only until other corpora are state-encoded")
        import vwm_qbe
        with _ENC_LOCK:
            t_enc = time.perf_counter()
            exclude = ((mid, t0, t1)
                       if body.get("exclude_self", True) and mid
                       and not upload else None)
            rows = vwm_qbe.search_frames(
                F, k=int(body.get("k") or 10), exclude=exclude)
            wm_ms = (time.perf_counter() - t_enc) * 1e3
        hits = [dict(media=m, t0=round(a, 2), t1=round(b, 2),
                     score=round(s, 4)) for m, a, b, s in rows]
        return dict(
            hits=hits,
            query=dict(store=key, media=mid, upload=upload, t0=t0,
                       t1=t1, frames=int(len(F)), sub_windows=None),
            stats=dict(returned=len(hits), mode="wm",
                       decode_ms=round(dec_ms, 1),
                       search_ms=round(wm_ms, 1),
                       excluded_self=exclude is not None))

    with _ENC_LOCK:
        warm()
        t_enc = time.perf_counter()
        try:
            Q = vqbe.Query(F)
        except ValueError as e:
            raise ValueError(f"{e} — widen the span") from None
        enc_ms = (time.perf_counter() - t_enc) * 1e3

        # Exclude the example's OWN moment by default. Handed a clip from
        # the store, the honest top hit is that clip itself, which tells
        # the user nothing they did not already know. The separation is
        # the context span, so a neighbour sharing the example's context
        # is excluded too - those are the near-duplicates that made every
        # inflated number in this project's history.
        exclude = None
        if body.get("exclude_self", True) and mid and not upload:
            exclude = (mid, t0, t1, vcore.CTX_MULT * max(vcore.SCALES))
        t_s = time.perf_counter()
        rows, info = vqbe.search(
            st, Q, k=(int(body["k"]) if body.get("k") else None),
            abstain=bool(body.get("abstain", True)),
            exclude=exclude, merge=True)
        search_ms = (time.perf_counter() - t_s) * 1e3

    corpus = st.corpus_bytes()
    hits = [dict(media=m, t0=round(a, 2), t1=round(b, 2), score=round(s, 4))
            for m, a, b, s in rows]
    return dict(
        hits=hits, query=dict(store=key, media=mid, upload=upload,
                              t0=t0, t1=t1, frames=int(len(F)),
                              sub_windows=Q.m),
        stats=dict(
            returned=len(hits), candidates=info["cand"], store_rows=st.n,
            threshold=round(float(info["thr"]), 4),
            weights={c: round(v, 3) for c, v in info["w"].items()},
            cut={c: round(v, 3) for c, v in info["cut"].items()},
            bytes_read=int(info["bytes"]), corpus_bytes=corpus,
            store_scanned=round(info["cand"] / max(st.n, 1), 4),
            elided=(round(1.0 - info["bytes"] / corpus, 6)
                    if corpus else None),
            decode_ms=round(dec_ms, 1), encode_ms=round(enc_ms, 1),
            search_ms=round(search_ms, 1),
            excluded_self=exclude is not None))


# ---------------------------------------------------------------- server

def build_id():
    h = hashlib.sha1()
    for f in ("vdesk.py", "vdesk_ui.html", "vqbe.py", "vcore.py",
              "vstore.py", "vsrc.py"):
        p = Path(__file__).parent / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


class Handler(BaseHTTPRequestHandler):
    server_version = "ElideDBVisionDesk"

    def log_message(self, *a):                # quiet
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _fail(self, e, code=400):
        self._json(dict(error=f"{type(e).__name__}: {e}"), code)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                p = Path(__file__).parent / "vdesk_ui.html"
                return self._send(200, p.read_bytes(),
                                  "text/html; charset=utf-8")
            if u.path == "/api/version":
                return self._json(dict(build=build_id()))
            if u.path == "/api/status":
                return self._json(api_status())
            if u.path == "/api/stores":
                return self._json([store_summary(k) for k in _STORES])
            if u.path == "/api/verdicts":
                return self._json(api_verdicts())
            if u.path == "/api/media":
                return self._json(api_media(q["store"]))
            if u.path == "/api/frame":
                b = api_frame(q.get("store", ""), q.get("media", ""),
                              float(q.get("t", 0)), int(q.get("w", 240)),
                              q.get("upload"))
                return self._send(200, b, "image/jpeg", cache=True)
            if u.path == "/api/clip":
                b = api_clip(q.get("store", ""), q.get("media", ""),
                             float(q["t0"]), float(q["t1"]),
                             q.get("upload"), int(q.get("w", 480)))
                return self._send(200, b, "video/mp4", cache=True)
            return self._json({"error": "not found"}, 404)
        except Exception as e:                            # noqa: BLE001
            traceback.print_exc()
            return self._fail(e)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        try:
            if u.path == "/api/upload":
                return self._json(api_upload(q.get("name", "clip.mp4"), raw))
            if u.path == "/api/qbe":
                return self._json(api_qbe(json.loads(raw or b"{}")))
            if u.path == "/api/verdict":
                return self._json(api_verdict(json.loads(raw or b"{}")))
            return self._json({"error": "not found"}, 404)
        except Exception as e:                            # noqa: BLE001
            traceback.print_exc()
            return self._fail(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("VDESK_PORT", "8788")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--no-warm", action="store_true",
                    help="skip the background encoder load")
    a = ap.parse_args()

    discover()
    if not _STORES:
        print("no vision stores found under stores/vision — build one with\n"
              "  python native/vstore.py --corpus sim", file=sys.stderr)
    if not a.no_warm:
        threading.Thread(target=warm, daemon=True).start()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    tot = sum(s.n for s in _STORES.values())
    print(f"ElideDB Desk (QbE): http://{a.host}:{a.port}   "
          f"{len(_STORES)} stores, {tot} windows   build {build_id()}")
    if a.open:
        threading.Timer(0.5, lambda: subprocess.run(
            ["open", f"http://{a.host}:{a.port}"])).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
