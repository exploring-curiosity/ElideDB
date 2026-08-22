"""The query-by-example Space: point at a moment, get its kind back.

WHY A SECOND DEPLOYMENT. The text Space answers "describe what you want"
and pays for it: five text towers, model weights on first boot, minutes
before the first query. This one answers the question a text encoder
provably cannot - direction - and costs no model at all. Everything it
ranks was computed at ingest; serving is numpy over parquet columns.
That is the whole argument for the storage layer, and it deserves to be
demonstrable in seconds rather than after a fifteen minute warm.

The read path is elidedb.qbe.search_like verbatim - multi-seed centroid,
per-channel seed-coherence weighting, z-fusion, and a return cut
calibrated from the query's own held-out seeds. Nothing here re-tunes it;
this file is a surface, not a second implementation.

Media (thumbnails, clips) reuses elidedb.desk's byte-range decode path,
which is the same one the text Space serves from - one media path, not a
copy that can drift.

    python deploy/qbe_serve.py [--port 7860] [--root lake]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, "python")

import numpy as np  # noqa: E402

from elidedb import desk as D  # noqa: E402  (media + store discovery)
from elidedb import qbe  # noqa: E402

HERE = Path(__file__).resolve().parent
UI = HERE / "qbe_ui.html"
_EPISODES: dict = {}
_SPACES: dict = {}


def episodes(key):
    """[(stream, ts, label)] for one store, ordered. The label is shown
    AFTER retrieval so a visitor can check the answer; it is never read
    by the ranker - search_like sees vectors and nothing else."""
    if key in _EPISODES:
        return _EPISODES[key]
    db = D.STORES[key]
    ep = db.table("episodes").scan()
    cols = ep.column_names
    lab = next((c for c in ("text", "task", "instruction", "label", "name")
                if c in cols), None)
    rows = list(zip(ep.column("stream").to_pylist(),
                    ep.column("ts").to_pylist(),
                    ep.column(lab).to_pylist() if lab
                    else [""] * ep.num_rows))
    rows = [(str(s), int(t), (str(x) if x else "")) for s, t, x in rows]
    rows.sort(key=lambda r: (r[0], r[1]))
    _EPISODES[key] = rows
    return rows


def warm(key):
    """Build the channel matrices once, at boot.

    spaces() reads every per-episode vector table and pools it. On the
    demo store that is a couple of seconds and a few hundred megabytes
    of parquet; paying it per query would make the first click look
    broken for no reason, and paying it at import means the platform
    reports RUNNING only when queries are genuinely warm - the same
    lesson the text Space learned the expensive way.
    """
    t0 = time.time()
    keys, M = qbe.spaces(D.STORES[key])
    _SPACES[key] = (keys, M)
    print(f"warm {key}: {len(keys)} episodes, "
          f"{len(M)} channels ({', '.join(sorted(M))}), "
          f"{time.time() - t0:.1f}s", flush=True)


def like(key, seeds, k_max):
    db = D.STORES[key]
    t0 = time.time()
    out = qbe.search_like(db, [(s, int(t)) for s, t in seeds], k_max=k_max)
    lab = {(s, t): x for s, t, x in episodes(key)}
    return dict(
        clips=[dict(stream=s, ts=t, label=lab.get((s, int(t)), ""))
               for s, t in out["clips"]],
        weights={c: round(float(w), 3)
                 for c, w in sorted(out.get("weights", {}).items(),
                                    key=lambda kv: -kv[1])},
        note=out.get("note", ""), ms=int((time.time() - t0) * 1000))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, cache=False):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code=200):
        self._send(code, obj, "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(),
                                  "text/html; charset=utf-8")
            if u.path == "/api/archive":
                key = q.get("store") or next(iter(D.STORES))
                rows = episodes(key)
                return self._json(dict(
                    store=key, stores=sorted(D.STORES),
                    channels=sorted(_SPACES.get(key, ([], {}))[1]),
                    episodes=[dict(stream=s, ts=t, label=x)
                              for s, t, x in rows]))
            if u.path == "/api/thumb":
                jpg = D.api_thumb(q.get("store") or next(iter(D.STORES)),
                                  q.get("stream", ""), int(q["t"]),
                                  int(q.get("w", "260")))
                if jpg is None:
                    return self._json({"error": "no frame"}, 404)
                return self._send(200, jpg, "image/jpeg", cache=True)
            if u.path == "/api/clip":
                mp4 = D.api_clip(q.get("store") or next(iter(D.STORES)),
                                 q.get("stream", ""), int(q["t0"]),
                                 int(q["t1"]), int(q.get("w", "480")))
                if isinstance(mp4, dict):
                    return self._json(mp4, 422)
                if mp4 is None:
                    return self._json({"error": "no frames"}, 404)
                return self._send(200, mp4, "video/mp4", cache=True)
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/like":
                seeds = body.get("seeds", [])
                if not seeds:
                    return self._json({"error": "pick at least one clip"},
                                      400)
                key = body.get("store") or next(iter(D.STORES))
                return self._json(like(key, seeds,
                                       int(body.get("k", 24))))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "7860")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--root", default=os.environ.get("ELIDEDB_LAKE", "lake"))
    a = ap.parse_args()
    D.ROOT = Path.cwd()
    D.LAKE = Path(a.root)
    D.discover()
    if not D.STORES:
        raise SystemExit(f"no store under {a.root}/ - set ELIDEDB_LAKE or "
                         f"download the demo store first")
    for key in D.STORES:
        warm(key)
    print(f"query by example on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
