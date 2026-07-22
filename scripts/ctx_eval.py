"""Shared evaluation harness for context retrieval.

The judge (VLM yes/no logprob margin) is the expensive part, so its verdicts
are cached on disk keyed by (stream, t0, t1, query). Every variant we try
re-uses the same verdicts, which makes the comparison both cheap and exactly
like-for-like: two methods are never scored by two different judgements.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

CACHE = Path("/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
             "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
             "scratchpad/judge_cache.json")

QUERIES = [
    "a pedestrian crossing the road in front of the car",
    "a cyclist riding along the street",
    "the car is stopped at an intersection",
    "people walking together on the pavement",
    "a bus or large vehicle passing by",
    "the car is driving past a row of parked cars",
]


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(c):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c))


def _key(w, q):
    return f"{w[0]}|{w[1]}|{w[2]}|{q}"


def judge(store, windows, query, cache):
    """Fill `cache` with the VLM margin for each window under `query`."""
    from PIL import Image

    from elidedb import rerank
    need = [w for w in windows if _key(w, query) not in cache]
    if not need:
        return
    imgs, keys = [], []
    rot = store.meta.get("display", {}).get("rotate", 0)
    for w in need:
        stream, t0, t1 = w
        mid = (t0 + t1) // 2
        win, _ = store.window(mid - 300_000_000, mid + 300_000_000,
                              tables=["frames"])
        fs = win.get("frames")
        dec = fs.decode(stream=stream, width=448, limit=1) if fs else []
        if not dec:
            cache[_key(w, query)] = 0.0
            continue
        im = Image.fromarray(dec[0][1])
        if rot:
            im = im.rotate(rot, expand=True)
        imgs.append(im)
        keys.append(w)
    if imgs:
        margins = rerank.score_images(imgs, rerank.as_question(query))
        for kk, m in zip(keys, margins):
            cache[_key(kk, query)] = float(m)


def evaluate(store, methods, windows, queries=None, k=5, verbose=True):
    """`methods` maps name -> callable(query) -> score array over `windows`.

    Returns {name: {judge_mean_topk, overlap_topk}}. Every method is judged on
    the same pooled verdicts.
    """
    queries = queries or QUERIES
    cache = load_cache()
    out = {n: {"judge": [], "overlap": []} for n in methods}
    for q in queries:
        scored = {n: np.asarray(f(q), dtype=float) for n, f in methods.items()}
        pool = set()
        for v in scored.values():
            pool |= {windows[i] for i in np.argsort(v)[::-1][:k]}
        judge(store, sorted(pool), q, cache)
        ranked = sorted(pool, key=lambda w: -cache[_key(w, q)])
        gold = set(ranked[:k])
        for n, v in scored.items():
            top = [windows[i] for i in np.argsort(v)[::-1][:k]]
            out[n]["judge"].append(np.mean([cache[_key(w, q)] for w in top]))
            out[n]["overlap"].append(len(set(top) & gold))
        if verbose:
            print(f"  judged '{q[:44]}' pool={len(pool)}", flush=True)
    save_cache(cache)
    return {n: {"judge_mean_top5": float(np.mean(d["judge"])),
                "overlap_top5": float(np.mean(d["overlap"]))}
            for n, d in out.items()}


def held_out(store, window_s=2.0, stride_s=0.5, frac=0.3):
    """The windows the tower was never trained on (time split, not random)."""
    from elidedb import context as C
    w = C.plan_windows(store, window_s, stride_s)
    t0s = np.array([x[1] for x in w])
    cut = np.quantile(t0s, 1.0 - frac)
    return [x for x in w if x[1] >= cut], float(cut)
