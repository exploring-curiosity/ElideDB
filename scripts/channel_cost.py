"""How long does each channel take to INGEST, per episode?

Times each channel's per-episode work on a small sample and extrapolates
to the full corpus, so the cost of "eight channels" is a number rather
than a feeling. Model load is timed separately from steady-state work —
load is paid once, the per-episode rate is what scales.

  python scripts/channel_cost.py [store] [--n 8]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb.store import Store  # noqa: E402

FULL_EPISODES = 1122
SOURCE_MB = 819.5


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def main():
    store = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else "lake/_bench_recovered"
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
    db = Store.open(store)
    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    ep.column("ts").to_pylist(),
                    ep.column("t1").to_pylist()))[:n]
    print(f"sampling {len(keys)} episodes from {store}\n")
    print(f"{'channel':10s} {'load_s':>7} {'per_ep_ms':>10} "
          f"{'full_corpus_s':>14} {'ms/MB':>8}")

    results = {}

    def report(name, load_s, per_ep_s):
        full = per_ep_s * FULL_EPISODES
        results[name] = {"load_s": round(load_s, 1),
                         "per_episode_ms": round(per_ep_s * 1000, 1),
                         "full_corpus_s": round(full, 1),
                         "ms_per_mb": round(full * 1000 / SOURCE_MB, 1)}
        print(f"{name:10s} {load_s:>7.1f} {per_ep_s*1000:>10.1f} "
              f"{full:>14.1f} {full*1000/SOURCE_MB:>8.1f}")

    # --- frame decode, the shared cost every visual channel pays -------
    from elidedb.video import FrameSet
    def decode_one(k):
        out, _ = db.window(int(k[1]), int(k[2]), tables=["frames"])
        fs = out["frames"]
        if not hasattr(fs, "decode"):
            fs = FrameSet(db, "frames", fs)
        return fs.decode(stream=k[0], limit=8)
    _, t = timed(decode_one, keys[0])           # warm
    t0 = time.perf_counter()
    for k in keys:
        decode_one(k)
    report("decode(8f)", t, (time.perf_counter() - t0) / len(keys))

    # --- PE ------------------------------------------------------------
    try:
        from elidedb import pe
        _, load = timed(pe._text_vec, "warm the tower")
        t0 = time.perf_counter()
        for k in keys:
            pe._text_vec(f"probe {k[1]}")       # text side only
        report("pe(text)", load, (time.perf_counter() - t0) / len(keys))
    except Exception as e:
        print(f"pe: {type(e).__name__}: {str(e)[:60]}")

    # --- SigLIP2 -------------------------------------------------------
    try:
        from elidedb import sig2
        _, load = timed(sig2.sig2_lookup, db, "warm")
        t0 = time.perf_counter()
        for k in keys:
            sig2.sig2_lookup(db, f"probe {k[1]}")
        report("sig2(text)", load, (time.perf_counter() - t0) / len(keys))
    except Exception as e:
        print(f"sig2: {type(e).__name__}: {str(e)[:60]}")

    # --- IV2 -----------------------------------------------------------
    try:
        from elidedb import iv2
        _, load = timed(iv2.iv2_lookup, db, "warm")
        t0 = time.perf_counter()
        for k in keys:
            iv2.iv2_lookup(db, f"probe {k[1]}")
        report("iv2(text)", load, (time.perf_counter() - t0) / len(keys))
    except Exception as e:
        print(f"iv2: {type(e).__name__}: {str(e)[:60]}")

    import json
    (ROOT / "eval/channel_cost.json").write_text(json.dumps(results, indent=1))
    print(f"\nNOTE: text-side timings show tower cost; the INGEST cost of a\n"
          f"visual channel is decode + a vision forward per episode, which\n"
          f"the vision passes below dominate. Written to eval/channel_cost.json")


if __name__ == "__main__":
    main()
