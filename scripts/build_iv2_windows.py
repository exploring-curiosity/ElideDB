"""IV2 at sub-episode granularity - the measured bottleneck of q03.

Per-channel yield at k=1.5x support, measured over 15 seed groups:

    channel     q03    q04    q05
    iv2        0.77   0.56   0.60      <- the only channel that carries q03
    every appearance channel  0.34-0.41 on q03

and every appearance channel sits at 0.35-0.40 because this corpus is
2,097 episodes of ONE kitchen: appearance similarity is near-constant,
so what separates q03 is video semantics, and IV2 is the only model
here that has them.

IV2 is also the most starved model in the store: `iv2_ingest.py` feeds
it FOUR frames spread over the WHOLE episode, so a 1B video model reads
a 30-frame demo as four stills seconds apart. Its checkpoint fixes the
clip at 4 frames, so the fix is not longer clips but MORE OF THEM:
three overlapping windows, each 4 frames drawn from a third of the
demo, which makes each clip temporally dense instead of a slideshow.

Set-matched at query time (max over windows), never mean-pooled -
pooling windows back into one vector would undo the granularity being
bought here, which is exactly what made the frame/objset channels
useless.

Cost is measured on this machine before the full run.

    python scripts/build_iv2_windows.py [--store lake/fresh_bench]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
from elidedb.iv2 import clip_vec, load_model, MDIR             # noqa: E402

NWIN = 3
# FOUR, not eight: this checkpoint's positional embedding is built for
# a 4-frame clip (1025 tokens = 4 x 256 + CLS) and 8 frames raises
# "size of tensor a (2049) must match tensor b (1025)". The starvation
# is real but it is the model's shape, not a setting - so the extra
# evidence has to come from more WINDOWS, not longer ones.
NFRAME = 4
CKPT_EVERY = 300


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    ep = db.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    ft = db.table("frames").scan()
    print(f"{len(recs):,} episodes x {NWIN} windows x {NFRAME} frames",
          flush=True)

    print("loading InternVideo2 (~4 GB, first use only)...", flush=True)
    t = time.time()
    load_model()
    print(f"  model ready in {time.time() - t:.0f}s", flush=True)

    # TIMING GATE: measure before committing to 6,291 forward passes
    s, a, b = recs[0]
    sel = ft.filter(pc.and_(
        pc.equal(ft.column("stream"), s),
        pc.and_(pc.greater_equal(ft.column("ts"), a),
                pc.less_equal(ft.column("ts"), b))))
    pick = np.unique(np.linspace(0, len(sel) - 1, NFRAME).round().astype(int))
    dec = FrameSet(db, "frames", sel.take(pick)).decode(width=224)
    fr = [d[1] for d in sorted(dec)]
    clip_vec(fr)                                       # warm up
    t = time.time()
    for _ in range(3):
        clip_vec(fr)
    per = (time.time() - t) / 3
    total = per * len(recs) * NWIN / 60
    print(f"  {per*1000:.0f} ms/window -> ~{total:.0f} min encode "
          f"+ ~6 min decode", flush=True)

    ck = db.dir / "_cache" / "iv2_win.npz"
    ck.parent.mkdir(exist_ok=True)
    rows, done = [], 0
    if ck.exists():
        d = np.load(ck, allow_pickle=True)
        rows, done = list(d["rows"]), int(d["done"])
        print(f"resuming at episode {done} ({len(rows):,} windows)")

    t0 = time.time()
    for ri in tqdm(range(done, len(recs)), initial=done, total=len(recs),
                   desc="iv2-win", unit="ep"):
        s, a, b = recs[ri]
        sel = ft.filter(pc.and_(
            pc.equal(ft.column("stream"), s),
            pc.and_(pc.greater_equal(ft.column("ts"), a),
                    pc.less_equal(ft.column("ts"), b))))
        n = len(sel)
        if n < 4:
            continue
        # three overlapping halves: [0,.5] [.25,.75] [.5,1]
        for w in range(NWIN):
            lo = int(w * (n - 1) / (NWIN + 1))
            hi = int((w + 2) * (n - 1) / (NWIN + 1))
            if hi - lo < 3:
                lo, hi = 0, n - 1
            pick = np.unique(np.linspace(lo, hi, NFRAME).round().astype(int))
            try:
                dec = FrameSet(db, "frames", sel.take(pick)).decode(width=224)
            except Exception:
                continue
            if len(dec) < 4:
                continue
            fr = [d[1] for d in sorted(dec)]
            while len(fr) < NFRAME:
                fr.append(fr[-1])
            v = clip_vec(fr[:NFRAME])
            if not np.isfinite(v).all():
                raise FloatingPointError(f"NaN iv2 window at {s} {a} w{w}")
            rows.append((str(s), int(a), int(b), int(w),
                         v.astype(np.float32)))
        if (ri + 1) % CKPT_EVERY == 0:
            np.savez(ck, rows=np.array(rows, object), done=ri + 1)

    V = np.stack([r[4] for r in rows])
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    tbl = pa.table({
        "ts": pa.array([r[1] for r in rows], pa.int64()),
        "t1": pa.array([r[2] for r in rows], pa.int64()),
        "stream": pa.array([r[0] for r in rows]),
        "window": pa.array([r[3] for r in rows], pa.int32()),
        "vector": pa.array([v for v in V], pa.list_(pa.float32(),
                                                    V.shape[1])),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    db.table("iv2_win_vectors").set_layout(
        "stream", sort_by=["stream", "ts", "window"], min_group_rows=1024)
    db.table("iv2_win_vectors").replace(
        tbl, kind="embeddings",
        meta={"model": MDIR, "dim": int(V.shape[1]),
              "windows_per_episode": NWIN, "frames_per_window": NFRAME,
              "pool": "set-matched at query, never mean-pooled"})
    got = len(db.table("iv2_win_vectors").scan())
    assert got == len(tbl)
    ck.unlink(missing_ok=True)
    print(json.dumps({"rows": got, "dim": int(V.shape[1]),
                      "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
