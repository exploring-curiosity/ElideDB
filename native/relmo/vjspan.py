"""Return a located SPAN inside a long recording, not a whole file.

Owner's requirement: "I want exact segments of clips from all the samples that
closely matches... recordings will be hours of data stitched in one clip. I
dont want that to be returned everytime I ask somethign nor whatever the input
time break down is being done to the clips (for eg: fixed 10s breakdown)."

So two constraints, and the second rules out the easy implementation:
  1. A result is (recording, t0, t1).
  2. t0 and t1 come from the MATCH, never from a grid. Cutting a recording into
     fixed windows and retrieving windows quantises every answer to a clock
     that has nothing to do with when the event happened.

MECHANISM - subsequence DTW with FREE endpoints
  The dense record (relmo/vjdense.py) gives one descriptor per timestep. Build
  a cost matrix between the query's steps and the recording's steps, then align
  with the START AND END IN THE RECORDING UNCONSTRAINED: row 0 may begin at any
  reference column at zero entry cost, and the answer is the column minimising
  the final row. Backtracking gives the first and last reference steps the
  query actually matched - and those ARE the span. No grid is involved at any
  point.
  It also subsumes the time problem rather than deferring it: the alignment
  warps, so a query where a door is thrown open can match a recording where it
  eases open, without either sharing a clock.

HONEST ENCODING OF THE LONG RECORDING
  The encoder takes 64 frames. A real recording is longer, so it is encoded in
  OVERLAPPING 64-frame windows at a fixed frame stride, and the per-timestep
  descriptors are concatenated with a frame index kept for each. Overlap
  matters: with non-overlapping windows an event straddling a boundary is split
  across two contexts and belongs to neither. Context never crosses a window,
  so there is no leakage between distant parts of the recording.

VALIDATION IS LABEL-FREE
  Stitch K episodes end to end into one recording and query with footage whose
  true location is known because we did the stitching. Ground truth is an
  offset, not an annotation. Reported: temporal IoU of the predicted span
  against the true one, and whether the span lands in the right episode at all.
  A trivial baseline is included - "return the whole recording" - because a
  span metric that a degenerate answer can win is not measuring localisation.

    python -m relmo.vjspan --episodes 8 --stride 32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, probe_dims, read_frames,  # noqa: E402
                       to_tensor)
from relmo.vjdense import MINCTX, WIN, dense_record  # noqa: E402
from relmo.vjeval import REC, l2, parse  # noqa: E402

WARPS = {"same": lambda u: u, "ease_out": lambda u: 1 - (1 - u) ** 2}


def encode_long(model, torch, dev, frames, n_frames, stride, cal, layer):
    """Dense descriptor over a long recording + the source frame of each step.

    Returns (D, idx): D is (T,1024) L2-normalised per step, idx[t] is the
    frame number in the ORIGINAL recording that step t is centred on, so a
    matched step range converts straight back to a time span.
    """
    from tqdm import tqdm
    n_t = n_frames // TUBELET
    covered = list(range(MINCTX, n_t, WIN))
    starts = list(range(0, max(len(frames) - n_frames, 0) + 1, stride))
    D, idx = [], []
    for s in tqdm(starts, unit="win", desc="encode", leave=False):
        _, what = dense_record(model, torch, dev, frames[s:s + n_frames],
                               n_frames, cal, layer)
        D.append(what)
        # step j of this window covers tubelet step covered[j//WIN] + j%WIN
        for j in range(what.shape[0]):
            tstep = covered[j // WIN] + (j % WIN)
            idx.append(s + tstep * TUBELET + TUBELET // 2)
    return l2(np.concatenate(D)), np.array(idx)


def subsequence_dtw(Q, Rf, band_frac=0.35):
    """Align query Q (m,D) inside reference Rf (n,D). Free start AND end.

    Returns (start, end, cost) as indices into Rf. The endpoints are outputs of
    the alignment, which is the whole point - nothing is quantised to a grid.
    A per-step warping penalty keeps the path from collapsing the query onto a
    single reference step, the same failure guarded against in vjeval.dtw_batch.
    """
    m, n = len(Q), len(Rf)
    C = 1.0 - Q @ Rf.T                                   # (m,n) cosine cost
    pen = 0.05
    INF = 1e18
    # Carry the START COLUMN forward alongside the cost instead of storing
    # pointers and walking back. Backtracking here is fiddly and easy to get
    # silently wrong - an earlier version returned an end index with a start
    # that was never actually on the winning path.
    D = C[0].astype(np.float64).copy()                    # FREE START
    st = np.arange(n)
    for i in range(1, m):
        diag = np.concatenate(([INF], D[:-1]))            # (i-1, j-1)
        up = D + pen                                      # (i-1, j)
        cand = np.stack([diag, up])
        k = cand.argmin(0)
        newD = C[i] + cand.min(0)
        newS = np.where(k == 0, np.concatenate(([0], st[:-1])), st)
        # in-row transition (i, j-1): reference advances, query does not.
        # Sequential by necessity - each cell depends on the one to its left.
        for j in range(1, n):
            if newD[j - 1] + pen < newD[j]:
                newD[j] = newD[j - 1] + pen
                newS[j] = newS[j - 1]
        D, st = newD, newS
    end = int(D.argmin())
    return int(st[end]), end, float(D[end] / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--layer", type=int, default=6)
    a = ap.parse_args()

    import torch
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = [p for p in sorted(d.glob("*.npz")) if not p.name.startswith("_")]
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), a.episodes, replace=False)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()

    # ---- stitch a long recording -----------------------------------------
    clips, bounds, names = [], [], []
    off = 0
    for i in pick:
        ep = R.dataset_dir(a.dataset) / "shard_0000" / files[i].stem / "frames.mp4"
        w_, h_ = probe_dims(ep)
        F = read_frames(ep, w_, h_)
        if len(F) < a.frames:
            continue
        clips.append(F)
        bounds.append((off, off + len(F)))
        names.append(files[i].stem)
        off += len(F)
    REC_F = np.concatenate(clips)
    fps = 20.0
    print(f"stitched {len(clips)} episodes -> {len(REC_F)} frames "
          f"({len(REC_F)/fps:.0f} s)", flush=True)

    t0 = time.time()
    D, idx = encode_long(model, torch, dev, REC_F, a.frames, a.stride, cal,
                         a.layer)
    enc_s = time.time() - t0
    print(f"dense index: {len(D)} steps, {enc_s:.0f}s "
          f"({enc_s/(len(REC_F)/fps):.2f}s of compute per s of video)\n")

    rows = []
    for k, (F, (b0, b1), nm) in enumerate(zip(clips, bounds, names)):
        for wname, fn in WARPS.items():
            u = np.linspace(0, 1, a.frames)
            q = F[np.clip((fn(u) * (len(F) - 1)).round().astype(int), 0,
                          len(F) - 1)]
            _, qw = dense_record(model, torch, dev, q, a.frames, cal, a.layer)
            s, e, cost = subsequence_dtw(l2(qw), D)
            p0, p1 = int(idx[s]), int(idx[e])
            inter = max(0, min(p1, b1) - max(p0, b0))
            union = max(p1, b1) - min(p0, b0)
            iou = inter / union if union else 0.0
            # trivial baseline: return the whole recording
            biou = (b1 - b0) / len(REC_F)
            # CONTAINMENT, not coverage. A correct answer here is a TIGHT
            # span inside the right episode; an IoU-style rule punishes it for
            # being tight, which is the behaviour that was asked for. Span
            # length and centre are reported so a degenerate 1-frame answer is
            # visible rather than scoring perfectly.
            rows.append(dict(ep=nm, warp=wname, iou=iou, base=biou,
                             hit=p0 >= b0 and p1 <= b1,
                             flen=(p1 - p0) / max(b1 - b0, 1),
                             fpos=((p0 + p1) / 2 - b0) / max(b1 - b0, 1),
                             pred=(p0, p1), true=(b0, b1)))
    print(f"{'episode':38s} {'warp':9s} {'true span':>14s} {'predicted':>14s} "
          f"{'IoU':>6s} {'hit':>4s}")
    print("-" * 92)
    for r in rows:
        ts = "%d-%d" % r["true"]
        ps = "%d-%d" % r["pred"]
        print(f"{r['ep'][:38]:38s} {r['warp']:9s} "
              f"{ts:>14s} {ps:>14s} "
              f"{r['iou']:6.3f} {'Y' if r['hit'] else '.':>4s}")
    print("-" * 92)
    for wname in WARPS:
        sub = [r for r in rows if r["warp"] == wname]
        print(f"{wname:9s} containment {sum(r['hit'] for r in sub)}/{len(sub)}"
              f"   span {100*np.mean([r['flen'] for r in sub]):.0f}% of episode"
              f"   centre {100*np.mean([r['fpos'] for r in sub]):.0f}% through"
              f"   IoU {np.mean([r['iou'] for r in sub]):.3f} "
              f"(baseline {np.mean([r['base'] for r in sub]):.3f})")
    R.log("vjspan", dataset=a.dataset, episodes=len(clips),
          frames=len(REC_F), steps=len(D), stride=a.stride,
          encode_s=round(enc_s, 1),
          **{f"iou_{w}": round(float(np.mean([r['iou'] for r in rows
                                              if r['warp'] == w])), 4)
             for w in WARPS})


if __name__ == "__main__":
    main()
