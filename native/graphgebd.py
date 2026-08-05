"""STEP 4, best available: GraphGEBD - recursive normalised cut.

Published unsupervised/zero-shot SOTA for generic event boundary
detection (F1@0.05 = 0.732 Kinetics-GEBD, above FlowGEBD's 0.713).
It also corrects an error I made earlier: it runs on SEMANTIC features
(ResNet50 / DINOv2), so embeddings do carry boundary signal - my
mistake was thresholding a per-frame score instead of solving for a
global partition.

Formulation: frames are graph nodes, edges are appearance similarity,
and boundaries are the CUTS that minimise the normalised-cut objective

    Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)

restricted to contiguous splits (a video segment is an interval), then
applied recursively. Every candidate split is scored by its own Ncut
value, so boundaries come out ranked by quality with no threshold of
mine; how many to emit is decided by the same distribution-fitted
emitter used elsewhere.

Kept even though the sensitivity study showed segmentation is not the
current bottleneck: it is the layer's best available implementation
and steps 5+ consume it.

    python native/graphgebd.py --corpus sim
    python native/graphgebd.py --corpus oxford
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import f1_at, media_jobs, arg                    # noqa: E402

FPS = 4.0
SIGMA = 0.25          # affinity bandwidth on cosine distance
LOCAL = 0             # 0 = full graph; >0 = temporal band width
MIN_SEG = 4           # frames


def frame_features(F):
    """Per-frame semantic features (the paper uses ResNet50/DINOv2;
    DINOv3 ConvNeXt-Tiny is the in-repo equivalent and is fast)."""
    from elidedb import dinov3
    dinov3._load()
    V = np.asarray(dinov3.embed(list(F)), np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True),
                          1e-8)


def affinity(V, sigma=SIGMA, local=LOCAL):
    S = V @ V.T
    W = np.exp(-(1.0 - S) / max(sigma, 1e-6))
    if local:
        n = len(V)
        i = np.arange(n)
        band = np.abs(i[:, None] - i[None, :]) <= local
        W = W * band
    np.fill_diagonal(W, 0.0)
    return W


def best_contiguous_cut(W, lo, hi, curve=False):
    """Minimum-Ncut contiguous split of [lo,hi). Returns (t, ncut).

    With curve=True also returns Ncut(t) for every candidate t, which
    is what tells a REAL boundary from a mere drift: a real boundary
    puts a sharp deep minimum in this curve, while a segment that is
    just slowly changing (a car driving straight) gives a shallow,
    smooth curve whose minimum is not special. The minimum's VALUE
    cannot distinguish those; its prominence can.
    """
    sub = W[lo:hi, lo:hi]
    n = hi - lo
    if n < 2 * MIN_SEG:
        return (None, np.inf, np.empty(0)) if curve else (None, np.inf)
    tot = sub.sum(1)                       # assoc to the segment
    best_t, best_v = None, np.inf
    # cut(A,B) for split at t = sum of W[:t, t:]
    csum = np.cumsum(np.cumsum(sub, axis=0), axis=1)

    def block(a0, a1, b0, b1):
        if a1 <= a0 or b1 <= b0:
            return 0.0
        r = csum[a1 - 1, b1 - 1]
        if a0 > 0:
            r -= csum[a0 - 1, b1 - 1]
        if b0 > 0:
            r -= csum[a1 - 1, b0 - 1]
        if a0 > 0 and b0 > 0:
            r += csum[a0 - 1, b0 - 1]
        return float(r)

    vs = []
    for t in range(MIN_SEG, n - MIN_SEG + 1):
        cut = block(0, t, t, n)
        aA = float(tot[:t].sum())
        aB = float(tot[t:].sum())
        if aA <= 0 or aB <= 0:
            vs.append(np.nan)
            continue
        v = cut / aA + cut / aB
        vs.append(v)
        if v < best_v:
            best_v, best_t = v, t
    bt = lo + best_t if best_t is not None else None
    return (bt, best_v, np.array(vs)) if curve else (bt, best_v)


def salience(vs):
    """How PROMINENT the best cut is within its own segment's curve.

    z-score of the minimum against the rest of the curve. Scale-free
    and computed per segment, so it does not care what the absolute
    Ncut level of a media is. A homogeneous stretch has a flat curve
    and scores ~0 however low its minimum sits.
    """
    v = vs[np.isfinite(vs)]
    if len(v) < 3:
        return 0.0
    sd = v.std()
    return float((v.mean() - v.min()) / sd) if sd > 1e-12 else 0.0


def recursive_ncut(W, depth=4, min_sal=0.0, rank="ncut"):
    """-> [(frame_index, ncut_value)] ranked by cut quality.

    min_sal > 0 stops recursion on segments with no prominent cut,
    instead of splitting blindly until `depth` runs out. Depth-only
    stopping is why a long uniform stretch got chopped up: the
    recursion had no way to say "this segment does not want splitting".
    """
    out = []
    stack = [(0, len(W), 0)]
    while stack:
        lo, hi, d = stack.pop()
        if d >= depth or hi - lo < 2 * MIN_SEG:
            continue
        t, v, vs = best_contiguous_cut(W, lo, hi, curve=True)
        if t is None or not np.isfinite(v):
            continue
        sal = salience(vs)
        if min_sal > 0.0 and sal < min_sal:
            continue                     # homogeneous: do not split
        out.append((t, v, sal))
        stack.append((lo, t, d + 1))
        stack.append((t, hi, d + 1))
    # Rank by PROMINENCE, not by raw Ncut. Raw Ncut rewards cutting a
    # long smooth drift (the value is low simply because the two halves
    # are far apart in time); prominence asks whether THIS split is
    # special within its own segment, which is the question a boundary
    # detector is actually asking.
    if rank == "sal":
        return sorted(out, key=lambda r: -r[2])
    return sorted(out, key=lambda r: r[1])       # best cuts first


def main():
    import encode as E
    corpus = arg("--corpus", "sim")
    limit = arg("--limit", 12, int)
    depth = arg("--depth", 4, int)
    jobs = media_jobs(corpus, limit)
    print(f"{corpus}: {len(jobs)} media, DINOv3 frames at {FPS} fps, "
          f"recursive Ncut depth {depth}", flush=True)
    res_m, res_t = [], []
    for name, path, gt in jobs:
        dur = E.probe_duration(path)
        F = E.decode(path, fps=FPS, w=256)
        V = frame_features(F)
        W = affinity(V)
        cuts = recursive_ncut(W, depth=depth)
        times = [c[0] / FPS for c in cuts]
        vals = np.array([c[1] for c in cuts])
        tol = 0.05 * dur
        # count-matched (placement) and emitter-decided (z-cut on the
        # Ncut values: good cuts are the low tail)
        pm = times[:max(len(gt), 1)]
        if len(vals) > 2:
            z = (vals - vals.mean()) / (vals.std() + 1e-8)
            pth = [t for t, zz in zip(times, z) if zz < 0.0]
        else:
            pth = times
        fm, _, _ = f1_at(pm, gt, tol)
        ft, _, _ = f1_at(pth, gt, tol)
        res_m.append(fm)
        res_t.append(ft)
        print(f"  {name:<8} dur {dur:5.1f}s gt {len(gt):2d} cuts "
              f"{len(cuts):2d}  matched {fm:.2f}  thresh {ft:.2f} "
              f"(emits {len(pth)})", flush=True)
    print(f"\nSTEP 4 GraphGEBD ({corpus}): matched "
          f"{np.mean(res_m):.3f}  thresh {np.mean(res_t):.3f}  "
          f"(n={len(res_m)})")


if __name__ == "__main__":
    main()
