"""EXTRACTION EXPERIMENT 1: how much structure do raw frames give for
free, and which COMPARISON OPERATOR uses it - before any training.

The directive: extract as much data as possible directly from the
images, and compare it without a basic cosine similarity. This script is
the zero-train floor for that program. Everything below is a function of
pixels; nothing is trained, labelled, or corpus-specific. The trained
world-model predictor later REPLACES the naive predictor here behind
the same interface - this file defines the interface and measures how
far pure extraction gets without it.

EXTRACTED per span (from DINOv3 ViT-S patch tokens, no pooling loss):

  grid     G_t : 4x4 cell grid per frame - parts and arrangement kept,
                 because one global vector cannot hold a relation
  states   causal EMAs of the grid at three time constants
                 (fast ~0.5s / mid ~2s / slow ~8s)
  surprise s_t : error of a constant-velocity predictor in feature
                 space - where the naive world model fails. The trained
                 predictor slots in exactly here.
  trans    D   : settled END minus settled START, per cell - a SIGNED
                 change field. Direction lives in the sign: reversing
                 an event negates it.
  motion   d_t : per-frame global feature velocity - the tempo profile

COMPARED with three non-flat operators:

  EMD      Hungarian assignment between the two spans' transition
           cells - the same change is found WHEREVER it happened in
           frame. Position-invariant relational matching.
  DTW      dynamic time warping over the motion profiles - same
           evolution at a different tempo still aligns.
  SIGN     EMD(Da, Db) vs EMD(Da, -Db): an event and its reversal
           should match NEGATED, not merely mismatch.

Output: the full pairwise matrix over the battery spans under each
operator, beside the flat-cosine baseline, and the two questions that
decide anything:
  1. do the match pairs beat every reject pair involving them?
  2. does SIGN see reversal (reject-direction pairs matching negated)?

    SDX_ENC=vits SDX_RES=320 python native/vext.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import vcore                                              # noqa: E402
import vsrc                                               # noqa: E402

CELLS = 4            # grid side; 320px / 16 = 20 patches -> 4x4 of 5x5
TAUS = (0.5, 2.0, 8.0)
SETTLE = 0.25        # fraction of the span treated as settled start/end


def _l2(V, axis=-1):
    return V / np.maximum(np.linalg.norm(V, axis=axis, keepdims=True),
                          1e-8)


def patch_grid(F):
    """(T,H,W,3) frames -> (T, CELLS, CELLS, d) cell grid, per-cell L2.

    Runs the encoder WITHOUT pooling: the pooled path throws away
    arrangement, and arrangement is the point.
    """
    m, proc, dev, dtype, torch = vcore.model()
    out = []
    B = vcore.BATCH
    for i in range(0, len(F), B):
        chunk = [np.ascontiguousarray(x[..., :3]) for x in F[i:i + B]]
        px = proc(images=chunk, return_tensors="pt",
                  size={"height": 320, "width": 320})["pixel_values"]
        with torch.no_grad():
            r = m(pixel_values=px.to(dev, dtype))
        P = vcore._patches(r, torch).float().cpu().numpy()   # (B,N,d)
        n = P.shape[1]
        side = int(round(n ** 0.5))
        P = P[:, :side * side].reshape(len(chunk), side, side, -1)
        # average side/CELLS x side/CELLS patches into each cell
        k = side // CELLS
        P = P[:, :k * CELLS, :k * CELLS]
        P = P.reshape(len(chunk), CELLS, k, CELLS, k, -1).mean((2, 4))
        out.append(P)
    return _l2(np.concatenate(out))


def extract(F, fps=vsrc.FPS):
    """The zero-train extraction: everything listed in the docstring."""
    G = patch_grid(F)                        # (T,C,C,d)
    T = len(G)
    g = _l2(G.reshape(T, -1).mean(0))        # span gist, baseline only
    f = _l2(G.reshape(T, CELLS * CELLS, -1).mean(1))     # (T,d) global
    states = {}
    for tau in TAUS:
        a = 1.0 - np.exp(-1.0 / (tau * fps))
        S = np.empty_like(G)
        S[0] = G[0]
        for t in range(1, T):
            S[t] = (1 - a) * S[t - 1] + a * G[t]
        states[tau] = _l2(S)
    # constant-velocity surprise: where the naive predictor fails
    s = np.zeros(T, np.float32)
    if T > 2:
        pred = 2 * f[1:-1] - f[:-2]
        s[2:] = np.linalg.norm(f[2:] - pred, axis=1)
    n = max(int(T * SETTLE), 2)
    D = G[-n:].mean(0) - G[:n].mean(0)       # (C,C,d) SIGNED change
    d = np.diff(f, axis=0)                   # (T-1,d) motion profile
    return dict(grid=G, gist=g, states=states, surprise=s,
                trans=D, motion=d)


# ---------------------------------------------------------- comparators

def emd(Da, Db):
    """Hungarian match of transition cells: position-invariant.

    Cells are unit-normed WITH their magnitude kept as a weight, so a
    cell where nothing changed cannot vote."""
    from scipy.optimize import linear_sum_assignment
    A = Da.reshape(-1, Da.shape[-1])
    B = Db.reshape(-1, Db.shape[-1])
    wa = np.linalg.norm(A, axis=1)
    wb = np.linalg.norm(B, axis=1)
    An, Bn = _l2(A), _l2(B)
    C = An @ Bn.T
    # weights on a SHARED scale. Normalizing each span by its own max
    # let a span where nothing changed vote at full strength - the
    # hover confuser scored 0.270 against a real event through exactly
    # that hole.
    m = max(wa.max(), wb.max(), 1e-8)
    W = np.sqrt(np.outer(wa / m, wb / m))
    r, c = linear_sum_assignment(-(C * W))
    return float((C[r, c] * W[r, c]).sum() / max(W[r, c].sum(), 1e-8))


def dtw(da, db):
    """Similarity of motion profiles under time warping."""
    A, B = _l2(da), _l2(db)
    S = A @ B.T
    n, m = S.shape
    acc = np.full((n + 1, m + 1), -np.inf)
    acc[0, 0] = 0.0
    L = np.zeros((n + 1, m + 1), np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            k = np.argmax([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
            pi, pj = [(i - 1, j - 1), (i - 1, j), (i, j - 1)][k]
            acc[i, j] = acc[pi, pj] + S[i - 1, j - 1]
            L[i, j] = L[pi, pj] + 1
    return float(acc[n, m] / max(L[n, m], 1))


def main():
    bat = json.loads((ROOT / "eval/battery.json").read_text())["sim"]
    srcs = {s.id: s for s in vsrc.sources("sim", 8)}
    print(f"encoder {vcore.ENCODER.split('/')[-1]}  res {vcore.RES}  "
          f"grid {CELLS}x{CELLS}")
    ex, order = {}, sorted(bat["spans"])
    import time
    for k in order:
        sp = bat["spans"][k]
        t0 = time.time()
        F = srcs[sp["media"]].cut(sp["t0"], sp["t1"])
        ex[k] = extract(F)
        print(f"  {k}: {sp['media']} [{sp['t0']},{sp['t1']}]  "
              f"{len(F)} frames  {time.time() - t0:.1f}s")

    ops = dict(
        cosine=lambda a, b: float(ex[a]["gist"] @ ex[b]["gist"]),
        emd_T=lambda a, b: emd(ex[a]["trans"], ex[b]["trans"]),
        dtw_M=lambda a, b: dtw(ex[a]["motion"], ex[b]["motion"]),
        sign=lambda a, b: emd(ex[a]["trans"], -ex[b]["trans"]))
    for name, op in ops.items():
        print(f"\n{name}")
        print("      " + "".join(f"{k:>8}" for k in order))
        for a in order:
            print(f"  {a}   " + "".join(f"{op(a, b):8.3f}" for b in order))

    print("\nthe two questions:")
    for a, b in bat["match"]:
        for name in ("cosine", "emd_T", "dtw_M"):
            v = ops[name](a, b)
            rej = [ops[name](x, y) for x, y in bat["reject"]
                   if a in (x, y) or b in (x, y)]
            verdict = "BEATS" if rej and v > max(rej) else "loses to"
            if rej:
                print(f"  {name:>7}: match({a},{b})={v:.3f} {verdict} "
                      f"its rejects (max {max(rej):.3f})")
    print("  reversal, via sign: a matched against negated d/e should "
          "EXCEED a against d/e unnegated:")
    for a in ("a", "b"):
        for r in ("d", "e"):
            print(f"    {a}-{r}: straight {ops['emd_T'](a, r):.3f}   "
                  f"negated {ops['sign'](a, r):.3f}")


if __name__ == "__main__":
    main()
