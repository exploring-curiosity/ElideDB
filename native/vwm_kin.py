"""KINEMATIC SUB-EVENT UNITS: cut where the change REVERSES.

Units v1 cut on motion energy and found a median of ONE unit per event -
the arm never pauses inside an event, so stillness separates events, not
phases. The phases are there, but they are marked by DIRECTION, not
magnitude: a pick is descend -> (reversal) -> lift, a place is carry ->
descend -> (reversal) -> retreat. In the trajectory of height-profile
change, those are corners.

So the cut is a corner detector: Douglas-Peucker over the trajectory in
latent space. Split at the point of maximum perpendicular deviation from
the chord, recurse while the deviation exceeds a tolerance read off the
trajectory's OWN radius of gyration (nothing fitted, nothing per-dataset,
no labels). What comes out is a piecewise-linear description of the
clip: each unit is a stretch during which the world was changing in one
consistent direction.

A clip is then a SET of unit descriptors, and similarity is symmetric
best-match between sets. This is what the composite classes need: an
unstack shares its grasp-high unit with other unstacks even when its
carry and release differ, which whole-span cosine cannot express.

    python native/vwm_kin.py                 # benchmark vs whole-span
    python native/vwm_kin.py --alpha 0.2
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "vwm"
FPS = 10
D_P = 768          # JL width for the per-frame profile
MIN_UNIT = 4       # frames; shorter is not a phase
ALPHA = 0.30       # corner tolerance, in radii of gyration
MAX_UNITS = 8


def _l2(V, ax=-1):
    V = np.asarray(V, np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=ax, keepdims=True), 1e-8)


def rproj(din, dout=D_P, seed=17):
    rs = np.random.RandomState(seed)
    return (rs.randn(din, dout) / np.sqrt(dout)).astype(np.float32)


def dp_cuts(X, alpha=ALPHA, min_len=MIN_UNIT, max_units=MAX_UNITS):
    """Douglas-Peucker corner cuts of trajectory X (T, D).

    Tolerance is alpha * radius of gyration of THIS trajectory, so a
    clip with little going on is not shredded into noise units and a
    busy clip is not forced into one.
    """
    T = len(X)
    if T < 2 * min_len:
        return [0, T]
    rg = float(np.mean(np.linalg.norm(X - X.mean(0), axis=1)))
    tol = alpha * max(rg, 1e-6)
    cuts = {0, T - 1}
    stack = [(0, T - 1)]
    while stack and len(cuts) < max_units + 1:
        a, b = stack.pop()
        if b - a < 2 * min_len:
            continue
        u = X[b] - X[a]
        nu = float(np.linalg.norm(u))
        if nu < 1e-8:
            continue
        u = u / nu
        rel = X[a:b + 1] - X[a]
        perp = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        inner = perp[min_len:b - a - min_len + 1]
        if not len(inner):
            continue
        i = int(np.argmax(inner)) + min_len
        if float(perp[i]) > tol:
            cuts.add(a + i)
            stack.append((a, a + i))
            stack.append((a + i, b))
    c = sorted(cuts)
    c[-1] = T
    return c


def unit_rep(X, s, e):
    """Descriptor of one unit: its net displacement plus its halves -
    direction of change, and how that direction itself evolved."""
    mid = s + (e - s) // 2
    return np.concatenate([
        _l2(X[e - 1] - X[s]),
        _l2(X[mid] - X[s]),
        _l2(X[e - 1] - X[mid])])


def span_rep(X, s, e):
    """Whole-span baseline in the same projected space (3+5+8)."""
    parts = []
    for n in (3, 5, 8):
        cuts = [s + (e - s) * i // n for i in range(n + 1)]
        cuts[-1] = e - 1
        parts += [_l2(X[cuts[i + 1]] - X[cuts[i]]) for i in range(n)]
    return np.concatenate(parts)


EPCUT = {}


def load_events(alpha=ALPHA, epcuts=False):
    """Extended ruler: every verified event, both views, units + span."""
    P = None
    out = []
    from tqdm import tqdm
    srcs = [("data/sim_chains", "sim_chains"),
            ("data/sim_eval_bal", "sim_eval_bal")]
    eps = [(r, p, ep) for r, p in srcs
           for ep in sorted((ROOT / r).glob("ep*"))]
    for root, pref, ep in tqdm(eps, unit="ep", desc="units"):
        views = sorted(CACHE.glob(f"{pref}_{ep.name}_cam*_v3.npz"))
        if len(views) < 2:
            continue
        Z = [np.load(v) for v in views]
        R = []
        for z in Z:
            r = z["r20"].astype(np.float32)
            if P is None:
                P = rproj(r.shape[1])
            R.append(_l2(r @ P))
        if epcuts:
            for vi, X in enumerate(R):
                EPCUT[(pref, ep.name, vi)] = dp_cuts(
                    X, alpha=alpha, max_units=64)
        meta = json.loads((ep / "meta.json").read_text())
        for e in meta["events"]:
            if not e["ok"]:
                continue
            T = min(len(x) for x in R)
            a, b = int(e["t0"] * FPS), min(T, int(e["t1"] * FPS))
            if b - a < 8:
                continue
            units, spans = [], []
            for vi, X in enumerate(R):
                W = X[a:b]
                if epcuts:
                    # SHIPPING GEOMETRY: the episode is segmented ONCE
                    # at write time and a window inherits the cuts that
                    # fall inside it. A live store cannot re-segment per
                    # candidate window, so the benchmark has to be run
                    # on the same geometry it would ship with.
                    ec = EPCUT[(pref, ep.name, vi)]
                    inner = [c for c in ec if a + MIN_UNIT <= c
                             <= b - MIN_UNIT]
                    cs = [0] + [c - a for c in inner] + [b - a]
                else:
                    cs = dp_cuts(W, alpha=alpha)
                us = [unit_rep(W, cs[i], cs[i + 1])
                      for i in range(len(cs) - 1)
                      if cs[i + 1] - cs[i] >= MIN_UNIT]
                if not us:
                    us = [unit_rep(W, 0, len(W))]
                units.append(_l2(np.stack(us)))
                spans.append(_l2(span_rep(W, 0, len(W))))
            out.append(dict(prim=e["prim"], ep=f"{pref}/{ep.name}",
                            units=units, span=spans))
    return out


def setmatch_all(events, vq, vc):
    """(n, n) symmetric best-match scores between unit sets, computed
    with segment reductions instead of per-pair python."""
    Uc = np.concatenate([e["units"][vc] for e in events])
    off = np.cumsum([0] + [len(e["units"][vc]) for e in events])
    cnt = np.diff(off).astype(np.float32)
    n = len(events)
    S = np.zeros((n, n), np.float32)
    for i, e in enumerate(events):
        M = e["units"][vq] @ Uc.T                    # (u_i, N_c)
        # for each candidate event: max over ITS units, mean over query
        best_c = np.maximum.reduceat(M, off[:-1], axis=1)   # (u_i, n)
        S[i] += best_c.mean(0)
        # and: max over QUERY units per candidate unit, mean over cand
        col = M.max(0)                                      # (N_c,)
        S[i] += np.add.reduceat(col, off[:-1]) / cnt
    return 0.5 * S


def align_all(events, vq, vc, mode="dtw"):
    """Order-consistent alignment between unit sequences.

    Set best-match is permissive - two clips that merely SHARE a
    descend phase score high, which is exactly the precision loss it
    measured. A phase sequence is ordered by construction (descend
    then lift, not lift then descend), so the score has to respect
    order: DTW over unit descriptors, normalized by path length.
    `cover` is the strictness knob instead: every query unit must find
    a partner (min, not mean).
    """
    m = max(len(e["units"][vc]) for e in events)
    n = len(events)
    C = np.zeros((n, m, events[0]["units"][vc].shape[1]), np.float32)
    L = np.zeros(n, np.int64)
    for j, e in enumerate(events):
        U = e["units"][vc]
        C[j, :len(U)] = U
        L[j] = len(U)
    S = np.zeros((n, n), np.float32)
    NEG = -1e4
    for i, e in enumerate(events):
        Q = e["units"][vq]                      # (u, d)
        u = len(Q)
        M = np.einsum("ud,jmd->ujm", Q, C)      # (u, n, m)
        pad = np.arange(m)[None, :] >= L[:, None]
        M[:, pad] = NEG
        if mode == "cover":
            best = M.max(2)                     # (u, n) best per query unit
            S[i] = best.min(0)                  # every unit must match
            continue
        D = np.full((u, n, m), NEG, np.float32)
        D[0, :, 0] = M[0, :, 0]
        for j2 in range(1, m):
            D[0, :, j2] = M[0, :, j2] + D[0, :, j2 - 1]
        for i2 in range(1, u):
            D[i2, :, 0] = M[i2, :, 0] + D[i2 - 1, :, 0]
            for j2 in range(1, m):
                prev = np.maximum(np.maximum(D[i2 - 1, :, j2],
                                             D[i2 - 1, :, j2 - 1]),
                                  D[i2, :, j2 - 1])
                D[i2, :, j2] = M[i2, :, j2] + prev
        end = D[u - 1, np.arange(n), L - 1]
        S[i] = end / (u + L - 1)                # path-length normalized
    return S


def bench(S, prims, eps_, tag):
    n = len(prims)
    ys, ps, aps = [], [], []
    per = defaultdict(lambda: [[], [], []])
    for i in range(n):
        mask = eps_ != eps_[i]
        sc = S[i][mask]
        lab = (prims[mask] == prims[i])
        support = int(lab.sum())
        if not support:
            continue
        order = np.argsort(-sc)
        lo = lab[order]
        tp = lo.cumsum()
        hp = np.where(lo)[0]
        ap = float(np.mean((np.arange(support) + 1) / (hp + 1)))
        kk = np.arange(1, len(lo) + 1)
        best = int(np.argmax(np.minimum(tp / support, tp / kk)))
        y = float(tp[best] / support)
        p = float(tp[best] / (best + 1))
        aps.append(ap); ys.append(y); ps.append(p)
        per[prims[i]][0].append(ap)
        per[prims[i]][1].append(y)
        per[prims[i]][2].append(p)
    print(f"  {tag:22s} AP {np.mean(aps):.3f}   "
          f"yield {np.mean(ys):.3f}  prec {np.mean(ps):.3f}")
    for p in ("pick", "place", "stack", "unstack", "push"):
        if per[p][0]:
            print(f"      {p:9s} AP {np.mean(per[p][0]):.3f}  "
                  f"y/p {np.mean(per[p][1]):.3f}/{np.mean(per[p][2]):.3f}")
    return float(np.mean(ys)), float(np.mean(ps))


def zrow(S):
    """Per-row standardization: the three scorers live on different
    scales, so a fused score has to compare like with like."""
    mu = S.mean(1, keepdims=True)
    sd = S.std(1, keepdims=True) + 1e-6
    return (S - mu) / sd


def adaptive_fuse(parts, selfs):
    """PER-QUERY weights, label-free: a scorer earns weight on a query
    when THIS query stands out from the corpus under it - its own
    cross-view agreement minus its mean agreement with everything else.
    A multi-phase query is distinctive under ordered alignment; a
    single-sweep query is not, and the weights say so without anyone
    naming a class. (Same principle as the read path's per-query
    channel weighting, which measured better than any fixed fusion.)
    """
    W = []
    for S, sv in zip(parts, selfs):
        info = sv - S.mean(1)
        W.append(info / (S.std(1) + 1e-6))
    W = np.stack(W)                                  # (c, n)
    W = np.maximum(W, 0)
    tot = W.sum(0, keepdims=True)
    # a query that stands out under NO scorer gets uniform weights;
    # dividing by ~0 zeroed its whole row and turned its ranking into
    # noise (measured: AP 0.394 -> 0.325, all of it from those rows)
    dead = tot[0] < 1e-6
    W[:, dead] = 1.0 / len(parts)
    tot = W.sum(0, keepdims=True)
    W = W / tot
    out = np.zeros_like(parts[0])
    for w, S in zip(W, parts):
        out += w[:, None] * zrow(S)
    return out, W


def csls(S, k=50):
    hub = np.sort(S, 1)[:, -k:].mean(1)
    return 2 * S - hub[None, :] - hub[:, None]


def main():
    alpha = ALPHA
    if "--alpha" in sys.argv:
        alpha = float(sys.argv[sys.argv.index("--alpha") + 1])
    ev = load_events(alpha, epcuts="--epcuts" in sys.argv)
    prims = np.array([e["prim"] for e in ev])
    eps_ = np.array([e["ep"] for e in ev])
    nu = [len(e["units"][0]) for e in ev]
    print(f"{len(ev)} events, alpha={alpha}; units/event "
          f"median {int(np.median(nu))} mean {np.mean(nu):.1f} "
          f"p90 {int(np.percentile(nu, 90))}")

    Sp = None
    for a in (0, 1):
        for b in (0, 1):
            M = np.stack([e["span"][a] for e in ev])
            N = np.stack([e["span"][b] for e in ev])
            X = M @ N.T
            Sp = X if Sp is None else np.maximum(Sp, X)
    bench(csls(Sp), prims, eps_, "whole-span (base)")

    Su = None
    for a in (0, 1):
        for b in (0, 1):
            X = setmatch_all(ev, a, b)
            Su = X if Su is None else np.maximum(Su, X)
    bench(csls(Su), prims, eps_, "kinematic units")

    for w in (0.3, 0.5):
        bench(csls((1 - w) * Sp + w * Su), prims, eps_,
              f"fused w_units={w}")

    Sd = None
    for a in (0, 1):
        for b in (0, 1):
            X = align_all(ev, a, b, "dtw")
            Sd = X if Sd is None else np.maximum(Sd, X)
    bench(csls(Sd), prims, eps_, "DTW ordered units")

    Sc = None
    for a in (0, 1):
        for b in (0, 1):
            X = align_all(ev, a, b, "cover")
            Sc = X if Sc is None else np.maximum(Sc, X)
    bench(csls(Sc), prims, eps_, "unit coverage (min)")

    for w in (0.3, 0.5, 0.7):
        bench(csls((1 - w) * Sp + w * Sd), prims, eps_,
              f"span+DTW w={w}")
    bench(csls(0.5 * Sp + 0.25 * Sd + 0.25 * Sc), prims, eps_,
          "span+DTW+cover")

    # cross-view self-scores: the query's own two cameras, per scorer
    n = len(ev)
    sp_self = np.array([float(e["span"][0] @ e["span"][1]) for e in ev])
    Sd01 = align_all(ev, 0, 1, "dtw")
    Sc01 = align_all(ev, 0, 1, "cover")
    Sm01 = setmatch_all(ev, 0, 1)
    d_self = Sd01[np.arange(n), np.arange(n)]
    c_self = Sc01[np.arange(n), np.arange(n)]
    u_self = Sm01[np.arange(n), np.arange(n)]

    np.savez(ROOT / "data/cache/kin_scores.npz", Sp=Sp, Sd=Sd, Su=Su,
             Sc=Sc, sp_self=sp_self, d_self=d_self, u_self=u_self,
             c_self=c_self, prims=prims, eps=eps_)
    print("  [score matrices cached -> data/cache/kin_scores.npz]")

    F, W = adaptive_fuse([Sp, Sd], [sp_self, d_self])
    print(f"  [adaptive weights span/DTW: {W.mean(1).round(2)}]")
    bench(csls(F), prims, eps_, "adaptive span+DTW")

    F2, W2 = adaptive_fuse([Sp, Sd, Su], [sp_self, d_self, u_self])
    print(f"  [weights span/DTW/set: {W2.mean(1).round(2)}]")
    bench(csls(F2), prims, eps_, "adaptive span+DTW+set")

    F3, W3 = adaptive_fuse([Sp, Sd, Su, Sc],
                           [sp_self, d_self, u_self, c_self])
    print(f"  [weights span/DTW/set/cover: {W3.mean(1).round(2)}]")
    bench(csls(F3), prims, eps_, "adaptive all four")


if __name__ == "__main__":
    main()
