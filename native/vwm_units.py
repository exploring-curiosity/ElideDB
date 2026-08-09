"""SUB-EVENT UNITS: motion segments as the unit of matching.

The supervised ceiling said the composite classes bound corpus-wide
yield/prec: an unstack IS a grasp-from-tower + carry + release on
film, so whole-span matching cannot both retrieve it and respect the
span taxonomy. The honest unit is the MOTION SEGMENT: contiguous
high-change stretches between stillness valleys of the trajectory's
own change energy - no labels, threshold from the recording itself
(Otsu on the energy distribution), the same trick the persistent-
change detector used.

A window's representation is its SEQUENCE of unit reps (each unit is
the same ordered height-profile change operator). Similarity is
symmetric best-match over units with an order-consistency weight.

    python native/vwm_units.py          # benchmark vs whole-span
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
SMOOTH = 5          # frames (0.5s) energy smoothing
MIN_UNIT = 6        # frames - shorter than this is jitter
MERGE_GAP = 3       # frames of stillness bridged inside one unit


def _l2(V, ax=-1):
    V = np.asarray(V, np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=ax, keepdims=True), 1e-8)


def dmulti(r, s, e):
    parts = []
    for n in (3, 5, 8):
        cuts = [s + (e - s) * i // n for i in range(n + 1)]
        cuts[-1] = e - 1
        parts += [_l2(r[cuts[i + 1]] - r[cuts[i]]) for i in range(n)]
    return np.concatenate(parts)


def otsu(x):
    h, edges = np.histogram(x, 64)
    mids = (edges[:-1] + edges[1:]) / 2
    w = h / max(h.sum(), 1)
    best, thr = -1.0, float(np.median(x))
    for i in range(1, 63):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = (w[:i] * mids[:i]).sum() / w0
        m1 = (w[i:] * mids[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(mids[i])
    return thr


def units_of(r):
    """(T, D) trajectory -> [(s, e)] motion units. The threshold comes
    from THIS recording's energy distribution (Otsu), not a constant."""
    e = np.linalg.norm(np.diff(r, axis=0), axis=1)
    k = np.ones(SMOOTH) / SMOOTH
    e = np.convolve(e, k, mode="same")
    thr = otsu(e)
    on = e > thr
    # bridge short stillness gaps
    segs, s = [], None
    gap = 0
    for t, v in enumerate(on):
        if v:
            if s is None:
                s = t
            gap = 0
        elif s is not None:
            gap += 1
            if gap > MERGE_GAP:
                if t - gap - s + 1 >= MIN_UNIT:
                    segs.append((s, t - gap + 1))
                s, gap = None, 0
    if s is not None and len(on) - s >= MIN_UNIT:
        segs.append((s, len(on)))
    return segs


def window_units(r, a, b):
    """Units clipped to window [a,b); whole window if none intersect."""
    out = []
    for s, e in units_of(r):
        s2, e2 = max(s, a), min(e, b)
        if e2 - s2 >= MIN_UNIT:
            out.append(_l2(dmulti(r, s2, e2)))
    if not out:
        out = [_l2(dmulti(r, a, b))]
    return out


def setmatch(Q, C):
    """Symmetric best-match over unit reps + order consistency."""
    S = np.stack(Q) @ np.stack(C).T
    f = 0.5 * (S.max(1).mean() + S.max(0).mean())
    if len(Q) > 1 and len(C) > 1:
        qi = S.argmax(1)
        mono = float(np.mean(np.diff(qi) >= 0))
        f = f * (0.75 + 0.25 * mono)
    return float(f)


def main():
    events = []
    from tqdm import tqdm
    eps_dirs = sorted((ROOT / "data/sim_chains").glob("ep*"))
    for ep in tqdm(eps_dirs, unit="ep", desc="units"):
        views = sorted(CACHE.glob(f"sim_chains_{ep.name}_cam*_v3.npz"))
        if len(views) < 2:
            continue
        Z = [np.load(v) for v in views]
        r20s = [z["r20"].astype(np.float32) for z in Z]
        meta = json.loads((ep / "meta.json").read_text())
        for e in meta["events"]:
            if not e["ok"]:
                continue
            T = len(r20s[0])
            a, b = int(e["t0"] * FPS), min(T, int(e["t1"] * FPS))
            if b - a < 6:
                continue
            events.append((
                e["prim"], ep.name,
                [_l2(dmulti(v, a, min(b, len(v)))) for v in r20s],
                [window_units(v, a, min(b, len(v))) for v in r20s]))
    n = len(events)
    prims = np.array([e[0] for e in events])
    eps_ = np.array([e[1] for e in events])
    nu = [len(e[3][0]) for e in events]
    print(f"{n} events; units/event median {int(np.median(nu))} "
          f"p90 {int(np.percentile(nu, 90))}")

    def bench(S, tag):
        ys, ps, aps = [], [], []
        per = defaultdict(list)
        for i in range(n):
            mask = eps_ != eps_[i]
            sc = S[i][mask]
            lab = (prims[mask] == prims[i])
            support = int(lab.sum())
            if not support:
                continue
            order = np.argsort(-sc)
            lo2 = lab[order]
            tp = lo2.cumsum()
            hitpos = np.where(lo2)[0]
            ap = float(np.mean((np.arange(support) + 1) / (hitpos + 1)))
            aps.append(ap); per[prims[i]].append(ap)
            kk = np.arange(1, len(lo2) + 1)
            best = np.argmax(np.minimum(tp / support, tp / kk))
            ys.append(float(tp[best] / support))
            ps.append(float(tp[best] / (best + 1)))
        pp = {p: round(float(np.mean(v)), 2) for p, v in per.items()}
        print(f"  {tag:14s} AP {np.mean(aps):.3f}  oracle "
              f"{np.mean(ys):.3f}/{np.mean(ps):.3f}  {pp}")

    # whole-span baseline
    S = None
    for vq in (0, 1):
        for vc in (0, 1):
            Mq = np.stack([e[2][vq] for e in events])
            Mc = np.stack([e[2][vc] for e in events])
            Sx = Mq @ Mc.T
            S = Sx if S is None else np.maximum(S, Sx)
    hub = np.sort(S, 1)[:, -50:].mean(1)
    bench(2 * S - hub[None, :] - hub[:, None], "whole-span")

    # unit set-match (view-max)
    from tqdm import tqdm
    Su = np.zeros((n, n), np.float32)
    for i in tqdm(range(n), unit="q", desc="setmatch"):
        for j in range(i, n):
            v = max(setmatch(events[i][3][a2], events[j][3][b2])
                    for a2 in (0, 1) for b2 in (0, 1))
            Su[i, j] = Su[j, i] = v
    hub = np.sort(Su, 1)[:, -50:].mean(1)
    bench(2 * Su - hub[None, :] - hub[:, None], "unit-setmatch")


if __name__ == "__main__":
    main()
