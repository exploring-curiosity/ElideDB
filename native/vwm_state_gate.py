"""THE SEGMENTATION GATE: does a candidate segmentation earn the
referenced-state metric?

Measured 2026-08-09 (BENCHMARKS "similarity lookup, triangulated"):
a ~16-dim subspace of the referenced-state space is worth +0.050 AP /
+0.064 yield / +0.043 prec fused with the shipped deltas - PROVEN by
cross-view CCA fitted on truth-event spans, holdout-stable across
episodes. But every label-free fit source fails to locate it: stride
windows 0.350, dp_cuts unit spans 0.334, vs raw 0.367 and event-fit
0.433. The differentiator is SPAN QUALITY.

So segmentation quality now has a retrieval-denominated acceptance
test instead of an aesthetic one:

    a segmentation is good enough when CCA fitted on ITS spans
    recovers the event-fit gain.

Usage: --spans {events|stride|units} picks the fit source; 'events'
is the supervised ceiling of the gate (uses meta.json, EVAL ONLY),
the others are label-free candidates. A future segmentation plugs in
as a new spans source and is judged by the same three numbers.

    python native/vwm_state_gate.py --spans events
    python native/vwm_state_gate.py --spans units
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
CACHE = ROOT / "data" / "cache" / "vwm"
FPS = 10
PRIMS = ("pick", "place", "stack", "unstack", "push")
CORPORA = ("sim_chains", "sim_eval_bal")
K, REG = 16, 1.0          # winning config; the subspace is tiny


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


def endmean(rr, t):
    a, b = max(0, t - 1), min(len(rr), t + 2)
    return rr[a:b].mean(0)


def statevec(rr, a, b):
    return _l2(np.concatenate([_l2(endmean(rr, a)),
                               _l2(endmean(rr, b - 1))]))


def episodes():
    for corp in CORPORA:
        for ep in sorted((ROOT / "data" / corp).glob("ep*")):
            views = sorted(CACHE.glob(f"{corp}_{ep.name}_cam*_v3.npz"))
            if len(views) < 2:
                continue
            Z = [np.load(v) for v in views[:2]]
            r20s = [z["r20"].astype(np.float32) for z in Z]
            rrefs = [r - np.median(r, 0) for r in r20s]
            yield corp, ep, r20s, rrefs


def truth_events():
    """The benchmark set (labels used for GRADING only)."""
    out = []
    for corp, ep, r20s, rrefs in episodes():
        meta = json.loads((ep / "meta.json").read_text())
        for e in meta["events"]:
            if not e["ok"]:
                continue
            a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
            if min(len(v) for v in r20s) - a < 6 or b - a < 6:
                continue
            ds = [_l2(dmulti(v, a, min(b, len(v)))) for v in r20s]
            ss = [statevec(rr, a, min(b, len(rr))) for rr in rrefs]
            out.append((e["prim"], f"{corp}_{ep.name}", ds, ss))
    return out


def fit_pairs(source):
    """(W0, W1) state-vector pairs from the chosen span source."""
    W0, W1 = [], []
    if source in ("units", "refunits"):
        import vwm_kin
        P = None
    for corp, ep, r20s, rrefs in episodes():
        T = min(len(r) for r in r20s)
        spans = []
        if source == "events":
            meta = json.loads((ep / "meta.json").read_text())
            spans = [(int(e["t0"] * FPS), int(e["t1"] * FPS))
                     for e in meta["events"] if e["ok"]]
        elif source == "stride":
            spans = [(s, s + 30) for s in range(0, T - 30 + 1, 15)]
        elif source == "stridesel":
            # composition hypothesis: 5 boundary schemes fail alike ->
            # the differentiator may be fit-set COMPOSITION, not
            # boundary placement. Truth events are all manipulation
            # spans; candidates dilute with idle/travel. Keep only
            # stride spans whose settled state actually CHANGED
            # (per-recording Otsu on end-start state-diff mass).
            import vwm_units
            rr = rrefs[0][:T]
            cand = [(s, s + 30) for s in range(0, T - 30 + 1, 15)]
            mass = np.array([float(np.linalg.norm(
                endmean(rr, b - 1) - endmean(rr, a)))
                for a, b in cand], np.float32)
            thr = vwm_units.otsu(mass)
            spans = [c for c, m in zip(cand, mass) if m >= thr]
        elif source in ("units", "refunits"):
            # refunits: corner the REFERENCED trajectory - the
            # extraction repair may improve the segmentation itself
            if P is None:
                P = vwm_kin.rproj(r20s[0].shape[1])
            base = r20s[0][:T] if source == "units" else rrefs[0][:T]
            rp = vwm_kin._l2(base @ P)
            cuts = vwm_kin.dp_cuts(rp, max_units=64)
            spans = [(cuts[i], cuts[i + 1])
                     for i in range(len(cuts) - 1)
                     if cuts[i + 1] - cuts[i] >= 6]
            spans += [(cuts[i], cuts[i + 2])
                      for i in range(len(cuts) - 2)
                      if cuts[i + 2] - cuts[i] >= 10]
        elif source == "motion":
            # motion-energy valleys (Otsu, per-recording) on the
            # referenced trajectory - vwm_units.units_of operator
            import vwm_units
            spans = [(s, e) for s, e in vwm_units.units_of(rrefs[0][:T])
                     if e - s >= 6]
        elif source == "statechange":
            # gate round 1 verdict: motion boundaries teach the
            # nuisance; event boundaries mark where SETTLED STATE
            # differs before/after. Cut exactly there: boundary score
            # = distance between the settled referenced state in the
            # past vs future 0.5s window, maxima above per-recording
            # Otsu, spans between consecutive boundaries (+ merges).
            import vwm_units
            rr = rrefs[0][:T]
            W = 5
            bs = np.zeros(T, np.float32)
            for t in range(W, T - W):
                bs[t] = float(np.linalg.norm(
                    rr[t:t + W].mean(0) - rr[t - W:t].mean(0)))
            thr = vwm_units.otsu(bs[W:T - W])
            cuts = [0]
            for t in range(W, T - W):
                if bs[t] >= thr and bs[t] == bs[max(0, t - 3):t + 4].max() \
                        and t - cuts[-1] >= 6:
                    cuts.append(t)
            cuts.append(T)
            spans = [(cuts[i], cuts[i + 1])
                     for i in range(len(cuts) - 1)
                     if cuts[i + 1] - cuts[i] >= 6]
            spans += [(cuts[i], cuts[i + 2])
                      for i in range(len(cuts) - 2)
                      if cuts[i + 2] - cuts[i] >= 10]
        for a, b in spans:
            b = min(b, T)
            if b - a < 6:
                continue
            W0.append(statevec(rrefs[0], a, b))
            W1.append(statevec(rrefs[1], a, b))
    return np.stack(W0), np.stack(W1)


def cca_fit(X, Y, k=K, reg=REG):
    """Dual (Gram-space) regularized linear CCA - d is large, n is
    small, so the eigenproblem lives in n x n."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    m = len(Xc)
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    r = reg * np.trace(Kx) / m
    ry = reg * np.trace(Ky) / m
    A = np.linalg.solve(Kx + r * np.eye(m), Ky)
    B = np.linalg.solve(Ky + ry * np.eye(m), Kx)
    w, V = sla.eig(A @ B)
    w, V = w.real, V.real
    order = np.argsort(-w)[:k]
    rho = np.sqrt(np.clip(w[order], 0, 1))
    alpha = V[:, order]
    beta = B @ alpha
    ux, uy = Kx @ alpha, Ky @ beta
    ux_s = np.maximum(ux.std(0), 1e-8)
    uy_s = np.maximum(uy.std(0), 1e-8)
    return (lambda v: ((v - mx) @ Xc.T @ alpha) / ux_s * rho,
            lambda v: ((v - my) @ Yc.T @ beta) / uy_s * rho)


def smat(E0, E1):
    S = None
    for a in (E0, E1):
        for b in (E0, E1):
            Sx = a @ b.T
            S = Sx if S is None else np.maximum(S, Sx)
    hub = np.sort(S, 1)[:, -min(50, len(S)):].mean(1)
    return 2 * S - hub[None, :] - hub[:, None]


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / np.maximum(
        S.std(1, keepdims=True), 1e-8)


def bench(S, tag, prims, eps_):
    n = len(prims)
    ap_a, fy_a, fp_a = (defaultdict(list) for _ in range(3))
    for i in range(n):
        mask = eps_ != eps_[i]
        lab = (prims[mask] == prims[i])
        support = int(lab.sum())
        if not support:
            continue
        lo = lab[np.argsort(-S[i][mask])]
        hitpos = np.where(lo)[0]
        ap_a[prims[i]].append(
            float(np.mean((np.arange(support) + 1) / (hitpos + 1))))
        tp = lo.cumsum()
        k = min(int(np.ceil(1.5 * support)), len(lo))
        fy_a[prims[i]].append(float(tp[k - 1] / support))
        fp_a[prims[i]].append(float(tp[k - 1] / k))
    allv = defaultdict(list)
    per = {}
    for p in PRIMS:
        for d, k2 in ((ap_a, 'ap'), (fy_a, 'fy'), (fp_a, 'fp')):
            allv[k2] += d[p]
        per[p] = f"{np.mean(ap_a[p]):.2f}" if ap_a[p] else "-"
    print(f"{tag:24s} AP {np.mean(allv['ap']):.3f}  fixed "
          f"{np.mean(allv['fy']):.3f}/{np.mean(allv['fp']):.3f}  {per}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="events",
                    choices=("events", "stride", "units", "refunits",
                             "motion", "statechange", "stridesel"))
    a = ap.parse_args()
    ev = truth_events()
    prims = np.array([e[0] for e in ev])
    eps_ = np.array([e[1] for e in ev])
    D0 = np.stack([e[2][0] for e in ev])
    D1 = np.stack([e[2][1] for e in ev])
    S0 = np.stack([e[3][0] for e in ev])
    S1 = np.stack([e[3][1] for e in ev])
    print(f"{len(ev)} events; fit source = {a.spans}", flush=True)
    Sdel = smat(_l2(D0), _l2(D1))
    bench(Sdel, "del (ship)", prims, eps_)
    bench(smat(_l2(S0), _l2(S1)), "st raw", prims, eps_)
    W0, W1 = fit_pairs(a.spans)
    if len(W0) > 3000:
        sel = np.random.RandomState(3).choice(len(W0), 3000,
                                              replace=False)
        W0, W1 = W0[sel], W1[sel]
    ex, ey = cca_fit(W0, W1)
    Sst = smat(_l2(ex(S0)), _l2(ey(S1)))
    bench(Sst, f"st cca[{a.spans}]", prims, eps_)
    bench(0.5 * zrow(Sdel) + zrow(Sst), "fuse 0.5del+cca", prims, eps_)


if __name__ == "__main__":
    main()
