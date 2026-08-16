"""Kinematic moment representation + correspondence matching over
CoTracker3 point tracks. No appearance, no labels, no class names, no
world axes, no per-dataset constants.

A moment is: the set of things that moved, how each moved, and how
they moved RELATIVE TO EACH OTHER.

  bundles   points that move together = one physical thing (common
            fate; L2's statistic, now on reliable kinematics). No
            detection, no segmentation, no naming.
  node      per-bundle motion shape, all rotation/translation
            invariant: normalized speed profile, path/net, arc,
            active fraction, displacement in own-cloud units.
  edge      per-pair distance-over-time in the pair's own extents -
            approach / contact / separation without any axis.
  match     best assignment between two moments' bundle sets,
            scoring node similarity + edge consistency. This is
            correspondence, not overlap-of-descriptions.

Everything self-calibrates per recording from its own statistics.
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "data" / "cache" / "ptrack"
NS = 12                  # samples per time series
MAXB = 4                 # bundles kept per moment (largest first)


def _resample(a, n=NS):
    if len(a) == n:
        return a
    x = np.linspace(0, 1, len(a))
    return np.interp(np.linspace(0, 1, n), x, a)


def bundles(xy, vis, a, b):
    """Points that MOVED in [a,b], grouped by common fate.

    Two points belong together when their velocity sequences agree
    (correlation over the window) and they stay at a stable offset.
    Threshold is the recording's own: the noise floor of stationary
    points, never a constant."""
    T, P, _ = xy.shape
    a = max(int(a), 0)
    b = min(int(b), T - 1)
    if b - a < 4:
        return []
    W = xy[a:b + 1]                                   # (t,P,2)
    ok = vis[a:b + 1].mean(0) > 0.5
    disp = np.linalg.norm(W[-1] - W[0], axis=-1)
    rngp = np.linalg.norm(W.max(0) - W.min(0), axis=-1)
    # the recording's own noise floor: most points are static, so the
    # low quantile IS the jitter scale
    floor = max(float(np.percentile(rngp, 60)) * 3.0, 4.0)
    idx = np.where(ok & (rngp > floor))[0]
    if len(idx) < 3:
        return []
    V = np.diff(W[:, idx], axis=0)                    # (t-1,k,2)
    F = V.transpose(1, 0, 2).reshape(len(idx), -1)
    F = F - F.mean(1, keepdims=True)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    C = F @ F.T                                       # velocity agreement
    # positional coherence: same thing keeps a stable relative offset
    D = W[:, idx]
    off = D - D.mean(1, keepdims=True)
    jit = np.zeros((len(idx), len(idx)), np.float32)
    for i in range(len(idx)):
        dd = np.linalg.norm(D[:, i][:, None, :] - D[:, :], axis=-1)
        jit[i] = dd.std(0)
    scale = max(float(np.median(rngp[idx])), 1.0)
    A = (C > 0.6) & (jit < 0.5 * scale)
    # connected components on the agreement graph
    lab = -np.ones(len(idx), int)
    c = 0
    for i in range(len(idx)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], c
        while stack:
            u = stack.pop()
            for v in np.where(A[u] & (lab < 0))[0]:
                lab[v] = c
                stack.append(v)
        c += 1
    out = []
    for k in range(c):
        m = idx[lab == k]
        if len(m) < 3:
            continue
        out.append(m)
    out.sort(key=lambda m: -len(m))
    return out[:MAXB]


def moment(xy, vis, a, b):
    """Nodes + edges for the window. Pure kinematics."""
    bs = bundles(xy, vis, a, b)
    if not bs:
        return None
    a, b = int(a), min(int(b), len(xy) - 1)
    W = xy[a:b + 1]
    # the recording's own length unit for this window: the spread of
    # everything that moved (never pixels, never a world axis)
    allm = np.concatenate(bs)
    unit = max(float(np.linalg.norm(
        W[:, allm].reshape(-1, 2).std(0)) ), 1.0)
    nodes, cent = [], []
    for m in bs:
        c = W[:, m].mean(1)                            # (t,2) centroid
        step = np.linalg.norm(np.diff(c, axis=0), axis=-1)
        path = float(step.sum())
        net = float(np.linalg.norm(c[-1] - c[0]))
        sp = _resample(step)
        spn = sp / max(sp.max(), 1e-6)
        # arc: deviation from the straight chord, own units
        ch = c[-1] - c[0]
        L = float(np.linalg.norm(ch))
        if L > 1e-6:
            perp = np.abs((c - c[0])[:, 0] * ch[1]
                          - (c - c[0])[:, 1] * ch[0]) / L
        else:
            perp = np.linalg.norm(c - c[0], axis=-1)
        moving = float((step > 0.15 * max(step.max(), 1e-6)).mean())
        # spread of the bundle = its own size
        size = max(float(np.linalg.norm(W[:, m].std(1).mean(0))), 1.0)
        nodes.append(dict(
            spn=spn.astype(np.float32),
            netu=min(net / unit, 4.0) / 4.0,
            pathnet=min(path / max(net, 1e-6), 6.0) / 6.0,
            arc=min(float(perp.max()) / max(L, 1e-6), 2.0) / 2.0,
            moving=moving,
            frac=len(m) / max(len(allm), 1),
            size=min(size / unit, 2.0) / 2.0))
        cent.append(c)
    edges = {}
    for i in range(len(cent)):
        for j in range(i + 1, len(cent)):
            d = np.linalg.norm(cent[i] - cent[j], axis=-1) / unit
            ds = _resample(d)
            edges[(i, j)] = np.minimum(ds, 4.0).astype(np.float32) / 4.0
    return dict(nodes=nodes, edges=edges, n=len(nodes))


def field(xy, vis, a, b, K=48):
    """Clustering-free alternative: the moving point cloud described
    by PERMUTATION-INVARIANT statistics of its own geometry over
    time. No bundles, no roles, no correspondence search, no axes.

      shape     histogram of pairwise distances at 4 time slices -
                the configuration's own form ("D2 shape distribution")
      dshape    how that histogram MOVED between slices = things
                came together / spread apart
      speed     distribution of per-point speed over time
      disp      distribution of per-point net displacement
      active    fraction of points moving, over time

    Everything is normalized by the window's own motion scale, so
    zoom, viewpoint distance and units cancel."""
    T = len(xy)
    a, b = max(int(a), 0), min(int(b), T - 1)
    if b - a < 4:
        return None
    W = xy[a:b + 1]
    ok = vis[a:b + 1].mean(0) > 0.5
    rngp = np.linalg.norm(W.max(0) - W.min(0), axis=-1)
    floor = max(float(np.percentile(rngp, 60)) * 3.0, 4.0)
    idx = np.where(ok & (rngp > floor))[0]
    if len(idx) < 6:
        return None
    if len(idx) > K:
        idx = idx[np.argsort(-rngp[idx])[:K]]
    C = W[:, idx]                                    # (t,k,2)
    unit = max(float(np.linalg.norm(C.reshape(-1, 2).std(0))), 1.0)
    ts = np.linspace(0, len(C) - 1, 4).astype(int)
    shape = []
    for t in ts:
        D = np.linalg.norm(C[t][:, None] - C[t][None, :], axis=-1)
        iu = np.triu_indices(len(idx), 1)
        h, _ = np.histogram(D[iu] / unit, 10, (0.0, 4.0))
        shape.append(h / max(h.sum(), 1))
    shape = np.stack(shape)
    dshape = np.diff(shape, axis=0)
    step = np.linalg.norm(np.diff(C, axis=0), axis=-1) / unit
    sp = np.stack([_resample(step[:, i]) for i in range(len(idx))])
    speed = np.percentile(sp, [10, 50, 90], axis=0)   # (3,NS)
    net = np.linalg.norm(C[-1] - C[0], axis=-1) / unit
    disp, _ = np.histogram(net, 8, (0.0, 3.0))
    disp = disp / max(disp.sum(), 1)
    thr = 0.15 * float(step.max()) if step.size else 0.0
    active = _resample((step > thr).mean(1))
    return np.concatenate([shape.ravel(), dshape.ravel() * 2.0,
                           speed.ravel(), disp,
                           active]).astype(np.float32)


def field_sim(f1, f2):
    if f1 is None or f2 is None:
        return 0.0
    return float(np.exp(-3.0 * np.abs(f1 - f2).mean()))


def phases(xy, vis, a, b):
    """Group points by WHEN they move, not by which way they move.

    Measured cause of the first readout's failure: during transport
    the agent and the carried thing move as one, so velocity-based
    common fate merges them (151/324 moments collapsed to a single
    bundle). But their ACTIVITY SUPPORTS differ - the agent is also
    moving before the pick-up and after the release, the carried
    thing is not. Temporal support separates what velocity cannot,
    and it is exactly the approach/release structure the probe says
    carries the signal.

    Per group: when it started, when it stopped, how long, its speed
    shape, its net displacement in the window's own units.
    Per pair: distance over time + how much their activity overlaps.
    No axes, no names, no classes."""
    T = len(xy)
    a, b = max(int(a), 0), min(int(b), T - 1)
    if b - a < 6:
        return None
    W = xy[a:b + 1]
    ok = vis[a:b + 1].mean(0) > 0.5
    rngp = np.linalg.norm(W.max(0) - W.min(0), axis=-1)
    floor = max(float(np.percentile(rngp, 60)) * 3.0, 4.0)
    idx = np.where(ok & (rngp > floor))[0]
    if len(idx) < 6:
        return None
    C = W[:, idx]
    step = np.linalg.norm(np.diff(C, axis=0), axis=-1)     # (t-1,k)
    unit = max(float(np.linalg.norm(C.reshape(-1, 2).std(0))), 1.0)
    # activity trace per point, self-normalized
    A = step / np.maximum(step.max(0, keepdims=True), 1e-6)
    An = A - A.mean(0, keepdims=True)
    An = An / np.maximum(np.linalg.norm(An, axis=0, keepdims=True), 1e-6)
    S = An.T @ An                                          # support agreement
    lab = -np.ones(len(idx), int)
    c = 0
    for i in range(len(idx)):
        if lab[i] >= 0:
            continue
        stack_, lab[i] = [i], c
        while stack_:
            u = stack_.pop()
            for v in np.where((S[u] > 0.5) & (lab < 0))[0]:
                lab[v] = c
                stack_.append(v)
        c += 1
    groups = [np.where(lab == k)[0] for k in range(c)]
    groups = [g for g in groups if len(g) >= 3]
    groups.sort(key=lambda g: -len(g))
    groups = groups[:MAXB]
    if not groups:
        return None
    nodes, cent, acts = [], [], []
    t = step.shape[0]
    for g in groups:
        act = A[:, g].mean(1)
        on = act > 0.3
        i0 = int(np.argmax(on)) if on.any() else 0
        i1 = int(t - 1 - np.argmax(on[::-1])) if on.any() else 0
        cc = C[:, g].mean(1)
        net = float(np.linalg.norm(cc[-1] - cc[0])) / unit
        sp = _resample(np.linalg.norm(np.diff(cc, axis=0), axis=-1))
        ch = cc[-1] - cc[0]
        L = float(np.linalg.norm(ch))
        perp = (np.abs((cc - cc[0])[:, 0] * ch[1]
                       - (cc - cc[0])[:, 1] * ch[0]) / L) if L > 1e-6 \
            else np.linalg.norm(cc - cc[0], axis=-1)
        nodes.append(dict(
            spn=(sp / max(sp.max(), 1e-6)).astype(np.float32),
            act=_resample(act).astype(np.float32),
            onset=i0 / max(t - 1, 1),
            offset=i1 / max(t - 1, 1),
            dur=float(on.mean()),
            netu=min(net, 4.0) / 4.0,
            arc=min(float(perp.max()) / max(L, 1e-6), 2.0) / 2.0,
            frac=len(g) / max(len(idx), 1)))
        cent.append(cc)
        acts.append(act)
    edges = {}
    for i in range(len(cent)):
        for j in range(i + 1, len(cent)):
            d = _resample(np.linalg.norm(cent[i] - cent[j], axis=-1)
                          / unit)
            ov = float(((acts[i] > 0.3) & (acts[j] > 0.3)).mean())
            edges[(i, j)] = (np.minimum(d, 4.0).astype(np.float32)
                             / 4.0, ov)
    return dict(nodes=nodes, edges=edges, n=len(nodes))


PK = ("onset", "offset", "dur", "netu", "arc", "frac")


def _psim(x, y):
    s = 1.0 - 0.4 * float(np.abs(x["spn"] - y["spn"]).mean())
    s *= 1.0 - 0.5 * float(np.abs(x["act"] - y["act"]).mean())
    for k in PK:
        w = 0.8 if k in ("onset", "offset", "dur") else 0.4
        s *= 1.0 - w * abs(x[k] - y[k])
    return max(s, 0.0)


def pmatch(M1, M2):
    if M1 is None or M2 is None:
        return 0.0
    small, big = (M1, M2) if M1["n"] <= M2["n"] else (M2, M1)
    ns, nb = small["n"], big["n"]
    best = 0.0
    for perm in permutations(range(nb), ns):
        node = np.mean([_psim(small["nodes"][i], big["nodes"][perm[i]])
                        for i in range(ns)])
        eds = []
        for i in range(ns):
            for j in range(i + 1, ns):
                e1 = small["edges"].get((i, j))
                p_, q_ = perm[i], perm[j]
                e2 = big["edges"].get((min(p_, q_), max(p_, q_)))
                if e1 is None or e2 is None:
                    continue
                d = 1.0 - float(np.abs(e1[0] - e2[0]).mean())
                o = 1.0 - abs(e1[1] - e2[1])
                eds.append(0.6 * d + 0.4 * o)
        edge = float(np.mean(eds)) if eds else None
        sc = node if edge is None else (node ** 0.5) * (edge ** 0.5)
        sc *= (ns / max(nb, 1)) ** 0.25
        best = max(best, sc)
    return best


NK = ("netu", "pathnet", "arc", "moving", "frac", "size")


def _nsim(x, y):
    s = 1.0 - 0.5 * float(np.abs(x["spn"] - y["spn"]).mean())
    for k in NK:
        w = 0.8 if k in ("pathnet", "arc", "netu") else 0.4
        s *= 1.0 - w * abs(x[k] - y[k])
    return max(s, 0.0)


def match(M1, M2):
    """Best partial correspondence between two moments' bundle sets."""
    if M1 is None or M2 is None:
        return 0.0
    n1, n2 = M1["n"], M2["n"]
    small, big = (M1, M2) if n1 <= n2 else (M2, M1)
    ns, nb = small["n"], big["n"]
    best = 0.0
    for perm in permutations(range(nb), ns):
        node = np.mean([_nsim(small["nodes"][i], big["nodes"][perm[i]])
                        for i in range(ns)])
        eds = []
        for i in range(ns):
            for j in range(i + 1, ns):
                e1 = small["edges"].get((i, j))
                p, q = perm[i], perm[j]
                e2 = big["edges"].get((min(p, q), max(p, q)))
                if e1 is None or e2 is None:
                    continue
                eds.append(1.0 - float(np.abs(e1 - e2).mean()))
        edge = float(np.mean(eds)) if eds else None
        sc = node if edge is None else (node ** 0.5) * (edge ** 0.5)
        # unmatched bundles are missing structure, not free
        sc *= (ns / max(nb, 1)) ** 0.25
        best = max(best, sc)
    return best


def load(corpus="data/prim_actions_v2"):
    from tqdm import tqdm
    name = Path(corpus).name
    evs = []
    eps = sorted((ROOT / corpus).glob("ep*"))
    for ep in tqdm(eps, unit="ep", desc="kin moments"):
        f = PT / f"{name}_{ep.name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        xy, vis = z["xy"].astype(np.float32), z["vis"]
        meta = json.loads((ep / "meta.json").read_text())
        for e in meta["events"]:
            if not e["ok"]:
                continue
            a, b = int(e["t0"] * 10), int(e["t1"] * 10)
            if b - a < 4:
                continue
            evs.append((e["prim"], ep.name, meta.get("arm", "?"),
                        moment(xy, vis, a - 2, b + 4),
                        field(xy, vis, a - 2, b + 4),
                        phases(xy, vis, a - 2, b + 4)))
    return evs


def bench(corpus="data/prim_actions_v2"):
    from collections import Counter, defaultdict
    from tqdm import tqdm
    evs = load(corpus)
    n = len(evs)
    prim = np.array([e[0] for e in evs])
    ep_ = np.array([e[1] for e in evs])
    arm = np.array([e[2] for e in evs])
    print(f"{n} events | no-moment "
          f"{sum(1 for e in evs if e[3] is None)} | no-field "
          f"{sum(1 for e in evs if e[4] is None)}")
    print("bundles/moment:", dict(Counter(e[3]["n"] for e in evs
                                          if e[3])))
    print("phase groups/moment:",
          dict(Counter(e[5]["n"] for e in evs if e[5])))
    Sm = np.zeros((n, n), np.float32)
    Sf = np.zeros((n, n), np.float32)
    Sp = np.zeros((n, n), np.float32)
    for i in tqdm(range(n), desc="match"):
        for j in range(i + 1, n):
            Sm[i, j] = Sm[j, i] = match(evs[i][3], evs[j][3])
            Sf[i, j] = Sf[j, i] = field_sim(evs[i][4], evs[j][4])
            Sp[i, j] = Sp[j, i] = pmatch(evs[i][5], evs[j][5])
    np.save(ROOT / "data/cache/kin_Sp.npy", Sp)
    np.save(ROOT / "data/cache/kin_Sm.npy", Sm)
    np.save(ROOT / "data/cache/kin_Sf.npy", Sf)
    np.save(ROOT / "data/cache/kin_meta.npy",
            np.array([(e[0], e[1], e[2]) for e in evs]))

    def rankz(S):
        return -np.argsort(np.argsort(-S, 1), 1).astype(np.float32) / n

    def csls_diffuse(S, topk=12, alpha=0.55, iters=8):
        r = np.sort(S, 1)[:, ::-1][:, 1:topk + 1].mean(1)
        C = 2 * S - r[None, :] - r[:, None]
        W = np.exp((C - C.max()) / 0.35)
        np.fill_diagonal(W, 0)
        thr = np.sort(W, 1)[:, ::-1][:, topk:topk + 1]
        W[W < thr] = 0
        W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
        F = S.copy()
        for _ in range(iters):
            F = (1 - alpha) * S + alpha * (W @ F)
        return F

    def run(S, name, qmask=None):
        y1, y15, p15 = defaultdict(list), defaultdict(list), \
            defaultdict(list)
        for i in range(n):
            if qmask is not None and not qmask[i]:
                continue
            mask = ep_ != ep_[i]
            if qmask is not None:
                mask = mask & ~qmask          # cross-embodiment store
            sup = int((prim[mask] == prim[i]).sum())
            if not sup:
                continue
            order = np.argsort(-S[i][mask])
            lab = (prim[mask] == prim[i])
            y1[prim[i]].append(lab[order[:sup]].sum() / sup)
            k15 = min(int(1.5 * sup), int(mask.sum()))
            t = lab[order[:k15]].sum()
            y15[prim[i]].append(t / sup)
            p15[prim[i]].append(t / k15)
        a1 = [v for p in y1 for v in y1[p]]
        per = " ".join(f"{p[:2]}{np.mean(y1[p]):.2f}"
                       for p in ("pick", "place", "stack", "unstack",
                                 "push") if y1[p])
        print(f"{name:32s} y@sup {np.mean(a1):.3f}  y@1.5 "
              f"{np.mean([v for p in y15 for v in y15[p]]):.3f}  "
              f"p@1.5 {np.mean([v for p in p15 for v in p15[p]]):.3f}"
              f"  [{per}]")
        return float(np.mean(a1))

    run(Sm, "kin bundles+correspondence")
    run(Sf, "kin field (perm-invariant)")
    run(Sp, "kin PHASE groups (temporal support)")
    Sc = rankz(Sm) + rankz(Sf)
    run(Sc, "kin bundles+field")
    Sc3 = rankz(Sp) + rankz(Sf)
    run(Sc3, "kin phase+field")
    Sc4 = rankz(Sp) + rankz(Sf) + rankz(Sm)
    run(Sc4, "kin all three")
    run(csls_diffuse((Sc3 - Sc3.mean()) / (Sc3.std() + 1e-9)),
        "kin phase+field + L8")
    best = Sc3
    for a_ in sorted(set(arm)):
        run(best, f"  cross-embodiment: {a_}", qmask=(arm == a_))
    return evs, Sm, Sf


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
