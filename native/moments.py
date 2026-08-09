"""L4-L5 + L7 of the general solution (native/PROBLEM.md).

L4  interactions: per view, the agent's contact spans with episodic
    evidence; topology = vanish/appear sites in the settled scene.
L5  moments: a window's content = its episode chain - for each
    involved entity: (vanished-from, moved-with-agent span,
    appeared-at), all state RELATIONAL (PROBLEM.md #5): displacement
    in units of the entity's own extent, vertical/lateral split
    relative to the scene's own gravity axis... no: even "vertical"
    presumes the image frame. V1 uses the only frames PROBLEM.md
    licenses: self-past (displacement magnitude/timing), pairwise
    (to the agent), scene (departed/arrived, site persistence).
L7  correspondence: two moments match by the best assignment of their
    episode structures; components scored separately (structure /
    patient appearance / agent appearance) so intent weighting stays
    possible. No component may consume absolute coordinates.

Problem-check: nothing names arm/block/tower; every quantity is
extent-normalized or a pure event-structure fact (site appeared vs
vanished, contact overlap fraction, correlation persisted or ceased).

    python native/moments.py --bench          # sim ruler + holdout
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENT = ROOT / "data" / "cache" / "ent"
FPS = 10.0
PRIMS = ("pick", "place", "stack", "unstack", "push")
MERGE = {"pick": "grasp", "unstack": "grasp",
         "place": "release", "stack": "release", "push": "push"}


def _l2(v):
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


NDIM = 14


def moment(rows, view, a, b):
    """L5 v2: the episode structure of window [a,b) in one view, as
    GRADED relational facts (the v1 8-bit skeleton tied massively and
    added noise to the graph - measured).

    All facts are relational (PROBLEM.md #5): the mover's own extent
    is the length unit; time is in thirds of its own in-window life;
    'other entity' is any different-looking episodic evidence, never a
    named thing. Returns dict(vec, patient_rgb, agent_rgb) or None.
    """
    vr = [r for r in rows if r["view"] == view]
    ents = [r for r in vr if r["role"] == "entity"]
    agents = [r for r in vr if r["role"] == "agent"]
    sites_all = [s for s in vr if s["role"] == "site"]
    sites = [s for s in sites_all
             if s["t0"] <= b + 10 and s["t1"] >= a - 10]

    agent_rgb = None
    if agents:
        biggest = max(agents, key=lambda r: r["extent"])
        agent_rgb = np.median(np.asarray(biggest["rgb"]), 0)

    # mover selection: among displaced episodic tracks, PREFER one
    # whose colour matches a settled-scene change in the window - the
    # sites are the near-perfect evidence (0.999 gate) and agent
    # shards never match them. Max displacement is only the
    # tiebreak/fallback, not the selector (the v2 mistake: with ~100
    # junk tracks/view, max displacement often picks an agent shard).
    cands = []
    for r in ents:
        m = (r["t"] >= a) & (r["t"] <= b)
        if m.sum() < 3:
            continue
        ext_c = max(r["extent"], 4.0)
        d = np.hypot(r["cx"][m].max() - r["cx"][m].min(),
                     r["cy"][m].max() - r["cy"][m].min()) / ext_c
        if d <= 0.35:
            continue
        crgb = np.median(np.asarray(r["rgb"])[m], 0)
        anchored = any(
            min(np.abs(s["pre_rgb"] - crgb).sum(),
                np.abs(s["post_rgb"] - crgb).sum()) < 200
            for s in sites)
        cands.append((anchored, d, r, m))
    best, mv = None, None
    if cands:
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        anchored, best, r0, m0 = cands[0]
        mv = (r0, m0)

    if mv is None or best is None or best <= 0.35:
        nv = len(sites)
        if nv == 0:
            return None
        vec = np.zeros(NDIM, np.float32)
        vec[10] = vec[11] = min(nv, 3) / 3.0
        return dict(vec=vec, patient_rgb=None, agent_rgb=agent_rgb)

    r, m = mv
    ext = max(r["extent"], 4.0)
    tt = r["t"][m]
    cx, cy = r["cx"][m], r["cy"][m]
    con = r["contact"][m]
    n = len(tt)
    mrgb = np.median(np.asarray(r["rgb"])[m], 0)

    # ordered life in thirds: motion profile and contact profile.
    # grasp = free->contact->contact; release = contact->..->free;
    # push = brief contact mid. The ORDER is the signature.
    thirds = np.array_split(np.arange(n), 3)
    step = np.hypot(np.diff(cx), np.diff(cy))
    moving = step > 0.05 * ext
    mo = [float(moving[ix[ix < len(moving)]].mean())
          if len(ix) and (ix < len(moving)).any() else 0.0
          for ix in thirds]
    co = [float(con[ix].mean()) if len(ix) else 0.0 for ix in thirds]

    # distance to the nearest agent at the window's endpoints, in own
    # extents: held ends near zero, released/pushed ends far
    def d_agent(t_ref, x, y):
        bestd = 6.0
        for ag in agents:
            i = int(np.clip(np.searchsorted(ag["t"], t_ref), 0,
                            len(ag["t"]) - 1))
            if abs(int(ag["t"][i]) - int(t_ref)) > 5:
                continue
            bestd = min(bestd, float(np.hypot(ag["cx"][i] - x,
                                              ag["cy"][i] - y)) / ext)
        return min(bestd, 6.0) / 6.0
    da0 = d_agent(tt[0], cx[0], cy[0])
    da1 = d_agent(tt[-1], cx[-1], cy[-1])

    # site-anchored chain: a vanish site is colour-matched near the
    # start, an appear site colour-matched near the end
    has_v = has_a = 0.0
    for s in sites:
        rad = 3 * ext + max(s["w"], s["h"])
        if np.abs(s["pre_rgb"] - mrgb).sum() < 200 and \
                np.hypot(s["cx"] - cx[0], s["cy"] - cy[0]) < rad:
            has_v = 1.0
        if np.abs(s["post_rgb"] - mrgb).sum() < 200 and \
                np.hypot(s["cx"] - cx[-1], s["cy"] - cy[-1]) < rad:
            has_a = 1.0

    # other-entity adjacency at the endpoints: evidence registry from
    # every OTHER track's last known position and settled sites up to
    # the window end. Generic pairwise relation - departed from next
    # to something / arrived next to something.
    pts = []
    for r2 in ents:
        if r2["tid"] == r["tid"]:
            continue
        mm2 = r2["t"] <= b
        if mm2.any():
            i2 = int(np.where(mm2)[0][-1])
            pts.append((float(r2["cx"][i2]), float(r2["cy"][i2]),
                        np.median(np.asarray(r2["rgb"]), 0)))
    for s in sites_all:
        if s["t1"] <= b:
            pts.append((s["cx"], s["cy"], s["post_rgb"]))

    def near_other(x, y):
        bestd = 6.0
        for px, py, prgb in pts:
            if np.abs(prgb - mrgb).sum() < 120:
                continue        # same-looking = likely its own history
            bestd = min(bestd, float(np.hypot(px - x, py - y)) / ext)
        return 1.0 - min(bestd, 6.0) / 6.0      # closeness, not dist
    no0 = near_other(cx[0], cy[0])
    no1 = near_other(cx[-1], cy[-1])

    vec = np.array([
        1.0, min(best, 8.0) / 8.0,
        mo[0], mo[1], mo[2],
        co[0], co[1], co[2],
        da0, da1,
        has_v, has_a,
        no0, no1,
    ], np.float32)
    return dict(vec=vec, patient_rgb=mrgb, agent_rgb=agent_rgb)


# ---------------- benchmark on the sim ruler ----------------

def load_events():
    evs = []
    for corp in ("sim_chains", "sim_eval_bal"):
        for ep in sorted((ROOT / "data" / corp).glob("ep*")):
            f = ENT / f"{corp}_{ep.name}.npy"
            if not f.exists():
                continue
            rows = np.load(f, allow_pickle=True)
            views = sorted({r["view"] for r in rows})[:2]
            if len(views) < 2:
                continue
            meta = json.loads((ep / "meta.json").read_text())
            for e in meta["events"]:
                if not e["ok"]:
                    continue
                a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
                if b - a < 6:
                    continue
                ms = [moment(rows, v, a, b) for v in views]
                evs.append((e["prim"], f"{corp}_{ep.name}", ms))
    return evs


def bench():
    evs = load_events()
    n = len(evs)
    prims = np.array([e[0] for e in evs])
    eps_ = np.array([e[1] for e in evs])
    print(f"{n} events", flush=True)
    # collect per-view vectors; standardize struct dims corpus-wide
    # (store-side statistics, the CSLS legal class) so cosine weights
    # the graded dims comparably
    V = np.zeros((n, 2, NDIM), np.float32)
    Prgb = np.full((n, 2, 3), np.nan, np.float32)
    for i, (_, _, ms) in enumerate(evs):
        for v in range(2):
            if ms[v] is not None:
                V[i, v] = ms[v]["vec"]
                if ms[v]["patient_rgb"] is not None:
                    Prgb[i, v] = ms[v]["patient_rgb"]
    mu = V.reshape(-1, NDIM).mean(0)
    sd = V.reshape(-1, NDIM).std(0) + 1e-6
    Vz = (V - mu) / sd
    Vz = Vz / np.maximum(
        np.linalg.norm(Vz, axis=-1, keepdims=True), 1e-8)
    comp = {}
    S = None
    for a in range(2):
        for b in range(2):
            X = Vz[:, a] @ Vz[:, b].T
            S = X if S is None else np.maximum(S, X)
    comp["struct"] = S
    S = None
    for a in range(2):
        for b in range(2):
            D = np.abs(Prgb[:, None, a] - Prgb[None, :, b]).sum(-1)
            X = np.exp(-np.nan_to_num(D, nan=765.0) / 180.0)
            S = X if S is None else np.maximum(S, X)
    comp["patient"] = S.astype(np.float32)

    def csls(S):
        hub = np.sort(S, 1)[:, -50:].mean(1)
        return 2 * S - hub[None, :] - hub[:, None]

    def zrow(S):
        return (S - S.mean(1, keepdims=True)) / np.maximum(
            S.std(1, keepdims=True), 1e-8)

    def diffuse(S, k=10, alpha=0.9):
        N = len(S)
        A = S.copy(); np.fill_diagonal(A, -9e9)
        thr = np.partition(A, N - k, axis=1)[:, N - k][:, None]
        Wp = np.where(A >= thr, np.maximum(A, 0.0), 0.0)
        Wp = np.maximum(Wp, Wp.T)
        d = np.sqrt(np.maximum(Wp.sum(1), 1e-8))
        Wn = Wp / d[:, None] / d[None, :]
        Y = np.eye(N, dtype=np.float32); F = Y.copy()
        for _ in range(30):
            F = alpha * (Wn @ F) + (1 - alpha) * Y
        return F

    def metr(S, sub=None):
        idx = np.arange(n) if sub is None else np.where(sub)[0]
        Ss = S[np.ix_(idx, idx)]
        P, E = prims[idx], eps_[idx]
        M = np.array([MERGE[p] for p in P])
        out = {}
        for nc, labels in ((5, P), (3, M)):
            ap, p1, ys, ps = [], [], [], []
            for ii in range(len(idx)):
                mask = E != E[ii]
                lab = labels[mask] == labels[ii]
                sup = int(lab.sum())
                if not sup:
                    continue
                lo = lab[np.argsort(-Ss[ii][mask])]
                hit = np.where(lo)[0]
                ap.append(float(np.mean(
                    (np.arange(sup) + 1) / (hit + 1))))
                p1.append(float(lo[0]))
                tp = lo.cumsum()
                k = min(int(np.ceil(1.5 * sup)), len(lo))
                ys.append(float(tp[k - 1] / sup))
                ps.append(float(tp[k - 1] / k))
            out[nc] = (np.mean(ap), np.mean(p1), np.mean(ys),
                       np.mean(ps))
        return out

    def show(o, tag):
        a, b3 = o[5], o[3]
        print(f"{tag:26s} 5cls AP {a[0]:.3f} P@1 {a[1]:.3f} y/p "
              f"{a[2]:.3f}/{a[3]:.3f} | 3cls y/p {b3[2]:.3f}/{b3[3]:.3f}",
              flush=True)

    hold = np.char.startswith(eps_, "sim_eval_bal")
    Ss = csls(comp["struct"])
    Sp = csls(comp["patient"])
    show(metr(Ss), "struct alone")
    show(metr(diffuse(Ss), sub=None), "struct+diffusion")
    show(metr(Ss, sub=hold), "[HOLD] struct")
    show(metr(diffuse(Ss), sub=hold), "[HOLD] struct+diff")
    # fusion probe against the delta channel (the standing baseline)
    rv = Path(__file__).parent.parent / \
        ("/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
         "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
         "scratchpad/ruler_vecs.npz").lstrip("/")
    rv = Path("/private/tmp/claude-501/-Users-sudharshanramesh-Studies"
              "-MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5"
              "fb3/scratchpad/ruler_vecs.npz")
    if rv.exists():
        z = np.load(rv, allow_pickle=True)
        D0 = z["D0"].astype(np.float32)
        D1 = z["D1"].astype(np.float32)
        if len(D0) == n:
            Sd = None
            for a in (D0, D1):
                for b2 in (D0, D1):
                    X = a @ b2.T
                    Sd = X if Sd is None else np.maximum(Sd, X)
            Sd = csls(Sd)
            show(metr(diffuse(Sd), sub=hold), "[HOLD] delta+diff base")
            for w in (0.15, 0.3, 0.6):
                F = diffuse(zrow(Sd) + w * zrow(Ss))
                show(metr(F, sub=hold), f"[HOLD] delta+{w}struct")
            for w in (0.3,):
                F = diffuse(zrow(Sd) + w * zrow(Ss)
                            + 0.15 * zrow(Sp))
                show(metr(F, sub=hold), f"[HOLD] +{w}st+0.15pa")
    print("\nbaseline to beat [HOLDOUT @1352]: diffusion-current "
          "AP 0.546 y/p 0.643/0.428 | 3cls 0.688/0.459", flush=True)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
