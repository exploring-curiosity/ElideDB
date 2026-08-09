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


NDIM = 12


def moment(rows, view, a, b):
    """L5 v3: the moment from SITE PAIRS + the agent's path.

    Measured basis: under the piecewise scene a manipulated entity has
    NO separate moving track - at rest it is scene, in motion it is
    fused with the agent's blob. Its identity is carried by the
    vanish/appear sites (core-colour 1.000 correct) and its motion by
    the agent's path between them (agent coverage 0.985). So the
    moment is built from the two near-perfect layers and needs no
    patient track at all.

    Independent entities (PROBLEM.md L2): a site with NO agent visit,
    or a self-moving track, is something that moved on its own -
    represented, not filtered.

    Facts (all relational, extent/size-normalized, graded):
      vanish-side: exists, agent-was-there
      appear-side: exists, agent-was-there
      pair: colour-linked vanish->appear, displacement in site units,
            time between them relative to window
      agent dwell near the involved sites
      appear-site adjacency to OTHER settled evidence (arrived next
      to something vs onto empty ground)
      independent motion present (site sans agent, or self-track)
    """
    vr = [r for r in rows if r["view"] == view]
    agents = [r for r in vr if r["role"] == "agent"]
    selfs = [r for r in vr if r["role"] == "entity"
             and r.get("selfmove", False)]
    sites_all = [s for s in vr if s["role"] == "site"]
    ws = [s for s in sites_all
          if s["t0"] <= b + 15 and s["t1"] >= a - 15]

    agent_rgb = None
    if agents:
        biggest = max(agents, key=lambda r: r["extent"])
        agent_rgb = np.median(np.asarray(biggest["rgb"]), 0)

    if not ws and not selfs:
        return None

    def agent_at(t_ref, x, y, size):
        """Graded: was any agent at (x,y) around t_ref?"""
        best = 0.0
        for ag in agents:
            i = int(np.clip(np.searchsorted(ag["t"], t_ref), 0,
                            len(ag["t"]) - 1))
            for jj in (max(0, i - 3), i, min(len(ag["t"]) - 1, i + 3)):
                d = np.hypot(ag["cx"][jj] - x, ag["cy"][jj] - y)
                rad = size + max(ag["w"][jj], ag["h"][jj])
                best = max(best, float(np.exp(-d / max(rad, 8.0))))
        return best

    # pair sites by core colour: a vanish (pre-side thing) matched to
    # an appear (post-side thing) of the same look, vanish first
    best_pair, pair = None, None
    for sv in ws:
        for sa in ws:
            if sa is sv or sa["t0"] < sv["t0"]:
                continue
            dcol = np.abs(sv["pre_rgb"] - sa["post_rgb"]).sum()
            if dcol > 200:
                continue
            score = np.exp(-dcol / 120.0)
            if best_pair is None or score > best_pair:
                best_pair, pair = score, (sv, sa)

    # the involved sites: the pair if found, else the strongest
    # single site (largest core change region)
    if pair is not None:
        sv, sa = pair
        prgb = 0.5 * (sv["pre_rgb"] + sa["post_rgb"])
    elif ws:
        sv = max(ws, key=lambda s: s["w"] * s["h"])
        sa = None
        prgb = sv["pre_rgb"]
    else:
        sv = sa = None
        prgb = None

    size_v = max(sv["w"], sv["h"]) if sv is not None else 1.0
    ag_v = agent_at(0.5 * (sv["t0"] + sv["t1"]), sv["cx"], sv["cy"],
                    size_v) if sv is not None else 0.0
    ag_a = 0.0
    pair_disp = dt_pair = 0.0
    if sa is not None:
        size_a = max(sa["w"], sa["h"])
        ag_a = agent_at(0.5 * (sa["t0"] + sa["t1"]), sa["cx"],
                        sa["cy"], size_a)
        unit = 0.5 * (size_v + size_a)
        pair_disp = min(float(np.hypot(sa["cx"] - sv["cx"],
                                       sa["cy"] - sv["cy"])) / unit,
                        10.0) / 10.0
        dt_pair = min(max(sa["t0"] - sv["t1"], 0)
                      / max(b - a, 1), 1.5) / 1.5

    # appear-site adjacency to OTHER settled evidence (different look):
    # arrived next to something, vs onto empty ground
    near_other = 0.0
    ref = sa if sa is not None else None
    if ref is not None:
        for s2 in sites_all:
            if s2 is ref or s2["t1"] > b + 15:
                continue
            if np.abs(s2["post_rgb"] - prgb).sum() < 120:
                continue
            unit = max(ref["w"], ref["h"])
            d = np.hypot(s2["cx"] - ref["cx"], s2["cy"] - ref["cy"])
            near_other = max(near_other,
                             float(np.exp(-d / max(1.5 * unit, 8.0))))

    # independent motion: a window site with NO agent visit, or a
    # self-moving track alive in the window
    indep = 0.0
    for s in ws:
        av = agent_at(0.5 * (s["t0"] + s["t1"]), s["cx"], s["cy"],
                      max(s["w"], s["h"]))
        indep = max(indep, 1.0 - av)
    self_alive = any(((r["t"] >= a) & (r["t"] <= b)).sum() >= 3
                     for r in selfs)

    vec = np.array([
        float(sv is not None),
        float(sa is not None),
        float(pair is not None),
        pair_disp, dt_pair,
        ag_v, ag_a,
        max(ag_v, ag_a),
        near_other,
        indep,
        float(self_alive),
        min(len(ws), 4) / 4.0,
    ], np.float32)
    return dict(vec=vec, patient_rgb=prgb, agent_rgb=agent_rgb)


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
