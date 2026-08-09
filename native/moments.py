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


def moment(rows, view, a, b):
    """L5: the episode structure of window [a,b) in one view.

    Returns None if nothing happened, else a dict of RELATIONAL facts:
      sites:   list of (kind, rgb) - kind in {vanish, appear, change}
               vanish = a settled thing left this spot,
               appear = a settled thing arrived,
               decided by which side matches the moving evidence
      mover:   dict(rgb, disp_ext, contact_frac, post_free_corr)
               disp_ext = displacement / own extent (scale-free)
               contact_frac = fraction of its moving life in contact
               post_free_corr = did it keep moving with the agent
               after last contact (held) or come to rest (released)
    """
    vr = [r for r in rows if r["view"] == view]
    ents = [r for r in vr if r["role"] == "entity"]
    agents = [r for r in vr if r["role"] == "agent"]
    sites = [s for s in vr if s["role"] == "site"
             and s["t0"] <= b + 10 and s["t1"] >= a - 10]

    # the window's mover: the episodic track with the largest
    # extent-normalized displacement inside the window
    best, mv = None, None
    for r in ents:
        m = (r["t"] >= a) & (r["t"] <= b)
        if m.sum() < 3:
            continue
        ext = max(r["extent"], 4.0)
        d = np.hypot(r["cx"][m].max() - r["cx"][m].min(),
                     r["cy"][m].max() - r["cy"][m].min()) / ext
        if best is None or d > best:
            best, mv = d, (r, m)
    out = dict(sites=[], mover=None, agent_rgb=None)
    if agents:
        biggest = max(agents, key=lambda r: r["extent"])
        out["agent_rgb"] = np.median(np.asarray(biggest["rgb"]), 0)
    for s in sites:
        # which side of the change holds the thing: the side whose
        # colour differs more from the OTHER side's surroundings is
        # ambiguous without the scene patch; V1 keeps both colours and
        # the kind is decided by the mover linkage below
        out["sites"].append(dict(pre=s["pre_rgb"], post=s["post_rgb"],
                                 cx=s["cx"], cy=s["cy"],
                                 w=s["w"], h=s["h"]))
    if mv is not None and best is not None and best > 0.35:
        r, m = mv
        contact = float(r["contact"][m].mean())
        # after its last in-window contact, did it keep moving (held)
        # or rest (released)? relational: own motion, own extent
        tt = r["t"][m]
        cx, cy = r["cx"][m], r["cy"][m]
        ci = np.where(r["contact"][m])[0]
        post_free = 0.0
        if len(ci) and ci[-1] < len(tt) - 3:
            j = ci[-1]
            ext = max(r["extent"], 4.0)
            post_free = float(np.hypot(cx[-1] - cx[j], cy[-1] - cy[j])
                              / ext)
        out["mover"] = dict(
            rgb=np.median(np.asarray(r["rgb"])[m], 0),
            disp_ext=float(best),
            contact_frac=contact,
            post_free=post_free,
            n_moving=int(m.sum()))
        # classify each site by colour agreement with the mover:
        # vanish if its PRE matches the mover, appear if its POST does
        for s in out["sites"]:
            dp = np.abs(s["pre"] - out["mover"]["rgb"]).sum()
            dq = np.abs(s["post"] - out["mover"]["rgb"]).sum()
            s["kind"] = ("vanish" if dp < dq else "appear") \
                if min(dp, dq) < 200 else "other"
    else:
        for s in out["sites"]:
            s["kind"] = "other"
    if not out["sites"] and out["mover"] is None:
        return None
    return out


def sim_moment(ma, mb):
    """L7: correspondence score between two single-view moments,
    per component. Structure never sees appearance; appearance never
    sees position; nothing sees absolute coordinates."""
    if ma is None or mb is None:
        return dict(struct=0.0, patient=0.0, agent=0.0)
    # STRUCTURE: compare the event pattern
    sa = _struct_vec(ma)
    sb = _struct_vec(mb)
    struct = float(_l2(sa) @ _l2(sb))
    # PATIENT appearance: mover colour (or best site colour)
    pa = ma["mover"]["rgb"] if ma["mover"] is not None else None
    pb = mb["mover"]["rgb"] if mb["mover"] is not None else None
    patient = 0.0
    if pa is not None and pb is not None:
        patient = float(np.exp(-np.abs(pa - pb).sum() / 180.0))
    agent = 0.0
    if ma["agent_rgb"] is not None and mb["agent_rgb"] is not None:
        agent = float(np.exp(
            -np.abs(ma["agent_rgb"] - mb["agent_rgb"]).sum() / 180.0))
    return dict(struct=struct, patient=patient, agent=agent)


def _struct_vec(m):
    """The moment's event pattern as pure relational facts."""
    nv = sum(1 for s in m["sites"] if s.get("kind") == "vanish")
    na = sum(1 for s in m["sites"] if s.get("kind") == "appear")
    if m["mover"] is not None:
        mv = m["mover"]
        return np.array([
            1.0,                            # something moved
            min(mv["disp_ext"], 8.0) / 8.0,  # how far, in own units
            mv["contact_frac"],             # carried vs free
            min(mv["post_free"], 4.0) / 4.0,  # kept moving after release?
            float(nv > 0), float(na > 0),   # settled world lost/gained
            min(nv, 3) / 3.0, min(na, 3) / 3.0,
        ], np.float32)
    return np.array([0.0, 0, 0, 0, float(nv > 0), float(na > 0),
                     min(nv, 3) / 3.0, min(na, 3) / 3.0], np.float32)


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
    comp = {k: np.zeros((n, n), np.float32)
            for k in ("struct", "patient", "agent")}
    from tqdm import tqdm
    for i in tqdm(range(n), unit="q", desc="pairs"):
        for j in range(i, n):
            best = dict(struct=0.0, patient=0.0, agent=0.0)
            for va in range(2):
                for vb in range(2):
                    s = sim_moment(evs[i][2][va], evs[j][2][vb])
                    if s["struct"] + s["patient"] > \
                            best["struct"] + best["patient"]:
                        best = s
            for k in comp:
                comp[k][i, j] = comp[k][j, i] = best[k]

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
    show(metr(Sp), "patient alone")
    fuse = zrow(Ss) + 0.3 * zrow(Sp)
    show(metr(fuse), "struct+0.3patient")
    D = diffuse(fuse)
    show(metr(D), "fused+diffusion")
    show(metr(Ss, sub=hold), "[HOLDOUT] struct")
    show(metr(D, sub=hold), "[HOLDOUT] fused+diff")
    print("\nbaseline to beat [HOLDOUT @1352]: diffusion-current "
          "AP 0.546 y/p 0.643/0.428 | 3cls 0.688/0.459", flush=True)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
