"""L5 + L7 done as PROBLEM.md actually specifies: a moment is a GRAPH
of entities and their relations, and two moments are compared by the
best ROLE CORRESPONDENCE between their entity sets - not by cosine
over a pooled descriptor.

Why this file exists (measured, 2026-08-10, BENCHMARKS.md):
  - The pooled 12-fact moment (moments.py v3) reaches 5cls AP 0.315,
    and the ORACLE test - restricting to moments that describe the
    RIGHT entity - lifts it only to 0.366, far under the incumbent
    delta channel's 0.546. So SELECTION was never the bound; the
    representation was.
  - PROBLEM.md predicts exactly this: "cosine over any pooled
    descriptor computes an OVERLAP OF DESCRIPTIONS. Overlap cannot
    express 'there exists a role assignment'."
  - The two structural gaps the pooled version could not close:
    (1) the primitives differ by a RELATION TO ANOTHER ENTITY
        (arrived on open ground vs onto a particular thing), which a
        scalar distance-to-anything cannot carry;
    (2) an entity that never moves had no representation at all -
        yet it is precisely the other end of that relation.
    ent.scene_objects (spatial coincidence within a settled snapshot)
    closes (2); this module closes (1).

Nothing here names a task, an object kind, or an axis. Nodes are
whatever the recording's own statistics produced; edges are the
pairwise reference frame PROBLEM.md #5 licenses (relative distance in
units of the entities' own extents); matching is positional.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENT = ROOT / "data" / "cache" / "ent"
FPS = 10.0
PRIMS = ("pick", "place", "stack", "unstack", "push")
MERGE = {"pick": "grasp", "unstack": "grasp",
         "place": "release", "stack": "release", "push": "push"}

NODE_F = 6      # per-entity facts
EDGE_F = 3      # per-pair facts


def _cfg(objects, t):
    """The scene's entity configuration at snapshot time t: every
    persistent object present, with its state THERE (raw, private
    frame - relations are formed later, never stored absolute)."""
    out = []
    for o in objects:
        ts = o["t"]
        i = int(np.argmin(np.abs(ts - t)))
        if abs(int(ts[i]) - int(t)) > ent.OBJ_CADENCE:
            continue
        out.append(dict(cx=float(o["cx"][i]), cy=float(o["cy"][i]),
                        w=float(o["w"][i]), h=float(o["h"][i]),
                        rgb=o["rgb"][i]))
    return out


def _match(A, B):
    """Correspondence between two configurations by appearance and
    size - identity by continuity where it exists, resemblance only
    to CONFIRM. Greedy over a small set (3-6 entities); returns pairs
    (ia, ib) plus the unmatched indices on each side."""
    cost = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            dcol = float(np.abs(a["rgb"] - b["rgb"]).sum())
            sa = max(a["w"], a["h"])
            sb = max(b["w"], b["h"])
            dsz = abs(sa - sb) / max(sa, sb, 1.0)
            if dcol > 150 or dsz > 0.6:
                continue
            cost.append((dcol + 200 * dsz, i, j))
    cost.sort()
    ua, ub, pairs = set(range(len(A))), set(range(len(B))), []
    for _, i, j in cost:
        if i in ua and j in ub:
            pairs.append((i, j))
            ua.discard(i)
            ub.discard(j)
    return pairs, sorted(ua), sorted(ub)


def _near(cfg, k):
    """Distance from entity k to its nearest OTHER entity, in units
    of the two entities' own extents (PROBLEM.md #5: scale
    self-calibrates from the things themselves)."""
    best = 9.9
    for j, o in enumerate(cfg):
        if j == k:
            continue
        unit = 0.5 * (max(cfg[k]["w"], cfg[k]["h"])
                      + max(o["w"], o["h"]))
        d = np.hypot(cfg[k]["cx"] - o["cx"], cfg[k]["cy"] - o["cy"])
        best = min(best, float(d) / max(unit, 1.0))
    return best


def moment(rows, view, a, b):
    """The window's sub-graph: nodes = entities whose state changed
    across the window's own settled scenes, each carrying its
    before/after relational facts; edges = pairwise relations among
    them. Returns None when the window has no settled pair to compare.
    """
    vr = [r for r in rows if r["view"] == view]
    objects = [r for r in vr if r["role"] == "object"]
    agents = [r for r in vr if r["role"] == "agent"]
    if not objects:
        return None
    snaps = sorted({int(t) for o in objects for t in o["t"]})
    pre = [t for t in snaps if t <= a + 4]
    post = [t for t in snaps if t >= b - 4]
    if not pre or not post:
        return None
    t0, t1 = pre[-1], post[0]
    if t1 <= t0:
        return None
    A, B = _cfg(objects, t0), _cfg(objects, t1)
    if not A and not B:
        return None
    pairs, only_a, only_b = _match(A, B)

    def agent_at(t_ref, x, y, size):
        best = 0.0
        for ag in agents:
            i = int(np.clip(np.searchsorted(ag["t"], t_ref), 0,
                            len(ag["t"]) - 1))
            for jj in (max(0, i - 3), i, min(len(ag["t"]) - 1, i + 3)):
                d = np.hypot(ag["cx"][jj] - x, ag["cy"][jj] - y)
                rad = size + max(ag["w"][jj], ag["h"][jj])
                best = max(best, float(np.exp(-d / max(rad, 8.0))))
        return best

    nodes = []
    for i, j in pairs:
        oa, ob = A[i], B[j]
        unit = max(oa["w"], oa["h"], ob["w"], ob["h"], 1.0)
        disp = float(np.hypot(ob["cx"] - oa["cx"],
                              ob["cy"] - oa["cy"])) / unit
        na, nb = _near(A, i), _near(B, j)
        nodes.append(dict(
            rgb=0.5 * (oa["rgb"] + ob["rgb"]),
            cx=ob["cx"], cy=ob["cy"], size=unit,
            moved=min(disp, 4.0) / 4.0,
            # persisted through the window (vs appeared / vanished)
            kind=0.0,
            near_pre=min(na, 4.0) / 4.0,
            near_post=min(nb, 4.0) / 4.0,
            # approach (<0) or separation (>0), self-calibrated
            drel=float(np.clip((nb - na) / 4.0, -1, 1)),
            agent=max(agent_at(t0, oa["cx"], oa["cy"], unit),
                      agent_at(t1, ob["cx"], ob["cy"], unit))))
    for i in only_a:                    # vanished from the settled world
        o = A[i]
        unit = max(o["w"], o["h"], 1.0)
        nodes.append(dict(
            rgb=o["rgb"], cx=o["cx"], cy=o["cy"], size=unit,
            moved=1.0, kind=-1.0,
            near_pre=min(_near(A, i), 4.0) / 4.0, near_post=1.0,
            drel=1.0,
            agent=agent_at(t0, o["cx"], o["cy"], unit)))
    for j in only_b:                    # appeared in the settled world
        o = B[j]
        unit = max(o["w"], o["h"], 1.0)
        nodes.append(dict(
            rgb=o["rgb"], cx=o["cx"], cy=o["cy"], size=unit,
            moved=1.0, kind=1.0,
            near_pre=1.0, near_post=min(_near(B, j), 4.0) / 4.0,
            drel=-1.0,
            agent=agent_at(t1, o["cx"], o["cy"], unit)))
    if not nodes:
        return None
    # ACTIVE nodes carry the moment; a node that neither moved nor
    # changed relation is scene furniture for this window (it still
    # serves as the OTHER end of relations, which is its whole job)
    for nd in nodes:
        nd["active"] = float(nd["moved"] > 0.15 or nd["kind"] != 0.0)
    return dict(nodes=nodes, t0=t0, t1=t1)


def _nodevec(nd):
    return np.array([nd["moved"], nd["kind"], nd["near_pre"],
                     nd["near_post"], nd["drel"], nd["agent"]],
                    np.float32)


def _edge(p, q):
    """Relation between two entities of the SAME moment: distance in
    their joint extent, and relative size. Absolute coordinates never
    escape this function (PROBLEM.md #5)."""
    unit = 0.5 * (p["size"] + q["size"])
    d = np.hypot(p["cx"] - q["cx"], p["cy"] - q["cy"]) / max(unit, 1.0)
    rs = p["size"] / max(q["size"], 1.0)
    return np.array([min(d, 4.0) / 4.0,
                     min(rs, 3.0) / 3.0,
                     float(p["agent"] > 0.5 and q["agent"] > 0.5)],
                    np.float32)


def sim(M1, M2, w_edge=1.0):
    """L7: the score of the BEST partial role correspondence between
    two moment graphs. Roles are positional - a node's identity in
    the match is what it DID and how it related, never a name and
    never its appearance (appearance is a separate component, so an
    intent can weight it or ignore it)."""
    if M1 is None or M2 is None:
        return 0.0
    N1 = [n for n in M1["nodes"] if n["active"]]
    N2 = [n for n in M2["nodes"] if n["active"]]
    if not N1 or not N2:
        return 0.0
    N1 = sorted(N1, key=lambda n: -n["moved"])[:3]
    N2 = sorted(N2, key=lambda n: -n["moved"])[:3]
    V1 = [_nodevec(n) for n in N1]
    V2 = [_nodevec(n) for n in N2]
    node = np.zeros((len(N1), len(N2)), np.float32)
    for i in range(len(N1)):
        for j in range(len(N2)):
            node[i, j] = 1.0 - float(
                np.abs(V1[i] - V2[j]).mean())

    best = -9e9
    # small sets: enumerate injective assignments exactly
    def assign(i, used, acc, chosen):
        nonlocal best
        if i == len(N1):
            sc = acc / max(len(chosen), 1)
            if chosen and w_edge > 0:
                es, ne = 0.0, 0
                for x in range(len(chosen)):
                    for y in range(x + 1, len(chosen)):
                        e1 = _edge(N1[chosen[x][0]], N1[chosen[y][0]])
                        e2 = _edge(N2[chosen[x][1]], N2[chosen[y][1]])
                        es += 1.0 - float(np.abs(e1 - e2).mean())
                        ne += 1
                if ne:
                    sc = (sc + w_edge * es / ne) / (1.0 + w_edge)
            best = max(best, sc)
            return
        for j in range(len(N2)):
            if j in used:
                continue
            assign(i + 1, used | {j}, acc + node[i, j],
                   chosen + [(i, j)])
        assign(i + 1, used, acc, chosen)   # leave i unmatched (partial)

    assign(0, set(), 0.0, [])
    return float(max(best, 0.0))


def load(corpora=("sim_chains", "sim_eval_bal")):
    evs = []
    for corp in corpora:
        for ep in sorted((ROOT / "data" / corp).glob("ep*")):
            f = ENT / f"{corp}_{ep.name}.npy"
            if not f.exists():
                continue
            rows = np.load(f, allow_pickle=True)
            if not len(rows) or rows[0].get("role") != "_ver":
                continue
            views = ent.views_of(rows)[:2]
            if len(views) < 2:
                continue
            meta = json.loads((ep / "meta.json").read_text())
            rgba = {b["name"]: 255 * np.array(b["rgba"][:3])
                    for b in meta["blocks"]}
            for e in meta["events"]:
                if not e["ok"]:
                    continue
                a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
                if b - a < 6:
                    continue
                ms = [moment(rows, v, a, b) for v in views]
                evs.append((e["prim"], f"{corp}_{ep.name}", ms,
                            rgba[e["block"]]))
    return evs


def bench(corpora=("sim_chains", "sim_eval_bal"), w_edge=1.0):
    from tqdm import tqdm
    evs = load(corpora)
    n = len(evs)
    prims = np.array([e[0] for e in evs])
    eps_ = np.array([e[1] for e in evs])
    print(f"{n} events", flush=True)

    # binding diagnostic (EVAL-ONLY): does the most-active node of
    # the graph describe the event's true entity?
    ok = []
    for _, _, ms, tgt in evs:
        hit = False
        for M in ms:
            if M is None:
                continue
            act = [nd for nd in M["nodes"] if nd["active"]]
            if not act:
                continue
            top = max(act, key=lambda nd: nd["moved"])
            hit = hit or float(np.abs(top["rgb"] - tgt).sum()) < 180
        ok.append(float(hit))
    print(f"binding (top-active node = true entity): {np.mean(ok):.3f}",
          flush=True)

    S = np.zeros((n, n), np.float32)
    for i in tqdm(range(n), unit="q", desc="corr"):
        for j in range(i, n):
            s = 0.0
            for x in range(2):
                for y in range(2):
                    s = max(s, sim(evs[i][2][x], evs[j][2][y], w_edge))
            S[i, j] = S[j, i] = s

    def csls(S):
        hub = np.sort(S, 1)[:, -50:].mean(1)
        return 2 * S - hub[None, :] - hub[:, None]

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
        a5, b3 = o[5], o[3]
        print(f"{tag:28s} 5cls AP {a5[0]:.3f} P@1 {a5[1]:.3f} y/p "
              f"{a5[2]:.3f}/{a5[3]:.3f} | 3cls y/p "
              f"{b3[2]:.3f}/{b3[3]:.3f}", flush=True)

    hold = np.char.startswith(eps_, "sim_eval_bal")
    show(metr(S), "corr alone")
    show(metr(csls(S)), "corr+csls")
    if hold.any():
        show(metr(S, hold), "[HOLD] corr")
        show(metr(csls(S), hold), "[HOLD] corr+csls")
    np.save(ROOT / "data" / "cache" / "corr_S.npy", S)
    print("\nincumbent [HOLD]: AP 0.546 y/p 0.643/0.428 | 3cls "
          "0.688/0.459   (pooled L5: AP 0.282)", flush=True)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        corp = (sys.argv[i + 1].split(",")
                if len(sys.argv) > i + 1
                and not sys.argv[i + 1].startswith("-")
                else ["sim_chains", "sim_eval_bal"])
        bench(tuple(corp))
