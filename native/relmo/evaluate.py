"""Held-out evaluation — the robot-arm corpus, SEALED.

Owner rule (2026-08-11): the arm domain is used EXCLUSIVELY for
evaluation. It never appears in training, its state is never read by
the model, and its class labels are used only to GRADE the output
after the fact - never inside the pipeline, never in a prompt, never
in selection.

The model sees exactly what it will see in production: point tracks
from video, nothing else. It produces groups and contacts; the
relational comparison consumes those; retrieval is scored.

Reported every cycle:
  groups/moment   did learned grouping fix the measured collapse
                  (hand-built common fate gave ONE group on 151/324)
  yield@support   the standing product metric (symbolic best 0.510)
  prec@1.5sup     precision of what is returned
  cross-embodiment  query one arm against a store of the others
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.device import DEVICE  # noqa: E402
from relmo.model import NT, RelMo  # noqa: E402

DEV = DEVICE
ARM_TRACKS = R.ROOT / "data" / "cache" / "ptrack"
ARM_CORPUS = R.ROOT / "data" / "prim_actions_v2"
PRIMS = ("pick", "place", "stack", "unstack", "push")


def load_model(run_id, step=None):
    out = R.MODELS / run_id
    cks = sorted(out.glob("ckpt_*.pt"))
    if not cks:
        return None, 0
    f = cks[-1] if step is None else out / f"ckpt_{step:07d}.pt"
    sd = torch.load(f, map_location=DEV)
    net = RelMo().to(DEV).eval()
    net.load_state_dict(sd["model"])
    return net, sd["step"]


@torch.no_grad()
def moment(net, xy, vis, a, b, thr=0.5):
    """Learned grouping + contact for one window. Deployment path:
    tracks in, structure out, no privileged anything."""
    T = len(xy)
    a, b = max(int(a), 0), min(int(b), T - 1)
    if b - a < 6:
        return None
    W, V = xy[a:b + 1], vis[a:b + 1]
    rng = np.linalg.norm(W.max(0) - W.min(0), axis=-1)
    ok = V.mean(0) > 0.5
    floor = max(float(np.percentile(rng, 60)) * 3.0, 4.0)
    idx = np.where(ok & (rng > floor))[0]
    if len(idx) < 6:
        return None
    if len(idx) > 320:
        idx = idx[np.argsort(-rng[idx])[:320]]
    x = torch.tensor(W[:, idx][None], dtype=torch.float32, device=DEV)
    v = torch.tensor(V[:, idx][None], device=DEV)
    tok = net.tokens(x, v)
    z = net.point_emb(tok)[0]
    S = (z @ z.T).cpu().numpy()
    # connected components on the learned same-thing affinity
    n = len(idx)
    lab = -np.ones(n, int)
    c = 0
    A = S > thr
    for i in range(n):
        if lab[i] >= 0:
            continue
        st, lab[i] = [i], c
        while st:
            u = st.pop()
            for w in np.where(A[u] & (lab < 0))[0]:
                lab[w] = c
                st.append(w)
        c += 1
    groups = [np.where(lab == k)[0] for k in range(c)]
    groups = [g for g in groups if len(g) >= 3]
    groups.sort(key=lambda g: -len(g))
    groups = groups[:4]
    if not groups:
        return None
    assign = torch.zeros(n, len(groups), device=DEV)
    for gi, g in enumerate(groups):
        assign[torch.tensor(g, device=DEV), gi] = 1.0
    gtok = net.group_pool(tok, assign[None])
    # per-group motion shape (normalized, axis-free) + pair contact
    C = W[:, idx]
    unit = max(float(np.linalg.norm(C.reshape(-1, 2).std(0))), 1.0)
    nodes, cent = [], []
    for g in groups:
        cc = C[:, g].mean(1)
        step = np.linalg.norm(np.diff(cc, axis=0), axis=-1)
        net_d = float(np.linalg.norm(cc[-1] - cc[0])) / unit
        sp = np.interp(np.linspace(0, 1, NT),
                       np.linspace(0, 1, len(step)), step) / unit
        nodes.append(dict(sp=sp.astype(np.float32),
                          net=min(net_d, 4.0) / 4.0,
                          frac=len(g) / n))
        cent.append(cc)
    edges = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            lgt = net.contact(gtok, torch.tensor([i], device=DEV),
                              torch.tensor([j], device=DEV))[0]
            d = np.linalg.norm(cent[i] - cent[j], axis=-1) / unit
            ds = np.interp(np.linspace(0, 1, NT),
                           np.linspace(0, 1, len(d)), d)
            edges[(i, j)] = (torch.sigmoid(lgt).cpu().numpy(),
                             np.minimum(ds, 4.0) / 4.0)
    emb = net.embed(tok)[0].cpu().numpy()
    return dict(nodes=nodes, edges=edges, n=len(groups), emb=emb)


def match(A, B):
    """Best correspondence between two moments' group sets.

    Hungarian assignment on node similarity, then edge agreement
    under that assignment - O(G^3) instead of the O(G!) permutation
    search, which dominated eval wall-clock and made the in-loop gate
    unusable."""
    from scipy.optimize import linear_sum_assignment
    if A is None or B is None:
        return 0.0
    sm, bg = (A, B) if A["n"] <= B["n"] else (B, A)
    ns, nb = sm["n"], bg["n"]
    C = np.zeros((ns, nb), np.float32)
    for i in range(ns):
        x = sm["nodes"][i]
        for j in range(nb):
            y = bg["nodes"][j]
            s = 1.0 - 0.5 * float(np.abs(x["sp"] - y["sp"]).mean())
            s *= 1.0 - 0.6 * abs(x["net"] - y["net"])
            s *= 1.0 - 0.3 * abs(x["frac"] - y["frac"])
            C[i, j] = max(s, 0.0)
    ri, ci = linear_sum_assignment(-C)
    node = float(C[ri, ci].mean())
    perm = {int(i): int(j) for i, j in zip(ri, ci)}
    es = []
    for i in range(ns):
        for j in range(i + 1, ns):
            e1 = sm["edges"].get((i, j))
            p_, q_ = perm.get(i), perm.get(j)
            if e1 is None or p_ is None or q_ is None:
                continue
            e2 = bg["edges"].get((min(p_, q_), max(p_, q_)))
            if e2 is None:
                continue
            cc = 1.0 - float(np.abs(e1[0] - e2[0]).mean())
            dd = 1.0 - float(np.abs(e1[1] - e2[1]).mean())
            es.append(0.6 * cc + 0.4 * dd)
    sc = node if not es else (node ** 0.5) * (float(np.mean(es)) ** 0.5)
    return sc * (ns / max(nb, 1)) ** 0.25


def evaluate(run_id, step=None, tag="", max_events=None):
    net, step = load_model(run_id, step)
    if net is None:
        return None
    evs = []
    for ep in sorted(ARM_CORPUS.glob("ep*")):
        f = ARM_TRACKS / f"prim_actions_v2_{ep.name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        xy, vis = z["xy"].astype(np.float32), z["vis"]
        meta = json.loads((ep / "meta.json").read_text())
        for e in meta["events"]:
            if not e["ok"]:
                continue
            a, b = int(e["t0"] * 10) - 2, int(e["t1"] * 10) + 4
            m = moment(net, xy, vis, a, b)
            evs.append((e["prim"], ep.name, meta.get("arm", "?"), m))
    if max_events and len(evs) > max_events:
        sel = np.random.default_rng(0).choice(
            len(evs), max_events, replace=False)
        evs = [evs[i] for i in sorted(sel)]
    n = len(evs)
    ok = [e for e in evs if e[3] is not None]
    gcount = defaultdict(int)
    for e in ok:
        gcount[e[3]["n"]] += 1
    prim = np.array([e[0] for e in evs])
    ep_ = np.array([e[1] for e in evs])
    arm = np.array([e[2] for e in evs])
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = match(evs[i][3], evs[j][3])

    def run(mask_q=None):
        y1, p15 = [], []
        for i in range(n):
            if evs[i][3] is None:
                continue
            if mask_q is not None and not mask_q[i]:
                continue
            m = ep_ != ep_[i]
            if mask_q is not None:
                m = m & ~mask_q
            sup = int((prim[m] == prim[i]).sum())
            if not sup:
                continue
            o = np.argsort(-S[i][m])
            lab = (prim[m] == prim[i])
            y1.append(lab[o[:sup]].sum() / sup)
            k = min(int(1.5 * sup), int(m.sum()))
            p15.append(lab[o[:k]].sum() / k)
        return (float(np.mean(y1)) if y1 else 0.0,
                float(np.mean(p15)) if p15 else 0.0)

    y, p = run()
    xres = {}
    for a_ in sorted(set(arm)):
        xres[a_] = run(arm == a_)[0]
    rec = dict(run=run_id, step=step, tag=tag, n_events=n,
               empty=n - len(ok),
               groups_hist={str(k): v for k, v in sorted(gcount.items())},
               single_group_frac=round(gcount[1] / max(len(ok), 1), 3),
               yield_at_sup=round(y, 4), prec_at_1p5=round(p, 4),
               cross_embodiment={k: round(v, 4) for k, v in xres.items()},
               baseline_symbolic=0.510)
    R.log("eval", **rec)
    (R.MODELS / run_id / "evals.jsonl").open("a").write(
        json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.run, a.step), indent=1))
