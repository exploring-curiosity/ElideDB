"""Gates for the world-model core. Ground truth is EVAL-ONLY here -
meta.json primitives/boundaries grade the latent, they never touch it.

Every number is printed NEXT TO the frozen-feature baseline that the
predictor must beat, because a number without a denominator is noise:

  probe      frame-state -> active primitive / carrying, linear probe.
             Baseline: same probe on raw frozen latents.
  surprise   prediction error vs event boundaries (AUC of near-boundary
             frames). Baseline: frozen-latent temporal derivative.
  qbe        event-span retrieval on the eval corpus: same-primitive
             P@10, mean-state cosine vs frozen-mean cosine.
  cross      A->B: query xarm7/vx300s train events against the panda
             eval corpus - the cross-embodiment memory claim.

    python native/vwm_gates.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
CACHE = ROOT / "data" / "cache" / "vwm"
FPS = 10
PRIMS = ("pick", "place", "stack", "unstack", "push")


def episodes(root):
    out = []
    for ep in sorted((ROOT / root).glob("ep*")):
        npz = CACHE / f"{'_'.join(ep.resolve().relative_to(ROOT / 'data').parts)}.npz"
        if npz.exists() and (ep / "meta.json").exists():
            out.append((ep, npz))
    return out


def _l2(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8)


def frame_labels(meta, T):
    """Per-frame primitive id (or -1) and carrying flag, from events."""
    lab = np.full(T, -1)
    carry = np.zeros(T, bool)
    for e in meta["events"]:
        a, b = int(e["t0"] * FPS), min(T, int(e["t1"] * FPS))
        lab[a:b] = PRIMS.index(e["prim"])
        if e["prim"] == "pick" and e["ok"]:
            carry[b:] = True          # held until the next release event
        elif e["prim"] in ("place", "stack", "unstack"):
            carry[min(T - 1, b):] = False
    return lab, carry


def probe_scores(reps, labs, seed=0):
    """Multinomial logistic probe, 5-fold by episode."""
    from sklearn.linear_model import LogisticRegression
    rs = np.random.RandomState(seed)
    order = rs.permutation(len(reps))
    folds = np.array_split(order, 5)
    accs = []
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        Xtr = np.concatenate([reps[i] for i in tr])
        ytr = np.concatenate([labs[i] for i in tr])
        Xte = np.concatenate([reps[i] for i in te])
        yte = np.concatenate([labs[i] for i in te])
        m = (ytr >= 0)
        mt = (yte >= 0)
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(Xtr[m], ytr[m])
        accs.append(float(clf.score(Xte[mt], yte[mt])))
    return float(np.mean(accs))


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    pos = np.asarray(pos); neg = np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def event_table(root, states, frozen):
    """(prim, ep, mean-state, mean-frozen) per verified event span."""
    rows = []
    for (ep, npz) in episodes(root):
        meta = json.loads((ep / "meta.json").read_text())
        h = states[str(ep)]
        g = frozen[str(ep)]
        for e in meta["events"]:
            if not e["ok"]:
                continue
            a, b = int(e["t0"] * FPS), min(len(h), int(e["t1"] * FPS))
            if b - a < 3:
                continue
            rows.append((e["prim"], str(ep),
                         _l2(h[a:b].mean(0)), _l2(g[a:b].mean(0)),
                         _l2(h[b - 1])))
    return rows


def p_at_10(rows, qrows=None):
    """Same-primitive P@10, self-episode excluded. qrows defaults to
    rows (within-corpus); pass a different list for cross-corpus."""
    qrows = qrows if qrows is not None else rows
    H = np.stack([r[2] for r in rows])
    G = np.stack([r[3] for r in rows])
    E = np.stack([r[4] for r in rows])
    cols = {"state": (2, H), "frozen": (3, G), "endstate": (4, E)}
    out = {}
    for col, (ci, M) in cols.items():
        Q = np.stack([r[ci] for r in qrows])
        S = Q @ M.T
        ps = []
        for i, (prim, epid, *_) in enumerate(qrows):
            mask = np.array([r[1] != epid for r in rows])
            s = np.where(mask, S[i], -np.inf)
            top = np.argsort(-s)[:10]
            ps.append(np.mean([rows[j][0] == prim for j in top]))
        out[col] = float(np.mean(ps))
    return out


def main():
    import torch
    import vwm
    model = vwm.load()
    roots_eval = ["data/sim_chains"]
    roots_cross = ["data/sim_train/xarm7", "data/sim_train/vx300s"]

    from tqdm import tqdm
    states, frozen = {}, {}
    all_eps = [e for r in roots_eval + roots_cross for e in episodes(r)]
    for ep, npz in tqdm(all_eps, unit="ep", desc="states"):
        g = np.load(npz)["g"].astype(np.float32)
        h, sur = model.states(g)
        states[str(ep)] = h
        frozen[str(ep)] = g
        states[str(ep) + "/sur"] = sur

    print("\n=== gate a: primitive probe (frame -> active primitive) ===")
    reps_h, reps_g, labs = [], [], []
    for ep, npz in episodes(roots_eval[0]):
        meta = json.loads((ep / "meta.json").read_text())
        h = states[str(ep)]; g = frozen[str(ep)]
        lab, _ = frame_labels(meta, len(h))
        reps_h.append(h); reps_g.append(g); labs.append(lab)
    acc_h = probe_scores(reps_h, labs)
    acc_g = probe_scores(reps_g, labs)
    base = max(np.bincount(np.concatenate(labs)[
        np.concatenate(labs) >= 0]).max() /
        (np.concatenate(labs) >= 0).sum(), 1e-9)
    print(f"  state {acc_h:.3f}   frozen {acc_g:.3f}   majority {base:.3f}")

    print("\n=== gate b: surprise vs event boundaries ===")
    pos_h, neg_h, pos_d, neg_d = [], [], [], []
    for ep, npz in episodes(roots_eval[0]):
        meta = json.loads((ep / "meta.json").read_text())
        sur = states[str(ep) + "/sur"]
        g = frozen[str(ep)]
        dg = np.zeros(len(g))
        dg[1:] = np.linalg.norm(np.diff(g, axis=0), axis=1)
        T = len(sur)
        near = np.zeros(T, bool)
        for e in meta["events"]:
            b0 = int(e["t0"] * FPS)
            near[max(0, b0 - 5):min(T, b0 + 5)] = True
        pos_h += list(sur[near]); neg_h += list(sur[~near])
        pos_d += list(dg[near]); neg_d += list(dg[~near])
    print(f"  state-surprise AUC {auc(pos_h, neg_h):.3f}   "
          f"frozen-derivative AUC {auc(pos_d, neg_d):.3f}")

    print("\n=== gate c: event QbE on eval corpus (same-prim P@10) ===")
    rows = event_table(roots_eval[0], states, frozen)
    n_by = defaultdict(int)
    for r in rows:
        n_by[r[0]] += 1
    print(f"  events: {dict(n_by)}")
    r = p_at_10(rows)
    print(f"  state {r['state']:.3f}   endstate {r['endstate']:.3f}   "
          f"frozen {r['frozen']:.3f}")

    print("\n=== gate e: cross-embodiment A->B (train arms -> panda eval) ===")
    for root in roots_cross:
        qrows = event_table(root, states, frozen)
        if not qrows:
            print(f"  {root}: no events cached yet")
            continue
        r = p_at_10(rows, qrows)
        print(f"  {Path(root).name:8s} q={len(qrows)}  "
              f"state {r['state']:.3f}   endstate {r['endstate']:.3f}   "
              f"frozen {r['frozen']:.3f}")


if __name__ == "__main__":
    main()
