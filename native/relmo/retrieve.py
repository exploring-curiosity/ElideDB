"""Does the world model's representation support "when did this happen"?

Everything measured so far scores FORECASTING. The product question is
RETRIEVAL: given a moment, find moments like it. Those are different
problems, and nothing in the trainer's loss (roll + mode + aot + seg)
pulls two similar events together - there is no similarity objective in
the system at all. So whether the learned representation is any good for
retrieval is, right now, an untested hope.

This measures it. For every window, build an embedding, query it against
windows from OTHER EPISODES, and ask whether the neighbours are the same
kind of event.

  WM        the world model's own trunk features, mean-pooled over
            points and context - the representation retrieval would
            actually use.
  TRAJ      the trajectory descriptor: direction, magnitude,
            straightness. What motion-R2 can see.
  REL       the relational descriptor: contact pattern + articulation.
            ORACLE (sim state), so this is a ceiling, not a system.
  RANDOM    shuffled labels, the floor.

Scoring: mAP and recall@5 with "same task" as the relevance label, and
SAME-EPISODE neighbours excluded so nothing wins by matching itself. The
label is used ONLY to score, exactly as sim GT scores a tracker; no
label is read to build any embedding.

    python -m relmo.retrieve --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.relsep import rel_feats, traj_feats  # noqa: E402

W = 24


def wm_embed(run_id, dataset, items, device="cpu"):
    """Mean-pooled trunk features from the trained world model."""
    import torch
    from relmo.gate_g4 import load_model
    from relmo.train_wm2 import TC, lift3d
    from relmo.wm3 import canon
    out = R.model_dir(run_id)
    p = out / "best_val.pt"
    if not p.exists():
        cks = sorted(out.glob("ckpt_*.pt"))
        if not cks:
            return None
        p = cks[-1]
    net, cfg, step = load_model(out, p, device)
    E = []
    for (f, t0) in items:
        z = np.load(f)
        X = lift3d(z)
        V = z["gvis"]
        a = int(t0)
        if a + TC > len(X):
            E.append(None)
            continue
        Xt = torch.tensor(X[a:a + TC], dtype=torch.float32)[None]
        Vt = torch.tensor(V[a:a + TC])[None]
        Xn, _, _ = canon(Xt, Vt, TC)
        with torch.no_grad():
            o = net(Xn, Vt, TC, noise=0.0)
        # pt is the per-point readout the slots and the decoder both read
        E.append(o["pt"][0].mean(0).cpu().numpy())
    return E, step


def norm(M):
    M = np.asarray(M, np.float64)
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def score(M, tasks, eps, k=5):
    """mAP and recall@k, excluding same-episode neighbours."""
    S = M @ M.T
    n = len(M)
    ap, rec = [], []
    for i in range(n):
        mask = np.array([eps[j] != eps[i] for j in range(n)])
        if mask.sum() < 2:
            continue
        idx = np.argsort(-S[i][mask])
        rel = (np.array(tasks)[mask][idx] == tasks[i]).astype(float)
        if rel.sum() == 0:
            continue
        hits = np.cumsum(rel)
        prec = hits / np.arange(1, len(rel) + 1)
        ap.append(float((prec * rel).sum() / rel.sum()))
        rec.append(float(rel[:k].max()))
    return float(np.mean(ap)), float(np.mean(rec)), len(ap)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", default="rcasa")
    ap_.add_argument("--run", default="wmR")
    ap_.add_argument("--per-ep", type=int, default=3)
    a = ap_.parse_args()

    man = R.read_manifest(a.dataset)
    by = {e["id"]: e for e in man["episodes"]}
    items, tasks, eps, TR, RE = [], [], [], [], []
    for f in sorted((R.TRACKS / a.dataset).glob("*.npz")):
        if f.stem not in by:
            continue
        z = np.load(f)
        if "window_starts" not in z.files or "contact_pairs" not in z.files:
            continue
        for t0 in z["window_starts"][:a.per_ep]:
            items.append((f, int(t0)))
            tasks.append(f.stem.split("_episode_")[0])
            eps.append(f.stem.split("__")[0])       # episode, not camera
            TR.append(traj_feats(z, int(t0), W))
            RE.append(rel_feats(z, int(t0), W))
    print(f"{a.dataset}: {len(items)} query windows, "
          f"{len(set(eps))} episodes, {len(set(tasks))} tasks")
    chance = np.mean([np.mean([t == q for t in tasks]) for q in tasks])
    print(f"chance precision (task prior) = {chance:.3f}\n")

    TRk = ("dir_x", "dir_y", "dir_z", "logmag", "straightness")
    REk = ("grip_frac", "third_frac", "n_partners", "contact_changes",
           "articulation")
    banks = {
        "TRAJ": norm([[r[k] for k in TRk] for r in TR]),
        "REL (oracle)": norm([[r[k] for k in REk] for r in RE]),
    }
    emb = wm_embed(a.run, a.dataset, items)
    if emb is not None:
        E, step = emb
        ok = [i for i, e in enumerate(E) if e is not None]
        if len(ok) == len(items):
            banks[f"WM {a.run}@{step}"] = norm([E[i] for i in ok])
    rng = np.random.default_rng(0)
    banks["RANDOM"] = norm(rng.normal(size=(len(items), 16)))

    print(f"{'representation':22s} {'mAP':>7s} {'recall@5':>9s} {'n':>5s}")
    print("-" * 46)
    res = {}
    for nm, M in banks.items():
        m, r, n = score(M, tasks, eps)
        res[nm] = dict(mAP=round(m, 4), recall_at_5=round(r, 4), n=n)
        print(f"{nm:22s} {m:7.3f} {r:9.3f} {n:5d}")
    R.log("retrieval", dataset=a.dataset, run=a.run,
          chance=round(float(chance), 4), results=res)
