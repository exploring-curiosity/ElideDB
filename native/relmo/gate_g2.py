"""G2 PARTS — is the slot binding discovering real rigid bodies?

WM_PLAN §5: "part segmentation vs sim body ids, beaten against
'everything is one body' AND 'every point its own body'."

Why this gate exists at all, and why it is being scored late: the
rollout (G4) predicts one SE(3) transform PER SLOT and blends the
results by slot attention. If the slots are not bodies, that decode is
averaging unrelated motions and no amount of training fixes it. The
project trained 11k steps of G4 with G2 unmeasured; this closes that.

METRIC CHOICE. Slot ids carry no meaning - slot 3 in one episode is not
slot 3 in another - so the metric must be permutation invariant:

  ARI     adjusted Rand index between slot argmax and sim body id.
          Chosen because BOTH trivial baselines score 0 BY
          CONSTRUCTION: a single cluster and all-singletons each have
          zero expected-adjusted agreement. That is exactly the pair
          WM_PLAN demands be beaten, so ARI > 0 IS the gate.
  FG-ARI  same, with the world/background body dropped. Standard in the
          slot-attention literature because a model can score well on
          ARI by merely separating "floor" from "everything else".
  mIoU    Hungarian-matched intersection-over-union, size-weighted.
          ARI's baselines being 0 is convenient but uninformative about
          magnitude, so mIoU is reported beside the honest non-trivial
          baseline: one slot covering every point, whose mIoU is the
          largest body's share of the points.

Nothing here trains. It reads a checkpoint and the sim body ids, which
are training-only scaffolding per WM_PLAN §4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.train_wm2 import TC, batch  # noqa: E402
from relmo.wm2 import RelMoWM2, canon  # noqa: E402


def ari(a, b):
    """Adjusted Rand index from the contingency table."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2:
        return float("nan")
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    n = np.zeros((len(ua), len(ub)), np.float64)
    np.add.at(n, (ia, ib), 1.0)

    def c2(x):
        return (x * (x - 1.0) / 2.0).sum()

    sij, sa, sb = c2(n), c2(n.sum(1)), c2(n.sum(0))
    tot = c2(np.array([float(len(a))]))
    exp = sa * sb / max(tot, 1e-12)
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / max(mx - exp, 1e-12))


def matched_iou(pred, true):
    """Size-weighted IoU under the best slot->body assignment.

    Greedy on the IoU matrix rather than Hungarian: with <=8 slots and
    a handful of bodies the two agree, and greedy has no dependency."""
    up, ut = np.unique(pred), np.unique(true)
    M = np.zeros((len(up), len(ut)))
    for i, p in enumerate(up):
        pm = pred == p
        for j, t in enumerate(ut):
            tm = true == t
            inter = np.logical_and(pm, tm).sum()
            M[i, j] = inter / max(np.logical_or(pm, tm).sum(), 1)
    tot, wsum = 0.0, 0.0
    used_p, used_t = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
    for i, j in order:
        if i in used_p or j in used_t:
            continue
        used_p.add(int(i))
        used_t.add(int(j))
        w = float((true == ut[j]).sum())
        tot += M[i, j] * w
        wsum += w
    for j, t in enumerate(ut):                      # unmatched bodies: IoU 0
        if j not in used_t:
            wsum += float((true == t).sum())
    return float(tot / max(wsum, 1e-9))


def run(run_id="relmowm2_fdnn", dataset="rcasa_v1", n=64, ckpt=None,
        device="cpu", seed=0):
    out = R.model_dir(run_id)
    p = Path(ckpt) if ckpt else (out / "best_val.pt")
    if not p.exists():
        cks = sorted(out.glob("ckpt_*.pt"))
        if not cks:
            return {"error": f"no checkpoint in {out}"}
        p = cks[-1]
    sd = torch.load(p, map_location=device)
    cfg = json.loads((out / "config.json").read_text())
    # readout MUST come from the run's own config: run A predates the
    # flag and has no "readout" key (correctly defaulting to "mean"),
    # run B is "last+mean" and carries an extra rd projection.
    net = RelMoWM2(head=cfg.get("head", "fdnn"),
                   readout=cfg.get("readout", "mean")).to(device).eval()
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    step = int(sd.get("step", -1))

    files = SP.partition(sorted((R.TRACKS / dataset).glob("*.npz")),
                         dataset)[SP.VAL]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        bt = batch(files, rng, 1)
        if bt is None:
            continue
        X, V, G = (t.to(device) for t in bt)
        Xn, _, _ = canon(X, V, TC)
        with torch.no_grad():
            h = net.encode(Xn[:, :TC], V[:, :TC])
            _, attn = net.parts(h, V[:, :TC])       # (B,K,P)
        slot = attn.argmax(1)[0].cpu().numpy()
        gb = G[0].cpu().numpy()
        vis = V[:, :TC].float().mean(1)[0].cpu().numpy() > 0.3
        keep = vis & (gb >= 0)
        if keep.sum() < 8 or len(np.unique(gb[keep])) < 2:
            continue
        s, t = slot[keep], gb[keep]
        # background/world body is 0 in MuJoCo and absent in Kubric
        fg = t > 0
        rows.append(dict(
            ari=ari(s, t),
            fg_ari=ari(s[fg], t[fg]) if fg.sum() > 8
            and len(np.unique(t[fg])) > 1 else np.nan,
            miou=matched_iou(s, t),
            one_body_miou=float(np.bincount(t).max() / len(t)),
            singleton_ari=ari(np.arange(len(t)), t),
            random_ari=ari(rng.integers(0, 8, len(t)), t),
            n_bodies=int(len(np.unique(t))),
            n_slots_used=int(len(np.unique(s))),
            n_points=int(len(t))))
    if not rows:
        return {"error": "no scorable episodes"}

    def m(k):
        v = np.array([r[k] for r in rows], float)
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")

    rep = dict(gate="G2", run=run_id, step=step, ckpt=p.name,
               dataset=dataset, split="val", episodes=len(rows),
               ari=round(m("ari"), 4), fg_ari=round(m("fg_ari"), 4),
               miou=round(m("miou"), 4),
               baseline_one_body_ari=0.0,
               baseline_one_body_miou=round(m("one_body_miou"), 4),
               baseline_singleton_ari=round(m("singleton_ari"), 4),
               baseline_random_ari=round(m("random_ari"), 4),
               n_bodies=round(m("n_bodies"), 2),
               n_slots_used=round(m("n_slots_used"), 2),
               n_points=round(m("n_points"), 1))
    rep["passed"] = bool(rep["ari"] > max(rep["baseline_singleton_ari"],
                                          rep["baseline_random_ari"], 0.0)
                         and rep["miou"] > rep["baseline_one_body_miou"])
    R.log("gate_g2", **rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="relmowm2_fdnn")
    ap.add_argument("--dataset", default="rcasa_v1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(json.dumps(run(a.run, a.dataset, a.n, a.ckpt, a.device), indent=1))
