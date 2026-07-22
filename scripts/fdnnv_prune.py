"""FDNN rules 2+3 on the trained video encoder.

    apoptosis -> fine-tune (re-settle) -> neurogenesis -> fine-tune

This is the cycle that could NOT be run on SigLIP itself: one-shot pruning
collapsed fidelity to 0.43 because there was nothing to fine-tune with. Here
the fine-tune step is ordinary distillation — every corpus frame has a teacher
vector — so survivors re-settle and reborn neurons get to earn a living.

The prune unit is the temporal-core channel (the FDNN neuron proper: a
KAN-sum over heterogeneous sub-functions). Utilization is the drop in
HELD-OUT embedding fidelity when the channel is silenced — the retrieval
system's own definition of "consumption". Reverse attention surfaces the
candidates, a PPO-style bandit searches keep-masks, and the selected state is
COMPACTED so dead channels stop costing FLOPs rather than being multiplied
by zero.

Honest scope note: the temporal core is where the FDNN neurons live, but most
of this model's wall-clock is the stem — so the cycle is reported for what it
gives (capacity right-sizing, params, any fidelity change), not sold as the
speed lever. The stem's speed dial is its width, chosen by the stage-1 race.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402
import mlx.nn as nn                                          # noqa: E402
import mlx.optimizers as optim                               # noqa: E402
from mlx.utils import tree_flatten                           # noqa: E402

from elidedb.fdnnvideo import (FDNNVideoEncoder, distill_loss,  # noqa: E402
                               load_encoder, save_encoder)
from fdnnv_train import eval_model, load_split, speed        # noqa: E402

OUT = Path("lake/bridge/models/fdnnv")


# ---------------------------------------------------------------------------
def get_mask(model):
    return np.array(model.cell.mask).copy()


def set_mask(model, m):
    model.cell.set_active_mask(m)


def val_fidelity(model, px, vec, sid, ts, val, cap=1200):
    """Streaming fidelity on a fixed prefix of each val stream — the ablation
    loop calls this hundreds of times, so it must be seconds, not minutes."""
    outs, tgts = [], []
    for s in np.unique(sid[val]):
        m = np.where(val & (sid == s))[0]
        m = m[np.argsort(ts[m])][:cap // len(np.unique(sid[val]))]
        outs.append(model.embed_frames_np(px[m], batch=128))
        tgts.append(vec[m])
    o = np.concatenate(outs)
    t = np.concatenate(tgts)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    return float((o * t).sum(1).mean())


def channel_stats(model, px, sid, ts, val, n=256):
    """Activation magnitude + downstream weight norm per channel."""
    m = np.where(val)[0][:n]
    x = mx.array(px[m].astype(np.float32) / 127.5 - 1.0)
    g = model.stem(x)
    h = model.init_state(len(m))
    acts = np.abs(np.array(model.cell.neuron_outputs(g, h))).mean(axis=0)
    down = (np.linalg.norm(np.array(model.head_h.weight), axis=0)
            + np.linalg.norm(np.array(model.cell.Wh), axis=1)
            + np.linalg.norm(np.array(model.cell.Uz), axis=1))
    return acts.astype(np.float32), down.astype(np.float32)


def reverse_attention(imp):
    z = (imp - imp.mean()) / (imp.std() + 1e-8)
    e = np.exp(-z - (-z).max())
    return e / e.sum()


def finetune(model, px, vec, sid, ts, val, steps=120, T=24, B=8, lr=3e-4,
             seed=0):
    """Distillation re-settle with aliveness pinned after every step."""
    rng = np.random.default_rng(seed)
    frozen = get_mask(model)
    starts = []
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        ok = ~val[m]
        for j in range(0, len(m) - T, 4):
            if ok[j:j + T].all():
                starts.append(m[j])
    starts = np.array(starts)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    def loss_fn(mdl, x, y):
        e, _ = mdl(x)
        return distill_loss(e[:, 4:], y[:, 4:])

    lg = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        b = rng.choice(starts, B, replace=False)
        x = np.stack([px[i:i + T] for i in b]).astype(np.float32)
        y = np.stack([vec[i:i + T] for i in b])
        _, grads = lg(model, mx.array(x / 127.5 - 1.0), mx.array(y))
        opt.update(model, grads)
        set_mask(model, frozen)
        mx.eval(model.parameters(), opt.state)
    return model


def neurogenesis(model, n_revive, seed=0):
    """Rebirth with fresh, spectrally diverse sub-functions; near-silent w2 so
    newborns cannot shock the forward pass."""
    rng = np.random.default_rng(seed)
    cell = model.cell
    mask = get_mask(model)
    dead = np.where(mask == 0.0)[0]
    if len(dead) == 0 or n_revive <= 0:
        return 0
    revive = dead[:min(n_revive, len(dead))]
    k = cell.k
    Wg, Wh = np.array(cell.Wg), np.array(cell.Wh)
    b1, ph = np.array(cell.b1), np.array(cell.phases)
    gs, la = np.array(cell.gabor_s), np.array(cell.log_alpha)
    w2 = np.array(cell.w2)
    om_full = np.repeat(cell.omegas_per_neuron, k)
    mean_om = float(cell.omegas_per_neuron.mean())
    lg = float(np.sqrt(6.0 / Wg.shape[0]) / mean_om)
    lh = float(np.sqrt(6.0 / cell.C) / mean_om)
    for c in revive:
        s, e = c * k, (c + 1) * k
        Wg[:, s:e] = rng.uniform(-lg, lg, (Wg.shape[0], k))
        Wh[:, s:e] = rng.uniform(-lh, lh, (Wh.shape[0], k))
        b1[s:e] = rng.uniform(-2.0, 2.0, k)
        ph[s:e] = rng.uniform(0, 2 * np.pi, k)
        gs[s:e] = rng.uniform(0.3, 1.5, k)
        lw = np.log(np.clip(om_full[s:e], 1e-3, None))
        # sub-unit omegas make log-omega negative; order the bounds (the
        # same crash the model init hit with the calmed recurrent bands)
        la[s:e] = rng.uniform(np.minimum(0.0, lw), np.maximum(0.0, lw) + 1e-6)
        w2[c] = rng.standard_normal(k) * 1e-6
    cell.Wg, cell.Wh = mx.array(Wg), mx.array(Wh)
    cell.b1, cell.phases = mx.array(b1), mx.array(ph)
    cell.gabor_s, cell.log_alpha = mx.array(gs), mx.array(la)
    cell.w2 = mx.array(w2)
    mask[revive] = 1.0
    set_mask(model, mask)
    return len(revive)


def compact(model):
    """Delete dead channels for real: slice every consumer and producer."""
    keep = np.where(get_mask(model) > 0.5)[0]
    cell = model.cell
    if len(keep) == cell.C:
        return model, keep
    cfg = dict(model.cfg)
    cfg["channels"] = int(len(keep))
    new = FDNNVideoEncoder(**cfg)
    # stem + glimpse head copy over unchanged
    from mlx.utils import tree_unflatten
    new.stem.update(tree_unflatten(
        [(k, mx.array(np.array(v)))
         for k, v in tree_flatten(model.stem.parameters())]))
    new.head_g.weight = mx.array(np.array(model.head_g.weight))
    new.head_g.bias = mx.array(np.array(model.head_g.bias))
    k = cell.k
    sub = np.concatenate([np.arange(c * k, (c + 1) * k) for c in keep])
    nc = new.cell
    nc.Wg = mx.array(np.array(cell.Wg)[:, sub])
    nc.Wh = mx.array(np.array(cell.Wh)[keep][:, sub])
    nc.b1 = mx.array(np.array(cell.b1)[sub])
    nc.phases = mx.array(np.array(cell.phases)[sub])
    nc.gabor_s = mx.array(np.array(cell.gabor_s)[sub])
    nc.log_alpha = mx.array(np.array(cell.log_alpha)[sub])
    # omegas is TRAINED: slice the live values, never rebuild from the design
    # band (the compaction bug found on the context tower).
    nc.omegas = mx.array(np.array(cell.omegas)[sub])
    nc.omegas_per_neuron = cell.omegas_per_neuron[keep]
    nc.basis_types = mx.array(np.array(cell.basis_types)[sub])
    nc.freeze(keys=["basis_types", "mask"], recurse=False)
    nc.w2 = mx.array(np.array(cell.w2)[keep])
    nc.Wz = mx.array(np.array(cell.Wz)[:, keep])
    nc.Uz = mx.array(np.array(cell.Uz)[keep][:, keep])
    nc.bz = mx.array(np.array(cell.bz)[keep])
    nc.set_active_mask(np.ones(len(keep), np.float32))
    new.head_h.weight = mx.array(np.array(model.head_h.weight)[:, keep])
    new.head_h.bias = mx.array(np.array(model.head_h.bias))
    mx.eval(new.parameters())
    return new, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.35)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--finetune-steps", type=int, default=120)
    ap.add_argument("--tolerance", type=float, default=0.003,
                    help="fidelity slack for the operating point")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, meta = load_encoder(OUT)
    px, vec, sid, ts, val = load_split()
    rng = np.random.default_rng(args.seed)

    fid0 = val_fidelity(model, px, vec, sid, ts, val)
    ms0 = speed(model)
    n0 = sum(int(np.prod(v.shape)) for _, v in tree_flatten(model.parameters()))
    print(f"[before] fid={fid0:.4f} channels={model.cell.C} "
          f"params={n0/1e6:.2f}M {ms0:.3f} ms/f", flush=True)

    # ---- utilization by ablation (the interpretable measure) --------------
    C = model.cell.C
    base_mask = get_mask(model)
    abl = np.zeros(C, np.float32)
    t0 = time.time()
    for c in range(C):
        m = base_mask.copy()
        m[c] = 0.0
        set_mask(model, m)
        abl[c] = max(fid0 - val_fidelity(model, px, vec, sid, ts, val,
                                         cap=400), 0.0)
    set_mask(model, base_mask)
    print(f"  ablated {C} channels in {time.time()-t0:.0f}s | "
          f"utilization p50={np.median(abl):.5f} max={abl.max():.5f}",
          flush=True)

    acts, down = channel_stats(model, px, sid, ts, val)
    ra = reverse_attention(abl)
    sal = (abl / (abl.max() + 1e-9)) + 0.3 * (acts * down) / \
        ((acts * down).max() + 1e-9)

    # ---- bandit over keep fractions along the saliency order --------------
    order = np.argsort(-sal)
    saved = []

    def try_keep(frac):
        n_keep = max(int(round(frac * C)), 8)
        m = np.zeros(C, np.float32)
        m[order[:n_keep]] = 1.0
        set_mask(model, m)
        f = val_fidelity(model, px, vec, sid, ts, val, cap=600)
        set_mask(model, base_mask)
        return f, m

    frac = 1.0 - args.sparsity
    best = (fid0, base_mask.copy(), 1.0)
    for it in range(args.iters):
        cand = float(np.clip(frac + 0.12 * rng.standard_normal(), 0.15, 1.0))
        f, m = try_keep(cand)
        reward = f - 0.02 * cand
        best_reward = best[0] - 0.02 * best[2]
        if reward > best_reward:
            best, frac = (f, m, cand), cand
        print(f"  bandit {it:2d} keep={cand:.2f} fid={f:.4f}", flush=True)

    stages = [("before", fid0, base_mask.copy())]

    # ---- apoptosis -> re-settle -------------------------------------------
    set_mask(model, best[1])
    fid_a = val_fidelity(model, px, vec, sid, ts, val)
    stages.append(("after_apoptosis", fid_a, get_mask(model)))
    print(f"[apoptosis] keep={int(best[1].sum())}/{C} fid={fid_a:.4f}",
          flush=True)
    finetune(model, px, vec, sid, ts, val, steps=args.finetune_steps)
    fid_f = val_fidelity(model, px, vec, sid, ts, val)
    stages.append(("after_finetune", fid_f, get_mask(model)))
    print(f"[finetune ] fid={fid_f:.4f}", flush=True)

    # ---- neurogenesis -> re-settle ----------------------------------------
    dead = int((get_mask(model) == 0).sum())
    born = neurogenesis(model, dead // 2, seed=args.seed)
    if born:
        finetune(model, px, vec, sid, ts, val, steps=args.finetune_steps)
        fid_r = val_fidelity(model, px, vec, sid, ts, val)
        stages.append(("after_rebirth_finetune", fid_r, get_mask(model)))
        print(f"[rebirth  ] +{born} channels, fid={fid_r:.4f}", flush=True)

    # ---- operating point: fewest channels within tolerance of best --------
    best_fid = max(s[1] for s in stages)
    ok = [s for s in stages if s[1] >= best_fid - args.tolerance]
    stage, fid_sel, mask_sel = min(ok, key=lambda s: s[2].sum())
    set_mask(model, mask_sel)
    print(f"[select   ] {stage}: {int(mask_sel.sum())}/{C} channels, "
          f"fid={fid_sel:.4f} (best {best_fid:.4f})", flush=True)

    pre_compact = val_fidelity(model, px, vec, sid, ts, val)
    model, keep = compact(model)
    post = val_fidelity(model, px, vec, sid, ts, val)
    if abs(post - pre_compact) > 1e-3:
        print(f"  WARNING: compaction moved fidelity "
              f"{pre_compact:.4f} -> {post:.4f}")
    fid1, p10, knn = eval_model(model, px, vec, sid, ts, val)
    ms1 = speed(model)
    n1 = sum(int(np.prod(v.shape)) for _, v in tree_flatten(model.parameters()))
    print(f"[after ] fid={fid1:.4f} p10={p10:.4f} knn={knn:.3f} "
          f"channels={len(keep)} params={n1/1e6:.2f}M {ms1:.3f} ms/f",
          flush=True)

    meta["prune"] = {
        "before": {"fid": fid0, "channels": C, "params": n0, "ms": ms0},
        "after": {"fid": fid1, "p10": p10, "knn": knn,
                  "channels": int(len(keep)), "params": n1, "ms": ms1},
        "stages": [(s, float(f), int(m.sum())) for s, f, m in stages],
        "utilization_p50": float(np.median(abl)),
    }
    save_encoder(model, meta, OUT)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
