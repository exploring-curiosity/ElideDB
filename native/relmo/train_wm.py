"""Self-supervised world-model training. No labels in the loss.

Objectives (both label-free):
  ROLLOUT   from a context window, predict where every point goes for
            the next H frames. Contact, support and momentum are the
            only way to get this right; point identity cannot be
            memorised into it.
  ARROW     forward or reversed? Direction of time is the thing every
            appearance representation measured here erased (open/close
            cosine 0.957), and it is free to supervise.

Splits are enforced (see splits.py): gradients only ever see `train`,
`val` drives selection, `test` is untouched until the end. That was
the missing control - the previous trainer discarded its own holdout.

    python -m relmo.train_wm --dataset physgen_v2 --steps 40000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.device import DEVICE  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.wm import RelMoWM  # noqa: E402

DEV = DEVICE
PMAX = 256
TC = 16                  # context frames
HZ = 8                   # predicted horizon


def load(f, rng, tc=TC, hz=HZ):
    z = np.load(f)
    xy = z["xy"].astype(np.float32)
    vis = z["vis"]
    T, P = xy.shape[0], xy.shape[1]
    if T < tc + hz:
        return None
    t0 = int(rng.integers(0, T - tc - hz + 1))
    xy, vis = xy[t0:t0 + tc + hz], vis[t0:t0 + tc + hz]
    if P > PMAX:
        keep = rng.choice(P, PMAX, replace=False)
        xy, vis = xy[:, keep], vis[:, keep]
    return xy, vis


def batch(files, rng, bs):
    out = [load(f, rng) for f in rng.choice(files, bs, replace=False)]
    out = [o for o in out if o is not None]
    if not out:
        return None
    P = max(o[0].shape[1] for o in out)
    T = out[0][0].shape[0]
    xy = np.zeros((len(out), T, P, 2), np.float32)
    vis = np.zeros((len(out), T, P), bool)
    for b, (x, v) in enumerate(out):
        xy[b, :, :x.shape[1]] = x
        vis[b, :, :x.shape[1]] = v
    return torch.tensor(xy), torch.tensor(vis)


def step_loss(net, xy, vis, tc=TC, hz=HZ):
    xy, vis = xy.to(DEV), vis.to(DEV)
    o = net(xy, vis, tc)
    xyn = o["xyn"]
    # target: displacement of each point from the last context frame,
    # in the context's own units - no absolute coordinates anywhere
    base = xyn[:, tc - 1]                                # (B,P,2)
    tgt = xyn[:, tc:tc + hz] - base[:, None]             # (B,H,P,2)
    tgt = tgt.permute(0, 2, 1, 3)                        # (B,P,H,2)
    m = vis[:, tc:tc + hz].permute(0, 2, 1)[..., None]   # (B,P,H,1)
    m = m & vis[:, tc - 1][..., None, None]
    # ONLY POINTS THAT ACTUALLY MOVE COUNT.
    # Measured the hard way: with every visible point in the loss, most
    # of them are static (background, resting objects) so "predict zero
    # displacement" is the minimum, and the model found exactly that -
    # its rollout error equalled the predict-nothing-moves baseline at
    # 1.00x while the loss curve looked healthy. A loss dominated by
    # the trivially predictable part of the field measures nothing.
    mag = tgt.norm(dim=-1, keepdim=True)                 # (B,P,H,1)
    thr = mag[m].mean().clamp(min=1e-4) * 0.5 if m.any() else 1e-4
    mv = m & (mag > thr)
    if mv.any():
        e = mv.expand_as(tgt)
        roll = F.smooth_l1_loss(o["dxy"][e], tgt[e], beta=0.02)
        # scale-free score: fraction of the MOTION that the model
        # explains. 0 = no better than predicting stillness, 1 = exact.
        sse = ((o["dxy"] - tgt)[e] ** 2).sum()
        sst = (tgt[e] ** 2).sum().clamp(min=1e-9)
        r2 = float(1.0 - sse / sst)
    else:
        roll = torch.zeros((), device=DEV)
        r2 = float("nan")
    # arrow of time on the context window
    B = len(xy)
    fwd = net.encode(xyn[:, :tc], vis[:, :tc])
    rev = net.encode(torch.flip(xyn[:, :tc], (1,)),
                     torch.flip(vis[:, :tc], (1,)))
    logit = torch.cat([net.arrow(fwd), net.arrow(rev)])
    lab = torch.cat([torch.ones(B, device=DEV),
                     torch.zeros(B, device=DEV)])
    aot = F.binary_cross_entropy_with_logits(logit, lab)
    acc = float(((logit > 0).float() == lab).float().mean())
    # slot occupancy: how many slots are actually used (diagnostic,
    # not a loss - a collapsed slot set would silently look fine)
    a = o["attn"].mean(-1)
    used = float((a > 0.05).float().sum(-1).mean())
    return roll, aot, acc, used, r2


def evaluate_val(net, files, rng, n=24, bs=4):
    net.eval()
    rs, as_, r2s = [], [], []
    with torch.no_grad():
        for _ in range(max(n // bs, 1)):
            bt = batch(files, rng, bs)
            if bt is None:
                continue
            r, a, acc, _, r2 = step_loss(net, *bt)
            rs.append(float(r))
            as_.append(acc)
            r2s.append(r2)
    net.train()
    return (float(np.mean(rs)) if rs else float("nan"),
            float(np.mean(as_)) if as_ else float("nan"),
            float(np.nanmean(r2s)) if r2s else float("nan"))


def train(dataset="physgen_v2", steps=40000, run_id="relmowm_v1",
          bs=4, lr=3e-4, log_every=250, ckpt_every=2500, head="mlp"):
    cfg = dict(dataset=dataset, bs=bs, lr=lr, d=128, blocks=4,
               slots=6, tc=TC, hz=HZ, pmax=PMAX, head=head,
               model="RelMoWM-v1", objective="rollout+arrow (no labels)")
    out = R.model_dir(run_id)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    net = RelMoWM(head=head).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    step0 = 0
    cks = sorted(out.glob("ckpt_*.pt"))
    if cks:
        sd = torch.load(cks[-1], map_location=DEV)
        net.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step0 = sd["step"]
    rng = np.random.default_rng(0)
    best = -float("inf")
    hist = []
    t0 = time.time()
    R.log("wm_train_start", run=run_id, dataset=dataset, step=step0,
          config=cfg)
    for step in range(step0 + 1, steps + 1):
        if step % 250 == 1:
            part = SP.partition(sorted(
                (R.TRACKS / dataset).glob("ep*.npz")))
            tr, va = part[SP.TRAIN], part[SP.VAL]
        if len(tr) < bs:
            R.log("wm_wait", have=len(tr))
            time.sleep(60)
            continue
        bt = batch(tr, rng, bs)
        if bt is None:
            continue
        roll, aot, acc, used, r2 = step_loss(net, *bt)
        loss = roll + 0.2 * aot
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        hist.append((float(roll), float(aot), acc, used,
                     r2 if r2 == r2 else 0.0))
        if step % log_every == 0:
            h = np.array(hist[-log_every:])
            vr, va_, vr2 = evaluate_val(net, va, rng) if va else (
                float("nan"), float("nan"), float("nan"))
            rec = dict(run=run_id, step=step,
                       train_rollout=round(float(h[:, 0].mean()), 5),
                       val_rollout=round(vr, 5),
                       val_motion_r2=round(vr2, 4),
                       train_motion_r2=round(float(h[:, 4].mean()), 4),
                       train_aot_acc=round(float(h[:, 2].mean()), 4),
                       val_aot_acc=round(va_, 4),
                       slots_used=round(float(h[:, 3].mean()), 2),
                       n_train=len(tr), n_val=len(va),
                       min_per_1k=round((time.time() - t0)
                                        / max(step - step0, 1) * 1000 / 60, 2))
            R.log("wm_train", **rec)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if vr2 == vr2 and vr2 > best:   # select on VAL motion-R2
                best = vr2
                torch.save(dict(model=net.state_dict(), step=step,
                                cfg=cfg, val_rollout=vr,
                                val_motion_r2=vr2),
                           out / "best_val.pt")
                R.log("wm_promote", run=run_id, step=step,
                      val_rollout=vr, val_motion_r2=vr2)
        if step % ckpt_every == 0:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(),
                            step=step, cfg=cfg,
                            data_fp=R.read_manifest(dataset).get("fingerprint")),
                       out / f"ckpt_{step:07d}.pt")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="physgen_v2")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--run", default="relmowm_v1")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--head", default="mlp", choices=["mlp", "fdnn"])
    a = ap.parse_args()
    print(train(a.dataset, a.steps, a.run, a.bs, head=a.head))
