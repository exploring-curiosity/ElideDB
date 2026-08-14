"""Can the model fit ONE batch? The cheapest gate that catches real bugs.

A model that cannot drive the loss to zero on a single fixed batch has a
defect in the architecture, the target, or the optimiser - not a data
problem, and no amount of steps will fix it. Running this before a long
job costs about a minute.

It reports BOTH numbers, because the difference between them is where
this project lost a training run:

  R2_best   the best of the model's motion modes, oracle-selected. This
            hit 0.9999 while the run was worthless.
  R2_top    the mode the model's own selector points at. This stayed
            near 0 - the model could predict the future perfectly and
            could not say which of its predictions it believed.

Only R2_top counts. The mode loss weight was raised from 0.1 to 1.0 to
fix exactly this, and this probe is the check that it worked, so the
pass condition is R2_top > 0.95 and the gap R2_best - R2_top is reported
next to it rather than buried.

Noise is off and the WTA temperature is held small: the point is to see
whether the machinery can fit anything at all, not to regularise.

    python -m relmo.overfit --dataset rcasa --steps 600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402


def run(dataset="rcasa", steps=600, d=256, blocks=4, heads=8, modes=4,
        bs=4, lr=3e-4, eps=0.05, seed=0, w_seg=0.5, w_mode=1.0, w_aot=0.2,
        decay=False):
    import torch
    from relmo import splits as SP
    from relmo.train_wm3 import DEV, RelMoWM3, batch, step_loss
    # No device override. step_loss does X.to(DEV) with train_wm3's
    # module-level DEV, so a net placed anywhere else meets its own data
    # on another device: "Tensor for argument weight is on cpu but
    # expected on mps". The probe must run where the trainer runs, which
    # is the point of a probe.
    dev = DEV
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    tr = SP.partition(files, dataset)[SP.TRAIN]
    if len(tr) < bs:
        return dict(error=f"only {len(tr)} train tracks, need {bs}")
    rng = np.random.default_rng(seed)
    bt = None
    for _ in range(50):                     # a batch that yields motion
        bt = batch(tr, rng, bs)
        if bt is not None:
            break
    if bt is None:
        return dict(error="no usable batch")

    net = RelMoWM3(d=d, blocks=blocks, heads=heads, modes=modes,
                   head="fdnn").to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    n_par = sum(p.numel() for p in net.parameters())
    t0 = time.time()
    curve = []
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
             if decay else None)
    for s in range(1, steps + 1):
        o = step_loss(net, *bt, 0.0, eps, aot_every=4, aot_step=s)
        if o is None:
            return dict(error="step_loss returned None on a fixed batch")
        loss = (o["roll"] + w_mode * o["mode"] + w_aot * o["aot"]
                + w_seg * o["seg"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if s % max(steps // 12, 1) == 0 or s == steps:
            net.eval()
            with torch.no_grad():
                e = step_loss(net, *bt, 0.0, 0.0, aot_every=1)
            net.train()
            p = e["parts"]
            r2 = {n_: 1.0 - p[k][0] / max(p[k][1], 1e-9)
                  for k, n_ in (("m", "top"), ("b", "best"),
                                ("cv", "cv"), ("st", "still"))}
            curve.append(dict(step=s, loss=round(float(loss), 5),
                              slots=round(float(o["used"]), 2),
                              mode=round(float(o["mode"]), 4),
                              seg=round(float(o["seg"]), 4),
                              **{k: round(float(v), 4)
                                 for k, v in r2.items()}))
            print(f"  {s:5d} loss {float(loss):8.4f} mode {float(o['mode']):7.4f} "
                  f"seg {float(o['seg']):6.4f} slots {float(o['used']):5.2f} "
                  f"R2_top {r2['top']:8.4f} R2_best {r2['best']:8.4f}", flush=True)
    last = curve[-1]
    peak = max(c["top"] for c in curve)
    return dict(dataset=dataset, steps=steps, params=n_par, device=str(dev),
                d=d, blocks=blocks, modes=modes, n_train=len(tr),
                w_seg=w_seg, w_mode=w_mode, w_aot=w_aot, decay=decay,
                R2_top_peak=round(peak, 4),
                monotonic=bool(last["top"] >= peak - 0.02),
                R2_top=last["top"], R2_best=last["best"],
                gap=round(last["best"] - last["top"], 4),
                PASS=bool(last["top"] > 0.95),
                minutes=round((time.time() - t0) / 60, 2), curve=curve)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seg", type=float, default=0.5)
    ap.add_argument("--mode", type=float, default=1.0)
    ap.add_argument("--aot", type=float, default=0.2)
    ap.add_argument("--decay", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = run(a.dataset, a.steps, d=a.d, blocks=a.blocks, w_seg=a.seg,
            w_mode=a.mode, w_aot=a.aot, decay=a.decay)
    r["tag"] = a.tag
    print("\n" + json.dumps({k: v for k, v in r.items() if k != "curve"},
                            indent=1))
    R.log("overfit_probe", **{k: v for k, v in r.items() if k != "curve"})
    if not r.get("PASS"):
        print("\nFAIL: R2_top did not exceed 0.95 — do not start a long run.")
