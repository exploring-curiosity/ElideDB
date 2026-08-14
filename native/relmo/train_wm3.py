"""RelMo-WM v3 trainer — per-point displacement, WTA modes, GNS noise.

Every eval still quotes the model beside the baselines it must beat, on
the SAME masked points, pooled sse/sst, per-episode threshold. Those
conventions were all bought with measurement errors and are not
negotiable: absolute-position denominators once made "predict stillness"
score +0.791, per-batch thresholds once made a 0.082 model read 0.208,
and averaging per-batch ratios once swung const-velocity +0.994/-7.858.

Four things differ from v2, each traceable to a measurement or a paper:

  DISPLACEMENT TARGETS. The loss now optimises the quantity the metric
    scores. v2 regressed absolute canonical position, where "stay put" is
    already ~99% right.
  WTA OVER M MODES (annealed). Deterministic regression on a multimodal
    future returns the conditional mean, which here is ~zero motion -
    exactly v2's collapse. eps anneals 1.0 -> 0.05 so modes separate
    instead of all chasing the mean from step one.
  GNS NOISE INJECTION on the context. v2 never met its own error
    distribution.
  SLOTS ARE AUXILIARY. They still earn G2/G3 and the retrieval
    descriptor; they no longer gate prediction.

    python -m relmo.daemon train_wm3 --datasets rcasa_v1,arctic_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.device import DEVICE  # noqa: E402
from relmo.train_wm2 import HZ, TC, batch  # noqa: E402
from relmo.wm3 import RelMoWM3, canon  # noqa: E402

DEV = DEVICE
# Predict the CONST-VELOCITY RESIDUAL instead of raw displacement.
#
# MEASURED on this corpus: const-vel alone scores +0.881 pooled on val and
# +0.676 on test, i.e. ~88% of the displacement target is a straight line.
# wmP spent 3000 steps re-deriving that line and failed - it emitted vectors
# 0.188x the target magnitude with cosine +0.032 to the truth, the classic
# collapse to the conditional mean of a multimodal future.
#
# With a residual target the model INHERITS const-vel's direction and
# magnitude and only has to learn the correction, so the same collapse
# (predict ~0 residual) scores +0.881 instead of -0.011: the baseline
# becomes the FLOOR rather than a bar the model never reaches.
#
# The REPORTED metric is unchanged. The loss is computed on the residual,
# but every R2 is scored on the original displacement with const-vel added
# back, so these numbers stay directly comparable to G0, ceiling.py, G4 and
# every arm measured before this.
RESIDUAL = os.environ.get("RELMO_RESIDUAL", "0") == "1"
# SOFT SELECTOR TARGET.
#
# l_mode regresses cross-entropy against argmin(per_mode) - a HARD label.
# MEASURED on wmR: the oracle-best mode beats const-velocity by +0.009..+0.026
# at every eval after step 1000 while the SELECTED mode sits at +-0.000, and
# the gap grows after the WTA anneal (+0.0149 +0.0191 +0.0273 +0.0260). The
# modes learn real dynamics; the selector cannot retrieve them.
#
# The cause is that once modes specialise their errors become near-tied, so
# argmin flips between steps and the selector is trained on a label that is
# mostly noise. A soft target fixes this without changing what is learned:
# it is peaked when one mode is genuinely better and near-uniform when they
# are tied, so ties contribute almost no gradient instead of a random one.
# Same scale normalisation as the roll softmin, so the two agree on what
# "clearly better" means.
SOFTSEL = os.environ.get("RELMO_SOFTSEL", "0") == "1"
SOFTSEL_TAU = float(os.environ.get("RELMO_SOFTSEL_TAU", "0.25"))


def masks(Xn, V):
    """Moving-point mask + displacement targets, the canonical way."""
    base = Xn[:, TC - 1]
    tgt = Xn[:, TC:TC + HZ]
    m = V[:, TC:TC + HZ] & V[:, TC - 1][:, None]
    d = tgt - base[:, None]                              # (B,H,P,3)
    mag = d.norm(dim=-1)
    if not m.any():
        return None
    ms = m.flatten(1).sum(1).clamp(min=1)
    thr = ((mag * m).flatten(1).sum(1) / ms).clamp(min=1e-5) * 0.5
    mv = m & (mag > thr[:, None, None])
    return (base, d, mv) if mv.any() else None


def pooled(pred, d, mv):
    """(sse, sst) on masked points - callers pool, never average ratios."""
    e = ((pred - d) ** 2).sum(-1)[mv]
    return float(e.sum()), float((d ** 2).sum(-1)[mv].sum())


def step_loss(net, X, V, G, noise, eps, aot_every=4, aot_step=0):
    X, V, G = X.to(DEV), V.to(DEV), G.to(DEV)
    Xn, _, _ = canon(X, V, TC)
    mk = masks(Xn, V)
    if mk is None:
        return None
    base, d, mv = mk
    # const-velocity extrapolation of the last context step
    v = Xn[:, TC - 1] - Xn[:, TC - 2]
    hsteps = torch.arange(1, HZ + 1, device=DEV, dtype=torch.float32)
    cv = v[:, None] * hsteps[None, :, None, None]
    # mv was computed on the TRUE displacement above and stays that way:
    # "which points are moving" must not depend on the parameterisation.
    d_fit = d - cv if RESIDUAL else d          # what the loss regresses
    o = net(Xn, V, TC, noise=noise)
    P = o["disp"]                                        # (B,M,H,P,3)
    err = ((P - d_fit[:, None]) ** 2).sum(-1)            # (B,M,H,P)
    msk = mv[:, None].expand_as(err)
    per_mode = (err * msk).flatten(2).sum(-1) / msk.flatten(2).sum(-1).clamp(min=1)
    win = per_mode.argmin(1)                             # (B,)
    # ANNEALED WTA done properly: softmin weighting over modes with a
    # temperature that falls to a winner-take-all.
    #
    # The previous form was roll = (1-eps)*winner + eps*MEAN(modes) with
    # eps annealed 1.0 -> 0.05, i.e. for the first 20% of training the
    # loss WAS the mean over modes. Averaging modes is the exact failure
    # multi-mode prediction exists to prevent - the conditional mean of a
    # multimodal future is "almost no motion" - so run D spent 8k steps
    # optimising the collapse it was built to avoid, and never got past
    # eps 0.47 before it was stopped. Softmin at high temperature still
    # differentiates the modes (the better one gets more gradient), so
    # they specialise from the first step instead of converging.
    sc = per_mode.mean(1, keepdim=True).detach().clamp(min=1e-8)
    w = torch.softmax(-per_mode / (sc * max(eps, 1e-3)), dim=1)
    roll = (w.detach() * per_mode).sum(1).mean()
    if SOFTSEL:
        soft = torch.softmax(-per_mode.detach() / (sc * SOFTSEL_TAU), dim=1)
        l_mode = -(soft * F.log_softmax(o["logit"], dim=1)).sum(1).mean()
    else:
        l_mode = F.cross_entropy(o["logit"], win)
    # honest eval prediction = the mode the model itself picks
    sel = o["logit"].argmax(1)
    top = P[torch.arange(len(P)), sel]                   # (B,H,P,3)
    best = P[torch.arange(len(P)), win]
    # ---- auxiliary heads
    g = o["pt"]
    _, attn = net.parts(g, V[:, :TC])
    seg = torch.zeros((), device=DEV)
    if (G >= 0).any():
        a = attn.transpose(1, 2)
        same = (G[:, :, None] == G[:, None, :]) & (G[:, :, None] >= 0)
        sim = torch.einsum("bpk,bqk->bpq", a, a).clamp(1e-4, 1 - 1e-4)
        pos = same.float()
        neg = 1.0 - pos
        seg = 0.5 * ((-torch.log(sim) * pos).sum() / pos.sum().clamp(min=1)
                     + (-torch.log(1 - sim) * neg).sum() / neg.sum().clamp(min=1))
    # Arrow of time costs an encoder pass, so: reuse the forward pass's
    # features for the forward direction (free) and only run the REVERSED
    # clip, and only every aot_every steps. Measured: the naive form ran
    # encode() three times per step - 3.33 s/step against 0.93 s for one
    # pass, i.e. 37 h for this run instead of ~13 h - to supervise a head
    # that reached 0.87 within 8k steps in v2 and gates nothing.
    aot = torch.zeros((), device=DEV)
    acc = float("nan")
    if aot_every and (aot_step % aot_every == 0):
        hf = o["feat"]
        hb = net.encode(torch.flip(Xn[:, :TC], (1,)),
                        torch.flip(V[:, :TC], (1,)))
        lg = torch.cat([net.arrow(hf), net.arrow(hb)])
        lab = torch.cat([torch.ones(len(X), device=DEV),
                         torch.zeros(len(X), device=DEV)])
        aot = F.binary_cross_entropy_with_logits(lg, lab)
        acc = float(((lg > 0).float() == lab).float().mean())
    with torch.no_grad():
        top_d = top + cv if RESIDUAL else top
        best_d = best + cv if RESIDUAL else best
        parts = dict(m=pooled(top_d, d, mv), b=pooled(best_d, d, mv),
                     cv=pooled(cv, d, mv),
                     st=pooled(torch.zeros_like(d), d, mv))
    return dict(roll=roll, mode=l_mode, aot=aot, seg=seg, acc=acc,
                used=float((attn.mean(-1) > 0.05).float().sum(-1).mean()),
                parts=parts)


def evaluate(net, files, rng, n=192, bs=4):
    net.eval()
    pool = {k: [0.0, 0.0] for k in ("m", "b", "cv", "st")}
    accs = []
    with torch.no_grad():
        for _ in range(max(n // bs, 1)):
            bt = batch(files, rng, bs)
            if bt is None:
                continue
            o = step_loss(net, *bt, 0.0, 0.0, aot_every=1)
            if o is None:
                continue
            for k, (a, b) in o["parts"].items():
                pool[k][0] += a
                pool[k][1] += b
            accs.append(o["acc"])
    net.train()
    out = {n_: (1.0 - pool[k][0] / max(pool[k][1], 1e-9))
           for k, n_ in (("m", "r2"), ("b", "r2_bestmode"),
                         ("cv", "r2_cv"), ("st", "r2_still"))}
    out["acc"] = float(np.mean(accs)) if accs else float("nan")
    return out


def train(datasets="rcasa_v1,arctic_v1", steps=40000, run_id=None, bs=4,
          lr=3e-4, head="fdnn", d=512, blocks=8, heads=8, modes=4,
          noise=0.003, log_every=250, ckpt_every=2500, decay=True):
    from tqdm import tqdm
    dss = [x.strip() for x in datasets.split(",") if x.strip()]
    run_id = run_id or f"relmowm3_{head}"
    out = R.model_dir(run_id)
    cfg = dict(datasets=dss, head=head, d=d, blocks=blocks, heads=heads,
               modes=modes, noise=noise, lr=lr, bs=bs, tc=TC, hz=HZ,
               lr_schedule="cosine" if decay else "constant",
               target="const_vel_residual" if RESIDUAL else "displacement",
               selector="soft" if SOFTSEL else "hard_argmin",
               selector_tau=SOFTSEL_TAU if SOFTSEL else None,
               model="RelMoWM3", eval_n=192, thr="per_episode",
               objective="per-point displacement, annealed WTA, aux slots")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    net = RelMoWM3(d=d, blocks=blocks, heads=heads, modes=modes,
                   hz=HZ, head=head).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    step0 = 0
    cks = sorted(out.glob("ckpt_*.pt"))
    if cks:
        sd = torch.load(cks[-1], map_location=DEV)
        net.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step0 = sd["step"]
    tr, va = [], []
    for ds in dss:
        p = SP.partition(sorted((R.TRACKS / ds).glob("*.npz")), ds)
        tr += p[SP.TRAIN]
        va += p[SP.VAL]
    R.log("wm3_start", run=run_id, step=step0, n_train=len(tr),
          n_val=len(va), params=sum(p.numel() for p in net.parameters()),
          config=cfg)
    if len(tr) < bs:
        R.log("wm3_no_data", have=len(tr))
        return run_id
    rng = np.random.default_rng(0)
    hist, best = [], -float("inf")
    t0 = time.time()
    bar = tqdm(range(step0 + 1, steps + 1), initial=step0, total=steps,
               unit="step", desc=f"wm3/{head}")
    for step in bar:
        # COSINE LR DECAY. Without it this objective destabilises itself:
        # the mode target is a hard argmin over per-mode error, so once the
        # modes specialise and their errors converge the winner flips
        # between steps, cross-entropy is handed a label that is noise, and
        # Adam's momentum amplifies it. MEASURED on one fixed batch,
        # 1200 steps, everything else equal:
        #   constant lr : peak R2_top 0.921 @832, collapses to -0.09,
        #                 mode loss 0.0000 -> 52.0
        #   cosine decay: monotonic 0.784 -> 0.9958, mode loss stays 0.0000
        # Dropping the seg loss instead made it far WORSE (slots pinned at
        # the maximum of 8, mode loss 52), so the rising slot count that
        # accompanied the collapse was a symptom, not the cause.
        # Computed from the absolute step so a resumed run continues on the
        # same schedule rather than restarting it.
        if decay:
            lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * min(step, steps)
                                              / max(steps, 1)))
            for gparam in opt.param_groups:
                gparam["lr"] = lr_t
        bt = batch(tr, rng, bs)
        if bt is None:
            continue
        # temperature, not a mixing weight: 1.0 = softly weighted across
        # modes, 0.02 = winner-take-all. Anneals over the first 30%.
        eps = float(np.clip(1.0 - 0.98 * step / (0.3 * steps), 0.02, 1.0))
        o = step_loss(net, *bt, noise, eps, aot_step=step)
        if o is None:
            continue
        # mode weight 1.0, not 0.1: the selector decides which mode is
        # REPORTED, and the overfit probe showed R2 of the selected mode
        # pinned near 0 while the best mode reached 0.9999 - a perfect
        # prediction the model could not point at is worth nothing.
        loss = o["roll"] + 1.0 * o["mode"] + 0.2 * o["aot"] + 0.5 * o["seg"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        a_ = o["acc"]
        hist.append((float(o["roll"]), a_ if a_ == a_ else np.nan,
                     o["used"], float(o["seg"])))
        if step % log_every == 0:
            h = np.array(hist[-log_every:])
            v = evaluate(net, va, rng) if va else {}
            rec = dict(run=run_id, step=step,
                       train_roll=round(float(h[:, 0].mean()), 5),
                       val_r2=round(v.get("r2", float("nan")), 4),
                       val_r2_bestmode=round(v.get("r2_bestmode",
                                                   float("nan")), 4),
                       val_r2_cv=round(v.get("r2_cv", float("nan")), 4),
                       val_r2_still=round(v.get("r2_still", float("nan")), 4),
                       aot=round(float(np.nanmean(h[:, 1])), 4),
                       slots_used=round(float(h[:, 2].mean()), 2),
                       seg=round(float(h[:, 3].mean()), 4),
                       eps=round(eps, 3), n_train=len(tr),
                       min_per_1k=round((time.time() - t0)
                                        / max(step - step0, 1) * 1000 / 60, 2))
            R.log("wm3", **rec)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            bar.set_postfix(r2=rec["val_r2"], cv=rec["val_r2_cv"])
            vr = v.get("r2", float("nan"))
            if vr == vr and vr > best:
                best = vr
                torch.save(dict(model=net.state_dict(), step=step, cfg=cfg,
                                val=v), out / "best_val.pt")
                R.log("wm3_promote", run=run_id, step=step, **v)
        if step % ckpt_every == 0:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(),
                            step=step, cfg=cfg), out / f"ckpt_{step:07d}.pt")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rcasa_v1,arctic_v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--head", default="fdnn", choices=["fdnn", "mlp"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.003)
    ap.add_argument("--no-decay", dest="decay", action="store_false",
                    help="constant LR (measured to collapse)")
    a = ap.parse_args()
    print(train(a.datasets, a.steps, a.run, head=a.head, d=a.d,
                blocks=a.blocks, heads=a.heads, modes=a.modes,
                noise=a.noise, decay=a.decay))
