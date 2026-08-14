"""RelMo-WM v2 trainer — 3D rollout, gated on trivial baselines.

THE RULE THIS FILE EXISTS TO ENFORCE: every reported number is quoted
beside the baseline it must beat. This project already spent 12k steps
on a model whose rollout error TIED "predict nothing moves" at 1.00x
while the loss curve looked healthy. So each eval reports

    model      vs   const-velocity   vs   predict-stillness

on the SAME masked points, and selection is on motion-R2 (the fraction
of actual motion explained), never on raw loss.

    python -m relmo.daemon train_wm2 --datasets rcasa_v1 --head fdnn
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
from relmo import splits as SP  # noqa: E402
from relmo.device import DEVICE  # noqa: E402
from relmo.wm2 import RelMoWM2, canon, kabsch  # noqa: E402

DEV = DEVICE
TC, HZ, PMAX = 16, 8, 192


from relmo.lift import MODES, lift3d, vis_of  # noqa: E402,F401

# WHERE THE 3D COMES FROM. "gt" is the original privileged path (gxy +
# gdist = the answer, handed to the model as input); "video" builds the
# same 3D from CoTracker xy and depth predicted from one RGB frame, and
# is the only serve-legal option. Kept as a flag rather than a swap so
# the two can be scored against each other on identical everything else
# — the delta IS the cost of being video-only, and nothing in this
# project had measured it. Default stays "gt" so no existing number
# silently changes meaning; runs must opt in with --lift video.
LIFT = "gt"


_WARNED_UNIFORM = False


def load(f, rng):
    z = np.load(f)
    from relmo import lift as LF
    if not LF.have(z, LIFT):
        return None
    # SERVE-TIME RULE, enforced here rather than trusted. cam_fovy is a
    # per-episode simulator value (45/60/75 deg across this corpus), so a
    # video-only run must not read it - it would be the same class of
    # leak as gdist. One nominal calibration stands in, which is what a
    # deployed camera gives you. Measured cost on the relational probe:
    # +0.021 [+0.004,+0.038] for articulation (i.e. nominal is slightly
    # BETTER), -0.012 [-0.021,-0.001] for contact change. Not load-bearing.
    fov = LF.NOMINAL_FOVY if LIFT in ("video", "flat") else None
    X = lift3d(z, LIFT, fixed_fov=fov)
    V = vis_of(z, LIFT)
    T, P = X.shape[0], X.shape[1]
    if T < TC + HZ:
        return None
    # SAMPLE FROM THE WINDOWS THAT CONTAIN MOTION, not uniformly.
    #
    # An atomic demo of ~200 frames holds only 3-4 windows in which the
    # manipulated object actually moves; the rest is the arm reaching.
    # A uniform t0 therefore draws a reach-only window most of the time,
    # and a reach is driven by the human's intent - unpredictable from
    # observation. That is precisely how rcasa_v1 made four runs score
    # zero, reproduced one level down on a corpus that is otherwise
    # correct. The replayer records window_starts for exactly this.
    ws = z["window_starts"] if "window_starts" in z.files else None
    if ws is not None and len(ws):
        ws = ws[ws + TC + HZ <= T]
    if ws is not None and len(ws):
        t0 = int(rng.choice(ws))
    else:
        # The fallback is legitimate for sources with no window record
        # (MOVi, arctic). It stayed silent once already, when tracks2 did
        # not copy window_starts into the track file and every rcasa
        # sample took this path unnoticed - harmlessly, as it turned out
        # (measured: uniform 0.0750 m median max-displacement vs 0.0722 m
        # for window_starts on that corpus), but only because the
        # replayer's gates happened to be strict. Announce it once per
        # process, naming the file, so a corpus that LOSES the field can
        # never again be mistaken for one that never had it.
        global _WARNED_UNIFORM
        if not _WARNED_UNIFORM:
            _WARNED_UNIFORM = True
            print(f"WARNING: no window_starts in {getattr(f, 'name', f)} - "
                  f"sampling t0 UNIFORMLY. Motion windows are not being "
                  f"targeted.", flush=True)
            R.log("wm_uniform_t0", file=str(f))
        t0 = int(rng.integers(0, T - TC - HZ + 1))
    X, V = X[t0:t0 + TC + HZ], V[t0:t0 + TC + HZ]
    gb = z["gbody"] if "gbody" in z.files else np.full(P, -1, np.int16)
    if P > PMAX:
        keep = rng.choice(P, PMAX, replace=False)
        X, V, gb = X[:, keep], V[:, keep], gb[keep]
    return X, V, gb


def batch(files, rng, bs):
    out = [load(f, rng) for f in rng.choice(files, bs, replace=False)]
    out = [o for o in out if o is not None]
    if not out:
        return None
    P = max(o[0].shape[1] for o in out)
    T = out[0][0].shape[0]
    X = np.zeros((len(out), T, P, 3), np.float32)
    V = np.zeros((len(out), T, P), bool)
    G = np.full((len(out), P), -1, np.int64)
    for b, (x, v, g) in enumerate(out):
        X[b, :, :x.shape[1]] = x
        V[b, :, :x.shape[1]] = v
        G[b, :len(g)] = g
    return (torch.tensor(X), torch.tensor(V), torch.tensor(G))


def r2_parts(pred, tgt, mask, base):
    """Raw (sse, sst) so callers can POOL across the eval set.

    Averaging per-batch R2 is averaging RATIOS: measured on real rcasa
    batches the const-velocity baseline swings +0.994 to -7.858 batch
    to batch while its POOLED value is +0.89. Selecting checkpoints on
    an averaged ratio would pick by luck, so evaluate() pools."""
    p = (pred - base)[mask]
    t = (tgt - base)[mask]
    return float(((p - t) ** 2).sum()), float((t ** 2).sum())


def motion_r2(pred, tgt, mask, base):
    """R2 over DISPLACEMENT from the last context frame.

    The denominator must be ||tgt - base||, NOT ||tgt||. With absolute
    positions in the denominator, "predict stillness" scored +0.791
    here - because a future position really is close to the current
    one - which would have flattered every model by ~0.8 and hidden a
    model that learned nothing. Measured during the smoke test; it is
    the same denominator error that cost this project 12k steps.
    With displacement, stillness scores EXACTLY 0 by construction and
    the number means "fraction of the actual motion explained"."""
    p = (pred - base)[mask]
    t = (tgt - base)[mask]
    sse = ((p - t) ** 2).sum()
    sst = (t ** 2).sum().clamp(min=1e-9)
    return float(1.0 - sse / sst)


def step_loss(net, X, V, G):
    X, V, G = X.to(DEV), V.to(DEV), G.to(DEV)
    Xn, _, _ = canon(X, V, TC)
    o = net.rollout(Xn, V, TC, HZ)
    base = Xn[:, TC - 1]
    tgt = Xn[:, TC:TC + HZ]                              # (B,H,P,3)
    m = V[:, TC:TC + HZ] & V[:, TC - 1][:, None]
    disp = tgt - base[:, None]
    mag = disp.norm(dim=-1)
    if not m.any():
        return None
    # PER-EPISODE threshold. This used to be one threshold over the whole
    # batch, which silently changed WHICH points get scored: pairing a
    # slow episode with a fast one raises the slow one's bar. Measured on
    # the same checkpoint (relmowm2_fdnn @ 15500, val): batch threshold
    # -> model 0.208 vs const-vel 0.204 (looks level); per-episode ->
    # model 0.082 vs const-vel 0.585 (beaten 0/8). G0 and gate_g4 use
    # per-episode, so the trainer was reporting a number its own gate
    # does not recognise.
    msum = m.flatten(1).sum(1).clamp(min=1)
    thr = ((mag * m).flatten(1).sum(1) / msum).clamp(min=1e-5) * 0.5
    mv = (m & (mag > thr[:, None, None]))[..., None].expand_as(tgt)
    if not mv.any():
        return None
    roll = F.smooth_l1_loss(o["pred"][mv], tgt[mv], beta=0.02)
    b4 = base[:, None].expand_as(tgt)
    r2 = motion_r2(o["pred"], tgt, mv, b4)
    # trivial baselines on the SAME masked points
    still = base[:, None].expand_as(tgt)
    v = Xn[:, TC - 1] - Xn[:, TC - 2]
    steps = torch.arange(1, HZ + 1, device=DEV, dtype=torch.float32)
    cv = base[:, None] + v[:, None] * steps[None, :, None, None]
    r2_still = motion_r2(still, tgt, mv, b4)
    r2_cv = motion_r2(cv, tgt, mv, b4)
    # arrow of time
    hf = net.encode(Xn[:, :TC], V[:, :TC])
    hb = net.encode(torch.flip(Xn[:, :TC], (1,)), torch.flip(V[:, :TC], (1,)))
    lg = torch.cat([net.arrow(hf), net.arrow(hb)])
    lab = torch.cat([torch.ones(len(X), device=DEV),
                     torch.zeros(len(X), device=DEV)])
    aot = F.binary_cross_entropy_with_logits(lg, lab)
    acc = float(((lg > 0).float() == lab).float().mean())
    # part supervision: points on the same body should share a slot
    attn = o["attn"]                                     # (B,K,P)
    seg = torch.zeros((), device=DEV)
    if (G >= 0).any():
        a = attn.transpose(1, 2)                          # (B,P,K)
        same = (G[:, :, None] == G[:, None, :]) & (G[:, :, None] >= 0)
        sim = torch.einsum("bpk,bqk->bpq", a, a)
        # CLASS-BALANCED pairwise BCE. The previous form divided the SUM
        # over all B*P^2 pairs by the count of SAME pairs and then again
        # by P, which shrank the term 38.9x below its mean-BCE value
        # (measured: 0.0134 vs 0.5225) and left it contributing 0.6% of
        # the total gradient. Meanwhile the mean BCE of 0.52 says the
        # binder sits near chance, and G2 measured ARI 0.235 against an
        # 0.857 ceiling. Dividing by pair COUNT per class also fixes the
        # imbalance properly: same-body pairs are a small minority, so a
        # plain mean is dominated by "these two points differ".
        pos = same.float()
        neg = 1.0 - pos
        sc = sim.clamp(1e-4, 1 - 1e-4)
        seg = 0.5 * ((-torch.log(sc) * pos).sum() / pos.sum().clamp(min=1)
                     + (-torch.log(1 - sc) * neg).sum()
                     / neg.sum().clamp(min=1))
    used = float((attn.mean(-1) > 0.05).float().sum(-1).mean())
    parts = dict(m=r2_parts(o["pred"], tgt, mv, b4),
                 cv=r2_parts(cv, tgt, mv, b4),
                 st=r2_parts(still, tgt, mv, b4))
    return dict(roll=roll, aot=aot, seg=seg, r2=r2, r2_cv=r2_cv,
                r2_still=r2_still, acc=acc, used=used, parts=parts)


def evaluate(net, files, rng, n=192, bs=4):
    # n was 48. Measured across 65 evals of run A: the const-velocity
    # baseline computed on those 48 windows had sd 0.51 while its true
    # pooled value is ~0.585 - so single evals landed anywhere from
    # -0.87 to +0.93, and one reported the model at +0.5038 when a
    # 200-window score of the same checkpoint was +0.082. Checkpoint
    # promotion selects on this number, so a noisy eval promotes by luck.
    """POOLED motion-R2 over the eval set, plus both trivial baselines
    on exactly the same masked points."""
    net.eval()
    pool = {k: [0.0, 0.0] for k in ("m", "cv", "st")}
    accs = []
    with torch.no_grad():
        for _ in range(max(n // bs, 1)):
            bt = batch(files, rng, bs)
            if bt is None:
                continue
            o = step_loss(net, *bt)
            if o is None:
                continue
            for k, (sse, sst) in o["parts"].items():
                pool[k][0] += sse
                pool[k][1] += sst
            accs.append(o["acc"])
    net.train()
    out = {n_: (1.0 - pool[k][0] / max(pool[k][1], 1e-9))
           for k, n_ in (("m", "r2"), ("cv", "r2_cv"), ("st", "r2_still"))}
    out["acc"] = float(np.mean(accs)) if accs else float("nan")
    return out


def train(datasets="rcasa_v1", steps=60000, run_id=None, bs=4, lr=3e-4,
          head="fdnn", log_every=250, ckpt_every=2500, readout="mean",
          lift="gt"):
    from tqdm import tqdm
    global LIFT
    LIFT = lift
    dss = [d.strip() for d in datasets.split(",") if d.strip()]
    # NOT "wm2_{head}": that collides with the PAUSED v1 run dirs
    # (wm2_fdnn / wm2_mlp from train_wm.py). The trainer resumes from
    # the newest ckpt in its dir, so the collision made it load a v1
    # checkpoint into a v2 model - size mismatch inp.weight 128x5 vs
    # 128x7 - and training never started. Verified by checking the log,
    # not the launch message.
    # The lift is IN THE RUN ID. A privileged-input run and a video-only
    # run differ only by this flag and produce numbers that are not
    # comparable; sharing a directory would let one resume from the
    # other's checkpoint and read as the wrong thing (the same class of
    # collision that once loaded a v1 checkpoint into a v2 model).
    run_id = run_id or (f"relmowm2_{head}" if lift == "gt"
                        else f"relmowm2_{head}_{lift}")
    out = R.model_dir(run_id)
    cfg = dict(datasets=dss, head=head, lr=lr, bs=bs, tc=TC, hz=HZ,
               pmax=PMAX, model="RelMoWM2", d=128, slots=8,
               readout=readout, eval_n=192, thr="per_episode", lift=lift,
               objective="3D rollout + arrow + part-binding")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    net = RelMoWM2(head=head, readout=readout).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    step0 = 0
    cks = sorted(out.glob("ckpt_*.pt"))
    if cks:
        sd = torch.load(cks[-1], map_location=DEV)
        net.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step0 = sd["step"]
    R.log("wm2_start", run=run_id, step=step0, config=cfg)
    tr, va = [], []
    for ds in dss:
        p = SP.partition(sorted((R.TRACKS / ds).glob("*.npz")), ds)
        tr += p[SP.TRAIN]
        va += p[SP.VAL]
    if len(tr) < bs:
        R.log("wm2_no_data", have=len(tr))
        return run_id
    rng = np.random.default_rng(0)
    hist, best = [], -float("inf")
    t0 = time.time()
    bar = tqdm(range(step0 + 1, steps + 1), initial=step0, total=steps,
               unit="step", desc=f"wm2/{head}")
    for step in bar:
        bt = batch(tr, rng, bs)
        if bt is None:
            continue
        o = step_loss(net, *bt)
        if o is None:
            continue
        loss = o["roll"] + 0.2 * o["aot"] + 0.5 * o["seg"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        hist.append((float(o["roll"]), o["r2"], o["r2_cv"], o["r2_still"],
                     o["acc"], o["used"], float(o["seg"])))
        if step % log_every == 0:
            h = np.array(hist[-log_every:])
            v = evaluate(net, va, rng) if va else {}
            rec = dict(run=run_id, step=step,
                       train_r2=round(float(h[:, 1].mean()), 4),
                       train_r2_cv=round(float(h[:, 2].mean()), 4),
                       train_r2_still=round(float(h[:, 3].mean()), 4),
                       val_r2=round(v.get("r2", float("nan")), 4),
                       val_r2_cv=round(v.get("r2_cv", float("nan")), 4),
                       val_r2_still=round(v.get("r2_still", float("nan")), 4),
                       aot=round(float(h[:, 4].mean()), 4),
                       slots_used=round(float(h[:, 5].mean()), 2),
                       seg=round(float(h[:, 6].mean()), 4),
                       n_train=len(tr),
                       min_per_1k=round((time.time() - t0)
                                        / max(step - step0, 1) * 1000 / 60, 2))
            R.log("wm2", **rec)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            bar.set_postfix(r2=rec["val_r2"], cv=rec["val_r2_cv"])
            vr = v.get("r2", float("nan"))
            if vr == vr and vr > best:
                best = vr
                torch.save(dict(model=net.state_dict(), step=step, cfg=cfg,
                                val=v), out / "best_val.pt")
                R.log("wm2_promote", run=run_id, step=step, **v)
        if step % ckpt_every == 0:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(),
                            step=step, cfg=cfg), out / f"ckpt_{step:07d}.pt")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rcasa_v1")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--head", default="fdnn", choices=["fdnn", "mlp"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--readout", default="mean",
                    choices=["mean", "last+mean"])
    ap.add_argument("--lift", default="gt", choices=list(MODES),
                    help="gt = privileged gxy+gdist (old); video = "
                         "CoTracker xy + predicted depth (serve-legal)")
    a = ap.parse_args()
    print(train(a.datasets, a.steps, a.run, head=a.head,
                readout=a.readout, lift=a.lift))
