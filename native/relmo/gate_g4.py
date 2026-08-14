"""G4 ROLLOUT — does the model beat the trivial baselines, per horizon?

WM_PLAN §5: "multi-step 3D trajectory R2 > 0 and rising with capacity,
beaten against const-velocity AND predict-stillness at EVERY horizon
step, not just the mean."

Why this exists separately from the trainer's inline eval: the inline
eval scores a sample, and a sample is not enough to resolve the question.
MEASURED - across 65 evals of relmowm2_fdnn the const-velocity baseline
computed on those 48 windows had sd 0.51 (min -0.87, max +0.93) while its
true pooled value on val is +0.585, and one eval reported the model at
+0.5038 when a 200-window score of the SAME checkpoint was +0.082. So
this scores a fixed checkpoint over many more windows with the baselines
computed ON THE SAME WINDOWS - a paired comparison.

bs=1 ON PURPOSE. The moving-point threshold is computed over whatever is
in the batch, so bs>1 silently becomes the per-BATCH protocol, measured
to flatter the model (run A: 0.208 vs const-vel 0.204 at bs=2, but 0.082
vs 0.585 at bs=1). bs=1 makes it per-episode, matching G0 and ceiling.py.

Reports per-horizon and pooled, because const-acceleration was measured
to be the BEST predictor at h=1 (+0.914) and pooled -0.363: a mean over
horizons hides exactly the failure mode this gate is meant to catch.

Architecture-agnostic: it asks whichever model the run's config names for
the SAME quantity - displacement from the last context frame - so v2's
per-slot SE(3) rollout and v3's per-point direct decode are comparable.
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
from relmo.train_wm2 import HZ, TC, batch  # noqa: E402
from relmo.wm2 import RelMoWM2, canon  # noqa: E402
from relmo.wm3 import RelMoWM3  # noqa: E402


def load_model(out: Path, ckpt: Path, device: str):
    sd = torch.load(ckpt, map_location=device)
    cfg = json.loads((out / "config.json").read_text())
    if cfg.get("model") == "RelMoWM3":
        net = RelMoWM3(d=cfg["d"], blocks=cfg["blocks"], heads=cfg["heads"],
                       modes=cfg["modes"], hz=cfg["hz"],
                       head=cfg["head"]).to(device).eval()
    else:
        # run A predates the readout flag and correctly defaults to "mean"
        net = RelMoWM2(head=cfg.get("head", "fdnn"),
                       readout=cfg.get("readout", "mean")).to(device).eval()
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    return net, cfg, int(sd.get("step", -1))


def predict_disp(net, cfg, Xn, V, base):
    """Displacement (B,H,P,3) from either architecture.

    v3 emits M modes; the honest single prediction is the mode the model
    itself selects with its own logit, never the oracle-best one."""
    if cfg.get("model") == "RelMoWM3":
        o = net(Xn, V, TC, noise=0.0)
        sel = o["logit"].argmax(1)
        return o["disp"][torch.arange(len(sel)), sel]
    o = net.rollout(Xn, V, TC, HZ)
    return o["pred"] - base[:, None]


def run(run_id="relmowm3_fdnn", dataset="rcasa", n=300, ckpt=None,
        device="cpu", seed=1, bs=1, split="val"):
    out = R.model_dir(run_id)
    p = Path(ckpt) if ckpt else (out / "best_val.pt")
    if not p.exists():
        cks = sorted(out.glob("ckpt_*.pt"))
        if not cks:
            return {"error": f"no checkpoint in {out}"}
        p = cks[-1]
    net, cfg, step = load_model(out, p, device)

    # A held-out set that is never scored is not held out, it is unused.
    # VAL here is 12 episodes over 7 of 12 tasks, so a verdict resting on
    # it alone is thin; TEST and the out-of-domain rcasa_eval corpus are
    # the checks that the number is not a split artefact.
    part = SP.partition(sorted((R.TRACKS / dataset).glob("*.npz")), dataset)
    key = {"val": SP.VAL, "test": SP.TEST, "train": SP.TRAIN}.get(split)
    files = part[key] if key is not None else sorted(
        (R.TRACKS / dataset).glob("*.npz"))
    if not files:
        return {"error": f"dataset {dataset!r} has no {split.upper()} tracks "
                         f"under {R.TRACKS / dataset} - nothing to score"}
    rng = np.random.default_rng(seed)
    names = ["model", "const_vel", "stillness"]
    sse = {k: np.zeros(HZ) for k in names}
    sst = np.zeros(HZ)
    used = 0
    for _ in range(n // bs):
        bt = batch(files, rng, bs)
        if bt is None:
            continue
        X, V = bt[0].to(device), bt[1].to(device)
        Xn, _, _ = canon(X, V, TC)
        base = Xn[:, TC - 1]
        with torch.no_grad():
            pdisp = predict_disp(net, cfg, Xn, V, base)
        tgt = Xn[:, TC:TC + HZ]
        m = V[:, TC:TC + HZ] & V[:, TC - 1][:, None]
        disp = (tgt - base[:, None]).cpu().numpy()
        mag = np.linalg.norm(disp, axis=-1)
        mnp = m.cpu().numpy()
        if not mnp.any():
            continue
        thr = max(mag[mnp].mean(), 1e-6) * 0.5
        mv = mnp & (mag > thr)
        if not mv.any():
            continue
        used += bs
        pm = pdisp.cpu().numpy()
        v1 = (Xn[:, TC - 1] - Xn[:, TC - 2]).cpu().numpy()
        # A residual-target run predicts (displacement - const_vel), so the
        # baseline has to be added back before scoring or the gate would
        # measure the correction alone against the full displacement and
        # report a spurious ~0. The config records which target was used.
        if cfg.get("target") == "const_vel_residual":
            pm = pm + v1[:, None] * np.arange(
                1, HZ + 1, dtype=np.float32).reshape(1, -1, 1, 1)
        for h in range(HZ):
            sel = mv[:, h]
            if not sel.any():
                continue
            d = disp[:, h][sel]
            sst[h] += (d ** 2).sum()
            preds = dict(model=pm[:, h][sel],
                         const_vel=(v1 * (h + 1))[sel],
                         stillness=np.zeros_like(d))
            for nm, q in preds.items():
                sse[nm][h] += ((q - d) ** 2).sum()

    # No scored window means sst is all zeros, and 1 - 0/1e-12 is 1.0 -
    # the gate would report a PERFECT r2 at every horizon off zero data.
    # Refuse instead. This is the same shape as tracks2 reporting 600/600
    # having written nothing.
    if used == 0 or sst.sum() <= 0:
        return {"error": f"no scorable windows in {dataset!r} {split.upper()} "
                         f"({len(files)} files, {n} draws) - refusing to "
                         f"report r2 computed from an empty sum"}
    r2 = {k: [round(float(1 - sse[k][h] / max(sst[h], 1e-12)), 4)
              for h in range(HZ)] for k in names}
    pooled = {k: round(float(1 - sse[k].sum() / max(sst.sum(), 1e-12)), 4)
              for k in names}
    beats = [bool(r2["model"][h] > r2["const_vel"][h]) for h in range(HZ)]
    rep = dict(gate="G4", run=run_id, step=step, ckpt=p.name,
               arch=cfg.get("model", "RelMoWM2"), dataset=dataset,
               split=split, windows=used, r2=r2, pooled=pooled,
               beats_const_vel_per_h=beats,
               n_horizons_beaten=int(sum(beats)),
               passed=bool(all(beats) and all(x > 0 for x in r2["model"])))
    R.log("gate_g4", **rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="relmowm3_fdnn")
    # rcasa, not rcasa_v1: that corpus was deleted as unusable, so the
    # old default named a directory with zero tracks and the gate died
    # inside numpy with "a cannot be empty" rather than saying so.
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="val",
                    choices=["val", "test", "train", "all"])
    a = ap.parse_args()
    rep = run(a.run, a.dataset, a.n, a.ckpt, a.device, split=a.split)
    if "error" in rep:
        print(rep)
        raise SystemExit(1)
    print(f"G4  {rep['run']} [{rep['arch']}] @ step {rep['step']} "
          f"({rep['ckpt']})  {rep['windows']} {rep['split']} windows\n")
    print(f"{'predictor':12s}" + "".join(f"   h={h}" for h in range(1, HZ + 1))
          + "  POOLED")
    for k in ("model", "const_vel", "stillness"):
        print(f"{k:12s}" + "".join(f"{x:+7.3f}" for x in rep["r2"][k])
              + f" {rep['pooled'][k]:+7.3f}")
    print(f"\nbeats const-vel at {rep['n_horizons_beaten']}/{HZ} horizons"
          f"   PASSED={rep['passed']}")
