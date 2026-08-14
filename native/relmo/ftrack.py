"""Finetune the point tracker on pixel-exact sim supervision.

WHY (owner, 2026-08-11): "validate and finetune the co-tracker too.
Cuz on real data there is no sim state." The tracker is the ONE
component that must work without privileged information at eval, so
it is the one component worth spending simulator ground truth on.

What the measurement says we are fixing - trackval on 400 textured
episodes, stock cotracker3_offline:
    moving bodies    EPE med 1.43px   p90 7.14px   ~23% of points jump
    resting bodies   EPE med 0.52px
    background       EPE med 0.45px
The centre is healthy; the defect is the TAIL - a point passes behind
an occluder and the tracker re-attaches to the occluder. So the
objective weights occlusion transitions, and the gate protects the
part that already works.

Scope, forced by measurement not by taste:
  fnet FROZEN      MPS has no grid_sampler_3d_backward, so gradients
                   cannot flow into the feature encoder here at all.
                   Convenient: it also means the visual features that
                   generalise to real video are untouched, and only
                   the tracking dynamics adapt. 22.7M of 25.4M
                   parameters remain trainable.
  low LR, gated    a sim-only finetune can trade real-world skill for
                   sim skill and nothing in this repo can measure real
                   -world regression. Mitigations: small LR, frozen
                   features, and a HARD gate - promote only if moving
                   EPE improves AND background/resting do not degrade.

Train/serve parity: the deployed predictor resizes 480x640 -> 384x512
and rescales queries. Training does the same, in the same order, or
the finetune would optimise a resolution the tracker never sees.

    python -m relmo.daemon ftrack --steps 4000
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
from relmo.pixelgt import body_speed  # noqa: E402
from relmo.tracks2 import frames as decode_frames  # noqa: E402

DEV = "mps"
H0, W0 = 480, 640
MRES = (384, 512)                 # the model's own resolution
T_WIN = 24                        # frames per training sample
N_PTS = 192                       # query points per sample
GAMMA = 0.8                       # iterative-refinement loss decay


def _scale_q(xy):
    """original pixels -> model-resolution pixels (predictor recipe)."""
    s = xy.new_tensor([(MRES[1] - 1) / (W0 - 1), (MRES[0] - 1) / (H0 - 1)])
    return xy * s


def prep_video(F_np):
    v = torch.tensor(F_np).permute(0, 3, 1, 2).float()
    v = F.interpolate(v, MRES, mode="bilinear", align_corners=True)
    return v[None].to(DEV)


class Sampler:
    """One training sample = one window of one episode.

    Query points are drawn from the SAME cached seeds the deployed
    pipeline used, and biased toward points that actually get
    occluded - that is where the measured error lives, and an
    unbiased sample would spend most of its gradient on the 1.4px
    case that already works."""

    def __init__(self, dataset="physgen_v3", split=SP.TRAIN):
        d = R.TRACKS / dataset
        self.files = SP.partition(sorted(d.glob("*.npz")), dataset)[split]
        self.dataset = dataset
        self.dir = R.dataset_dir(dataset)
        man = R.read_manifest(dataset)
        self.shard = {e["id"]: e["shard"] for e in man["episodes"]}

    def ep_video(self, stem):
        return decode_frames(self.dir / self.shard[stem] / stem
                             / "frames.mp4")

    def sample(self, rng):
        f = self.files[int(rng.integers(len(self.files)))]
        z = np.load(f)
        g, gv = z["gxy"].astype(np.float32), z["gvis"]
        T = g.shape[0]
        if T < T_WIN + 1:
            return None
        t0 = int(rng.integers(0, T - T_WIN + 1))
        gw, gvw = g[t0:t0 + T_WIN], gv[t0:t0 + T_WIN]
        ok = gvw[0] & (gvw.sum(0) >= 4)
        idx = np.where(ok)[0]
        if len(idx) < 16:
            return None
        # occlusion transitions: the measured failure mode
        occl = (gvw[:-1, idx] != gvw[1:, idx]).any(0)
        hard, easy = idx[occl], idx[~occl]
        n_hard = min(len(hard), N_PTS // 2)
        n_easy = min(len(easy), N_PTS - n_hard)
        sel = np.concatenate([
            rng.choice(hard, n_hard, replace=False) if n_hard else
            np.zeros(0, int),
            rng.choice(easy, n_easy, replace=False) if n_easy else
            np.zeros(0, int)]).astype(int)
        if len(sel) < 16:
            return None
        V = self.ep_video(f.stem)[t0:t0 + T_WIN]
        return (V, gw[:, sel], gvw[:, sel],
                float(occl.mean()))


def seq_loss(td, tgt, vis, valid):
    """CoTracker's own recipe: L1 over refinement iterations with a
    geometric weight, plus visibility BCE. Position error is only
    defined where the point is genuinely visible."""
    coords, viss = td[0][0], td[1][0]
    n = len(coords)
    m = (vis & valid)[..., None]
    lp = torch.zeros((), device=DEV)
    for i, c in enumerate(coords):
        w = GAMMA ** (n - i - 1)
        d = ((c - tgt).abs() * m).sum() / m.sum().clamp(min=1) / 2
        lp = lp + w * d
    lv = torch.zeros((), device=DEV)
    for i, v in enumerate(viss):
        w = GAMMA ** (n - i - 1)
        lv = lv + w * F.binary_cross_entropy(
            v.clamp(1e-4, 1 - 1e-4), vis.float())
    return lp, lv


@torch.no_grad()
def validate(pred, dataset="physgen_v3", n_ep=10, seed=0,
             split=SP.VAL):
    """Deployment path on held-out episodes, scored against GT.

    Compares like for like: the cache already stores the FROZEN
    tracker's output for the same queries, so the baseline needs no
    recomputation and cannot drift.

    split=VAL drives promotion during training. split=TEST is the
    FINAL read - 215 episodes no trainer and no gate has ever seen -
    and must be run exactly once, after selection is over, or it
    stops being a test set."""
    d = R.TRACKS / dataset
    files = SP.partition(sorted(d.glob("*.npz")), dataset)[split]
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.choice(len(files),
                                          min(n_ep, len(files)),
                                          replace=False)]
    man = R.read_manifest(dataset)
    shard = {e["id"]: e["shard"] for e in man["episodes"]}
    acc = {k: [] for k in ("ft_mov", "fz_mov", "ft_rest", "fz_rest",
                           "ft_bg", "fz_bg", "ft_jump", "fz_jump")}
    for f in files:
        z = np.load(f)
        g, gv = z["gxy"].astype(np.float32), z["gvis"]
        xy0 = z["xy"].astype(np.float32)
        ident = z["ident"]
        V = decode_frames(R.dataset_dir(dataset) / shard[f.stem]
                          / f.stem / "frames.mp4")
        vid = torch.tensor(V).permute(0, 3, 1, 2)[None].float().to(DEV)
        q = torch.tensor(np.concatenate(
            [np.zeros((len(xy0[0]), 1), np.float32), xy0[0]], 1))[None]
        tr, _ = pred(vid, queries=q.to(DEV))
        ft = tr[0].cpu().numpy()
        T = min(len(ft), len(g))
        sp = body_speed(R.dataset_dir(dataset) / shard[f.stem]
                        / f.stem, T)
        for p in range(g.shape[1]):
            m = gv[:T, p]
            if m.sum() < 8:
                continue
            b = int(ident[p])          # ident IS the segment value
            moving = (0 < b < sp.shape[1]
                      and sp[:T, b].max() > 0.02)
            cls = "mov" if moving else ("rest" if b > 0 else "bg")
            for tag, arr in (("ft", ft), ("fz", xy0)):
                e = np.linalg.norm(arr[:T, p] - g[:T, p], axis=-1)[m]
                acc[f"{tag}_{cls}"].append(float(np.median(e)))
                if cls == "mov":
                    acc[f"{tag}_jump"].append(float((e > 4).mean()))
        del vid, tr
        torch.mps.empty_cache()
    out = {k: (round(float(np.median(v)), 4) if v else None)
           for k, v in acc.items()}
    out["jump_ft"] = (round(float(np.mean(acc["ft_jump"])), 4)
                      if acc["ft_jump"] else None)
    out["jump_fz"] = (round(float(np.mean(acc["fz_jump"])), 4)
                      if acc["fz_jump"] else None)
    return out


def promote(v):
    """The gate. Improve the tail without paying for it anywhere
    else - a tracker that wins on moving points by smearing the
    static field is not an improvement, it is a different bug."""
    if v["ft_mov"] is None or v["fz_mov"] is None:
        return False, "no data"
    better = v["ft_mov"] < v["fz_mov"] * 0.97
    keep_bg = v["ft_bg"] <= v["fz_bg"] * 1.10 + 0.05
    keep_rest = (v["ft_rest"] is None or v["fz_rest"] is None
                 or v["ft_rest"] <= v["fz_rest"] * 1.10 + 0.05)
    ok = bool(better and keep_bg and keep_rest)
    why = (f"moving {v['fz_mov']}->{v['ft_mov']} "
           f"bg {v['fz_bg']}->{v['ft_bg']} "
           f"rest {v['fz_rest']}->{v['ft_rest']} "
           f"jump {v['jump_fz']}->{v['jump_ft']}")
    return ok, why


def train(dataset="physgen_v3", steps=4000, lr=1e-5, run_id="ftrack_v1",
          log_every=100, val_every=500, accum=4):
    from tqdm import tqdm
    out = R.model_dir(run_id)
    pred = torch.hub.load("facebookresearch/co-tracker",
                          "cotracker3_offline").to(DEV)
    m = pred.model
    for p in m.fnet.parameters():
        p.requires_grad = False
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    step0 = 0
    cks = sorted(out.glob("ckpt_*.pt"))
    if cks:
        sd = torch.load(cks[-1], map_location=DEV)
        m.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step0 = sd["step"]
    cfg = dict(dataset=dataset, lr=lr, steps=steps, t_win=T_WIN,
               n_pts=N_PTS, accum=accum, frozen="fnet",
               base="cotracker3_offline", mres=list(MRES))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    R.log("ftrack_start", run=run_id, step=step0, config=cfg)
    smp = Sampler(dataset)
    rng = np.random.default_rng(0)
    hist, best = [], None
    bar = tqdm(range(step0 + 1, steps + 1), initial=step0, total=steps,
               unit="step", desc="ftrack")
    opt.zero_grad(set_to_none=True)
    for step in bar:
        s = smp.sample(rng)
        if s is None:
            continue
        V, g, gv, hard = s
        vid = prep_video(V)
        tgt = _scale_q(torch.tensor(g).to(DEV))[None]
        vis = torch.tensor(gv).to(DEV)[None]
        q = torch.cat([torch.zeros(len(g[0]), 1, device=DEV),
                       _scale_q(torch.tensor(g[0]).to(DEV))], 1)[None]
        _, _, _, td = m(vid, q, iters=4, is_train=True)
        valid = td[3].bool()
        lp, lv = seq_loss(td, tgt, vis, valid)
        loss = (lp + 0.5 * lv) / accum
        loss.backward()
        if step % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        hist.append((float(lp), float(lv), hard))
        del vid, tgt, vis, q, td, loss
        if step % 20 == 0:
            torch.mps.empty_cache()
        if step % log_every == 0:
            h = np.array(hist[-log_every:])
            rec = dict(run=run_id, step=step,
                       pos_l1=round(float(h[:, 0].mean()), 4),
                       vis_bce=round(float(h[:, 1].mean()), 4),
                       occl_frac=round(float(h[:, 2].mean()), 3))
            bar.set_postfix(pos=rec["pos_l1"], vis=rec["vis_bce"])
            R.log("ftrack", **rec)
            with open(out / "metrics.jsonl", "a") as f_:
                f_.write(json.dumps(rec) + "\n")
        if step % val_every == 0:
            m.eval()
            v = validate(pred, dataset)
            ok, why = promote(v)
            m.train()
            R.log("ftrack_val", run=run_id, step=step, promote=ok,
                  why=why, **v)
            with open(out / "evals.jsonl", "a") as f_:
                f_.write(json.dumps(dict(step=step, promote=ok,
                                         **v)) + "\n")
            if ok and (best is None or v["ft_mov"] < best):
                best = v["ft_mov"]
                torch.save(dict(model=m.state_dict(), step=step,
                                cfg=cfg, val=v),
                           out / "best_val.pt")
                R.log("ftrack_promote", run=run_id, step=step, **v)
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(),
                            step=step, cfg=cfg), out / f"ckpt_{step:07d}.pt")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="physgen_v3")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--run", default="ftrack_v1")
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--ckpt", default=None,
                    help="load finetuned weights before scoring")
    ap.add_argument("--split", default=SP.VAL,
                    choices=[SP.TRAIN, SP.VAL, SP.TEST])
    ap.add_argument("--n-ep", type=int, default=10)
    a = ap.parse_args()
    if a.val_only:
        pr = torch.hub.load("facebookresearch/co-tracker",
                            "cotracker3_offline").to(DEV)
        if a.ckpt:
            sd = torch.load(a.ckpt, map_location=DEV)
            pr.model.load_state_dict(sd["model"])
            print(f"loaded {a.ckpt} @ step {sd.get('step')}")
        pr.model.eval()
        v = validate(pr, a.dataset, n_ep=a.n_ep, split=a.split)
        ok, why = promote(v)
        R.log("ftrack_final", ckpt=a.ckpt, split=a.split,
              n_ep=a.n_ep, passes=ok, why=why, **v)
        print(json.dumps(dict(split=a.split, n_ep=a.n_ep,
                              passes_gate=ok, **v), indent=1))
    else:
        print(train(a.dataset, a.steps, a.lr, a.run))
