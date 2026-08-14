"""Attach PREDICTED depth to every tracked point — the video-only lift.

THE GATE THIS FILE EXISTS TO OPEN. train_wm2.lift3d builds its 3D from
gxy (GT projections of body-fixed points) + gdist (GT planar-z). Those
two ARE the answer: handed both, the model never has to recover geometry
from anything. At serve time neither exists. So until the 3D the model
consumes is built from video, every 3D number this project has produced
is privileged and its transfer to real footage is unmeasured.

This computes depth for every tracked point from ONE RGB FRAME using
relmo/depthnet.py (trained from scratch on this corpus, no pretrained
teacher — standing rule), and caches it into the track file so training
never re-decodes video.

TWO fields are written, deliberately, because they decompose the loss:

    ddist    sampled at xy  (CoTracker pixels)   -> depth error + tracker error
    ddist_g  sampled at gxy (GT pixels)          -> depth error ALONE

Without the second, a drop in a downstream score cannot be attributed;
with it, the tracker's contribution is the difference between the two.
ddist_g is a DIAGNOSTIC, not a serve-time field — it needs gxy.

CONVENTION: depthnet regresses standardised log of the rendered depth
buffer, and pixelgt's `dist` is `-z_cam` read against that same buffer
(relmo/depthcheck.py settled planar-z vs ray-length by mj_ray). So
exp(p*sd + mu) is directly comparable to gdist, no rescaling.

SPLIT HONESTY: depthnet trained on TRAIN episodes only, so ddist on a
TRAIN episode is partly memorised and ddist on val/test/rcasa_eval is
not. Every downstream score must be read on held-out episodes; this
module prints the per-split depth accuracy so the difference is visible
rather than assumed.

    python -m relmo.preddepth --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402


def _frames(mp4, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def load_depthnet(device):
    """The trained encoder-decoder plus its target normalisation."""
    import torch
    from relmo.depthnet import build_net
    ck = R.model_dir("depthnet") / "depthnet.pt"
    if not ck.exists():
        raise SystemExit(f"no depthnet checkpoint at {ck} — run "
                         f"`python -m relmo.depthnet` first")
    sd = torch.load(ck, map_location=device)
    net = build_net(device)
    net.load_state_dict(sd["model"])
    net.eval()
    return net, float(sd["mu"]), float(sd["sd"]), sd.get("cfg", {})


def _bilinear(maps, xy, scale):
    """Sample (T,h,w) at (T,P,2) FULL-RES pixels. Bilinear, not nearest:
    the map is half-res, so nearest quantises depth to a 2 px grid and
    puts a step discontinuity across every object boundary — which is
    exactly where the tracked points sit."""
    T, h, w = maps.shape
    u = xy[..., 0] * scale - 0.5
    v = xy[..., 1] * scale - 0.5
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - u0, v - v0
    t = np.arange(T)[:, None]
    a = maps[t, v0, u0]
    b = maps[t, v0, u0 + 1]
    c = maps[t, v0 + 1, u0]
    d = maps[t, v0 + 1, u0 + 1]
    return ((a * (1 - fu) + b * fu) * (1 - fv)
            + (c * (1 - fu) + d * fu) * fv)


def predict_episode(net, mu, sd, F, xy, gxy, device, bs=16):
    """(T,P) predicted metric depth at xy and at gxy."""
    import torch
    T = len(F)
    outs = []
    with torch.no_grad():
        for k in range(0, T, bs):
            x = torch.tensor(F[k:k + bs]).permute(0, 3, 1, 2).float()
            x = x.div_(255.).to(device)
            outs.append(net(x).cpu().numpy())
    P = np.concatenate(outs).astype(np.float32)          # (T,h,w) standardised
    Lm = P * sd + mu                                     # log metres
    scale = P.shape[1] / F.shape[1]                      # half-res map
    lg = _bilinear(Lm, xy[:T].astype(np.float64), scale)
    lg_g = _bilinear(Lm, gxy[:T].astype(np.float64), scale)
    return np.exp(lg).astype(np.float32), np.exp(lg_g).astype(np.float32)


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum()
                 / (((y - y.mean()) ** 2).sum() + 1e-12))


def si_r2(p, y):
    """Scale-and-shift-invariant R2 in log space.

    A monocular predictor from a single view cannot fix absolute scale —
    that is the classical ambiguity, and it is why depthnet measured
    metric R2 +0.669 in-domain and -0.004 out-of-domain. Aligning one
    global (a, b) on log-depth before scoring separates "the shape of
    the scene is wrong" from "the whole scene is the wrong size", which
    are very different failures for a relational model: relations built
    from RATIOS survive the second, absolute distances do not."""
    # float64 throughout: in float32 the normal equations on ~50k points
    # overflowed and numpy warned "divide by zero / overflow in matmul"
    # while still returning a plausible-looking number. A believable
    # figure out of a diverged solve is this project's most expensive
    # recurring bug (an AUC 0.769 once came from one).
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    lp, ly = np.log(np.clip(p, 1e-3, 50)), np.log(np.clip(y, 1e-3, 50))
    A = np.stack([lp, np.ones_like(lp)], 1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    out = r2(A @ coef, ly)
    assert np.isfinite(out), "si_r2 solve did not produce a finite number"
    return out


def run(dataset="rcasa", force=False):
    import torch
    from tqdm import tqdm
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net, mu, sd, cfg = load_depthnet(dev)
    print(f"depthnet loaded on {dev}: {cfg}", flush=True)
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    which = {f.stem: nm for nm, key in (("train", SP.TRAIN), ("val", SP.VAL),
                                        ("test", SP.TEST))
             for f in part[key]}
    done = skipped = 0
    per_split = {}
    for f in tqdm(files, unit="ep", desc=f"preddepth/{dataset}"):
        z = np.load(f)
        e = by.get(f.stem)
        if e is None:
            skipped += 1
            continue
        if "ddist" in z.files and not force:
            # still score it, so the report covers the whole corpus
            g, dd = z["gdist"], z["ddist_g"]
            ok = z["gvis"] & (g > 0.05) & (g < 20)
            if ok.sum() > 100:
                s = per_split.setdefault(which.get(f.stem, "train"), [])
                s.append((r2(dd[ok], g[ok]), si_r2(dd[ok], g[ok])))
            skipped += 1
            continue
        mp4 = R.dataset_dir(dataset) / e["shard"] / f.stem / "frames.mp4"
        if not mp4.exists():
            skipped += 1
            continue
        W_, H_ = int(z["width"]), int(z["height"])
        F = _frames(mp4, W_, H_)
        T = min(len(F), len(z["xy"]))
        dd, ddg = predict_episode(net, mu, sd, F[:T], z["xy"], z["gxy"], dev)
        # pad to full track length if the mp4 is short of the track
        Tt = len(z["xy"])
        if T < Tt:
            dd = np.concatenate([dd, np.repeat(dd[-1:], Tt - T, 0)])
            ddg = np.concatenate([ddg, np.repeat(ddg[-1:], Tt - T, 0)])
        pay = {k: z[k] for k in z.files}
        pay["ddist"] = dd
        pay["ddist_g"] = ddg
        tmp = f.with_name(f".dp_{f.name}")
        np.savez_compressed(tmp, **pay)
        tmp.rename(f)
        g = z["gdist"]
        ok = z["gvis"] & (g > 0.05) & (g < 20)
        if ok.sum() > 100:
            s = per_split.setdefault(which.get(f.stem, "train"), [])
            s.append((r2(ddg[ok], g[ok]), si_r2(ddg[ok], g[ok])))
        done += 1
    rep = dict(dataset=dataset, files=len(files), added=done, skipped=skipped,
               device=dev)
    for k, v in sorted(per_split.items()):
        a = np.array(v)
        rep[f"metric_r2_{k}"] = round(float(a[:, 0].mean()), 4)
        rep[f"si_r2_{k}"] = round(float(a[:, 1].mean()), 4)
        rep[f"n_{k}"] = len(v)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rep = run(a.dataset, a.force)
    print("\n" + json.dumps(rep, indent=1))
    R.log("preddepth", **rep)
    files = sorted((R.TRACKS / a.dataset).glob("*.npz"))
    have = sum("ddist" in np.load(f).files for f in files)
    print(f"VERIFIED on disk: {have}/{len(files)} track files carry "
          f"predicted depth")
    print("NOTE: depthnet TRAINED on this dataset's train split — read the "
          "val/test rows, the train row is partly memorised.")
