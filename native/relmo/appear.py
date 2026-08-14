"""Attach APPEARANCE to every tracked point, from the video only.

Measured on this corpus: from 2D tracks alone, depth is recoverable at
R2 +0.650 when the camera moves (eye_in_hand) and only +0.130 when it
does not — and 85% of the corpus is static-camera, where the camera
translates 1.6 mm over an entire episode. With no parallax there is no
geometric depth cue, so a point's (u,v) trajectory cannot say how far
away it is. The +0.130 is mostly image-row position: memorising one
fixed viewpoint's perspective, not geometry.

The world model is supposed to predict the ground truth FROM VIDEO. It
never had to, because lift3d hands it gxy+gdist — the answer — as input.
Take that away and, with coordinates alone, it has nothing to work with
on 85% of the data.

What a static monocular view does carry is APPEARANCE: apparent size,
texture scale, shading, occlusion, the look of the surface. None of it
reaches the model today - the track files hold coordinates and sim state
and not one photometric value.

So this samples, per tracked point per frame, a small local descriptor
from the raw frame:

    rgb mean (3)      what the surface looks like
    rgb std  (3)      how textured it is
    grad mag (1)      edge energy — texture SCALE shrinks with distance
    laplacian(1)      focus/blur proxy

Deliberately NOT a pretrained encoder. The standing rule is no
pretrained teacher on the write or read path, so this is raw pixel
statistics only — computable on any video, no model, no labels.

Idempotent, and reads the mp4 that is already beside each episode, so no
re-tracking is needed.

    python -m relmo.appear --dataset rcasa
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

PATCH = 7          # odd; the local window sampled around each point
FIELDS = 8         # rgb mean(3) + rgb std(3) + grad(1) + lap(1)


def _frames(mp4, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def describe(F, xy, vis):
    """(T,P,8) local appearance at each tracked point, video only."""
    T, H, W, _ = F.shape
    P = xy.shape[1]
    r = PATCH // 2
    g = F.mean(-1)                                   # (T,H,W) grayscale
    gy, gx = np.gradient(g, axis=(1, 2))
    gm = np.hypot(gx, gy)
    lap = np.abs(np.gradient(gx, axis=2) + np.gradient(gy, axis=1))
    out = np.zeros((T, P, FIELDS), np.float32)
    Fp = F.astype(np.float32) / 255.0
    for t in range(T):
        u = np.clip(xy[t, :, 0].astype(int), r, W - r - 1)
        v = np.clip(xy[t, :, 1].astype(int), r, H - r - 1)
        # gather the PATCH x PATCH neighbourhood for every point at once
        du, dv = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        uu = (u[:, None] + du.ravel()[None]).clip(0, W - 1)
        vv = (v[:, None] + dv.ravel()[None]).clip(0, H - 1)
        pat = Fp[t][vv, uu]                          # (P, patch^2, 3)
        out[t, :, 0:3] = pat.mean(1)
        out[t, :, 3:6] = pat.std(1)
        out[t, :, 6] = gm[t][vv, uu].mean(1)
        out[t, :, 7] = lap[t][vv, uu].mean(1)
    out[~vis.astype(bool)] = 0.0
    return out


def run(dataset="rcasa"):
    from tqdm import tqdm
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    done = skipped = 0
    for f in tqdm(files, unit="ep", desc=f"appear/{dataset}"):
        z = np.load(f)
        if "appear" in z.files:
            skipped += 1
            continue
        e = by.get(f.stem)
        if e is None:
            skipped += 1
            continue
        mp4 = R.dataset_dir(dataset) / e["shard"] / f.stem / "frames.mp4"
        if not mp4.exists():
            skipped += 1
            continue
        W_, H_ = int(z["width"]), int(z["height"])
        F = _frames(mp4, W_, H_)
        T = min(len(F), len(z["xy"]))
        A = describe(F[:T], z["xy"][:T], z["vis"][:T])
        pay = {k: z[k] for k in z.files}
        pay["appear"] = A.astype(np.float16)
        tmp = f.with_name(f".ap_{f.name}")
        np.savez_compressed(tmp, **pay)
        tmp.rename(f)
        done += 1
    return dict(dataset=dataset, files=len(files), added=done,
                skipped=skipped, fields=FIELDS, patch=PATCH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    a = ap.parse_args()
    r = run(a.dataset)
    print("\n" + json.dumps(r, indent=1))
    R.log("appear", **r)
    files = sorted((R.TRACKS / a.dataset).glob("*.npz"))
    have = sum("appear" in np.load(f).files for f in files)
    print(f"VERIFIED on disk: {have}/{len(files)} track files carry appearance")
