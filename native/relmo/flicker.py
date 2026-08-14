"""Does the render FLICKER? The temporal test the F gate should have been.

gtcheck's F gate looks at one frame and calls a region suspect if it is
saturated AND locally flat, because that is what untextured collision
geometry looked like. On this corpus it fires on 33/448 episodes - and
every one inspected is a kitchen whose cabinets are painted a solid red
or orange. Flat and saturated is what paint looks like.

The bug F was built to catch was never really "saturated": it was
collision geoms z-fighting the visual mesh, so a surface that is not
moving changes colour between consecutive frames. That is a temporal
property, and paint does not have it. So:

  FLICKER   for pixels whose segmentation id is UNCHANGED between t and
            t+1 - i.e. the same body is still there, nothing moved into
            or out of the pixel - how often does the colour jump? A
            static surface under a static light should be near-identical.

  HUES      z-fighting showed magenta AND green AND blue at once, because
            MuJoCo assigns collision geoms arbitrary colours. A kitchen
            style is one or two dominant hues. So count how many distinct
            saturated hue clusters cover a meaningful area.

Run it over the F-failures and a control sample. If they are
indistinguishable, F is a false positive and the corpus is clean.

    python -m relmo.flicker --dataset rcasa --ids a,b,c
    python -m relmo.flicker --dataset rcasa --sample 40
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

FLICKER_TOL = 30.0     # per-channel jump that counts as a colour change
MAX_FLICKER = 0.02     # a clean render holds still on static pixels
MIN_SAT = 90           # same saturation notion gtcheck's F uses


def _frames(mp4, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def check(ep_dir: Path, n_pairs=8):
    z = np.load(ep_dir / "state.npz", allow_pickle=True)
    W, H = int(z["width"]), int(z["height"])
    ds = int(z["seg_ds"])
    seg = z["seg"]
    F = _frames(ep_dir / "frames.mp4", W, H)
    T = min(len(F), len(seg))
    # frames at seg resolution so a seg id and a pixel refer to the same place
    Fd = F[:T, ::ds, ::ds].astype(np.int16)[:, :seg.shape[1], :seg.shape[2]]

    ts = np.linspace(0, T - 2, min(n_pairs, max(T - 1, 1))).astype(int)
    flick, static_n = [], 0
    for t in ts:
        same = seg[t] == seg[t + 1]           # body did not change here
        if same.sum() < 50:
            continue
        d = np.abs(Fd[t + 1] - Fd[t]).max(-1)
        flick.append(float((d[same] > FLICKER_TOL).mean()))
        static_n += int(same.sum())

    # hue diversity over saturated area, mid frame
    f = F[T // 2].astype(np.int16)
    mx, mn = f.max(-1), f.min(-1)
    sat = (mx - mn) > MIN_SAT
    hues = 0
    if sat.sum() > 100:
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        c = np.maximum(mx - mn, 1)
        h = np.where(mx == r, ((g - b) / c) % 6,
                     np.where(mx == g, (b - r) / c + 2, (r - g) / c + 4)) * 60
        hh = h[sat]
        hist, _ = np.histogram(hh % 360, bins=12, range=(0, 360))
        hues = int((hist > 0.05 * hist.sum()).sum())

    fl = float(np.mean(flick)) if flick else 0.0
    return dict(id=ep_dir.name, T=int(T), static_px=static_n,
                flicker=round(fl, 5), sat_frac=round(float(sat.mean()), 4),
                hue_modes=hues, FLICKER_OK=bool(fl < MAX_FLICKER))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--ids", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = R.dataset_dir(a.dataset)
    eps = sorted(p.parent for p in root.glob("shard_*/*/state.npz"))
    want = [s for s in a.ids.split(",") if s]
    if want:
        pick = [e for e in eps if any(e.name.startswith(w) for w in want)]
    else:
        rng = np.random.default_rng(a.seed)
        pick = [eps[i] for i in rng.choice(len(eps),
                                           min(a.sample or 20, len(eps)),
                                           replace=False)]
    hdr = f"{'episode':50s} {'flicker':>8s} {'sat':>6s} {'hues':>5s}  OK"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for e in pick:
        r = check(e)
        rows.append(r)
        print(f"{r['id'][:50]:50s} {r['flicker']:8.5f} {r['sat_frac']:6.3f} "
              f"{r['hue_modes']:5d}  {'ok' if r['FLICKER_OK'] else 'FLICKER'}",
              flush=True)
    fl = np.array([r["flicker"] for r in rows])
    print(f"\n{sum(r['FLICKER_OK'] for r in rows)}/{len(rows)} steady   "
          f"flicker median {np.median(fl):.5f} max {fl.max():.5f}")
