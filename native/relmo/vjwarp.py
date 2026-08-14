"""Is the experience record invariant to HOW FAST the event happened?

The owner's point, which the design has to survive: a door thrown open in half
a second and a fridge easing open over four seconds are the same experience.
And the stretch is not uniform - things start slow and finish fast, stall in
the middle, snap shut at the end. If the descriptor only matches events that
happen on the same clock, it is matching timing, not experience.

This takes ONE episode, re-renders its own footage under several time warps,
and asks whether the warped clip still retrieves the ORIGINAL above all 400+
other clips in the corpus. Nothing is labelled: the correct answer is "itself",
which is known without any annotation.

WARPS  output frame i takes source frame w(i/N) * T
  identity   w(u)=u            the reference
  ease_in    w(u)=u^2          slow at the start, fast at the end
  ease_out   w(u)=1-(1-u)^2    fast at the start, slow at the end
  sigmoid    slow-fast-slow    the natural shape of a hinge swinging
  zoom       middle 60% only   the whole event stretched ~1.7x slower

Global duration is ALREADY normalised upstream (64 frames across the whole
episode, so a 3 s and a 14 s clip share a time base). These warps therefore
test the part that is NOT free: the differential speed profile within a clip.

COMPARED THREE WAYS, because the choice of comparison is the thing under test
  pooled  cosine of the time-averaged trace - order destroyed
  direct  step t against step t - ordered, but assumes a shared clock
  dtw     warped alignment - ordered, tolerates a changing clock

READ IT AS: rank of the true original among all corpus records. Rank 1 is
perfect. If dtw holds rank 1 under ease_in/ease_out while direct does not, the
alignment is earning its cost. If pooled also holds rank 1, the ordering is
doing no work and the extra machinery is unjustified.

    python -m relmo.vjwarp --episodes 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, forward_pair, probe_dims,  # noqa: E402
                       read_frames)
from relmo.vjeval import REC, dtw_batch, l2  # noqa: E402
from relmo.vjrec2 import OUT2, build_record  # noqa: E402

WARPS = {
    "identity": lambda u: u,
    "ease_in": lambda u: u ** 2,
    "ease_out": lambda u: 1 - (1 - u) ** 2,
    "sigmoid": lambda u: 0.5 - 0.5 * np.cos(np.pi * u),
    "zoom": lambda u: 0.2 + 0.6 * u,
}


def warped_clip(F, n, fn):
    u = np.linspace(0.0, 1.0, n)
    idx = np.clip((fn(u) * (len(F) - 1)).round().astype(int), 0, len(F) - 1)
    return F[idx]


def describe(model, torch, dev, clip, n_frames, cal, layer=6):
    """v2 construction (relmo/vjrec2.build_record) on a supplied clip.

    v1 pooled the layer-24 residual by its OWN magnitude. That descriptor
    broke badly under a differential time warp (rank 52.5 under ease_out)
    because the residual scales with apparent motion speed, so replaying an
    event faster changed the trace's CONTENT rather than merely its timing.
    v2 gates on early-layer CHANGE instead, which is a different quantity and
    may not carry that speed dependence - which is exactly what this re-run
    is for."""
    return build_record(model, torch, dev, clip, n_frames, cal, layer)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--layer", type=int, default=6)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
    if len(files) < 100:
        raise SystemExit(f"only {len(files)} records - let vjcache finish first")
    d2 = OUT2 / f"{a.dataset}_L{a.layer}"
    files = [p for p in files if (d2 / f"{p.stem}.npz").exists()]
    SEQ = l2(np.stack([np.load(d2 / f"{p.stem}.npz")["what_seq"]
                       for p in files]))
    POOL = l2(SEQ.mean(1))
    names = [p.stem for p in files]
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"])

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print(f"  loaded | corpus {len(files)} records", flush=True)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), min(a.episodes, len(files)), replace=False)
    acc = {w: {"pooled": [], "direct": [], "dtw": []} for w in WARPS}

    for i in tqdm(pick, unit="ep", desc="warp"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / names[i] / "frames.mp4"
        if not ep.exists():
            continue
        w_, h_ = probe_dims(ep)
        F = read_frames(ep, w_, h_)
        if len(F) < a.frames:
            continue
        for wname, fn in WARPS.items():
            q = describe(model, torch, dev, warped_clip(F, a.frames, fn),
                         a.frames, cal, a.layer)
            qn = l2(q)
            s_pool = POOL @ l2(q.mean(0))
            s_dir = np.einsum("sd,nsd->n", qn, SEQ) / qn.shape[0]
            s_dtw = dtw_batch(qn, SEQ)
            for k, s in (("pooled", s_pool), ("direct", s_dir), ("dtw", s_dtw)):
                # rank of the TRUE original, 1 = best
                acc[wname][k].append(int((s > s[i]).sum()) + 1)

    print("\n" + "=" * 64)
    print(f"rank of the true original among {len(files)} clips "
          f"(1 = perfect), median")
    print(f"{'warp':12s} {'pooled':>10s} {'direct':>10s} {'dtw':>10s}")
    print("-" * 64)
    out = {}
    for w in WARPS:
        r = {k: float(np.median(v)) if v else float("nan")
             for k, v in acc[w].items()}
        print(f"{w:12s} {r['pooled']:10.1f} {r['direct']:10.1f} "
              f"{r['dtw']:10.1f}")
        out.update({f"{w}_{k}": round(v, 2) for k, v in r.items()})
    print("=" * 64)
    print("identity should be rank 1 everywhere - it is the same clip.")
    print("ease_in / ease_out / sigmoid are the differential-speed cases:")
    print("  dtw rank 1 and direct worse -> alignment is earning its cost")
    print("  pooled also rank 1          -> ordering is doing no work")
    R.log("vjwarp", dataset=a.dataset, episodes=len(pick),
          corpus=len(files), **out)


if __name__ == "__main__":
    main()
