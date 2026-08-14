"""THE TIME AXIS, measured per content channel and per matcher.

Owner: "a door opening fast and fridge opening slow is still the same context.
not just fast/slow, the time can be differential. slow at start and fast at
end."

Global duration is already free - relmo/vjs.sample_clip spreads 64 frames over
the whole episode, so a 3 s and a 14 s clip share a time base. What is NOT free
is the differential profile inside a clip, and that is what this measures.

METHOD, and it needs no labels: re-render ONE episode's own footage under a
time warp, then ask whether the warped clip still retrieves the ORIGINAL out of
the whole corpus. The correct answer is "itself", known without annotation.

  identity   w(u)=u            control - must be rank 1
  ease_in    w(u)=u^2          slow start, fast end
  ease_out   w(u)=1-(1-u)^2    fast start, slow end
  sigmoid    slow-fast-slow    the natural shape of a hinge swinging
  zoom       middle 60%        the event stretched ~1.7x

WHAT IS NEW HERE
  1. All THREE content channels, not just error. pred_change and obs_change are
     built from a subtraction against actual(t), so they may not inherit the
     speed dependence that broke v1: prediction error scales with apparent
     motion, so replaying an event faster changed the trace's CONTENT rather
     than merely its timing, and no amount of alignment can undo that.
  2. FIXED-length context upstream (v4). v3's growing context put a ramp in
     every trace, and two ramps align to each other regardless of event - so
     earlier warp numbers were partly measuring the ramp.
  3. Both matchers. Previously measured: pooling is warp-INVARIANT but
     order-blind and 3x more object-biased, because order is what encodes
     direction. So invariance alone is not the goal - a descriptor that
     survives warping by discarding order has not solved anything.

    python -m relmo.vjtime --episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import MODEL, probe_dims, read_frames  # noqa: E402
from relmo.vjeval import REC, l2  # noqa: E402
from relmo.vjeval4 import subseq_batch  # noqa: E402
from relmo.vjrec4 import CHANNELS, OUT4, record  # noqa: E402

WARPS = {
    "identity": lambda u: u,
    "ease_in": lambda u: u ** 2,
    "ease_out": lambda u: 1 - (1 - u) ** 2,
    "sigmoid": lambda u: 0.5 - 0.5 * np.cos(np.pi * u),
    "zoom": lambda u: 0.2 + 0.6 * u,
}


def warped(F, n, fn):
    u = np.linspace(0.0, 1.0, n)
    return F[np.clip((fn(u) * (len(F) - 1)).round().astype(int), 0, len(F) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d4 = OUT4 / f"{a.dataset}_L{a.layer}"
    files = sorted(d4.glob("*.npz"))
    if len(files) < 100:
        raise SystemExit(f"only {len(files)} v4 records - let vjrec4 finish")
    Z = [np.load(p) for p in files]
    corpus = {k: l2(np.stack([z[k] for z in Z])) for k in CHANNELS}
    names = [p.stem for p in files]
    z = np.load(REC / a.dataset / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print(f"  loaded | corpus {len(files)} v4 records", flush=True)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), min(a.episodes, len(files)), replace=False)
    acc = {(c, m, w): [] for c in CHANNELS for m in ("cos", "span")
           for w in WARPS}

    for i in tqdm(pick, unit="ep", desc="time"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / names[i] / "frames.mp4"
        if not ep.exists():
            continue
        w_, h_ = probe_dims(ep)
        F = read_frames(ep, w_, h_)
        if len(F) < a.frames:
            continue
        for wn, fn in WARPS.items():
            rec = record(model, torch, dev, warped(F, a.frames, fn),
                         a.frames, cal, a.layer)
            for c in CHANNELS:
                q = l2(rec[c])
                B = corpus[c]
                s_cos = np.einsum("sd,nsd->n", q, B) / q.shape[0]
                s_spn = -subseq_batch(q, B)
                for m, s in (("cos", s_cos), ("span", s_spn)):
                    acc[(c, m, wn)].append(int((s > s[i]).sum()) + 1)

    print(f"\nrank of the TRUE original among {len(files)} clips "
          f"(1 = perfect), median")
    print(f"{'channel':13s} {'matcher':8s} " +
          " ".join(f"{w:>9s}" for w in WARPS))
    print("-" * (23 + 10 * len(WARPS)))
    out = {}
    for c in CHANNELS:
        for m in ("cos", "span"):
            row = [np.median(acc[(c, m, w)]) if acc[(c, m, w)] else np.nan
                   for w in WARPS]
            print(f"{c:13s} {m:8s} " + " ".join(f"{v:9.1f}" for v in row))
            for w, v in zip(WARPS, row):
                out[f"{c}_{m}_{w}"] = round(float(v), 2)
    print("\nidentity must be 1.0 - it is the same clip. The differential cases")
    print("are ease_in / ease_out / sigmoid; zoom is a uniform stretch and was")
    print("already free from the 64-frames-over-the-whole-episode sampling.")
    R.log("vjtime", dataset=a.dataset, layer=a.layer,
          episodes=len(pick), corpus=len(files), **out)


if __name__ == "__main__":
    main()
