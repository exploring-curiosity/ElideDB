"""SigLIP appearance, sampled at STREAM steps rather than clip positions.

The v4 companion (relmo/vjsig.py) embedded the first frame of each tubelet of a
whole-episode-normalised 64-frame clip. Under v6 the sampling rate is fixed and
the trace length varies with duration, so alignment can no longer be recomputed
from a frame count - it has to come from the record itself.

relmo/vjrec6 writes `frame0`, the absolute SOURCE-frame index of the first
frame of every tubelet it described. This module embeds exactly those frames.
Step t of the SigLIP sequence and step t of the prediction trace therefore
describe the same instant by construction, and cannot drift if the window
geometry changes.

    python -m relmo.vjsig6 --dataset rcasa
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
from relmo.vjrec6 import OUT6, episode_paths  # noqa: E402
from relmo.vjs import probe_dims, read_frames  # noqa: E402
from relmo.vjsig import MODEL, RES  # noqa: E402

OUT = R.BASE / "vjsig6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as Fn
    from tqdm import tqdm
    from transformers import AutoModel

    rec = OUT6 / f"{a.dataset}_L{a.layer}{a.suffix}"
    paths = {i: p for i, p, _ in episode_paths(a.dataset)}
    files = [p for p in sorted(rec.glob("*.npz"))
             if not p.name.startswith(".")]
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no v6 records in {rec} - run relmo.vjrec6 first")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {a.model} onto {dev}...", flush=True)
    model = AutoModel.from_pretrained(a.model,
                                      dtype=torch.float16).to(dev).eval()
    print(f"  loaded | {sum(p.numel() for p in model.parameters())/1e6:.0f}M "
          f"params | fp16", flush=True)

    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    out = OUT / f"{a.dataset}{a.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in files if not (out / p.name).exists()]
    print(f"{len(files)} records, {len(todo)} to do", flush=True)

    t0, done, failed, nsteps = time.time(), 0, 0, 0
    for p in tqdm(todo, unit="ep", desc="siglip6"):
        mp4 = paths.get(p.stem)
        if mp4 is None or not mp4.exists():
            failed += 1
            continue
        try:
            f0 = np.load(p)["frame0"].astype(int)
            w_, h_ = probe_dims(mp4)
            F = read_frames(mp4, w_, h_)
            sel = F[np.clip(f0, 0, len(F) - 1)]
            V = []
            for s in range(0, len(sel), a.batch):
                x = torch.tensor(sel[s:s + a.batch]).permute(0, 3, 1, 2) \
                    .float().div_(255.)
                x = Fn.interpolate(x, size=(RES, RES), mode="bilinear",
                                   align_corners=False)
                x = ((x - mean) / std).to(dev, torch.float16)
                with torch.no_grad():
                    V.append(model.get_image_features(pixel_values=x)
                             .float().cpu().numpy())
            v = np.concatenate(V).astype(np.float32)
            assert len(v) == len(f0), f"{len(v)} != {len(f0)}"
        except Exception as ex:                                # noqa: BLE001
            tqdm.write(f"  {p.stem}: {type(ex).__name__}: {ex}")
            failed += 1
            continue
        tmp = out / f".w_{p.stem}.npz"
        np.savez_compressed(tmp, sig=v)
        tmp.rename(out / p.name)
        done += 1
        nsteps += len(v)
    have = [q for q in out.glob("*.npz") if not q.name.startswith(".")]
    rep = dict(dataset=a.dataset, model=a.model, records=len(files),
               written=done, failed=failed, on_disk=len(have),
               steps_written=nsteps,
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {len(have)}/{len(files)} siglip6 records")
    R.log("vjsig6", **rep)


if __name__ == "__main__":
    main()
