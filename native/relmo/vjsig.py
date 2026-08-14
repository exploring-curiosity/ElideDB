"""A SigLIP appearance channel, aligned step-for-step with the prediction trace.

WHY, and why SigLIP specifically.

STRAP (arXiv 2412.15182) is the published system closest to this project: encode
frames with a vision foundation model, segment demos, match sub-trajectories
with subsequence DTW, retrieve, then train a policy on what came back. It
reports +24.7% over BehaviorRetrieval and +25.0% over FlowRetrieval on
LIBERO-10, and it explicitly retrieves "even if object appearance and pose, or
the task differ" - the cross-object case this project has been stuck on.

It uses DINOv2 or CLIP. We rejected appearance features early and never
measured against them, so that is an unpaid debt: the demonstrated baseline for
retrieval-driven transfer is appearance, and our prediction channel has only
ever been compared to itself.

SigLIP rather than DINOv2 because of the owner's stated next use case, text
queries. DINOv2 has no text tower and can never serve them; SigLIP's image and
text encoders share a space, so the same index answers both a clip query and a
text query. That is the whole reason to prefer a weaker-but-aligned model over
a stronger vision-only one.

NOTE this supersedes, for the retrieval channel only, the standing rule
"no text identity - no CLIP/SigLIP/DINOv2, no naming at write" (2026-07-31).
The owner asked for it directly and the text use case requires it. It does NOT
license naming objects at write time; nothing here emits a label.

ALIGNMENT. The prediction trace covers tubelet steps CTX..31 of 32, i.e. frames
2*CTX..63 of the 64 sampled across the episode. This embeds the FIRST frame of
each of those tubelets, so step t of the SigLIP sequence and step t of the
prediction sequence describe the same instant. Without that the two channels
cannot be fused per-step or aligned by the same DTW path.

    python -m relmo.vjsig --dataset rcasa
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
from relmo.vjs import TUBELET, probe_dims, read_frames, sample_clip  # noqa: E402
from relmo.vjeval import REC  # noqa: E402
from relmo.vjrec4 import CTX  # noqa: E402

OUT = R.BASE / "vjsig"
MODEL = "google/siglip2-base-patch16-224"
RES = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--from-manifest", action="store_true",
                    help="enumerate the dataset manifest instead of a prior "
                         "vjrec pass - needed for OOD corpora, which have none")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as Fn
    from tqdm import tqdm
    from transformers import AutoModel

    d = REC / a.dataset
    shard = {}
    if a.from_manifest:
        man = R.read_manifest(a.dataset)
        files = [Path(e["id"] + ".npz") for e in man["episodes"]]
        shard = {e["id"]: e["shard"] for e in man["episodes"]}
    else:
        files = [p for p in sorted(d.glob("*.npz"))
                 if not p.name.startswith("_")]
    if a.limit:
        files = files[:a.limit]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {a.model} onto {dev}...", flush=True)
    model = AutoModel.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    npar = sum(p.numel() for p in model.parameters())
    n_t = a.frames // TUBELET
    steps = list(range(CTX, n_t))
    print(f"  loaded | {npar/1e6:.0f}M params | {len(steps)} steps aligned to "
          f"the prediction trace", flush=True)

    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in files if not (out / f"{p.stem}.npz").exists()]
    print(f"{len(files)} episodes, {len(todo)} to do", flush=True)
    t0, done, failed = time.time(), 0, 0
    for p in tqdm(todo, unit="ep", desc="siglip"):
        ep = (R.dataset_dir(a.dataset) / shard.get(p.stem, "shard_0000")
              / p.stem / "frames.mp4")
        if not ep.exists():
            failed += 1
            continue
        try:
            w_, h_ = probe_dims(ep)
            F = read_frames(ep, w_, h_)
            if len(F) < a.frames:
                failed += 1
                continue
            clip = sample_clip(F, a.frames)
            # first frame of each tubelet the prediction trace covers
            sel = clip[[t * TUBELET for t in steps]]
            x = torch.tensor(sel).permute(0, 3, 1, 2).float().div_(255.)
            x = Fn.interpolate(x, size=(RES, RES), mode="bilinear",
                               align_corners=False)
            x = ((x - mean) / std).to(dev)
            with torch.no_grad():
                e = model.get_image_features(pixel_values=x)
            v = e.float().cpu().numpy().astype(np.float32)
        except Exception as ex:                              # noqa: BLE001
            tqdm.write(f"  {p.stem}: {type(ex).__name__}: {ex}")
            failed += 1
            continue
        tmp = out / f".w_{p.stem}.npz"
        np.savez_compressed(tmp, sig=v)
        tmp.rename(out / f"{p.stem}.npz")
        done += 1
    have = len(list(out.glob("*.npz")))
    rep = dict(dataset=a.dataset, model=a.model, steps=len(steps),
               episodes=len(files), written=done, failed=failed, on_disk=have,
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {have}/{len(files)} siglip records")
    R.log("vjsig", **rep)


if __name__ == "__main__":
    main()
