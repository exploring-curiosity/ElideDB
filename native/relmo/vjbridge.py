"""ood_test records from bridge - REAL robot video, read once.

The strongest domain shift available without a download: real cameras, real
kitchens and tabletops, a WidowX arm instead of a simulated Panda, natural
lighting, 5 fps, and free-text task strings written by whoever collected the
demo. Nothing about it resembles the RoboCasa renderer the recurrence trained
on, and no sim state exists here at all - which is the point, since the physics
supervision is training-time only and this is what serve time looks like.

WHAT THIS TEST CAN AND CANNOT SHOW. Of 50,415 bridge episodes only 1,499 run to
64 frames, and those are dominated by two event types: "sweep into pile" (955)
and a family of "put X in pot/pan, put pot/pan on stove" (~500). So this is a
TWO-CLASS problem, balanced by subsampling. It is real evidence that the
representation survives the domain, and it is NOT evidence about fine-grained
event discrimination - sweeping and placing look very different, and a weak
appearance feature would also separate them. Reported with that limit stated,
not quoted as a headline.

Calibration travels from rcasa unchanged. Refitting the affine on the target
domain would be per-domain tuning, which an out-of-domain claim may not use.

    python -m relmo.vjbridge --per-class 250
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
from relmo.vjs import MODEL, TUBELET, probe_dims, sample_clip  # noqa: E402
from relmo.vjrec4 import CTX, OUT4, WIN, record  # noqa: E402

BRIDGE = Path(__file__).resolve().parents[2] / "data" / "bridge"
STREAM = "observation.images.image_0"
OUT = OUT4 / "bridge_L6"
SIGOUT = R.BASE / "vjsig" / "bridge"


def klass(task: str) -> str | None:
    """Event class from the free-text task string. GRADING ONLY."""
    t = task.strip().lower()
    if t.startswith("sweep"):
        return "sweep"
    if t.startswith("put") and ("pot" in t or "pan" in t) and "stove" in t:
        return "put_on_stove"
    return None


def episodes(per_class):
    import pandas as pd
    f = sorted((BRIDGE / "meta" / "episodes").glob("*/*.parquet"))
    cols = ["episode_index", "tasks", "length",
            f"videos/{STREAM}/chunk_index", f"videos/{STREAM}/file_index",
            f"videos/{STREAM}/from_timestamp", f"videos/{STREAM}/to_timestamp"]
    d = pd.concat([pd.read_parquet(x, columns=cols) for x in f])
    d.columns = ["ep", "tasks", "length", "chunk", "file", "t0", "t1"]
    d = d[d.length >= 64].copy()
    d["task"] = d["tasks"].map(lambda x: str(x[0]) if not isinstance(x, str)
                               else str(x))
    d["cls"] = d["task"].map(klass)
    d = d[d["cls"].notna()]
    rng = np.random.default_rng(0)
    keep = []
    for c, g in d.groupby("cls"):
        idx = rng.choice(len(g), min(per_class, len(g)), replace=False)
        keep.append(g.iloc[sorted(idx)])
    import pandas as pd2  # noqa: F401
    return __import__("pandas").concat(keep)


def read_range(mp4, w, h, t0, t1):
    """Decode only [t0,t1) of a multi-episode bridge file."""
    import subprocess
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
         "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--sig", action="store_true", help="also build SigLIP")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = episodes(a.per_class)
    print(f"bridge ood_test: {len(d)} episodes "
          f"{dict(d['cls'].value_counts())}", flush=True)
    z = np.load(R.BASE / "vjrec" / "rcasa" / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev} (calibration travels from rcasa)...",
          flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    sig = None
    if a.sig:
        from transformers import AutoModel
        from relmo.vjsig import MODEL as SM
        sig = AutoModel.from_pretrained(SM, dtype=torch.float32).to(dev).eval()
        SIGOUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("  loaded", flush=True)

    rows = [r for _, r in d.iterrows()
            if not (OUT / f"{r['cls']}_ep{int(r['ep']):06d}.npz").exists()]
    print(f"{len(rows)} to do (~{len(rows)*11/60:.0f} min)", flush=True)
    t0, done, failed = time.time(), 0, 0
    dims = {}
    for r in tqdm(rows, unit="ep", desc="bridge"):
        name = f"{r['cls']}_ep{int(r['ep']):06d}"
        mp4 = (BRIDGE / "videos" / STREAM / f"chunk-{int(r['chunk']):03d}"
               / f"file-{int(r['file']):03d}.mp4")
        try:
            if mp4 not in dims:
                dims[mp4] = tuple(probe_dims(mp4))
            w_, h_ = dims[mp4]
            F = read_range(mp4, w_, h_, float(r["t0"]), float(r["t1"]))
            if len(F) < a.frames:
                failed += 1
                continue
            clip = sample_clip(F, a.frames)
            rec = record(model, torch, dev, clip, a.frames, cal, 6)
            if sig is not None:
                import torch.nn.functional as Fn
                steps = list(range(CTX, a.frames // TUBELET))
                x = torch.tensor(clip[[t * TUBELET for t in steps]]) \
                    .permute(0, 3, 1, 2).float().div_(255.)
                x = Fn.interpolate(x, size=(224, 224), mode="bilinear",
                                   align_corners=False)
                x = ((x - 0.5) / 0.5).to(dev)
                with torch.no_grad():
                    e = sig.get_image_features(pixel_values=x)
                np.savez_compressed(SIGOUT / f"{name}.npz",
                                    sig=e.float().cpu().numpy().astype(np.float32))
        except Exception as ex:                              # noqa: BLE001
            tqdm.write(f"  {name}: {type(ex).__name__}: {ex}")
            failed += 1
            continue
        tmp = OUT / f".w_{name}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(OUT / f"{name}.npz")
        done += 1
    have = len(list(OUT.glob("*.npz")))
    rep = dict(written=done, failed=failed, on_disk=have,
               minutes=round((time.time() - t0) / 60, 1), win=WIN)
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {have} bridge records")
    R.log("vjbridge", **rep)


if __name__ == "__main__":
    main()
