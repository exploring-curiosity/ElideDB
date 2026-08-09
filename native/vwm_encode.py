"""Per-frame frozen-encoder latents for sim episodes, cached to disk.

The write path of the world-model experiments: every episode becomes
its per-frame vits latents (global g and 5x5 coarse grid c - the same
two label-free views vdyn stores). Nothing else is extracted; the
predictor is trained ON these, and at read time the same operator runs
over any query clip. Ground truth in meta.json is never touched here.

One camera per episode: embodiment/viewpoint variety lives ACROSS
episodes (2 of 4 jittered cams were recorded; we take the first
present). Resumable: an existing npz is skipped, so an interrupt costs
one episode, not the run.

    SDX_ENC=vits SDX_RES=320 python native/vwm_encode.py \
        data/sim_chains data/sim_train/panda data/sim_train/xarm7 ...
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import vcore                                              # noqa: E402
from vtrans import encode                                 # noqa: E402

CACHE = ROOT / "data" / "cache" / "vwm"
W, H = 640, 480          # sim_stack RES, every sim episode


def read_frames(mp4: Path) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, check=True)
    buf = p.stdout
    n = len(buf) // (W * H * 3)
    assert n * W * H * 3 == len(buf), f"{mp4}: not {W}x{H} rgb24"
    return np.frombuffer(buf, np.uint8).reshape(n, H, W, 3)


def episode_dirs(roots):
    for r in roots:
        r = Path(r).resolve()
        for ep in sorted(r.glob("ep*")):
            if (ep / "meta.json").exists():
                yield ep


def cache_path(ep: Path) -> Path:
    rel = ep.relative_to(ROOT / "data")
    return CACHE / f"{'_'.join(rel.parts)}.npz"


def main():
    roots = sys.argv[1:] or ["data/sim_chains"]
    eps = list(episode_dirs(roots))
    todo = [e for e in eps if not cache_path(e).exists()]
    print(f"{len(eps)} episodes, {len(todo)} to encode "
          f"({vcore.ENCODER.split('/')[-1]})")
    if not todo:
        return
    vcore.feature_dim()          # load the model BEFORE the bar exists
    CACHE.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    for ep in tqdm(todo, unit="ep", desc="encode"):
        cams = sorted(ep.glob("cam*.mp4"))
        if not cams:
            continue
        F = read_frames(cams[0])
        G = encode(F)                        # (T, GRID, GRID, d)
        T = len(G)
        g = G.reshape(T, -1, G.shape[-1]).mean(1)
        g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
        k = G.shape[1] // 5
        c = G.reshape(T, 5, k, 5, k, -1).mean((2, 4)).reshape(T, -1)
        np.savez(cache_path(ep), g=g.astype(np.float16),
                 c=c.astype(np.float16), cam=cams[0].name)


if __name__ == "__main__":
    main()
