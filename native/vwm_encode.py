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


# fixed JL projections for the finer grids (seed-pinned, never fitted).
# The 5x5 cache hit its ceiling at 0.565-0.569 P@10 across every
# temporal refinement - a ~30px block smears into ~100px cells. The
# encoder computes 20x20 anyway; keeping 10x10/20x20 projections costs
# the pass nothing.
def _proj(din, dout, seed):
    rs = np.random.RandomState(seed)
    return (rs.randn(din, dout) / np.sqrt(dout)).astype(np.float32)


_P10 = None
_P20 = None


def projs():
    global _P10, _P20
    if _P10 is None:
        _P10 = _proj(10 * 10 * 384, 3072, 11)
        _P20 = _proj(20 * 20 * 384, 4096, 13)
    return _P10, _P20


def main():
    roots = sys.argv[1:] or ["data/sim_chains"]
    eps = list(episode_dirs(roots))

    def done(e):
        fp = cache_path(e)
        if not fp.exists():
            return False
        try:
            return "p10" in np.load(fp)
        except Exception:
            return False

    # V3 cache: EVERY recorded camera per episode (retrieval measured
    # 92% same-camera bias at +0.53 over chance - the second view is
    # raw data the store was ignoring), plus row marginals at 10 and 20
    # rows: at 5 rows "table level" and "one block up" share a cell,
    # which is precisely the stack/place confusion.
    jobs = []
    for e in eps:
        for cam in sorted(e.glob("cam*.mp4")):
            fp = CACHE / (cache_path(e).stem + f"_{cam.stem}_v3.npz")
            if not fp.exists():
                jobs.append((e, cam, fp))
    print(f"{len(eps)} episodes, {len(jobs)} cam-files to encode "
          f"({vcore.ENCODER.split('/')[-1]})")
    if not jobs:
        return
    vcore.feature_dim()          # load the model BEFORE the bar exists
    CACHE.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    for ep, cam, fp in tqdm(jobs, unit="cam", desc="encode"):
        F = read_frames(cam)
        G = encode(F)                        # (T, GRID, GRID, d)
        T = len(G)
        g = G.reshape(T, -1, G.shape[-1]).mean(1)
        g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
        k = G.shape[1] // 5
        c = G.reshape(T, 5, k, 5, k, -1).mean((2, 4)).reshape(T, -1)
        r10 = G.reshape(T, 10, 2, 20, -1).mean((2, 3)).reshape(T, -1)
        r20 = G.mean(2).reshape(T, -1)       # (T, 20*384) row marginal
        np.savez(fp, g=g.astype(np.float16), c=c.astype(np.float16),
                 r10=r10.astype(np.float16), r20=r20.astype(np.float16),
                 cam=cam.name)


if __name__ == "__main__":
    main()
