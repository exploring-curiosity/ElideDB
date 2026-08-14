"""Track extraction + supervision assembly.

Frozen CoTracker3 turns every episode into point trajectories. The
simulator's segmentation then says which BODY each tracked point sat
on, and the contact solver says which bodies touched when. Those two
become the training targets:

    grouping  : do points i and j belong to the same physical thing?
    contact   : are bodies A and B touching at time t?

Both are computed from privileged simulator state, which is a LABEL -
that is allowed at training time and nowhere else. Nothing here ever
runs on the evaluation corpus.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

GRID = 32                 # 32x32 = 1024 query points
SEG_DS = 2
VER = 1
_M = None


def model():
    global _M
    if _M is None:
        import torch
        _M = torch.hub.load("facebookresearch/co-tracker",
                            "cotracker3_offline").to("mps").eval()
    return _M


def frames(mp4):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 480, 640, 3)


def extract(ep_dir: Path, out: Path):
    """tracks + per-point body id + per-frame body contact matrix."""
    import gc

    import torch
    F = frames(ep_dir / "frames.mp4")
    st = np.load(ep_dir / "state.npz")
    seg = st["seg"]                                   # (T,240,320)
    V = torch.tensor(F).permute(0, 3, 1, 2)[None].float().to("mps")
    with torch.no_grad():
        tr, vis = model()(V, grid_size=GRID)
    xy = tr[0].cpu().numpy()                          # (T,P,2)
    vs = vis[0].cpu().numpy()
    del tr, vis, V
    gc.collect()
    torch.mps.empty_cache()
    T = min(len(xy), len(seg))
    xy, vs, seg = xy[:T], vs[:T], seg[:T]
    # body id under each point, per frame (0 = background/floor)
    H, W = seg.shape[1:]
    ix = np.clip((xy[..., 0] / SEG_DS).astype(int), 0, W - 1)
    iy = np.clip((xy[..., 1] / SEG_DS).astype(int), 0, H - 1)
    bid = np.take_along_axis(
        seg.reshape(T, -1), (iy * W + ix), axis=1).astype(np.uint8)
    # a point's identity = the body it sits on MOST of the time while
    # visible (a single frame's lookup lands on edges and shadows)
    P = xy.shape[1]
    ident = np.zeros(P, np.uint8)
    for p in range(P):
        b = bid[:, p][vs[:, p] > 0.5]
        if len(b) == 0:
            continue
        vals, cnt = np.unique(b, return_counts=True)
        ident[p] = vals[np.argmax(cnt)]
    np.savez_compressed(
        out, ver=VER, xy=xy.astype(np.float16), vis=vs.astype(bool),
        bid=bid, ident=ident,
        contact=st["contact"].astype(np.float16),
        xpos=st["xpos"], xvel=st["xvel"])
    return dict(P=int(P), T=int(T),
                on_body=int((ident > 0).sum()),
                bodies=int(len(np.unique(ident[ident > 0]))))


def build(name="physgen_v1", limit=None):
    from tqdm import tqdm
    man = R.read_manifest(name)
    out_dir = R.TRACKS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model()
    eps = man["episodes"] if limit is None else man["episodes"][:limit]
    todo = [e for e in eps
            if not (out_dir / f"{e['id']}.npz").exists()]
    if not todo:
        return 0
    made = 0
    for e in tqdm(todo, unit="ep", desc=f"tracks/{name}"):
        ep_dir = R.dataset_dir(name) / e["shard"] / e["id"]
        tmp = out_dir / f".tmp_{e['id']}.npz"
        try:
            extract(ep_dir, tmp)
            tmp.rename(out_dir / f"{e['id']}.npz")
            made += 1
        except Exception as exc:
            R.log("track_error", dataset=name, id=e["id"],
                  error=str(exc)[:200])
            tmp.unlink(missing_ok=True)
    R.log("tracks_done", dataset=name, made=made,
          total=len(list(out_dir.glob("ep*.npz"))))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="physgen_v1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print("new tracks:", build(a.name, a.limit))
