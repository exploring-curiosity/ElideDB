"""G1 smoke test (PLAN §5.2): does the tracker follow the blocks?

Tracks one episode-view end to end, prints shapes + wall time, and
saves an overlay PNG that MUST be looked at before any corpus loop.
Settings chosen here are frozen for the whole corpus.

    python native/smoke_track.py [--grid 30] [--scale 1.0] [--stride 1]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

SCRATCH = Path(os.environ.get(
    "ELIDEDB_SCRATCH",
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad"))


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def main():
    import cv2
    import torch
    from elidedb import Store
    import chain_delta as cd

    grid = arg("--grid", 30, int)
    scale = arg("--scale", 1.0, float)
    stride = arg("--stride", 1, int)
    ep_want = arg("--ep", 0, int)

    db = Store.open(str(ROOT / "lake/sim_chains"))
    views = [v for v in cd.episode_views(db)
             if v[0] == ep_want and v[1] == "simA"]
    ts, F = cd.decode_view(db, views[0][2])
    F = F[::stride]
    ts = ts[::stride]
    if scale != 1.0:
        F = np.stack([cv2.resize(f, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
                      for f in F])
    T, H, W = F.shape[:3]
    print(f"clip: {T} frames {W}x{H} "
          f"{(ts[-1]-ts[0])/1e9:.1f}s  grid={grid} scale={scale} "
          f"stride={stride}", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading cotracker3_offline on {dev}...", flush=True)
    t0 = time.time()
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline").to(dev).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    vid = torch.tensor(F, dtype=torch.float32, device=dev) \
        .permute(0, 3, 1, 2)[None]              # 1,T,3,H,W
    t0 = time.time()
    with torch.no_grad():
        tracks, vis = model(vid, grid_size=grid)
    dt = time.time() - t0
    tracks = tracks[0].float().cpu().numpy()     # T,N,2
    vis = vis[0].cpu().numpy()                   # T,N
    print(f"tracks {tracks.shape}  vis {vis.shape}  "
          f"{dt:.1f}s for {T} frames ({dt/T*1000:.0f} ms/frame)",
          flush=True)

    # displacement profile: how many tracks actually travel?
    disp = np.linalg.norm(tracks[-1] - tracks[0], axis=1)
    net = np.abs(np.diff(tracks, axis=0)).sum(0).sum(1)
    print(f"end-to-end displacement: p50 {np.percentile(disp,50):.1f} "
          f"p90 {np.percentile(disp,90):.1f} max {disp.max():.1f} px")
    print(f"path length: p50 {np.percentile(net,50):.1f} "
          f"p90 {np.percentile(net,90):.1f} max {net.max():.1f} px")
    print(f"tracks with path > 50px: {int((net>50).sum())} of "
          f"{tracks.shape[1]}")

    # overlay: the 60 most-travelled tracks, 4 frames across the clip
    idx = np.argsort(-net)[:60]
    picks = [int(T * f) for f in (0.05, 0.35, 0.65, 0.95)]
    tiles = []
    for fi in picks:
        im = F[min(fi, T - 1)].copy()
        for k in idx:
            trail = tracks[max(fi - 25, 0):fi + 1, k]
            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(im, tuple(np.int32(a)), tuple(np.int32(b)),
                         (255, 255, 0), 1)
            p = tracks[min(fi, T - 1), k]
            ok = vis[min(fi, T - 1), k]
            cv2.circle(im, tuple(np.int32(p)), 3,
                       (0, 255, 0) if ok else (255, 0, 0), -1)
        cv2.putText(im, f"t={(ts[min(fi,T-1)]-ts[0])/1e9:.1f}s",
                    (8, 22), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 0), 2)
        tiles.append(im)
    out = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    p = SCRATCH / f"smoke_tracks_ep{ep_want}.png"
    cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
