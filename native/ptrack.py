"""Kinematic write path: CoTracker3 dense point tracks per recording.

WHY (measured, 2026-08-10): every appearance route sat at ~chance on
this corpus (frozen V-JEPA2 pooled 0.309), the only informative frozen
signal was motion (flow 0.380), the best system computes kinematic
facts explicitly (0.510), and the oracle test proved TRACK FIDELITY is
the bound - box-centroid tracking corrupts through grip occlusion,
merges and mask theft. Point trackers are the class built for exactly
that: per-point (x,y,t) with visibility, no boxes, no identity rules,
and 25px is irrelevant because points have no extent.

Content-free throughout: a uniform grid is tracked, nothing is
detected, named, or classified. Downstream layers see only
coordinates over time.

    python native/ptrack.py --build data/prim_actions_v2
    python native/ptrack.py --verify ep0030   # overlay sheet
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "ptrack"
GRID = 30                # 30x30 = 900 tracked points per recording
VER = 1

_M = None


def model():
    global _M
    if _M is None:
        import torch
        m = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline")
        _M = m.to("mps").eval()
    return _M


def frames(mp4):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 480, 640, 3)


def view_of(ep):
    """The view this corpus's earlier stage selected (per-recording
    self-calibration, already computed); falls back to the first cam."""
    f = ROOT / "data/cache/entsam" / \
        f"prim_actions_v2_{ep.name}.npy"
    if f.exists():
        rows = np.load(f, allow_pickle=True)
        v = next((x.get("view") for x in rows
                  if x["role"] == "_ver"), None)
        if v:
            return ep / v
    return sorted(ep.glob("cam*.mp4"))[0]


def build_one(ep, out):
    """One recording -> tracks. MPS cache is released per episode:
    without it allocations accumulate and throughput collapsed 20x
    (6 s/ep -> 139 s/ep, measured) - the tracker is not slow, the
    allocator was."""
    import gc

    import torch
    F = frames(view_of(ep))
    V = torch.tensor(F).permute(0, 3, 1, 2)[None].float().to("mps")
    with torch.no_grad():
        tr, vis = model()(V, grid_size=GRID)
    xy = tr[0].cpu().numpy().astype(np.float16)
    vs = vis[0].cpu().numpy().astype(bool)
    del tr, vis, V
    gc.collect()
    torch.mps.empty_cache()
    np.savez_compressed(out, ver=VER, xy=xy, vis=vs)
    return out


def build(corpus):
    from tqdm import tqdm
    CACHE.mkdir(parents=True, exist_ok=True)
    model()
    eps = sorted((ROOT / corpus).glob("ep*"))
    name = Path(corpus).name
    for ep in tqdm(eps, unit="ep", desc=f"ptrack/{name} (~30 min)"):
        out = CACHE / f"{name}_{ep.name}.npz"
        if out.exists():
            try:
                if int(np.load(out)["ver"]) == VER:
                    continue
            except Exception:
                pass
        build_one(ep, out)


def verify(epn, corpus="data/prim_actions_v2"):
    """Overlay the MOVING tracked points on frames - the eyes-on gate
    before any corpus run. Moving is defined by the recording's own
    noise floor, not a constant."""
    from PIL import Image, ImageDraw
    ep = ROOT / corpus / epn
    z = np.load(CACHE / f"{Path(corpus).name}_{epn}.npz")
    xy, vis = z["xy"].astype(np.float32), z["vis"]
    T, P, _ = xy.shape
    F = frames(view_of(ep))
    d = np.linalg.norm(np.diff(xy, axis=0), axis=-1)      # (T-1,P)
    rng = np.linalg.norm(xy.max(0) - xy.min(0), axis=-1)  # (P,)
    thr = max(np.percentile(rng, 75) * 2.0, 6.0)
    mov = np.where(rng > thr)[0]
    sheet = Image.new("RGB", (4 * 320, 2 * 240))
    fis = np.linspace(0, T - 1, 8).astype(int)
    for k, fi in enumerate(fis):
        img = Image.fromarray(F[min(fi, len(F) - 1)]).resize((320, 240))
        dr = ImageDraw.Draw(img)
        for p in mov:
            if not vis[fi, p]:
                continue
            x, y = xy[fi, p] / 2
            dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill="red")
        dr.text((4, 4), f"f{fi}", fill="white")
        sheet.paste(img, ((k % 4) * 320, (k // 4) * 240))
    o = ROOT / f"data/cache/ptrack/verify_{epn}.png"
    sheet.save(o)
    print(f"{epn}: {len(mov)}/{P} moving points (thr {thr:.1f}px) -> {o}")
    return o


if __name__ == "__main__":
    if "--build" in sys.argv:
        build(sys.argv[sys.argv.index("--build") + 1])
    if "--verify" in sys.argv:
        verify(sys.argv[sys.argv.index("--verify") + 1])
