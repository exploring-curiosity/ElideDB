"""[T] Dense point tracks for every episode-view. PLAN §5.3.

Pretrained CoTracker3 (offline), grid-seeded, frozen settings from the
smoke test: grid 40, temporal stride 2 (5 fps - timing precision needed
downstream is ~0.3 s), full resolution. Nothing is fitted here and
nothing is interpreted; this stage only produces trajectories.

Resumable: one .npz per view under the cache dir, so an interrupt
costs one view. Version string in the cache dir name - bump VER when a
change would alter the contents (stale-cache rule, PLAN §2 DON'T-8).

    python native/track.py [--store lake/sim_chains] [--limit N]
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

VER = "v1"
GRID = 40
STRIDE = 2

SCRATCH = Path(os.environ.get(
    "ELIDEDB_SCRATCH",
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad"))


def cache_dir(store_name):
    d = SCRATCH / f"native_tracks_{VER}_{store_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_view(cdir, ep, sv):
    p = cdir / f"{ep:05d}_{sv}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["tracks"], z["vis"], z["ts"]


def main():
    import torch
    from tqdm import tqdm
    from elidedb import Store
    import chain_delta as cd

    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else 0
    db = Store.open(str(store))
    views = cd.episode_views(db)
    if limit:
        views = [v for v in views if v[0] < limit]
    cdir = cache_dir(store.name)
    todo = [v for v in views
            if not (cdir / f"{v[0]:05d}_{v[1]}.npz").exists()]
    print(f"{store.name}: {len(views)} episode-views, "
          f"{len(todo)} to track (grid {GRID}, stride {STRIDE})",
          flush=True)
    if not todo:
        return

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading cotracker3_offline on {dev}...", flush=True)
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline").to(dev).eval()
    print(f"  projected {len(todo)*22/3600:.1f} h at 22 s/view",
          flush=True)

    t_all = time.time()
    for ep, sv, sl in tqdm(todo, desc="track", unit="view"):
        ts, F = cd.decode_view(db, sl)
        F = F[::STRIDE]
        ts = ts[::STRIDE]
        vid = torch.tensor(F, dtype=torch.float32, device=dev) \
            .permute(0, 3, 1, 2)[None]
        with torch.no_grad():
            tr, vi = model(vid, grid_size=GRID)
        np.savez_compressed(
            cdir / f"{ep:05d}_{sv}.npz",
            tracks=tr[0].float().cpu().numpy().astype(np.float16),
            vis=vi[0].cpu().numpy().astype(bool),
            ts=np.asarray(ts, np.int64))
        del vid, tr, vi
    print(f"tracked {len(todo)} views in "
          f"{(time.time()-t_all)/60:.1f} min -> {cdir}", flush=True)


if __name__ == "__main__":
    main()
