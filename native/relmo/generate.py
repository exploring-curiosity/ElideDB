"""Data service: keeps generating physics episodes and extending an
immutable, versioned dataset. Runs forever or for a fixed count.

    python -m relmo.generate --name physgen_v1 --target 8000

Crash-safe: an episode is built in `.tmp_<id>` and renamed only when
both frames.mp4 and state.npz exist, so an interrupted run never
leaves a partial episode that later looks complete. Resumes by
reading the manifest.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import physgen, registry as R  # noqa: E402


def _gen_module(gen: int):
    if gen == 3:
        from relmo import physgen3
        return physgen3
    if gen == 2:
        from relmo import physgen2
        return physgen2
    return physgen


def generate(name="physgen_v1", target=8000, seed_base=1_000_000,
             gen=1):
    G = _gen_module(gen)
    man = R.read_manifest(name)
    man.setdefault("episodes", [])
    man["gen_version"] = G.GEN_VERSION
    man["config"] = dict(fps=G.FPS, seg_ds=G.SEG_DS,
                         shapes=list(G.SHAPES),
                         drivers=list(getattr(G, "DRIVERS", [])),
                         textured=gen >= 3)
    have = {e["id"] for e in man["episodes"]}
    d = R.dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    made = 0
    t0 = time.time()
    for i in range(target):
        eid = f"ep{i:06d}"
        if eid in have:
            continue
        shard = d / f"shard_{i // 500:04d}"
        tmp = shard / f".tmp_{eid}"
        final = shard / eid
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        try:
            stats = G.run_episode(seed_base + i, tmp)
        except Exception as exc:                       # never die
            R.log("gen_error", dataset=name, id=eid, error=str(exc)[:200])
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        if not ((tmp / "frames.mp4").exists()
                and (tmp / "state.npz").exists()):
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        tmp.rename(final)
        man["episodes"].append(dict(id=eid, shard=shard.name, **stats))
        made += 1
        if made % 100 == 0:
            R.write_manifest(name, man)
            R.log("gen_progress", dataset=name, made=made,
                  total=len(man["episodes"]),
                  eps_per_min=round(made / max(time.time() - t0, 1) * 60, 1))
    man = R.write_manifest(name, man)
    R.log("gen_done", dataset=name, made=made,
          total=man["n_episodes"], fingerprint=man["fingerprint"],
          minutes=round((time.time() - t0) / 60, 1))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="physgen_v1")
    ap.add_argument("--target", type=int, default=8000)
    ap.add_argument("--gen", type=int, default=1,
                    help="generator version (1, 2, 3=textured)")
    a = ap.parse_args()
    m = generate(a.name, a.target, gen=a.gen)
    print(f"{a.name}: {m['n_episodes']} episodes, "
          f"fingerprint {m['fingerprint']}")
