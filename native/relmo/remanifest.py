"""Rebuild a dataset manifest from what is ACTUALLY on disk.

rcreplay.build() used to write a manifest containing only the rows from
its own invocation. genall calls build() once per task and every refill
calls it again, so the file ended up describing the last call: rcasa had
447 episodes on disk and a manifest listing 128, all from the PickPlace
refill. That is a silent defect of the worst kind, because tracks2 reads
the manifest - it would have tracked 128 episodes, written 128 files and
reported success with two thirds of the corpus invisible.

build() now merges, but the rows lost before that fix have to come back
from somewhere, and the source of truth is state.npz. Every field the
manifest carries is recomputed from the episode itself rather than
copied from a stale record, so the rebuilt manifest cannot inherit a
wrong value from the thing it is replacing.

    python -m relmo.remanifest --name rcasa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

MIN_VIS = 0.0001
MIN_GRIP = 0.002


def row(ep_dir: Path):
    z = np.load(ep_dir / "state.npz", allow_pickle=True)
    tg = z["target_bodies"]
    xp = z["xpos"]
    tv, gv = z["target_visible"], z["gripper_visible"]
    nw = int(z["n_windows"]) if "n_windows" in z.files else 0
    ng = int(z["n_windows_with_gripper"]) if "n_windows_with_gripper" \
        in z.files else 0
    return dict(
        shard=ep_dir.parent.name,
        task=ep_dir.name.split("_episode_")[0],
        ok=True,
        id=ep_dir.name,
        T=int(len(z["seg"])),
        camera=str(z["camera"]),
        target_move=round(float(np.linalg.norm(
            xp[-1][tg] - xp[0][tg], axis=-1).max()), 4),
        target_vis=round(float(tv.mean()), 4),
        grip_vis=round(float(gv.mean()), 4),
        seen=round(float((tv >= MIN_VIS).mean()), 4),
        grip_seen=round(float((gv >= MIN_GRIP).mean()), 4),
        win=nw,
        win_grip=ng,
        instruction=str(z["instruction"]),
    )


def rebuild(name: str, split="pretrain"):
    from tqdm import tqdm
    root = R.dataset_dir(name)
    eps = sorted(p.parent for p in root.glob("shard_*/*/state.npz"))
    rows, bad = [], []
    for e in tqdm(eps, unit="ep", desc=f"remanifest/{name}"):
        try:
            rows.append(row(e))
        except Exception as exc:
            bad.append((e.name, str(exc)[:120]))
    prev = R.read_manifest(name) or {}
    man = R.write_manifest(name, dict(
        name=name, source=prev.get("source", "robocasa_replay"),
        tasks=sorted({r["task"] for r in rows}),
        split=prev.get("split", split), episodes=rows))
    return man, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rcasa")
    a = ap.parse_args()
    man, bad = rebuild(a.name)
    import collections
    c = collections.Counter(e["task"] for e in man["episodes"])
    print(f"\n{man['name']}: {man['n_episodes']} episodes, "
          f"{len(man['tasks'])} tasks, fingerprint {man['fingerprint']}")
    for t, n in sorted(c.items()):
        print(f"  {t:30s} {n:4d}")
    if bad:
        print(f"\n{len(bad)} unreadable:")
        for i, e in bad[:10]:
            print(f"  {i}: {e}")
