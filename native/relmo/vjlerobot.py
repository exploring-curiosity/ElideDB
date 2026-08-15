"""Ingest a RoboCasa LeRobot release directly. No simulation, no re-render.

WHY THIS EXISTS. The 447-episode rcasa corpus was built by replaying every
demonstration through MuJoCo to recover exact GT - segmentation, depth,
contacts, joint state - at ~5 s/episode with THREE render passes per frame and
3.3 MB of compressed state per episode, 6.4x the size of the video itself.

That GT is no longer used. The ranker trains on latents with a graded relevance
built from task identity, so the only things needed per episode are the VIDEO
and WHICH TASK it is. RoboCasa ships both: every release carries rendered mp4s
at 20 fps for three camera views plus a task label. So corpus construction
stops being a simulation job and becomes an indexing job.

Available locally without any download: 63 task releases, 26,674 episodes,
80,022 rendered mp4s. rcasa used 12 tasks and 447 episodes.

WHAT IS LOST, and the owner accepted it: the visibility and gripper gates were
computed from segmentation, so they are gone. Episodes are taken as they come.
That is the production shape anyway - real video arrives ungated.

    python -m relmo.vjlerobot --task TurnOnMicrowave --split target --kind atomic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

DEMOS = (Path(__file__).resolve().parents[2] / "vendor_robocasa" / "datasets"
         / "v1.0")
# one physical event recorded from three places; the same rollout, so these
# must never be split across train/test and never train each other (cross-view
# is barred). The manifest records the rollout id so both rules stay enforceable.
CAMS = ("robot0_agentview_left", "robot0_agentview_right",
        "robot0_eye_in_hand")


def releases():
    out = []
    for info in sorted(DEMOS.glob("*/*/*/*/lerobot/meta/info.json")):
        rel = info.parent.parent
        parts = info.relative_to(DEMOS).parts
        out.append(dict(split=parts[0], kind=parts[1], task=parts[2],
                        root=rel))
    return out


def build(task, split=None, kind=None, cams=CAMS, limit=0, name=None):
    rels = [r for r in releases() if r["task"] == task
            and (split is None or r["split"] == split)
            and (kind is None or r["kind"] == kind)]
    if not rels:
        raise SystemExit(f"no release for task {task!r} "
                         f"(have: {sorted({r['task'] for r in releases()})})")
    rel = rels[0]
    root = rel["root"]
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info.get("fps", 20)

    # Two LeRobot layouts exist in this tree. The RoboCasa releases use
    # meta/episodes.jsonl with one mp4 per episode under
    # videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4; newer
    # releases use meta/episodes/*.parquet with episodes concatenated into
    # file-NNN.mp4. Handle the jsonl form here - it is what is on disk.
    rows = []
    ep_jsonl = root / "meta" / "episodes.jsonl"
    if ep_jsonl.exists():
        rows = [json.loads(l) for l in ep_jsonl.read_text().splitlines() if l]
    else:
        import pandas as pd
        df = pd.concat([pd.read_parquet(f) for f in
                        sorted((root / "meta" / "episodes").glob("*/*.parquet"))])
        rows = df.to_dict("records")

    def find_mp4(cam, idx):
        for ch in sorted((root / "videos").glob("chunk-*")):
            f = ch / f"observation.images.{cam}" / f"episode_{idx:06d}.mp4"
            if f.exists():
                return f
        return None

    episodes = []
    for r in rows:
        idx = int(r["episode_index"])
        t = r.get("tasks") or r.get("task")
        instr = str(t[0]) if hasattr(t, "__len__") and not isinstance(t, str) \
            else str(t)
        for cam in cams:
            mp4 = find_mp4(cam, idx)
            if mp4 is None:
                continue
            episodes.append(dict(
                id=f"{task}_episode_{idx:06d}__{cam}", task=task, camera=cam,
                rollout=f"{task}#{idx:06d}", instruction=instr,
                length=int(r.get("length", 0)), fps=fps,
                video=str(mp4),
                shard="lerobot", kind=rel["kind"], split_src=rel["split"]))
        if limit and len({e["rollout"] for e in episodes}) >= limit:
            break

    ds = name or f"lr_{task}"
    R.write_manifest(ds, dict(name=ds, source="robocasa_lerobot", task=task,
                              kind=rel["kind"], src_split=rel["split"],
                              fps=fps, cameras=list(cams), episodes=episodes))
    return ds, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="max ROLLOUTS (each yields up to 3 camera episodes)")
    ap.add_argument("--name", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        rs = releases()
        print(f"{len(rs)} releases on disk")
        for r in rs:
            print(f"  {r['split']:9s} {r['kind']:10s} {r['task']}")
        return
    ds, eps = build(a.task, a.split, a.kind, limit=a.limit,
                    name=a.name or None)
    n_roll = len({e["rollout"] for e in eps})
    print(json.dumps(dict(dataset=ds, task=a.task, rollouts=n_roll,
                          episodes=len(eps),
                          instructions=len({e["instruction"] for e in eps})),
                     indent=1))
    print(f"VERIFIED: {len(eps)} episode entries over {n_roll} rollouts")
    R.log("vjlerobot", dataset=ds, task=a.task, rollouts=n_roll,
          episodes=len(eps))


if __name__ == "__main__":
    main()
