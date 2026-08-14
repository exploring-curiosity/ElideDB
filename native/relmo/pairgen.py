"""L5.0b — generate the SCENE-MATCHED corpus.

THE PROPERTY THIS CORPUS HAS AND rcasa DOES NOT. On rcasa, verbs per
kitchen is exactly 1.00: every episode was staged in its own kitchen and
no kitchen was ever reused across verbs, so scene identity is a perfect
predictor of the label and no relabeling can decouple them (measured in
L5.0). Here every scene hosts AT LEAST TWO VERBS, so within a scene the
room is constant while the label varies, and a room descriptor is forced
to chance BY CONSTRUCTION rather than by hoping.

SCENE = (layout_id, style_id) from the source demo's ep_meta.json.
Verified before generating (L5.0b gating check): two demos sharing that
pair share the kitchen INSTANCE - fixture-name Jaccard 1.000 against
0.163 for a control of different scenes. Distractor objects and robot
base pose are re-randomised per demo, but not in a verb-biased way
(paired t on object count p = 0.529), so they add noise, not bias.
DO NOT use gen_textures or cam_configs as evidence of scene sharing:
gen_textures is {} for every demo in the corpus and cam_configs is a
fixed dict, so both read "identical" for matched AND unmatched pairs.
Only the fixture names discriminate.

SCOPE, and the trade that set it. Tracking dominates cost at a measured
~48 s/episode (192-244 ms/frame, median 226 frames). Under a 3 h budget
cameras and scenes trade directly:
    110 scenes x 1 camera = 241 episodes = 3.2 h
    110 scenes x 2 cameras = 482 episodes = 6.4 h
     49 scenes x 2 cameras = 222 episodes = 3.0 h
The power calculation depends on SCENES, not episodes - the bootstrap
resamples scenes because that is the independent unit - and n=49 sits
below the n~100 sufficiency line while n=110 clears it. So one camera is
the default. The scenes hosting all three verbs get two cameras: they
are few (11), they yield the most contrasts per episode, and they supply
the same-event/different-viewpoint pairs the benchmark otherwise lacks.

A COUNT I HAD WRONG. I earlier reported 213 multi-verb scenes. That
counted the composite tasks (StackBowlsCabinet, LoadDishwasher,
PrepareCoffee) as verbs of their own. Restricted to the three verbs that
actually factor - open, close, move - it is 121 scenes, 11 of them with
all three. The scope above is set on the corrected number.

RESUMABLE, AND VERIFIED ON DISK. An episode already written is skipped,
so an interrupt costs one episode rather than the run. The manifest is
MERGED, never overwritten: build() once wrote only its own call's rows
and left 447 episodes on disk described by a manifest listing 128, and
tracks2 reads the manifest, so it would have tracked 128 and reported
success. Counts are checked against the directory, never against an
exit code.

    python -m relmo.pairgen --name rcasa_pairs --scenes 110
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

VERBS = {"Open": "open", "Close": "close", "PickPlaceCounterTo": "move"}


def verb_of(task):
    for p, v in VERBS.items():
        if task.startswith(p):
            return v
    return None


def scan(split="pretrain"):
    """(layout, style) -> [(task, verb, demo_dir, object)] over real verbs."""
    from relmo.rcreplay import DEMOS, episodes
    root = DEMOS / "v1.0" / split
    tasks = set()
    for kind in ("atomic", "composite"):
        p = root / kind
        if p.is_dir():
            tasks |= {d.name for d in p.iterdir() if d.is_dir()}
    sc = collections.defaultdict(list)
    for t in sorted(tasks):
        v = verb_of(t)
        if v is None:
            continue
        obj = next((o for o in ("Cabinet", "Drawer", "Microwave", "Sink")
                    if t.endswith(o)), "?")
        for d in episodes(t, split):
            f = d / "ep_meta.json"
            if not f.exists():
                continue
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            sc[(j.get("layout_id"), j.get("style_id"))].append((t, v, d, obj))
    return sc


def select(sc, n_scenes):
    """Multi-verb scenes, richest first: 3 verbs before 2, then more demos."""
    multi = {k: v for k, v in sc.items() if len({x[1] for x in v}) >= 2}
    ranked = sorted(multi.items(),
                    key=lambda kv: (-len({x[1] for x in kv[1]}),
                                    -len(kv[1]), str(kv[0])))
    return ranked[:n_scenes]


def run(name, n_scenes, split="pretrain", budget_h=None):
    import warnings
    warnings.filterwarnings("ignore")
    import robocasa  # noqa: F401  - importing REGISTERS the kitchen envs
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from tqdm import tqdm
    from relmo import rcreplay as RC

    sel = select(scan(split), n_scenes)
    n3 = sum(1 for _, v in sel if len({x[1] for x in v}) >= 3)
    jobs = []
    for (lay, sty), items in sel:
        ncam = 2 if len({x[1] for x in items}) >= 3 else 1
        for task, verb, d, obj in items:
            jobs.append(dict(task=task, verb=verb, dir=d, obj=obj,
                             lay=lay, sty=sty, ncam=ncam))
    print(f"{len(sel)} scenes ({n3} with all three verbs), {len(jobs)} demos, "
          f"~{sum(j['ncam'] for j in jobs)} episodes", flush=True)

    out_root = R.dataset_dir(name)
    prev = R.read_manifest(name) or {}
    have = {e["id"] for e in prev.get("episodes", [])}
    cfg = load_composite_controller_config(robot="PandaOmron")
    rows, rej, t0 = [], [], time.time()
    by_task = collections.defaultdict(list)
    for j in jobs:
        by_task[j["task"]].append(j)

    for task, js in by_task.items():
        # RESUME: skip demos already fully on disk before paying for an env
        # episode dirs are "{eid}__{camera}", so the resume check must
        # GLOB - testing for a directory literally named {eid} would
        # never match and every demo would be replayed again.
        def done(j):
            eid = f"{task}_{j['dir'].name}"
            return (any((out_root / "shard_0000").glob(f"{eid}__*"))
                    or any(e.startswith(eid + "__") for e in have))
        todo = [j for j in js if not done(j)]
        if not todo:
            continue
        env = robosuite.make(env_name=task, robots="PandaOmron",
                             controller_configs=cfg, has_renderer=False,
                             has_offscreen_renderer=False,
                             use_camera_obs=False, control_freq=RC.FPS,
                             ignore_done=True)
        env.reset()
        for j in tqdm(todo, unit="demo", desc=f"pairgen/{task}"):
            if budget_h and (time.time() - t0) / 3600 > budget_h:
                print("wall-clock budget reached; stopping cleanly",
                      flush=True)
                break
            RC.N_CAMS = j["ncam"]
            eid = f"{task}_{j['dir'].name}"
            try:
                res, err = RC.replay(j["dir"], env, out_root, eid)
            except Exception as exc:
                res, err = [], str(exc)[:160]
            if err:
                rej.append((eid, err))
            for r in res:
                if r.get("ok"):
                    # scene id rides in the manifest so the benchmark
                    # never has to re-derive it from the vendor demos
                    rows.append(dict(shard="shard_0000", task=task,
                                     verb=j["verb"], object=j["obj"],
                                     scene=f"{j['lay']}_{j['sty']}",
                                     layout_id=j["lay"], style_id=j["sty"],
                                     **r))
                else:
                    rej.append((f"{eid}/{r.get('cam','?')}",
                                r.get("why", "?")))
        env.close()

    # MERGE, never overwrite (see module docstring)
    keep = {e["id"]: e for e in prev.get("episodes", [])}
    keep.update({r["id"]: r for r in rows})
    all_rows = [keep[k] for k in sorted(keep)]
    R.write_manifest(name, dict(
        name=name, source="robocasa_replay_scene_matched", split=split,
        n_episodes=len(all_rows), tasks=sorted({e.get("task", "?")
                                                for e in all_rows}),
        scene_matched=True, episodes=all_rows))
    return all_rows, rej


def audit(name):
    """Does the corpus actually have the property it was built for?"""
    man = R.read_manifest(name)
    eps = man.get("episodes", [])
    by_scene = collections.defaultdict(set)
    for e in eps:
        if e.get("scene"):
            by_scene[e["scene"]].add(e.get("verb"))
    n = len(by_scene)
    multi = sum(1 for v in by_scene.values() if len(v) >= 2)
    vps = (sum(len(v) for v in by_scene.values()) / n) if n else 0
    return dict(episodes=len(eps), scenes=n, scenes_multi_verb=multi,
                verbs_per_scene=round(vps, 3),
                frac_multi=round(multi / n, 3) if n else 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rcasa_pairs")
    ap.add_argument("--scenes", type=int, default=110)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--budget-h", type=float, default=None)
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()
    if not a.audit_only:
        rows, rej = run(a.name, a.scenes, a.split, a.budget_h)
        print(f"\nmanifest now lists {len(rows)} episodes")
        if rej:
            print(f"REJECTED {len(rej)}:")
            for i, w in rej[:15]:
                print(f"  {i:44s} {w}")
    rep = audit(a.name)
    # VERIFY ON DISK, not from the manifest we just wrote
    on_disk = len(list((R.dataset_dir(a.name)).glob("shard_*/*/state.npz")))
    rep["on_disk_state_npz"] = on_disk
    print("\n" + json.dumps(rep, indent=1))
    print(f"\nTHE PROPERTY: verbs per scene = {rep['verbs_per_scene']} "
          f"(rcasa = 1.00). {rep['scenes_multi_verb']}/{rep['scenes']} "
          f"scenes host >=2 verbs.")
    R.log("pairgen", dataset=a.name, **rep)
