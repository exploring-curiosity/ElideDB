"""CHAIN EPISODES: programs of 5-8 spatial events, not pick-stack loops.

The user's target is CONTEXTUAL retrieval: chains of 4-5+ events linked
in time AND space ("picked from the left, stacked on the cylinder,
later unstacked and pushed to the far right"). Two-event temporal is
just semantics; context lives in long chains. So an episode here is a
sampled PROGRAM over primitives, executed by the same arm machinery as
sim_stack and verified event by event from sim state:

    pick(block)              carried check
    place(block, zone)       block inside the named table zone
    stack(block, on_block)   resting on the live top, height-verified
    unstack(top, zone)       removed from its tower to a zone
    push(block, direction)   slid along the table, displacement-verified

Zones are a 3x2 grid over the table (left/mid/right x near/far) -
generator-side names, eval-only like every other label. TEMPLATES give
chain SHAPES support (~episodes/templates each); random block/zone
bindings refine them into attribute-bound chains. One template is the
PRECARIOUS build (random stack order, box-on-cylinder possible, 4-story
attempts) with COLLAPSE ATTRIBUTION: after every stack the standing set
is snapshotted, so the truth records which placement toppled what -
making "box on cylinder, another box added, it topples" gradable.

This is the v2 episode runner; sim_stack.py remains the v1 generator
whose scene/arm/IK it imports. Truth: meta.json per episode + rows
appended to data/sim_chains/truth.parquet.

    python scripts/sim_chains.py --episodes 3
    python scripts/sim_chains.py --episodes 150
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sim_stack import (Arm, PALETTE, RES, RFPS, build_scene)   # noqa: E402

Z_TOP = 0.2
ZONES = {                              # 3x2 grid over the table
    "left_near": (0.36, -0.24), "left_far": (0.58, -0.24),
    "mid_near": (0.36, 0.0), "mid_far": (0.58, 0.0),
    "right_near": (0.36, 0.24), "right_far": (0.58, 0.24),
}

# templates: chain shapes; `n` blocks needed. Steps reference blocks by
# index and zones by draw. Bindings randomised per episode.
TEMPLATES = {
    "relocate_build": dict(n=3, steps=[
        ("pick", 0), ("place", 0, "Z0"),
        ("pick", 1), ("stack", 1, 0),
        ("pick", 2), ("stack", 2, 1)]),
    "build_unstack_move": dict(n=3, steps=[
        ("pick", 0), ("place", 0, "Z0"),
        ("pick", 1), ("stack", 1, 0),
        ("unstack", 1, "Z1"),
        ("pick", 2), ("stack", 2, 0)]),
    "push_then_build": dict(n=3, steps=[
        ("push", 0, "Z0"),
        ("pick", 1), ("stack", 1, 0),
        ("pick", 2), ("stack", 2, 1)]),
    "two_sites_merge": dict(n=3, steps=[
        ("pick", 0), ("place", 0, "Z0"),
        ("pick", 1), ("place", 1, "Z1"),
        ("pick", 2), ("stack", 2, 0),
        ("unstack", 2, "Z1")]),
    "swap": dict(n=2, steps=[
        ("pick", 0), ("place", 0, "Z2"),
        ("pick", 1), ("place", 1, "Z0"),
        ("pick", 0), ("place", 0, "Z1")]),
    "precarious": dict(n=4, steps=[
        ("pick", 0), ("place", 0, "Z0"),
        ("pick", 1), ("stack", 1, 0),
        ("pick", 2), ("stack", 2, 1),
        ("pick", 3), ("stack", 3, 2)], precarious=True),
}


def zone_of(x, y):
    best, bd = None, 1e9
    for name, (zx, zy) in ZONES.items():
        d = np.hypot(x - zx, y - zy)
        if d < bd:
            best, bd = name, d
    return best


def run_episode(ep_id, rng, out_dir, log):
    tname = list(TEMPLATES)[rng.integers(len(TEMPLATES))]
    T = TEMPLATES[tname]
    scene, meta, _ = build_scene(rng, T["n"])
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    arm = Arm(m, d)
    arm.to_home()
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, RES[0], RES[1])
    cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{i}")
            for i in range(4)]
    ep_dir = out_dir / f"ep{ep_id:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    enc = [subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{RES[1]}x{RES[0]}", "-r", str(RFPS), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", str(ep_dir / f"cam{i}.mp4")],
        stdin=subprocess.PIPE) for i in range(4)]
    spf = max(1, int(round(1 / (RFPS * m.opt.timestep))))
    frame_n = 0

    def frames():
        nonlocal frame_n
        for cam, p in zip(cams, enc):
            r.update_scene(d, camera=cam)
            p.stdin.write(r.render().tobytes())
        frame_n += 1

    def sim(seconds):
        n = int(seconds / m.opt.timestep)
        for s in range(n):
            mujoco.mj_step(m, d)
            if s % spf == 0:
                frames()

    def move_to(target, tol=0.012, timeout=3.0):
        n = int(timeout / m.opt.timestep)
        for s in range(n):
            e = arm.step_ik(np.asarray(target))
            mujoco.mj_step(m, d)
            if s % spf == 0:
                frames()
            if e < tol:
                break

    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"])
           for b in meta]
    # bind zone slots to distinct random zones
    zslots = list(rng.permutation(list(ZONES)))
    zone_bind = {f"Z{i}": zslots[i] for i in range(3)}

    def grasp(i):
        for attempt in range(3):
            bp = d.xpos[bid[i]].copy()
            move_to([bp[0], bp[1], bp[2] + 0.22])
            arm.gripper(True)
            sim(0.2)
            n = int(3.5 / m.opt.timestep)
            for st in range(n):
                bp = d.xpos[bid[i]]
                zt = bp[2] + arm.tip_off - meta[i]["half_h"] * 0.4
                hp = d.xpos[arm.hand]
                xy = float(np.hypot(hp[0] - bp[0], hp[1] - bp[1]))
                z = max(hp[2] - 0.0025, zt) if xy < 0.012 else hp[2]
                arm.step_ik(np.array([bp[0], bp[1], z]))
                mujoco.mj_step(m, d)
                if st % spf == 0:
                    frames()
                if hp[2] <= zt + 0.002 and xy < 0.008:
                    break
            arm.gripper(False)
            sim(0.6)
            cur = d.xpos[arm.hand].copy()
            move_to([cur[0], cur[1], 0.45], timeout=2.0)
            if d.xpos[bid[i]][2] > Z_TOP + meta[i]["half_h"] + 0.05:
                return True
        return False

    def lower_release(i, txy, slot_z):
        cur = d.xpos[arm.hand].copy()
        move_to([(cur[0] + txy[0]) / 2, (cur[1] + txy[1]) / 2, 0.45],
                timeout=2.0)
        move_to([txy[0], txy[1], 0.45], tol=0.006, timeout=2.5)
        n = int(4.0 / m.opt.timestep)
        contact_at = None
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gblk{i}")
        for st in range(n):
            hp = d.xpos[arm.hand]
            xy = float(np.hypot(hp[0] - txy[0], hp[1] - txy[1]))
            z = hp[2] - 0.002 if xy < 0.010 else hp[2]
            arm.step_ik(np.array([txy[0], txy[1], z]))
            mujoco.mj_step(m, d)
            if st % spf == 0:
                frames()
            hit = any(gid in (d.contact[c].geom1, d.contact[c].geom2)
                      and d.contact[c].geom1 not in arm.finger_geoms
                      and d.contact[c].geom2 not in arm.finger_geoms
                      for c in range(d.ncon))
            if hit or d.xpos[bid[i]][2] <= slot_z + 0.003:
                if contact_at is None:
                    contact_at = st
                if st - contact_at > int(0.25 / m.opt.timestep):
                    break
        arm.gripper(True)
        sim(0.5)
        cur = d.xpos[arm.hand].copy()
        move_to([cur[0], cur[1], 0.50], timeout=1.5)

    def standing_set():
        out = []
        for j in range(len(meta)):
            p = d.xpos[bid[j]]
            out.append([round(float(x), 4) for x in p])
        return out

    sim(0.5)
    events = []
    held = None
    for step in T["steps"]:
        prim, i = step[0], step[1]
        t0 = frame_n / RFPS
        ok = False
        extra = {}
        if prim == "pick":
            ok = grasp(i)
            held = i if ok else None
        elif prim in ("place", "unstack"):
            zname = zone_bind[step[2]]
            zx, zy = ZONES[zname]
            if prim == "unstack" and held != i:
                ok0 = grasp(i)
                if not ok0:
                    events.append({"i": len(events), "prim": prim,
                                   "block": meta[i]["name"], "ok": False,
                                   "t0": t0, "t1": frame_n / RFPS,
                                   "zone": zname})
                    continue
            if held == i or prim == "unstack":
                lower_release(i, (zx, zy), Z_TOP + meta[i]["half_h"])
                p = d.xpos[bid[i]]
                ok = (zone_of(p[0], p[1]) == zname
                      and p[2] < Z_TOP + meta[i]["half_h"] + 0.02)
                held = None
            extra = {"zone": zname}
        elif prim == "stack":
            j = step[2]
            if held == i:
                top = d.xpos[bid[j]]
                slot = float(top[2]) + meta[j]["half_h"] + meta[i]["half_h"]
                before = standing_set()
                lower_release(i, (float(top[0]), float(top[1])), slot)
                p = d.xpos[bid[i]]
                ok = (np.hypot(p[0] - d.xpos[bid[j]][0],
                               p[1] - d.xpos[bid[j]][1]) < 0.04
                      and abs(p[2] - slot) < 0.028)
                held = None
                after = standing_set()
                # COLLAPSE ATTRIBUTION: which blocks moved >3cm during
                # this placement - the causal label for topple queries
                moved = [meta[k]["name"] for k in range(len(meta))
                         if k != i and np.linalg.norm(
                             np.array(after[k]) - np.array(before[k])) > 0.03]
                extra = {"on": meta[j]["name"], "toppled": moved}
        elif prim == "push":
            zname = zone_bind[step[2]]
            zx, zy = ZONES[zname]
            bp = d.xpos[bid[i]].copy()
            v = np.array([zx - bp[0], zy - bp[1]])
            dist = float(np.linalg.norm(v))
            u = v / max(dist, 1e-6)
            start = bp[:2] - u * (meta[i]["half_w"] + 0.05)
            arm.gripper(False)
            sim(0.2)
            move_to([start[0], start[1], bp[2] + 0.15])
            move_to([start[0], start[1], bp[2] + 0.01], tol=0.008,
                    timeout=2.0)
            push_d = min(dist, 0.12)
            end = start + u * (push_d + meta[i]["half_w"] + 0.06)
            n = int(2.5 / m.opt.timestep)
            for st in range(n):
                hp = d.xpos[arm.hand]
                frac = min(1.0, (st * m.opt.timestep) / 2.0)
                tgt = start + (end - start) * frac
                arm.step_ik(np.array([tgt[0], tgt[1], bp[2] + 0.01]))
                mujoco.mj_step(m, d)
                if st % spf == 0:
                    frames()
            cur = d.xpos[arm.hand].copy()
            move_to([cur[0], cur[1], 0.45], timeout=1.5)
            p = d.xpos[bid[i]]
            ok = float(np.hypot(p[0] - bp[0], p[1] - bp[1])) > 0.05
            extra = {"zone": zname,
                     "moved_mm": round(float(np.hypot(
                         p[0] - bp[0], p[1] - bp[1])) * 1000)}
        events.append({"i": len(events), "prim": prim,
                       "block": meta[i]["name"],
                       "shape": meta[i]["shape"], "color": meta[i]["color"],
                       "ok": bool(ok), "t0": round(t0, 1),
                       "t1": round(frame_n / RFPS, 1), **extra})
    move_to([0.35, 0.0, 0.6], timeout=2.0)
    sim(0.8)

    for p in enc:
        p.stdin.close()
    for p in enc:
        p.wait()
    ok_flags = [e["ok"] for e in events]
    chain_ok = 0
    for f in ok_flags:
        if f:
            chain_ok += 1
        else:
            break
    rec = {"episode": ep_id, "template": tname,
           "blocks": meta, "events": events,
           "events_ok": sum(ok_flags), "events_total": len(events),
           "chain_prefix_ok": chain_ok,
           "success": all(ok_flags),
           "frames_per_cam": frame_n,
           "seconds": round(frame_n / RFPS, 1)}
    (ep_dir / "meta.json").write_text(json.dumps(rec, indent=1))
    log.append(rec)
    return rec


def main():
    argv = sys.argv
    n_ep = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv \
        else 3
    out = ROOT / (argv[argv.index("--out") + 1] if "--out" in argv
                  else "data/sim_chains")
    start = int(argv[argv.index("--start") + 1]) if "--start" in argv else 0
    out.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    from tqdm import tqdm
    for ep in tqdm(range(start, start + n_ep), desc="chains", unit="ep"):
        rng = np.random.default_rng(5000 + ep)
        # precarious template needs NON-sorted stacking: monkey-free -
        # build_scene randomises shapes already; order comes from the
        # template's step list, so box-on-cylinder occurs naturally
        rec = run_episode(ep, rng, out, log)
        tqdm.write(f"  ep{ep:04d} {rec['template']:<18} "
                   f"events {rec['events_ok']}/{rec['events_total']} "
                   f"chain_prefix {rec['chain_prefix_ok']}")
    full = sum(1 for r in log if r["success"])
    print(json.dumps({"episodes": len(log), "full_chain": full,
                      "mean_events_ok": round(float(np.mean(
                          [r["events_ok"] / r["events_total"]
                           for r in log])), 2),
                      "wall_min": round((time.time() - t0) / 60, 1)}))
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    for rec in log:
        for e in rec["events"]:
            rows.append({"episode": rec["episode"],
                         "template": rec["template"], **{
                             k: (json.dumps(v) if isinstance(v, list)
                                 else v) for k, v in e.items()}})
    t = pa.Table.from_pylist(rows)
    f = out / "truth.parquet"
    if f.exists():
        t = pa.concat_tables([pq.read_table(f), t], promote_options="default")
    pq.write_table(t, f)
    print(f"truth sidecar: {len(t)} rows -> {f}")


if __name__ == "__main__":
    main()
