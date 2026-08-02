"""CHAIN EPISODES: programs of 5-8 spatial events, not pick-stack loops.

The user's target is CONTEXTUAL retrieval: chains of 4-5+ events linked
in time AND space ("picked from the left, stacked on the cylinder,
later unstacked and pushed to the far right"). Two-event temporal is
just semantics; context lives in long chains. So an episode here is a
sampled PROGRAM over primitives, executed by the same arm machinery as
sim_stack and verified event by event from sim state:

    pick(block)              carried check (z gain, not absolute z)
    place(block, zone)       block inside the named table zone
    stack(block, on_block)   resting on the live top, height-verified
    unstack(top, zone)       removed from its tower to a zone
    push(block, direction)   slid along the table, displacement-verified

Zones are a 3x2 grid over the table (left/mid/right x near/far) -
generator-side names, eval-only like every other label. TEMPLATES give
chain SHAPES support; bindings (shapes, colours, zones) are sampled per
episode against a persistent SIGNATURE REGISTRY so no two episodes of
the same template ever show the same-looking setup - colours are drawn
without replacement, camera poses jitter per episode, and 2 of the 4
viewpoints are recorded (random pair): half the render/encode cost of
v1 and more visual variety, not less.

Speed: v1 averaged ~9s of video per event (conservative gains, fixed
0.45m transfers, long settles). Here unloaded moves run at FREE_GAIN,
loaded moves at CARRY_GAIN (smooth beats fast while holding - transit
drops were the #1 visible failure), and every clearance height is
computed from what is actually standing instead of a fixed altitude,
so flat-table transfers are low and quick while tower transfers clear.

Truth: meta.json per episode + rows appended to truth.parquet.
EVAL-ONLY sidecars; the store gets pixels.

    python scripts/sim_chains.py --episodes 12          # pilot
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

from sim_stack import (Arm, PALETTE, RES, RFPS, SHAPES,        # noqa: E402
                       build_scene)

Z_TOP = 0.2
NCAMS = 2
FREE_GAIN = 5.0          # nothing held: get there
CARRY_GAIN = 3.0         # block held: acceleration is what sheds blocks
ZONES = {                              # 3x2 grid, inside dexterous reach
    "left_near": (0.34, -0.20), "left_far": (0.52, -0.20),
    "mid_near": (0.34, 0.0), "mid_far": (0.52, 0.0),
    "right_near": (0.34, 0.20), "right_far": (0.52, 0.20),
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


def sample_binding(ep, rng, registry):
    """Template round-robin (even support per chain shape) + bindings
    redrawn until the (shapes, colours, zones) signature is one the
    registry has never seen - the no-identical-setup rule."""
    tname = list(TEMPLATES)[ep % len(TEMPLATES)]
    T = TEMPLATES[tname]
    zused = sorted({s[2] for s in T["steps"] if len(s) > 2
                    and isinstance(s[2], str)})
    for _ in range(80):
        shapes = [SHAPES[rng.integers(2)] for _ in range(T["n"])]
        colors = list(rng.choice(list(PALETTE), size=T["n"], replace=False))
        zperm = list(rng.permutation(list(ZONES)))
        zone_bind = {f"Z{k}": zperm[k] for k in range(3)}
        key = "\t".join((tname,
                         "|".join(f"{s}:{c}" for s, c in zip(shapes, colors)),
                         "|".join(zone_bind[z] for z in zused)))
        if key not in registry:
            break
    registry.add(key)
    return tname, list(zip(shapes, colors)), zone_bind, key


def run_episode(ep_id, tname, spec, zone_bind, rng, out_dir, log):
    T = TEMPLATES[tname]
    scene, meta, _ = build_scene(rng, T["n"], spec=spec)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    arm = Arm(m, d)
    arm.to_home()
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, RES[0], RES[1])
    rec = sorted(int(c) for c in rng.choice(4, NCAMS, replace=False))
    cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{c}")
            for c in rec]
    ep_dir = out_dir / f"ep{ep_id:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    enc = [subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{RES[1]}x{RES[0]}", "-r", str(RFPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", str(ep_dir / f"cam{c}.mp4")],
        stdin=subprocess.PIPE) for c in rec]
    dt = m.opt.timestep
    spf = max(1, int(round(1 / (RFPS * dt))))
    frame_n = 0
    wall0 = time.time()

    def frames():
        nonlocal frame_n
        for cam, p in zip(cams, enc):
            r.update_scene(d, camera=cam)
            p.stdin.write(r.render().tobytes())
        frame_n += 1

    def sim(seconds):
        n = int(seconds / dt)
        for s in range(n):
            mujoco.mj_step(m, d)
            if s % spf == 0:
                frames()

    def move_to(target, tol=0.012, timeout=2.5, gain=FREE_GAIN):
        n = int(timeout / dt)
        for s in range(n):
            e = arm.step_ik(np.asarray(target), gain=gain)
            mujoco.mj_step(m, d)
            if s % spf == 0:
                frames()
            if e < tol:
                break

    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"])
           for b in meta]

    def obs_top(exclude=()):
        """Highest standing surface among blocks not in `exclude` -
        clearance is computed from the scene as it stands, not from a
        fixed altitude (0.45 barely cleared a 3-story tower; a flat
        table never needed it)."""
        tops = [float(d.xpos[bid[j]][2]) + meta[j]["half_h"]
                for j in range(len(meta)) if j not in exclude]
        return max(tops + [Z_TOP])

    def grasp(i):
        for attempt in range(3):
            arm.gripper(True)
            bp = d.xpos[bid[i]].copy()
            az = min(0.55, max(0.34, obs_top((i,)) + arm.tip_off + 0.02))
            move_to([bp[0], bp[1], az], timeout=2.0)
            sim(0.12)              # kill the approach swing pre-descend
            n = int(2.5 / dt)
            for st in range(n):
                bp = d.xpos[bid[i]]
                zt = bp[2] + arm.tip_off - meta[i]["half_h"] * 0.4
                hp = d.xpos[arm.hand]
                xy = float(np.hypot(hp[0] - bp[0], hp[1] - bp[1]))
                # capture offset becomes GRIP offset: whatever xy error
                # exists when the pads close is frozen into the grasp
                # and inherited by every later placement (a 1cm-off
                # grip toppled 40%% of stacks). With command-side IK
                # integration the arm can hold 1-2mm, so gate tight.
                z = max(hp[2] - 0.0035, zt) if xy < 0.008 else hp[2]
                arm.step_ik(np.array([bp[0], bp[1], z]))
                mujoco.mj_step(m, d)
                if st % spf == 0:
                    frames()
                if hp[2] <= zt + 0.003 and xy < 0.005:
                    break
            arm.gripper(False)
            sim(0.3)
            # carried = the block GAINED height with the lift; the old
            # absolute-z check called a block still sitting on its
            # tower "carried". Threshold sits well under the lift
            # height: the probe measured a real grasp rising 67mm and a
            # 70mm bar failed the entire pilot.
            z0 = float(d.xpos[bid[i]][2])
            cur = d.xpos[arm.hand].copy()
            move_to([cur[0], cur[1], cur[2] + 0.14], timeout=1.6,
                    gain=CARRY_GAIN)
            if float(d.xpos[bid[i]][2]) > z0 + 0.05:
                return True
        return False

    def lower_release(i, txy, slot_z, track=None):
        tx_, ty_ = float(txy[0]), float(txy[1])
        cz = min(0.58, max(0.36, obs_top((i,)) + arm.tip_off
                           + 0.6 * meta[i]["half_h"] + 0.03))
        cur = d.xpos[arm.hand].copy()
        move_to([(cur[0] + tx_) / 2, (cur[1] + ty_) / 2, cz],
                timeout=1.5, gain=CARRY_GAIN)
        move_to([tx_, ty_, cz], tol=0.007, timeout=2.2, gain=CARRY_GAIN)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gblk{i}")
        n = int(3.0 / dt)
        contact_at = None
        z_hold = None
        for st in range(n):
            if track is not None:      # follow the live top as it drifts
                tx_, ty_ = float(d.xpos[track][0]), float(d.xpos[track][1])
            hp = d.xpos[arm.hand]
            bp_ = d.xpos[bid[i]]
            # servo the HELD BLOCK over the target, not the hand: any
            # residual capture offset in the grip would otherwise land
            # the block on the base's edge and torque it over - the
            # measured topple mode of this pilot's failed stacks
            ox, oy = bp_[0] - hp[0], bp_[1] - hp[1]
            xy = float(np.hypot(bp_[0] - tx_, bp_[1] - ty_))
            # after contact, FREEZE the z command +1mm: descending
            # through the hold made the integrator wind up and press
            # the tower, and the ramp-open released that stored press
            # as a pop that shed the block
            if z_hold is not None:
                z = z_hold
            else:
                # fine phase for the last 15mm: a base kissed at 3mm/step
                # while 8mm off-centre is how cylinder towers toppled
                near = float(d.xpos[bid[i]][2]) - slot_z < 0.015
                rate, lim = (0.0015, 0.005) if near else (0.003, 0.010)
                z = hp[2] - rate if xy < lim else hp[2]
            arm.step_ik(np.array([tx_ - ox, ty_ - oy, z]), gain=CARRY_GAIN)
            mujoco.mj_step(m, d)
            if st % spf == 0:
                frames()
            hit = any(gid in (d.contact[c].geom1, d.contact[c].geom2)
                      and d.contact[c].geom1 not in arm.finger_geoms
                      and d.contact[c].geom2 not in arm.finger_geoms
                      for c in range(d.ncon))
            # the landed test only counts ABOVE the target: a held
            # block dangling at hover already sits below a 1-story
            # slot_z, and that false break released 5cm off-target
            if (hit or d.xpos[bid[i]][2] <= slot_z + 0.003) and xy < 0.015:
                if contact_at is None:
                    contact_at = st
                    z_hold = float(d.xpos[arm.hand][2]) + 0.001
                if st - contact_at > int(0.2 / dt):
                    break
        # RAMP the fingers open while the arm holds still: snap-open at
        # 255 flicked top stories off (29% of v1 blocks placed then
        # toppled)
        hold = d.xpos[arm.hand].copy()
        n = int(0.25 / dt)
        for st in range(n):
            d.ctrl[arm.grip] = 255.0 * (st + 1) / n
            arm.step_ik(hold, gain=CARRY_GAIN)
            mujoco.mj_step(m, d)
            if st % spf == 0:
                frames()
        cur = d.xpos[arm.hand].copy()
        move_to([cur[0], cur[1], cur[2] + 0.10], timeout=1.0)

    def free_spot(i, zx, zy):
        """Drop point inside the zone that clears every other block:
        zones get occupied at spawn and mid-chain, and setting a block
        down ON an occupier was a measured place-failure (the cylinder
        landed on a box and rolled off). Max nudge 5.5cm keeps the
        point nearest to this zone's centre (zones are 18-20cm apart)."""
        cands = [(zx, zy)]
        for rad in (0.035, 0.055):
            for a in range(8):
                th = a * np.pi / 4
                cands.append((zx + rad * np.cos(th), zy + rad * np.sin(th)))
        for cx, cy in cands:
            if all(np.hypot(cx - d.xpos[bid[j]][0], cy - d.xpos[bid[j]][1])
                   > meta[i]["half_w"] + meta[j]["half_w"] + 0.012
                   for j in range(len(meta)) if j != i):
                return cx, cy
        return zx, zy

    def standing_set():
        return [[round(float(x), 4) for x in d.xpos[bid[j]]]
                for j in range(len(meta))]

    sim(0.3)
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
                if not grasp(i):
                    events.append({"i": len(events), "prim": prim,
                                   "block": meta[i]["name"],
                                   "shape": meta[i]["shape"],
                                   "color": meta[i]["color"], "ok": False,
                                   "t0": round(t0, 1),
                                   "t1": round(frame_n / RFPS, 1),
                                   "zone": zname})
                    continue
            if held == i or prim == "unstack":
                lower_release(i, free_spot(i, zx, zy),
                              Z_TOP + meta[i]["half_h"])
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
                lower_release(i, (float(top[0]), float(top[1])), slot,
                              track=bid[j])
                p = d.xpos[bid[i]]
                # verify against the LIVE top, not the pre-placement
                # slot: the target block drifts a few mm under contact
                exp_z = (float(d.xpos[bid[j]][2]) + meta[j]["half_h"]
                         + meta[i]["half_h"])
                ok = (np.hypot(p[0] - d.xpos[bid[j]][0],
                               p[1] - d.xpos[bid[j]][1]) < 0.04
                      and abs(p[2] - exp_z) < 0.02)
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
            # fingertips at block mid-height; the old bp+0.01 hand
            # target put the tips below the TABLE and pushed by pressing
            # into it
            pz = bp[2] + arm.tip_off - meta[i]["half_h"] * 0.2
            arm.gripper(False)
            az = min(0.55, max(0.34, obs_top((i,)) + arm.tip_off + 0.02))
            move_to([start[0], start[1], az], timeout=2.0)
            move_to([start[0], start[1], pz], tol=0.008, timeout=1.6)
            push_d = min(dist, 0.12)
            end = start + u * (push_d + meta[i]["half_w"] + 0.06)
            n = int(1.8 / dt)
            for st in range(n):
                frac = min(1.0, (st * dt) / 1.5)
                tgt = start + (end - start) * frac
                arm.step_ik(np.array([tgt[0], tgt[1], pz]))
                mujoco.mj_step(m, d)
                if st % spf == 0:
                    frames()
            cur = d.xpos[arm.hand].copy()
            move_to([cur[0], cur[1], cur[2] + 0.15], timeout=1.0)
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
    move_to([0.35, 0.0, 0.55], timeout=1.5)
    sim(0.3)

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
    rec_out = {"episode": ep_id, "template": tname,
               "blocks": meta, "events": events,
               "zone_bind": zone_bind, "cams": rec,
               "events_ok": sum(ok_flags), "events_total": len(events),
               "chain_prefix_ok": chain_ok,
               "success": all(ok_flags),
               "frames_per_cam": frame_n,
               "seconds": round(frame_n / RFPS, 1),
               "wall_s": round(time.time() - wall0, 1)}
    (ep_dir / "meta.json").write_text(json.dumps(rec_out, indent=1))
    log.append(rec_out)
    return rec_out


def main():
    argv = sys.argv
    n_ep = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv \
        else 3
    out = ROOT / (argv[argv.index("--out") + 1] if "--out" in argv
                  else "data/sim_chains")
    start = int(argv[argv.index("--start") + 1]) if "--start" in argv else 0
    out.mkdir(parents=True, exist_ok=True)
    sig_f = out / "signatures.json"
    registry = set(json.loads(sig_f.read_text())) if sig_f.exists() else set()
    log = []
    t0 = time.time()
    from tqdm import tqdm
    for ep in tqdm(range(start, start + n_ep), desc="chains", unit="ep"):
        rng = np.random.default_rng(5000 + ep)
        tname, spec, zone_bind, key = sample_binding(ep, rng, registry)
        rec = run_episode(ep, tname, spec, zone_bind, rng, out, log)
        sig_f.write_text(json.dumps(sorted(registry)))
        tqdm.write(f"  ep{ep:04d} {rec['template']:<18} "
                   f"events {rec['events_ok']}/{rec['events_total']} "
                   f"chain_prefix {rec['chain_prefix_ok']} "
                   f"({rec['seconds']}s film, {rec['wall_s']}s wall)")
    full = sum(1 for r in log if r["success"])
    print(json.dumps({"episodes": len(log), "full_chain": full,
                      "mean_events_ok": round(float(np.mean(
                          [r["events_ok"] / r["events_total"]
                           for r in log])), 2),
                      "film_min": round(sum(r["seconds"] for r in log) / 60,
                                        1),
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
