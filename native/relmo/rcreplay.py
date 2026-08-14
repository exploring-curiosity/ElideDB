"""Replay RoboCasa's own human demonstrations and log our own ground truth.

WHY THIS REPLACES rcgen.py. rcgen drove the arm with a hand-written
"generic 5-phase contact seeker" policy, because I concluded RoboCasa's
LeRobot-format demos carried no simulator state. That was wrong: every
demo ships extras/episode_*/ with

    model.xml.gz   the exact MJCF for that episode
    states.npz     the full per-frame MuJoCo state [time, qpos, qvel]
    ep_meta.json   scene spec, cam_configs, and the language instruction

so the demonstrations can be replayed exactly and rendered any way we
like. The cost of the shortcut, measured over 200 rcgen episodes: the
manipulation target moved in 9 of them (4.5%), a scene object of any kind
moved in 44%, and 16% had the camera buried inside geometry. Four
training runs were spent on that corpus before anyone looked at it.

No policy of mine appears anywhere in this file. The motion is human.

FOUR SOURCE-LEVEL FIXES, one per observed defect:

  textures    robosuite/robocasa setup_macros.py had never been run, so
              materials fell back to the missing-texture magenta that
              looked like "flickering camouflage". Fixed outside this
              file; asserted here so it cannot regress silently.
  no real op  replay human demos instead of a heuristic policy.
  buried cam  rcgen cycled a hardcoded camera list and never checked what
              it saw. Cameras are now SCORED per episode and the episode
              is rejected if none is usable.
  occlusion   the manipulated body must actually be visible, for most of
              the episode, or the episode is rejected.

The target body is discovered by replaying the states first and taking
the non-robot bodies that move - no name matching, no task-specific
rules, so it works for drawers, doors, and pick-and-place alike.

    python -m relmo.rcreplay --tasks OpenDrawer CloseDrawer --limit 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.rcgen import FPS, GEN_VERSION, H, MAXC, SEG_DS, W  # noqa: E402
from relmo.rcgen import _mp4, structure  # noqa: E402

DEMOS = Path(__file__).resolve().parents[2] / "vendor_robocasa" / "datasets"
ROBOT = ("robot", "gripper", "mount", "mobilebase", "world")
# acceptance thresholds - an episode that fails any of these is DROPPED,
# and the reason is recorded so a rejection rate can be read off later
MIN_MOVE = 0.05      # m, the manipulated body must actually travel
# ~30 pixels at 640x480, NOT a percentage. As a fraction (0.004) this
# gate was size-dependent: a cabinet door covers 10% of the frame while a
# lemon being carried to the sink covers ~0.3%, and it dips below any
# percentage threshold exactly when the gripper closes on it. That cost
# PickPlaceCounterToSink 95% of its episodes ("0 usable windows") while
# the identical pipeline kept 100% of the articulation tasks. A small
# object that occupies 30 pixels is still a trackable object.
MIN_VIS = 0.0001     # target must cover >=~30 pixels...
MIN_VIS_FRAC = 0.5   # ...in >=50% of frames
MIN_DEPTH_STD = 0.15  # camera not buried in a surface
MAX_VIS = 0.60       # target must not ENGULF the frame (wrist-cam case)
MIN_BODIES = 6       # distinct bodies sharing the view = a relation exists
MIN_GRIP = 0.002     # the gripper must be visible - it is where contact is
N_CAMS = 3           # how many viewpoints to emit per demonstration
                     # (overridable: rcreplay.N_CAMS = 2 before build)
# NO CAP. 200 truncated 57-95% of atomic demos; 900 still truncated 73%
# of ArrangeTea (composite median 695-967, max 1265). Truncation is not
# merely lost footage - _targets() reads the truncated array, so the
# "target" silently becomes whatever moved before the cut.
T_MAX = 10_000
# The model trains on TC+HZ frame windows, so THAT is what a recording
# has to supply. A global "target visible in >=50% of frames" gate is the
# wrong question for a 900-frame multi-phase task where the arm
# legitimately leaves the fixture for a whole phase - it rejected
# PrepareCoffee on every camera at 0.34-0.41. Count usable windows
# instead: it is exactly what the trainer will consume.
WIN = 24             # TC + HZ
MIN_WINDOWS = 3      # MEASURED: an atomic demo of ~200 frames yields only
                     # 3-9 windows containing object motion (CloseDrawer 3,
                     # OpenDrawer 4, CloseCabinet 6, OpenMicrowave 9);
                     # composite yields 14-48. A floor of 8 threw away most
                     # of the atomic corpus. Motion is cleanly bimodal - the
                     # counts are identical at 0.01/0.005/0.002 m - so the
                     # magnitude threshold is not what is doing the work.
MIN_WIN_VIS = 0.5    # target visible in half a window's frames
MIN_WIN_MOVE = 0.01  # and actually moving within it, in metres


def episodes(task, split="pretrain"):
    """All demo directories for a task, newest capture date first.

    Searches atomic AND composite. This hardcoded "atomic" once, so the
    three composite tasks downloaded successfully (exit 0, files on disk
    under v1.0/*/composite/) and then enumerated as zero episodes - a
    task silently contributing nothing rather than failing.
    """
    out = []
    for kind in ("atomic", "composite"):
        root = DEMOS / "v1.0" / split / kind / task
        if not root.is_dir():
            continue
        for date in sorted(root.iterdir(), reverse=True):
            ex = date / "lerobot" / "extras"
            if ex.is_dir():
                out += sorted(d for d in ex.iterdir() if d.is_dir())
    return out


def _visual_only(mujoco):
    """Render ONLY the visual geoms.

    THE bug in every rcasa_v1 frame, and in rcasa_v2 until now: MuJoCo's
    Renderer shows all geom groups by default, and robosuite puts
    COLLISION geometry in group 0 with no material. This model has 976
    such geoms. Untextured, MuJoCo paints them flat default colours, and
    because they are coincident with the visual meshes the winner flips
    per pixel as the view moves - which is exactly the "magenta, green
    and blue overlapping normal surfaces, flickering" that was reported.
    The flat red cabinets, the flat green arm and the blue/magenta blobs
    on the marble were ALL collision geometry. Group 1 is the visual
    layer; showing only it gives oak cabinets, a white Panda and clean
    marble. Applied to the RGB, depth AND segmentation passes so all
    three describe the same surfaces."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    return opt


def _targets(sim, st, nb):
    """Which non-robot bodies move? Discovered, not named."""
    names = [sim.model.body_id2name(i) or "" for i in range(nb)]
    sim.set_state_from_flattened(st[0])
    sim.forward()
    a = sim.data.body_xpos.copy()
    sim.set_state_from_flattened(st[-1])
    sim.forward()
    b = sim.data.body_xpos.copy()
    disp = np.linalg.norm(b - a, axis=-1)
    scene = np.array([i for i in range(nb)
                      if not any(k in names[i].lower() for k in ROBOT)])
    if not len(scene):
        return [], 0.0
    order = scene[np.argsort(-disp[scene])]
    top = [int(i) for i in order if disp[i] >= MIN_MOVE]
    return top, float(disp[order[0]])


def _score_cameras(sim, mujoco, ren, st, cams, targets, opt, grip):
    """Pick the camera that actually SHOWS the manipulation.

    Scores each candidate on three probe frames: what fraction of pixels
    belong to a target body, and whether the depth image has structure
    (a camera wedged inside a cabinet returns a near-constant depth)."""
    probes = [len(st) // 4, len(st) // 2, 3 * len(st) // 4]
    out = []
    for cam in cams:
        cid = mujoco.mj_name2id(sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA,
                                cam)
        if cid < 0:
            continue
        vis, dstd, nbod, gv = [], [], [], []
        for t in probes:
            sim.set_state_from_flattened(st[t])
            sim.forward()
            ren.enable_segmentation_rendering()
            ren.update_scene(sim.data._data, camera=cid, scene_option=opt)
            s = ren.render()[..., 0]
            ok = (s >= 0) & (s < sim.model._model.ngeom)
            bid = np.zeros_like(s)
            bid[ok] = sim.model._model.geom_bodyid[s[ok]]
            vis.append(float(np.isin(bid, targets).mean()))
            # how many DISTINCT bodies occupy a real share of the frame
            u, c = np.unique(bid, return_counts=True)
            nbod.append(int((c / bid.size >= 0.005).sum()))
            gv.append(float(np.isin(bid, grip).mean()))
            ren.disable_segmentation_rendering()
            ren.enable_depth_rendering()
            ren.update_scene(sim.data._data, camera=cid, scene_option=opt)
            dstd.append(float(ren.render().std()))
            ren.disable_depth_rendering()
        v, ds, nb_ = float(np.mean(vis)), float(np.mean(dstd)), float(np.mean(nbod))
        g_ = float(np.mean(gv))
        # A camera is only useful if it shows the RELATION, not just the
        # target. Maximising target pixels alone selected
        # robot0_eye_in_hand for 9 of the first 10 episodes - a wrist cam
        # pressed flat against a cabinet door, which fills the frame with
        # one body and shows no scene at all. So: the target must be
        # visible but NOT engulf the view, and enough distinct bodies must
        # share the frame for a relation to exist in it.
        # the GRIPPER must be visible too: contact is where the story is,
        # and every one of the first nine episodes occluded the pincers.
        if (ds < MIN_DEPTH_STD or v > MAX_VIS or nb_ < MIN_BODIES
                or g_ < MIN_GRIP):
            continue
        score = nb_ + 4.0 * min(v, 0.25) + 6.0 * min(g_, 0.05)
        out.append((cam, score, ds, cid, v, nb_, g_))
    # rank, and hand back the best N so one demo yields several viewpoints
    return sorted(out, key=lambda z: -z[1])[:N_CAMS]


def replay(ep_dir: Path, env, out_root: Path, eid: str):
    """Replay one demonstration and emit ONE EPISODE PER CAMERA.

    Several viewpoints of the same sequence is what was asked for, and it
    is also free: the state trajectory is identical, only the render
    differs. Downstream code sees ordinary independent episodes."""
    import mujoco
    meta = json.loads((ep_dir / "ep_meta.json").read_text())
    xml = gzip.open(ep_dir / "model.xml.gz", "rt").read()
    st = np.load(ep_dir / "states.npz")["states"]
    env.set_ep_meta(meta)
    env.reset()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    sim = env.sim
    m = sim.model._model
    if st.shape[1] != 1 + m.nq + m.nv:
        return [], f"state width {st.shape[1]} != {1 + m.nq + m.nv}"
    T = min(len(st), T_MAX)
    st = st[:T]
    targets, moved = _targets(sim, st, m.nbody)
    if not targets:
        return [], f"no scene body moved (max {moved:.3f} m)"
    names = [sim.model.body_id2name(i) or "" for i in range(m.nbody)]
    grip = [i for i in range(m.nbody) if "gripper" in names[i].lower()]

    opt = _visual_only(mujoco)
    ren = mujoco.Renderer(m, H, W)
    cams = list(meta.get("cam_configs", {}).keys()) or ["robot0_agentview_left"]
    picks = _score_cameras(sim, mujoco, ren, st, cams, targets, opt, grip)
    if not picks:
        ren.close()
        return [], "no camera passed (buried / engulfed / gripper hidden)"

    H2, W2 = H // SEG_DS, W // SEG_DS
    nb = m.nbody
    made = []
    for cam, _sc, dstd, cid, vis0, nbod, gvis in picks:
        xpos = np.zeros((T, nb, 3), np.float32)
        xquat = np.zeros((T, nb, 4), np.float32)
        xvel = np.zeros((T, nb, 6), np.float32)
        qpos = np.zeros((T, m.nq), np.float32)
        cam_pos = np.zeros((T, 3), np.float32)
        cam_mat = np.zeros((T, 9), np.float32)
        # uint16, NOT uint8. These kitchens have 318-507 bodies and the
        # old buffer clipped every id above 255 to a single value. Two
        # consequences, both silent: CloseDrawer's target bodies are
        # 323/324/325/329, so the visibility gate matched nothing and
        # rejected every camera as "occluded 0.00"; and the saved GT
        # segmentation merged all high-id bodies into one, which is the
        # gbody that tracks2 reads and G2 was scored on.
        segs = np.zeros((T, H2, W2), np.uint16)
        deps = np.zeros((T, H2, W2), np.float16)
        con = np.zeros((T, MAXC, 3), np.float32)
        ncon = np.zeros(T, np.int32)
        n_over = 0
        visf = np.zeros(T, np.float32)
        gripf = np.zeros(T, np.float32)
        frames = []
        for t in range(T):
            sim.set_state_from_flattened(st[t])
            sim.forward()
            d = sim.data._data
            xpos[t], xquat[t] = d.xpos, d.xquat
            # cvel is expressed about subtree_com, but xpos is the body
            # ORIGIN, so the raw linear part is NOT the velocity of the
            # position stored beside it. MEASURED on OpenCabinet: raw
            # cvel vs d(xpos)/dt is 1.43 relative error; shifted to the
            # origin it is 0.076, which is finite-difference noise. Store
            # the shifted value so (xpos, xvel) is a consistent pair.
            ang = d.cvel[:, :3]
            lin = d.cvel[:, 3:] + np.cross(
                ang, d.xpos - d.subtree_com[m.body_rootid])
            xvel[t, :, :3], xvel[t, :, 3:] = lin, ang
            qpos[t] = d.qpos[:m.nq]
            cam_pos[t], cam_mat[t] = d.cam_xpos[cid], d.cam_xmat[cid]
            n = 0
            for c in range(min(d.ncon, MAXC)):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, f)
                con[t, n] = (m.geom_bodyid[d.contact[c].geom1],
                             m.geom_bodyid[d.contact[c].geom2],
                             float(np.linalg.norm(f[:3])))
                n += 1
            ncon[t] = n
            n_over += int(d.ncon > MAXC)
            ren.disable_depth_rendering()
            ren.disable_segmentation_rendering()
            ren.update_scene(d, camera=cid, scene_option=opt)
            frames.append(ren.render().copy())
            ren.enable_segmentation_rendering()
            ren.update_scene(d, camera=cid, scene_option=opt)
            s = ren.render()[..., 0]
            sg = np.zeros_like(s, np.uint16)
            ok = (s >= 0) & (s < m.ngeom)
            sg[ok] = m.geom_bodyid[s[ok]].astype(np.uint16)
            visf[t] = float(np.isin(sg, targets).mean())
            gripf[t] = float(np.isin(sg, grip).mean())
            segs[t] = sg[::SEG_DS, ::SEG_DS]
            ren.disable_segmentation_rendering()
            ren.enable_depth_rendering()
            ren.update_scene(d, camera=cid, scene_option=opt)
            deps[t] = ren.render()[::SEG_DS, ::SEG_DS].astype(np.float16)
            ren.disable_depth_rendering()

        seen = float((visf >= MIN_VIS).mean())
        gseen = float((gripf >= MIN_GRIP).mean())
        # usable windows: target visible for half the window AND moving
        tmov = np.linalg.norm(np.diff(xpos[:, targets], axis=0), axis=-1).max(1)
        starts, gwin = [], 0
        for a in range(0, T - WIN + 1, WIN // 2):
            w = slice(a, a + WIN)
            if (visf[w] >= MIN_VIS).mean() < MIN_WIN_VIS:
                continue
            if float(tmov[a:a + WIN - 1].sum()) < MIN_WIN_MOVE:
                continue
            starts.append(a)
            gwin += int((gripf[w] >= MIN_GRIP).mean() >= MIN_WIN_VIS)
        nwin = len(starts)
        if nwin < MIN_WINDOWS:
            made.append(dict(ok=False, cam=cam,
                             why=f"only {nwin} usable windows"))
            continue
        if gwin / max(nwin, 1) < 0.3:
            made.append(dict(ok=False, cam=cam,
                             why=f"gripper unseen in {nwin - gwin}/{nwin} windows"))
            continue
        od = out_root / "shard_0000" / f"{eid}__{cam}"
        od.mkdir(parents=True, exist_ok=True)
        _mp4(frames, od / "frames.mp4")
        np.savez_compressed(
            od / "state.npz", source="robocasa_replay",
            gen_version=GEN_VERSION, fps=FPS, width=W, height=H,
            seg_ds=SEG_DS, camera=cam, seg=segs, depth=deps, xpos=xpos,
            xquat=xquat, xvel=xvel, qpos=qpos, cam_pos=cam_pos,
            cam_mat=cam_mat, cam_fovy=np.float32(m.cam_fovy[cid]),
            contact_pairs=con, contact_n=ncon,
            contact_overflow_frames=np.int32(n_over),
            target_bodies=np.array(targets, np.int32), target_visible=visf,
            gripper_visible=gripf, n_windows=np.int32(nwin),
            n_windows_with_gripper=np.int32(gwin),
            # THE START INDICES THEMSELVES, not just the count. A ~200
            # frame atomic demo contains 3-4 windows where the object
            # actually moves; the rest is the arm reaching. Sampling t0
            # uniformly - which is what the trainer did - lands on
            # reach-only windows almost every time, which is the
            # rcasa_v1 failure one level down. The trainer must sample
            # from here.
            window_starts=np.array(starts, np.int32),
            window_len=np.int32(WIN),
            instruction=str(meta.get("lang", "")), demo=str(ep_dir.name),
            **structure(m))
        made.append(dict(ok=True, id=f"{eid}__{cam}", T=T, camera=cam,
                         target_move=round(moved, 4),
                         target_vis=round(float(visf.mean()), 4),
                         grip_vis=round(float(gripf.mean()), 4),
                         seen=round(seen, 2), grip_seen=round(gseen, 2),
                         win=nwin, win_grip=gwin,
                         instruction=str(meta.get("lang", ""))))
    ren.close()
    return made, None


def build(tasks, name="rcasa_v2", limit=4, split="pretrain"):
    import warnings
    warnings.filterwarnings("ignore")
    import robocasa  # noqa: F401  - importing REGISTERS the kitchen envs
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from tqdm import tqdm
    out_root = R.dataset_dir(name)
    cfg = load_composite_controller_config(robot="PandaOmron")
    made, rej, rows = 0, [], []
    for task in tasks:
        eps = episodes(task, split)[:limit]
        if not eps:
            rej.append((task, "no demos on disk"))
            continue
        env = robosuite.make(env_name=task, robots="PandaOmron",
                             controller_configs=cfg, has_renderer=False,
                             has_offscreen_renderer=False,
                             use_camera_obs=False, control_freq=FPS,
                             ignore_done=True)
        env.reset()
        for ep in tqdm(eps, unit="ep", desc=f"replay/{task}"):
            eid = f"{task}_{ep.name}"
            try:
                res, err = replay(ep, env, out_root, eid)
            except Exception as exc:
                res, err = [], str(exc)[:160]
            if err:
                rej.append((eid, err))
            for r in res:
                if r.get("ok"):
                    made += 1
                    rows.append(dict(shard="shard_0000", task=task, **r))
                else:
                    rej.append((f"{eid}/{r.get('cam','?')}", r.get("why", "?")))
            R.log("rcreplay_ep", dataset=name, id=eid, n=len(res))
        env.close()
    # MERGE, never overwrite. build() is called once per task by genall
    # and again by every refill, so writing only this call's rows leaves
    # a manifest describing the LAST invocation. It did: after the
    # PickPlace refill, rcasa held 447 episodes on disk and a manifest
    # listing 128 - and tracks2 reads the manifest, so it would have
    # tracked 128, written 128 files and reported success. Episodes are
    # keyed by id, and this call's rows win, so a re-replayed episode
    # updates in place rather than duplicating.
    prev = R.read_manifest(name) or {}
    keep = {e["id"]: e for e in prev.get("episodes", [])}
    keep.update({r["id"]: r for r in rows})
    all_rows = [keep[k] for k in sorted(keep)]
    all_tasks = sorted({e.get("task", "?") for e in all_rows})
    R.write_manifest(name, dict(name=name, source="robocasa_replay",
                                n_episodes=len(all_rows), tasks=all_tasks,
                                split=split, episodes=all_rows))
    return made, rej, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rcasa_v2")
    ap.add_argument("--tasks", nargs="+",
                    default=["OpenDrawer", "CloseDrawer", "OpenCabinet",
                             "CloseCabinet"])
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--split", default="pretrain")
    a = ap.parse_args()
    made, rej, rows = build(a.tasks, a.name, a.limit, a.split)
    print(f"\nwrote {made} episodes to {R.dataset_dir(a.name)}")
    for r in rows:
        print(f"  {r['id'][:52]:52s} T={r['T']:3d} move={r['target_move']:.2f} "
              f"tgt={r['target_vis']:.3f}/{r['seen']:.2f} "
              f"grip={r['grip_vis']:.3f}/{r['grip_seen']:.2f}")
    if rej:
        print(f"\nREJECTED {len(rej)}:")
        for i, w in rej[:12]:
            print(f"  {i:34s} {w}")
