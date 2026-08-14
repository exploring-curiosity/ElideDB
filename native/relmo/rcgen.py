"""RoboCasa episode generator — our own rollouts, exact GT.

WHY NOT REPLAY THE SHIPPED DEMOS (measured, 2026-08-12): RoboCasa
v1.0 distributes demonstrations in LeRobot format, whose schema is
    observation.images.*  (video)
    observation.state     float64[16]   robot proprioception ONLY
    action                float64[12]
There is NO MuJoCo state and no object pose, so `set_state_from_
flattened` replay - the whole reason RoboCasa looked attractive - is
impossible from those files. Verified by reading meta/info.json out of
the distributed tar.

So we use the ENVIRONMENTS instead of the recordings. RoboCasa's value
was never its 500 videos; it is 300+ task definitions, 3500 object
assets, 705 articulated fixtures and a scene sampler - and we own the
simulator, so we get exact GT for free.

WHAT DRIVES THE ARM. Not a per-task script (that would be me
hand-authoring the event vocabulary again, which the owner barred).
One generic contact-seeking controller, identical for every task:
reach toward a sampled object, close on contact, lift, carry to a
sampled site, release. Those five phases are what a gripper physically
does; they are not task semantics. Everything else - which object,
which site, speeds, pauses, whether the grasp succeeds - is sampled.
A failed grasp is KEPT: a slip is a physical event, and a corpus of
only successes teaches a model that things never slip.

The world model never sees any of this. It sees point tracks. The
policy exists solely to make interesting things happen in front of a
camera.

    python -m relmo.daemon rcgen --episodes 600
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

GEN_VERSION = 201
FPS = 20
SEG_DS = 2
MAXC = 512   # was 64. Scenes carry a median of 58 contacts per frame and
             # peak at 68+, so 64 silently dropped contacts in 35.8% of
             # frames across the sample corpus (100% in the worst
             # episode). Contact is the channel that says WHEN something
             # touched; a truncated one is worse than none.
W, H = 320, 240

# tasks chosen for PHYSICAL variety, not semantics: transport, and
# articulation (doors/drawers) which is where joints become visible
TASKS = ["PickPlaceCounterToCabinet", "PickPlaceCounterToSink",
         "PickPlaceCounterToMicrowave", "PickPlaceCabinetToCounter",
         "OpenDrawer", "CloseDrawer", "OpenCabinet", "CloseCabinet",
         "OpenMicrowave", "CloseMicrowave", "TurnOnStove", "TurnOnSinkFaucet"]
CAMS = ["robot0_agentview_left", "robot0_agentview_right",
        "robot0_eye_in_hand"]


def _mj(env):
    m = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    d = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    return m, d


def structure(m):
    """The static answer key the world model must rediscover."""
    import mujoco
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}"
             for b in range(m.nbody)]
    return dict(body_parentid=np.asarray(m.body_parentid, np.int32),
                body_names=np.array(names),
                jnt_type=np.asarray(m.jnt_type, np.int32),
                jnt_axis=np.asarray(m.jnt_axis, np.float32),
                jnt_range=np.asarray(m.jnt_range, np.float32),
                jnt_bodyid=np.asarray(m.jnt_bodyid, np.int32),
                jnt_qposadr=np.asarray(m.jnt_qposadr, np.int32),
                geom_bodyid=np.asarray(m.geom_bodyid, np.int32),
                body_mass=np.asarray(m.body_mass, np.float32))


def movable_bodies(env):
    """Bodies with a FREE joint = the things that can be picked up.

    Deliberately not env.objects[...]: that resolves to template
    bodies, which measured 20 m away from the end-effector (object at
    [-0.3,-5.5] while the eef sat at [10.1,15.2]) and would have sent
    the policy chasing a ghost forever. A free joint is the physical
    definition of "movable" and needs no naming convention, so this
    works unchanged across all 300+ tasks."""
    import mujoco
    m, _ = _mj(env)
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            b = int(m.jnt_bodyid[j])
            if m.body_mass[b] < 5.0:
                out.append(b)
    return out


def _target_pos(env, cache):
    """Nearest movable body to the hand (recomputed only when the
    policy re-targets, so one episode chases one thing)."""
    _, d = _mj(env)
    b = cache.get("body")
    if b is None:
        return None
    return np.array(d.xpos[b], np.float64)


def _eef(env):
    """Live end-effector position, from the SITE.

    NOT robots[0]._hand_pos: measured to report [10.07,15.15,1.39]
    (a stale/base-frame value near robot0_base at [10,10,0]) while the
    real eef site sat at [-0.10,-5.15,1.30], 0.35 m from the target.
    That 23 m phantom gap kept the policy in phase 0 for entire
    episodes."""
    try:
        _, d = _mj(env)
        sid = env.robots[0].eef_site_id
        sid = sid["right"] if isinstance(sid, dict) else sid
        return np.array(d.site_xpos[sid], np.float64)
    except Exception:
        return None


def policy(env, t, T, rng, phase_state):
    """Generic 5-phase contact seeker. Returns a 12-dim action.

    Layout (PandaOmron/HybridMobileBase): 0:6 right arm delta pose,
    6 gripper, 7:10 base, 10:12 torso. We drive the arm and gripper
    and leave the base still - a moving base would make every episode
    an ego-motion episode, and MOVi already covers that."""
    a = np.zeros(env.action_dim)
    eef = _eef(env)
    if phase_state.get("body") is None and eef is not None:
        cands = movable_bodies(env)
        if cands:
            _, d = _mj(env)
            dist = [np.linalg.norm(d.xpos[b] - eef) for b in cands]
            order = np.argsort(dist)
            k = int(order[min(int(rng.integers(0, 2)), len(order) - 1)])
            phase_state["body"] = cands[k]
    tgt = _target_pos(env, phase_state)
    ph = phase_state["ph"]
    if eef is None or tgt is None:
        a[:3] = rng.normal(0, 0.25, 3)              # blind wander
        a[6] = -1.0 if (t // 25) % 2 else 1.0
        return a
    goal = tgt.copy()
    if ph == 0:                                     # approach from above
        goal[2] += 0.10
        if np.linalg.norm(eef - goal) < 0.07:
            phase_state["ph"] = 1
    elif ph == 1:                                   # descend
        if np.linalg.norm(eef - goal) < 0.05:
            phase_state["ph"], phase_state["t0"] = 2, t
    elif ph == 2:                                   # close
        if t - phase_state["t0"] > 12:
            phase_state["ph"] = 3
            phase_state["carry"] = tgt + np.array(
                [rng.uniform(-.25, .25), rng.uniform(-.25, .25),
                 rng.uniform(.10, .28)])
    elif ph == 3:                                   # transport
        goal = phase_state["carry"]
        if np.linalg.norm(eef - goal) < 0.08:
            phase_state["ph"], phase_state["t0"] = 4, t
    elif ph == 4:                                   # release
        if t - phase_state["t0"] > 15:
            phase_state["ph"] = 0                   # go again
    d = np.clip((goal - eef) * 8.0, -1, 1)
    a[:3] = d + rng.normal(0, 0.04, 3)
    a[6] = 1.0 if ph in (2, 3) else -1.0            # +1 closes
    return a


def episode(env, out_dir: Path, rng, cam, T=120):
    import mujoco
    m, d = _mj(env)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    cid = cid if cid >= 0 else 0
    ren = mujoco.Renderer(m, H, W)
    H2, W2 = H // SEG_DS, W // SEG_DS
    nb = m.nbody
    xpos = np.zeros((T, nb, 3), np.float32)
    xquat = np.zeros((T, nb, 4), np.float32)
    xvel = np.zeros((T, nb, 6), np.float32)
    qpos = np.zeros((T, m.nq), np.float32)
    cam_pos = np.zeros((T, 3), np.float32)
    cam_mat = np.zeros((T, 9), np.float32)
    segs = np.zeros((T, H2, W2), np.uint8)
    deps = np.zeros((T, H2, W2), np.float16)
    con = np.zeros((T, MAXC, 3), np.float32)
    ncon = np.zeros(T, np.int32)
    acts = np.zeros((T, env.action_dim), np.float32)
    phase = np.zeros(T, np.int8)
    ps = dict(ph=0, t0=0, carry=None, body=None)
    frames = []
    for t in range(T):
        a = policy(env, t, T, rng, ps)
        env.step(a)
        m, d = _mj(env)
        acts[t], phase[t] = a, ps["ph"]
        xpos[t], xquat[t] = d.xpos, d.xquat
        xvel[t, :, :3], xvel[t, :, 3:] = d.cvel[:, 3:], d.cvel[:, :3]
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
        ren.disable_depth_rendering()
        ren.disable_segmentation_rendering()
        ren.update_scene(d, camera=cid)
        frames.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=cid)
        s = ren.render()[..., 0]
        sg = np.zeros_like(s, np.uint8)
        ok = (s >= 0) & (s < m.ngeom)
        sg[ok] = np.clip(m.geom_bodyid[s[ok]], 0, 255).astype(np.uint8)
        segs[t] = sg[::SEG_DS, ::SEG_DS]
        ren.disable_segmentation_rendering()
        ren.enable_depth_rendering()
        ren.update_scene(d, camera=cid)
        deps[t] = ren.render()[::SEG_DS, ::SEG_DS].astype(np.float16)
    ren.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    _mp4(frames, out_dir / "frames.mp4")
    np.savez_compressed(
        out_dir / "state.npz", source="robocasa", gen_version=GEN_VERSION,
        fps=FPS, width=W, height=H, seg_ds=SEG_DS, camera=cam,
        seg=segs, depth=deps, xpos=xpos, xquat=xquat, xvel=xvel,
        qpos=qpos, cam_pos=cam_pos, cam_mat=cam_mat,
        cam_fovy=np.float32(m.cam_fovy[cid]), action=acts, phase=phase,
        contact_pairs=con, contact_n=ncon, **structure(m))
    mov = int((np.linalg.norm(xvel[:, :, :3], axis=-1).max(0) > 0.02).sum())
    return dict(frames=T, n_bodies=int(nb), n_joints=int(m.njnt),
                moving=mov, contact_frames=int((ncon > 0).sum()),
                phases=int(len(np.unique(phase))))


def _mp4(frames, path):
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v",
         # yuv444p, not yuv420p: 4:2:0 halves chroma resolution, which
         # shimmers along the hard edges of RoboCasa's saturated
         # generative textures. MEASURED against raw renders on 30
         # frames: 420p costs 2.472/255, 444p 1.727/255, while the
         # renderer itself is exactly deterministic (max diff 1, mean
         # 0.0000 for the same state rendered twice). Point tracking
         # keys on exactly those edges, so the 30% is worth the bytes.
         "libx264", "-crf", "14", "-pix_fmt", "yuv444p", "-bf", "0",
         str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def build(name="rcasa_v1", episodes=600, T=120, seed0=7_000_000):
    from tqdm import tqdm
    from robocasa.utils.env_utils import create_env
    man = R.read_manifest(name)
    man.setdefault("episodes", [])
    man["gen_version"] = GEN_VERSION
    man["config"] = dict(source="robocasa", fps=FPS, seg_ds=SEG_DS,
                         tasks=TASKS, cameras=CAMS, articulated=True,
                         structural_gt=True, T=T)
    have = {e["id"] for e in man["episodes"]}
    root = R.dataset_dir(name)
    made = 0
    bar = tqdm(range(episodes), unit="ep", desc=f"rcgen/{name}")
    env, cur = None, None
    for i in bar:
        eid = f"ep{i:06d}"
        if eid in have:
            continue
        rng = np.random.default_rng(seed0 + i)
        task = TASKS[i % len(TASKS)]
        cam = CAMS[(i // len(TASKS)) % len(CAMS)]
        try:
            if env is None or cur != task:
                if env is not None:
                    env.close()
                env = create_env(env_name=task, robots="PandaOmron",
                                 camera_widths=W, camera_heights=H,
                                 seed=int(seed0 + i))
                cur = task
            env.reset()
            shard = f"shard_{made // 200:04d}"
            tmp = root / shard / f".tmp_{eid}"
            st = episode(env, tmp, rng, cam, T=T)
            tmp.rename(root / shard / eid)
            man["episodes"].append(dict(id=eid, shard=shard, task=task,
                                        camera=cam, **st))
            made += 1
            if made % 20 == 0:
                R.write_manifest(name, man)
                bar.set_postfix(made=made)
        except Exception as exc:
            R.log("rcgen_error", id=eid, task=task, error=str(exc)[:200])
            env, cur = None, None
    man = R.write_manifest(name, man)
    R.log("rcgen_done", dataset=name, made=made, total=man["n_episodes"],
          fingerprint=man["fingerprint"])
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rcasa_v1")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--T", type=int, default=120)
    a = ap.parse_args()
    m = build(a.name, a.episodes, a.T)
    print(f"{a.name}: {m['n_episodes']} episodes, fp {m['fingerprint']}")
