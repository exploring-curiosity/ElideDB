"""Relational physical targets, aligned to the prediction trace. TRAINING ONLY.

These come from sim state and are used to fit the recurrence. They are never
available at serve, never enter an index, and never appear in a ranked result -
the same posture the oracle gate already has. The task-family labels used for
GRADING never touch this file.

WHY THESE TARGETS. The retrieval metric has to transfer to a real kitchen, so
the supervision must be about physics rather than vocabulary. "A hinged panel
swinging open while contact holds at the handle" is the same fact whether the
panel is a rcasa cabinet, an unseen fridge, or someone's actual cupboard. Fit
to that, a model has no reason to require having seen the object before.

THE OPENNESS CONVENTION, and the check that earned it. MuJoCo joint ranges here
are not consistently signed - cabinets run [-1.18, 0.39], drawers [-0.65, 0],
microwaves [-1.57, 0] - so neither raw qpos nor (q-lo)/(hi-lo) means the same
thing across families; the latter reads a closed drawer as fully open. Measured
over 12 episodes of each Open/Close family, CLOSED IS EXACTLY qpos 0 in every
one of them:

    task            open_start  open_end        (open = |q| / max(|lo|,|hi|))
    CloseCabinet         0.947     0.001
    CloseDrawer          0.474     0.002
    CloseMicrowave       0.934     0.000
    OpenCabinet          0.000     1.000
    OpenDrawer           0.000     0.792
    OpenMicrowave        0.000     0.999

So `open = |q| / max(|lo|,|hi|)` is convention-free, works for hinge and slide
alike, and needs no per-family table. Its time derivative is the signed fact
that matters: opening is positive, closing is negative, for every articulated
object including ones never seen.

THE MOVER is whichever NON-ROBOT body moves furthest over the episode. Not by
name, not by a per-task rule, and no longer via state["target_bodies"] - that
field is rcasa declaring which object the task is about, i.e. a per-task
annotation, and it has no business inside a training target.

    python -m relmo.vjphys --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import TUBELET  # noqa: E402
from relmo.vjrec4 import CTX  # noqa: E402

OUT = R.BASE / "vjphys"
N_T, FRAMES = 32, 64
STEPS = list(range(CTX, N_T))
# order is the contract with the trainer; changing it invalidates a checkpoint
KEYS = ("open", "d_open", "has_art", "speed", "d_rot", "rot_cum")

# DROPPED 2026-08-14, on the owner's rule that a target must describe the EVENT
# and not the robot: contact (target<->gripper), grip_dist (target->eef) and
# d_rel_x/y/z (mover motion expressed IN THE GRIPPER FRAME). All three are
# defined by a Panda gripper that does not exist on a WidowX, and ood_test
# showed exactly that failure - the trained model lost to the frozen one on
# real robot video (0.804 vs 0.908). Val leave-one-out rated grip_dist among
# the MOST valuable channels; that value is the domain-specific value we are
# deliberately giving up.

# d_rot / rot_cum were added after the first trained model was diagnosed on
# val: it improved open-vs-close but made hinged-vs-sliding WORSE than the
# frozen baseline (Close/hinged returning Close/sliding went 7 -> 20 of 120).
# The nine original targets say which DIRECTION something moved and never say
# HOW it moves, so a door and a drawer were being asked to look alike. A hinged
# panel rotates about an edge; a drawer translates with its orientation fixed.
# That is a physical fact, not a naming convention, and it holds for any real
# cupboard or any real drawer - so it belongs in the targets rather than in a
# per-object rule.


def quat_angle(q1, q2):
    """Rotation angle between two MuJoCo quaternions, radians. Sign-free:
    a hinge gives a large value, a slide ~0 regardless of which way it went."""
    d = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return 2.0 * float(np.arccos(d))


def targets(st):
    """state.npz -> (24, len(KEYS)) float32, or None if unusable."""
    names = [str(x) for x in st["body_names"]]
    xpos, xquat, xvel = st["xpos"], st["xquat"], st["xvel"]
    T = len(xpos)
    if T < 8:
        return None
    idx = np.linspace(0, T - 1, FRAMES).round().astype(int)

    # THE MOVER, chosen WITHOUT the simulator's task annotation. The previous
    # version picked among state["target_bodies"], which is rcasa declaring
    # which object the task is about - a per-task hand-off sitting inside the
    # training targets. Largest displacement among all non-robot bodies is what
    # the video itself would show and needs no annotation.
    robot = np.array([i for i, n in enumerate(names)
                      if n.startswith(("robot", "gripper"))], dtype=int)
    cand = np.setdiff1d(np.arange(xpos.shape[1]), robot)
    disp = np.linalg.norm(xpos[:, cand, :] - xpos[0, cand, :], axis=-1).sum(0)
    if not len(cand) or float(disp.max()) <= 0:
        return None
    mover = int(cand[int(np.argmax(disp))])

    # articulated joint on THE MOVER (hinge=3, slide=2); a free body has none
    jb, jt, ja, jr = (st["jnt_bodyid"], st["jnt_type"], st["jnt_qposadr"],
                      st["jnt_range"])
    q = np.zeros(T, np.float64)
    has_art = 0.0
    for k in range(len(jb)):
        if int(jb[k]) == mover and int(jt[k]) in (2, 3):
            d = max(abs(float(jr[k][0])), abs(float(jr[k][1])))
            if d > 1e-6:
                q = np.abs(st["qpos"][:, int(ja[k])]) / d      # see docstring
                has_art = 1.0
                break

    out = np.zeros((len(STEPS), len(KEYS)), np.float32)
    for si, t in enumerate(STEPS):
        f = int(min(idx[t * TUBELET], T - 1))
        f1 = int(min(idx[min(t + 1, N_T - 1) * TUBELET], T - 1))
        out[si] = (q[f], q[f1] - q[f], has_art,
                   float(np.linalg.norm(xvel[f][mover][:3])),
                   quat_angle(xquat[f][mover], xquat[f1][mover]),
                   quat_angle(xquat[0][mover], xquat[f][mover]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    a = ap.parse_args()

    from tqdm import tqdm
    man = R.read_manifest(a.dataset)
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    ok, skip = 0, 0
    for e in tqdm(man["episodes"], unit="ep", desc=f"phys:{a.dataset}"):
        f = out / f"{e['id']}.npy"
        if f.exists():
            ok += 1
            continue
        p = R.dataset_dir(a.dataset) / e["shard"] / e["id"] / "state.npz"
        if not p.exists():
            skip += 1
            continue
        try:
            y = targets(np.load(p, allow_pickle=True))
        except Exception as ex:                              # noqa: BLE001
            tqdm.write(f"  {e['id']}: {type(ex).__name__}: {ex}")
            skip += 1
            continue
        if y is None:
            skip += 1
            continue
        np.save(f, y)
        ok += 1
    have = len(list(out.glob("*.npy")))
    print(json.dumps(dict(dataset=a.dataset, ok=ok, skipped=skip,
                          on_disk=have, keys=list(KEYS)), indent=1))
    print(f"VERIFIED on disk: {have} target files")
    R.log("vjphys", dataset=a.dataset, ok=ok, skipped=skip, on_disk=have)


def load(dataset, ep_id):
    p = OUT / dataset / f"{ep_id}.npy"
    return np.load(p) if p.exists() else None


if __name__ == "__main__":
    main()
