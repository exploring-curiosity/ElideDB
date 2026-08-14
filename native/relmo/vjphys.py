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

THE MOVER is chosen as the target body with the largest displacement over the
episode. Not by name, not by a per-task rule - rcasa lists 1-4 target bodies
(door, handle, inner box, ...) and which one carries the event differs by task.

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
KEYS = ("open", "d_open", "has_art", "contact", "speed", "grip_dist",
        "d_rel_x", "d_rel_y", "d_rel_z", "d_rot", "rot_cum")

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


def quat_to_mat(q):
    """MuJoCo (w,x,y,z) -> 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def gripper_bodies(names):
    return np.array([i for i, n in enumerate(names)
                     if "gripper0" in n or n.endswith("_hand")], dtype=int)


def eef_body(names):
    for i, n in enumerate(names):
        if n.endswith("eef"):
            return i
    g = gripper_bodies(names)
    return int(g[0]) if len(g) else 0


def targets(st):
    """state.npz -> (24, len(KEYS)) float32, or None if unusable."""
    names = [str(x) for x in st["body_names"]]
    tb = np.asarray(st["target_bodies"], dtype=int)
    xpos, xquat, xvel = st["xpos"], st["xquat"], st["xvel"]
    T = len(xpos)
    if T < 8 or len(tb) == 0:
        return None
    idx = np.linspace(0, T - 1, FRAMES).round().astype(int)

    # THE MOVER: largest total displacement among the declared targets
    disp = np.linalg.norm(xpos[:, tb, :] - xpos[0, tb, :], axis=-1).sum(0)
    mover = int(tb[int(np.argmax(disp))])

    # articulated joint on any target body (hinge=3, slide=2); free=0 has none
    jb, jt, ja, jr = (st["jnt_bodyid"], st["jnt_type"], st["jnt_qposadr"],
                      st["jnt_range"])
    q = np.zeros(T, np.float64)
    has_art = 0.0
    for k in range(len(jb)):
        if int(jb[k]) in set(tb.tolist()) and int(jt[k]) in (2, 3):
            d = max(abs(float(jr[k][0])), abs(float(jr[k][1])))
            if d > 1e-6:
                q = np.abs(st["qpos"][:, int(ja[k])]) / d      # see docstring
                has_art = 1.0
                break

    grip = gripper_bodies(names)
    eef = eef_body(names)
    cp, cn = st["contact_pairs"], st["contact_n"]
    gset = set(grip.tolist())
    tset = set(tb.tolist())

    out = np.zeros((len(STEPS), len(KEYS)), np.float32)
    for si, t in enumerate(STEPS):
        f = int(min(idx[t * TUBELET], T - 1))
        f1 = int(min(idx[min(t + 1, N_T - 1) * TUBELET], T - 1))
        # contact between a target body and a gripper body
        k = int(cn[f])
        c = 0.0
        if k:
            pr = cp[f][:k, 1:3]
            for b1, b2 in pr:
                if (int(b1) in tset and int(b2) in gset) or \
                   (int(b2) in tset and int(b1) in gset):
                    c = 1.0
                    break
        Rg = quat_to_mat(xquat[f][eef])
        dw = xpos[f1][mover] - xpos[f][mover]
        drel = Rg.T @ dw                       # motion in the GRIPPER frame
        out[si] = (q[f], q[f1] - q[f], has_art, c,
                   float(np.linalg.norm(xvel[f][mover][:3])),
                   float(np.linalg.norm(xpos[f][mover] - xpos[f][eef])),
                   drel[0], drel[1], drel[2],
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
