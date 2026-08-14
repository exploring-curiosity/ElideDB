"""Does the RELATIONAL channel separate what the trajectory cannot?

The world model exists to answer "when did something like this happen".
The training signal it has been given is motion-R2 over per-point
displacement — and that quantity is INVARIANT to the distinction the
question turns on. Pushing a block left-to-right and closing a drawer
left-to-right produce the same point displacements, so a model scoring
1.000 on motion-R2 still cannot tell them apart. Optimising it harder
cannot fix that; it is not a matter of degree.

So before building anything on a relational target, measure whether the
relation actually carries the distinction. This is an ORACLE test in the
same spirit as ceiling.py: it uses privileged sim state (contacts, joint
ranges, body ids) to ask "is this information sufficient?", NOT to serve
a query. If the answer is yes, the world model's job becomes predicting
these relations from pixels. If it is no, nothing downstream can work
and we have saved ourselves the run.

Two descriptors over the same episodes and the same windows:

  TRAJ  what motion-R2 sees: the moving points' mean unit direction and
        log displacement magnitude, plus the shape of the displacement
        profile over the horizon.
  REL   who touches what, and is the mover hinged: contact with the
        gripper, contact with a third body, how many distinct bodies the
        target touches, whether contacts CHANGE during the window, and
        the fraction of a joint's range the mover traverses.

Scored as separability (AUC) between two families that look alike in
motion and differ in kind:

  ARTICULATED  Open/Close Drawer, Cabinet, Microwave — a hinged or
               prismatic part swings/slides through a bounded range.
  TRANSPORT    PickPlaceCounterTo* — a free body is carried between
               supports.

Task names are used ONLY as the scoring label, exactly as sim GT is used
to score a tracker. Nothing here is read at serve time.

    python -m relmo.relsep --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

ARTIC = ("OpenDrawer", "CloseDrawer", "OpenCabinet", "CloseCabinet",
         "OpenMicrowave", "CloseMicrowave", "OpenFridge")
TRANS = ("PickPlaceCounterToCabinet", "PickPlaceCounterToSink",
         "PickPlaceCounterToDrawer")
ROBOT_MAX = 40          # body ids below this are robot/gripper/mount


def auc(pos, neg):
    """Ties count as half, or a binary feature reads as a perfect AUC."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def traj_feats(z, t0, W):
    """Everything motion-R2 can see."""
    xp = z["xpos"].astype(np.float64)
    tg = z["target_bodies"]
    T = len(xp)
    a, b = t0, min(t0 + W, T - 1)
    d = xp[b][tg] - xp[a][tg]
    mag = np.linalg.norm(d, axis=-1)
    k = int(mag.argmax())
    v = d[k]
    n = np.linalg.norm(v) + 1e-9
    # direction, magnitude, and how straight the path is
    path = np.linalg.norm(np.diff(xp[a:b + 1, tg[k]], axis=0), axis=-1).sum()
    return dict(dir_x=v[0] / n, dir_y=v[1] / n, dir_z=v[2] / n,
                logmag=float(np.log10(n + 1e-6)),
                straightness=float(n / (path + 1e-9)))


def rel_feats(z, t0, W):
    """Who touches what, and is the mover hinged?"""
    cp, cn = z["contact_pairs"], z["contact_n"]
    tg = set(z["target_bodies"].tolist())
    T = len(cn)
    a, b = t0, min(t0 + W, T)
    grip = third = 0
    partners, sets = set(), []
    for t in range(a, b):
        n = int(cn[t])
        if not n:
            sets.append(frozenset())
            continue
        pr = cp[t, :n, :2].astype(int)
        hit = pr[np.isin(pr, list(tg)).any(1)]
        others = {int(x) for x in hit.ravel() if x not in tg}
        partners |= others
        sets.append(frozenset(others))
        if any(o < ROBOT_MAX for o in others):
            grip += 1
        if any(o >= ROBOT_MAX for o in others):
            third += 1
    nf = max(b - a, 1)
    # does the contact SET change during the window? a transport episode
    # swaps supports; an articulated part keeps the same neighbours.
    changes = sum(1 for i in range(1, len(sets)) if sets[i] != sets[i - 1])
    # articulation: what fraction of a joint's range does the mover cover?
    art = 0.0
    if all(k in z.files for k in ("qpos", "jnt_bodyid", "jnt_qposadr",
                                  "jnt_range", "jnt_type")):
        q, jb = z["qpos"], z["jnt_bodyid"]
        adr, rng, jt = z["jnt_qposadr"], z["jnt_range"], z["jnt_type"]
        for j in range(len(jb)):
            if int(jb[j]) not in tg or int(jt[j]) not in (2, 3):
                continue                       # slide=2, hinge=3
            i = int(adr[j])
            if i >= q.shape[1]:
                continue
            lo, hi = float(rng[j][0]), float(rng[j][1])
            span = abs(hi - lo)
            if span <= 1e-6:
                continue
            art = max(art, float(abs(q[b - 1, i] - q[a, i]) / span))
    return dict(grip_frac=grip / nf, third_frac=third / nf,
                n_partners=float(len(partners)),
                contact_changes=changes / nf, articulation=art)


def collect(dataset, W=24):
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    out = []
    for f in sorted((R.TRACKS / dataset).glob("*.npz")):
        if f.stem not in by:
            continue
        task = f.stem.split("_episode_")[0]
        fam = ("artic" if task in ARTIC else
               "trans" if task in TRANS else None)
        if fam is None:
            continue
        z = np.load(f)
        if "contact_pairs" not in z.files:
            continue
        ws = z["window_starts"] if "window_starts" in z.files else []
        for t0 in (ws[:6] if len(ws) else [0]):
            r = dict(task=task, fam=fam, id=f.stem, t0=int(t0))
            r.update(traj_feats(z, int(t0), W))
            r.update(rel_feats(z, int(t0), W))
            out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--win", type=int, default=24)
    a = ap.parse_args()
    rows = collect(a.dataset, a.win)
    A = [r for r in rows if r["fam"] == "artic"]
    B = [r for r in rows if r["fam"] == "trans"]
    print(f"{a.dataset}: {len(rows)} windows  "
          f"({len(A)} articulated, {len(B)} transport)\n")
    TR = ("dir_x", "dir_y", "dir_z", "logmag", "straightness")
    RE = ("grip_frac", "third_frac", "n_partners", "contact_changes",
          "articulation")
    print(f"{'feature':18s} {'AUC':>7s}  {'|AUC-.5|':>8s}  family")
    print("-" * 50)
    best = {"TRAJ": 0.0, "REL": 0.0}
    for fam, keys in (("TRAJ", TR), ("REL", RE)):
        for k in keys:
            v = auc([r[k] for r in A], [r[k] for r in B])
            sep = abs(v - 0.5)
            best[fam] = max(best[fam], sep)
            print(f"{k:18s} {v:7.3f}  {sep:8.3f}  {fam}")
    print(f"\nbest separation  TRAJ {best['TRAJ']:.3f}   REL {best['REL']:.3f}")
    print(f"=> AUC equivalent: TRAJ {0.5 + best['TRAJ']:.3f}, "
          f"REL {0.5 + best['REL']:.3f}")
    R.log("relsep", dataset=a.dataset, windows=len(rows),
          n_artic=len(A), n_trans=len(B),
          best_traj=round(best["TRAJ"], 4), best_rel=round(best["REL"], 4))


def matched(rows, TR, RE, k=1, tol=None):
    """The honest version of the question: hold TRAJECTORY fixed.

    Comparing whole families is too easy - a transport episode lifts and
    an articulated one does not, so even the trajectory separates them a
    bit (straightness 0.828). The claim under test is sharper: for
    windows that LOOK THE SAME IN MOTION, does the relation still tell
    them apart? So pair each articulated window with its nearest
    transport window in normalised trajectory space, keep only pairs that
    are genuinely close, and score the relational features on that
    matched set. If the relation still separates there, the distinction
    is carried by relations and is invisible to motion-R2 by
    construction."""
    A = [r for r in rows if r["fam"] == "artic"]
    B = [r for r in rows if r["fam"] == "trans"]
    XA = np.array([[r[k_] for k_ in TR] for r in A], float)
    XB = np.array([[r[k_] for k_ in TR] for r in B], float)
    mu = np.concatenate([XA, XB]).mean(0)
    sd = np.concatenate([XA, XB]).std(0) + 1e-9
    ZA, ZB = (XA - mu) / sd, (XB - mu) / sd
    D = np.linalg.norm(ZA[:, None] - ZB[None], axis=-1)
    j = D.argmin(1)
    d = D[np.arange(len(ZA)), j]
    if tol is None:
        tol = float(np.median(d))              # the closest half of pairs
    keep = d <= tol
    pa = [A[i] for i in range(len(A)) if keep[i]]
    pb = [B[j[i]] for i in range(len(A)) if keep[i]]
    return pa, pb, float(d[keep].mean()), int(keep.sum())
