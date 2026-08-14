"""Audit every claim the pipeline makes, against sim ground truth.

Each stage of this design asserts that some latent quantity recovers something
about the world. Most of those assertions were never tested - they were argued
for, and then the retrieval number was used as if it confirmed them. It does
not: retrieval can work for reasons unrelated to whether the gate finds the
object or the content describes the event.

CLAIMS UNDER TEST, each against sim state used for SCORING ONLY:

  A  WHERE   the gate's hot patch is on the object being manipulated
             GT: seg masks + target_bodies
  B  WHEN    the gate rises when that object actually moves
             GT: |xvel| of the target bodies, per frame
  C  WHAT    the content channel encodes what happened to it - direction of
             travel and how far
             GT: signed displacement of the target bodies over each step
  D  CONTACT the trace registers contact being made or broken
             GT: contact_pairs count involving a target body
  E  SPAN    the interval the trace calls active is the interval the object
             actually moves in
             GT: overlap of the above-median gate interval with the
             above-threshold motion interval

Every number is reported against the baseline it has to beat - the chance rate
for that quantity - because a lift of 1.0 on a 0.9 raw score means nothing, and
this project has already published one such number.

    python -m relmo.vjaudit --episodes 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import GRID, TUBELET  # noqa: E402
from relmo.vjrec4 import CTX, OUT4  # noqa: E402


def to_grid(m):
    import torch
    import torch.nn.functional as Fn
    t = torch.tensor(m.astype(np.float32))[None, None]
    return Fn.interpolate(t, size=(GRID, GRID), mode="area")[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=60)
    a = ap.parse_args()

    d4 = OUT4 / f"{a.dataset}_L{a.layer}"
    man = R.read_manifest(a.dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = [p for p in sorted(d4.glob("*.npz")) if p.stem in by]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), min(a.episodes, len(files)), replace=False)
    steps = list(range(CTX, 32))

    A = [0, 0, []]                       # hits, n, chance
    B, C, Cn, D, E = [], [], [], [], []
    XC, YC = [], []
    for i in pick:
        p = files[i]
        e = by[p.stem]
        z = np.load(R.dataset_dir(a.dataset) / e["shard"] / p.stem / "state.npz")
        if "seg" not in z.files:
            continue
        seg, tb = z["seg"], z["target_bodies"]
        xpos, xvel = z["xpos"], z["xvel"]
        cp, cn = z["contact_pairs"], z["contact_n"]
        T = len(seg)
        idx = np.linspace(0, T - 1, 64).round().astype(int)
        rec = np.load(p)
        gm = np.clip(rec["where_map"], 0, None)
        what = rec["pred_change"]
        gmass, tspeed, tdisp, ctouch = [], [], [], []
        for si, t in enumerate(steps):
            f = min(idx[t * TUBELET], T - 1)
            f0 = min(idx[max(t - 1, 0) * TUBELET], T - 1)
            # A - WHERE
            m = np.isin(seg[f], tb).astype(np.float32)
            if m.sum() >= 8:
                mk = to_grid(m) > 0.10
                if mk.sum():
                    g = gm[si].ravel()
                    A[0] += int(mk.ravel()[g.argmax()])
                    A[1] += 1
                    A[2].append(mk.mean())
            gmass.append(gm[si].sum())
            # B - WHEN: speed of the target bodies
            tspeed.append(np.abs(xvel[f][tb][:, :3]).sum())
            # C - WHAT: signed displacement of the target between steps
            tdisp.append((xpos[f][tb][:, :3] - xpos[f0][tb][:, :3]).mean(0))
            # D - CONTACT involving a target body
            k = int(cn[f])
            pr = cp[f][:k]
            ctouch.append(float(np.isin(pr[:, 1:3], tb).any(1).sum()) if k else 0.0)
        gmass = np.array(gmass)
        tspeed = np.array(tspeed)
        tdisp = np.array(tdisp)
        ctouch = np.array(ctouch)
        if gmass.std() > 0 and tspeed.std() > 0:
            B.append(np.corrcoef(gmass, tspeed)[0, 1])
            # E - does the gate's active interval overlap the motion interval
            ga = gmass > np.median(gmass)
            ta = tspeed > np.median(tspeed)
            inter = (ga & ta).sum()
            E.append(inter / max((ga | ta).sum(), 1))
        XC.append(what)
        YC.append(tdisp)
        if ctouch.std() > 0 and gmass.std() > 0:
            D.append(np.corrcoef(gmass, ctouch)[0, 1])

    X = np.concatenate(XC)
    Y = np.concatenate(YC)
    ep = np.concatenate([np.full(len(x), k) for k, x in enumerate(XC)])
    tr = ep < ep.max() / 2
    Xz = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)
    Xz = np.c_[Xz, np.ones(len(Xz))]
    w = np.linalg.lstsq(Xz[tr], Y[tr], rcond=None)[0]
    pr = Xz[~tr] @ w
    r2 = 1 - ((pr - Y[~tr]) ** 2).sum() / (((Y[~tr] - Y[~tr].mean(0)) ** 2).sum())
    sgn = (np.sign(pr) == np.sign(Y[~tr])).mean()

    ch = float(np.mean(A[2]))
    print(f"AUDIT - {len(pick)} episodes, sim state used for SCORING only\n")
    print(f"A  WHERE   gate peak on the manipulated object   "
          f"{A[0]/A[1]:.3f}   chance {ch:.3f}   lift {(A[0]/A[1])/ch:.2f}x")
    print(f"B  WHEN    corr(gate mass, target speed)         "
          f"{np.mean(B):+.3f}   (0 = no relation)")
    print(f"C  WHAT    held-out R2, content -> target displacement  "
          f"{r2:+.3f}")
    print(f"           sign of displacement correct          "
          f"{sgn:.3f}   chance 0.500")
    print(f"D  CONTACT corr(gate mass, contacts on target)   "
          f"{np.mean(D):+.3f}")
    print(f"E  SPAN    IoU of gate-active vs actually-moving "
          f"{np.mean(E):.3f}   chance ~0.333")
    R.log("vjaudit", dataset=a.dataset, episodes=len(pick),
          where=round(A[0]/A[1], 4), where_chance=round(ch, 4),
          when_corr=round(float(np.mean(B)), 4),
          what_r2=round(float(r2), 4), what_sign=round(float(sgn), 4),
          contact_corr=round(float(np.mean(D)), 4),
          span_iou=round(float(np.mean(E)), 4))


if __name__ == "__main__":
    main()
