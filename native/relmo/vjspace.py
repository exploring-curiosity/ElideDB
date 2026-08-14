"""WHERE the surprise was, not just how much. Geometry of the error map.

The owner's point: "a surprise may have risen in the top right for opening a
cabinet vs center or top center for a microwave". So absolute position is
NUISANCE - the same kind of event lands in different parts of the frame
depending on where the object happens to be. What should be shared is the
RELATIVE structure: a compact disturbance at the hinge that spreads outward
along the swinging panel, and the DIRECTION it sweeps.

relmo/vjcache.py already throws all of this away. res_seq pools the 16x16 error
map into one magnitude-weighted vector per timestep, so every trace of "how the
disturbed region moved and changed shape" is gone before retrieval sees it.
That is very likely why open and close are still confused: they have the same
magnitude profile and the OPPOSITE sweep, and only the sweep was discarded.

res_map (S,16,16) is already on disk, so this costs no model runs.

ARMS, each a different answer to "what part of the geometry matters"
  mass      total surprise per step. The magnitude-only story, as a floor.
  pos       ABSOLUTE centroid track. Should be WEAK if the owner is right -
            it is the arm that hard-codes where in the frame things happen.
  rel       centroid MINUS this clip's own mean centroid. Translation
            invariant: cabinet top-right and microwave top-centre collapse
            onto the same description.
  flow      centroid VELOCITY. The sweep. This is the arm that can tell an
            opening from a closing, because they differ in sign, not size.
  shape     spread, elongation and orientation of the disturbed region from
            its second moments - a swinging panel is a growing streak, a
            carried object is a moving blob.
  mirror    flow, but x-flipped so the sweep always runs one way. A
            left-hinged and a right-hinged door are the same event mirrored;
            this asks whether that should be normalised away.
  geom      rel + flow + shape together.
  geom+surp geom concatenated with the existing pooled residual direction.

Protocol is identical to relmo/vjeval.py - same pool exclusions, same k, same
grading-only labels - so the numbers are comparable to the 0.451 already
measured, and to scene-only 0.445 and the 0.279 base rate.

    python -m relmo.vjspace --dataset rcasa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjeval import REC, l2, parse  # noqa: E402

G = 16


def moments(M):
    """(S,G,G) positive surprise map -> per-step geometry.

    Only the POSITIVE part is used. res_map is already centred per patch over
    time (relmo/vjcache.py), so a negative value means "less surprising than
    this patch usually is", which is not a disturbance and must not pull the
    centroid around.
    """
    S = len(M)
    P = np.clip(M, 0, None).reshape(S, -1)
    tot = P.sum(1) + 1e-9
    ys, xs = np.mgrid[0:G, 0:G]
    xs, ys = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)
    cx = (P * xs).sum(1) / tot
    cy = (P * ys).sum(1) / tot
    dx, dy = xs[None] - cx[:, None], ys[None] - cy[:, None]
    sxx = (P * dx * dx).sum(1) / tot
    syy = (P * dy * dy).sum(1) / tot
    sxy = (P * dx * dy).sum(1) / tot
    tr = sxx + syy
    det = sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2_ = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-9)
    return dict(cx=cx, cy=cy, mass=tot / (G * G),
                spread=np.sqrt(np.maximum(tr, 0)),
                elong=np.sqrt(l1 / l2_),
                # orientation as (cos2t, sin2t): an axis has period pi, so
                # feeding the raw angle would make 0 and pi look maximally
                # different when they are the same orientation
                ori_c=np.cos(2 * np.arctan2(2 * sxy, sxx - syy)),
                ori_s=np.sin(2 * np.arctan2(2 * sxy, sxx - syy)))


def build(maps):
    """List of (S,G,G) -> dict of arm name -> (N,S,F) descriptor."""
    mo = [moments(m) for m in maps]
    S = len(maps[0])

    def stk(*keys):
        return np.stack([np.stack([m[k] for k in keys], -1) for m in mo])

    pos = stk("cx", "cy")
    rel = pos - pos.mean(1, keepdims=True)
    flow = np.diff(pos, axis=1, prepend=pos[:, :1])
    shape = stk("spread", "elong", "ori_c", "ori_s")
    mass = stk("mass")
    mir = flow.copy()
    # canonical sweep: if the net horizontal drift is negative, mirror x.
    # A left-hinged and a right-hinged door are the same event reflected.
    flip = (flow[:, :, 0].sum(1) < 0)
    mir[flip, :, 0] *= -1
    # each block standardised over the corpus so no block dominates by unit
    def nz(x):
        return (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-9)
    pos, rel, flow, shape, mass, mir = map(nz, (pos, rel, flow, shape, mass,
                                                mir))
    return dict(mass=mass, pos=pos, rel=rel, flow=flow, shape=shape,
                mirror=mir,
                geom=np.concatenate([rel, flow, shape], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    d = REC / a.dataset
    files = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
    meta = [parse(p.stem) for p in files]
    maps, seqs, ffs = [], [], []
    for p in files:
        z = np.load(p)
        maps.append(z["res_map"])
        seqs.append(z["res_seq"])
        ffs.append(z["f_first"])
    arms = build(maps)
    SEQ = l2(np.stack(seqs))
    FF = l2(np.stack(ffs))
    arms["geom+surp"] = np.concatenate(
        [arms["geom"], SEQ * np.sqrt(arms["geom"].shape[-1] / SEQ.shape[-1])],
        -1)
    arms = {k: l2(v) for k, v in arms.items()}

    obj = np.array([m["obj"] for m in meta])
    verb = np.array([m["verb"] for m in meta])
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    order = ["mass", "pos", "rel", "flow", "mirror", "shape", "geom",
             "geom+surp"]
    res = {k: [] for k in order}
    res["surprise (old)"] = []
    res["scene-only"] = []
    ctrl = {k: [] for k in ("geom", "geom+surp", "flow")}
    base = []

    for i in range(len(files)):
        cand = np.where((obj != obj[i]) & (epid != epid[i]))[0]
        if len(cand) < a.k:
            continue
        y = (verb[cand] == verb[i])
        if y.sum() == 0:
            continue
        base.append(y.mean())
        for k in order:
            M = arms[k]
            s = np.einsum("sd,nsd->n", M[i], M[cand]) / M.shape[1]
            res[k].append(y[np.argsort(-s)[:a.k]].mean())
        s = np.einsum("sd,nsd->n", SEQ[i], SEQ[cand]) / SEQ.shape[1]
        res["surprise (old)"].append(y[np.argsort(-s)[:a.k]].mean())
        res["scene-only"].append(
            y[np.argsort(-(FF[cand] @ FF[i]))[:a.k]].mean())
        full = np.where(epid != epid[i])[0]
        for k in ctrl:
            M = arms[k]
            sf = np.einsum("sd,nsd->n", M[i], M[full]) / M.shape[1]
            top = full[np.argsort(-sf)[:a.k]]
            ctrl[k].append((((obj[top] == obj[i]) & (verb[top] != verb[i])).mean(),
                            ((obj[top] != obj[i]) & (verb[top] == verb[i])).mean()))

    b = float(np.mean(base))
    print(f"{len(base)} queries | k={a.k} | pool excludes same object and "
          f"other views of the same episode\n")
    print(f"{'arm':16s} {'P@k':>8s} {'lift':>7s}")
    print("-" * 34)
    for k in ["scene-only", "surprise (old)"] + order:
        v = float(np.mean(res[k]))
        print(f"{k:16s} {v:8.3f} {v/b:7.2f}")
    print(f"{'random':16s} {b:8.3f} {1.0:7.2f}")
    print(f"\nCONTROL (full pool, top-{a.k}) - event must beat object:")
    print(f"{'arm':12s} {'same obj/wrong verb':>21s} {'diff obj/right verb':>21s}")
    out = {}
    for k, v in ctrl.items():
        c = np.array(v)
        ok = "EVENT" if c[:, 1].mean() > c[:, 0].mean() else "object"
        print(f"{k:12s} {c[:,0].mean():21.3f} {c[:,1].mean():21.3f}   {ok}")
        out[f"ctrl_{k}_obj"] = round(float(c[:, 0].mean()), 4)
        out[f"ctrl_{k}_event"] = round(float(c[:, 1].mean()), 4)
    R.log("vjspace", dataset=a.dataset, queries=len(base), k=a.k,
          base_rate=round(b, 4),
          **{k.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_"):
             round(float(np.mean(v)), 4) for k, v in res.items()}, **out)


if __name__ == "__main__":
    main()
