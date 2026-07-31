"""DISCOVERED transitions. No verb is written in this file, on purpose.

What it replaces
----------------
The write path used to type every participant transition through a
hand-authored ladder:

    k = ("adjust"   if disp < REL_MIN
         else "take_out" if inside(c0, abox) and not inside(c1, abox)
         else "put_into" if inside(c1, abox)
         else "put_on")

with FLAG_KINDS naming the eight outcomes. Two things were wrong with
that, and the second is worse than the first.

1. It is a task prior. "put_into" is a fact about table-top
   manipulation. A driving corpus has no put_into and every lane change
   would be forced into `put_on`; a warehouse camera has no drawers.
   The system could not recognise a domain it had not been told about.

2. `adjust` was the `else` branch, so it was not a class - it was
   everything unclassified, which is why 82% of participant motion
   landed there and why it scores AUC 0.553 against 0.898 for put_on.
   A catch-all guarantees a majority class that means nothing.

WHAT REPLACES IT
----------------
Geometry already produces continuous, nameless quantities. Keep them
continuous, describe each transition as a vector of them, and let the
corpus say how many kinds there are and where the boundaries fall.

The descriptor is deliberately domain-free - every component is a
measurement any moving-object corpus can produce:

    displacement      how far the thing went, in its own body lengths,
                      so a lane change and a spoon lift are comparable
    direction         unit vector of travel; the axis a corpus cares
                      about is its own business
    duration          how long the transition took, in seconds
    scale change      did it grow or shrink in view - approach, recede,
                      or occlusion, without deciding which
    enclosure         change in how much of it sits inside the largest
                      surrounding structure; a drawer, a bin, a garage
                      and a lane are all "an enclosure" at this level
    persistence       is it still visible afterwards
    agent proximity   distance to the self-moving thing at onset and at
                      offset, in the same body-length units

CLUSTERING WITH NOISE, not a partition. HDBSCAN leaves genuinely
unusual transitions unlabelled (-1) instead of sweeping them into a
majority bucket. That is the direct fix for `adjust`: a transition the
corpus cannot type stays untyped, and a query for it finds nothing
rather than finding everything.

Types get INTEGER IDS, never names. A name is a read-time convenience -
if a human wants one, it can be attached at the surface from whatever
language happens to be available, and the store is unaffected.
"""
from __future__ import annotations

import numpy as np

# The descriptor's field order. Named so the code is readable; none of
# these names is a category, they are measurements.
FIELDS = ("disp", "dir_x", "dir_y", "duration", "scale_ratio",
          "enclosure_delta", "persists", "agent_d0", "agent_d1")


def descriptor(origin, dest, onset, life, fps, frame_wh,
               enclosure=None, agent_xy=None, persists=True):
    """One transition -> a nameless vector of physical measurements.

    `origin` and `dest` are boxes (x0, y0, x1, y1); `enclosure` is the
    box of the largest surrounding structure if one was detected, else
    None. Everything is normalised by the moving object's own size, so
    the descriptor does not depend on resolution or on how big things
    happen to be in this corpus.
    """
    o = np.asarray(origin, np.float64)
    d = np.asarray(dest, np.float64)
    c0 = np.array([(o[0] + o[2]) / 2, (o[1] + o[3]) / 2])
    c1 = np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])
    body = max(float(np.hypot(o[2] - o[0], o[3] - o[1])), 1.0)
    v = c1 - c0
    disp = float(np.linalg.norm(v)) / body
    dirv = v / (np.linalg.norm(v) + 1e-8)
    a0 = max((o[2] - o[0]) * (o[3] - o[1]), 1.0)
    a1 = max((d[2] - d[0]) * (d[3] - d[1]), 1.0)

    def frac_inside(box, reg):
        if reg is None:
            return 0.0
        x0, y0 = max(box[0], reg[0]), max(box[1], reg[1])
        x1, y1 = min(box[2], reg[2]), min(box[3], reg[3])
        inter = max(x1 - x0, 0) * max(y1 - y0, 0)
        return float(inter / max((box[2] - box[0]) * (box[3] - box[1]), 1))

    # ENCLOSURE OVER CANDIDATE STRUCTURES, strongest change wins.
    #
    # `enclosure` was one box: articulated_box, the bbox of all coherent
    # non-agent motion. That box spans 96% of the frame at the median,
    # so containment tested true 97% of the time and the field was dead
    # - measured ~0.00 for four of seven discovered types, which is why
    # 84% of transitions came back unclustered.
    #
    # Now it is a LIST of detected regions (segmented structures), and
    # the descriptor reports the largest containment change against any
    # of them. No region is designated "the container": whichever
    # structure the object actually moved into or out of is the one that
    # scores, which is as true of a lane or a shelf as of a drawer.
    regs = ([enclosure] if enclosure is not None
            and not isinstance(enclosure, (list, tuple))
            or (enclosure is not None and len(enclosure) == 4
                and np.isscalar(enclosure[0]))
            else list(enclosure or ()))
    enc = 0.0
    for r in regs:
        if r is None:
            continue
        dd = frac_inside(d, r) - frac_inside(o, r)
        if abs(dd) > abs(enc):
            enc = dd
    ad0 = ad1 = 0.0
    if agent_xy is not None:
        g0, g1 = np.asarray(agent_xy[0], float), np.asarray(agent_xy[1], float)
        ad0 = float(np.linalg.norm(c0 - g0)) / body
        ad1 = float(np.linalg.norm(c1 - g1)) / body
    return np.array([disp, dirv[0], dirv[1], life / max(fps, 1e-6),
                     np.log(a1 / a0), enc, 1.0 if persists else 0.0,
                     ad0, ad1], np.float32)


def _standardise(D):
    """Z-score per field. Without it `duration` in seconds and
    `direction` in [-1, 1] are not comparable and the clustering is
    really clustering whichever field happens to have the largest
    units."""
    mu, sd = D.mean(0), D.std(0) + 1e-6
    return (D - mu) / sd, mu, sd


def discover(D, min_size=None, seed=0):
    """Descriptors -> (labels, centroids, report). Labels of -1 are
    genuinely untyped and MUST stay untyped.

    min_size defaults to a fraction of the corpus rather than a
    constant, so the same call behaves sensibly on 500 transitions and
    on 500,000.
    """
    D = np.asarray(D, np.float32)
    if len(D) < 20:
        return (np.full(len(D), -1, np.int32),
                np.zeros((0, D.shape[1]), np.float32),
                {"reason": "too few transitions to discover types",
                 "n": int(len(D))})
    Z, mu, sd = _standardise(D)
    if min_size is None:
        min_size = max(10, int(0.01 * len(Z)))
    try:
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=int(min_size),
                      metric="euclidean").fit_predict(Z)
    except Exception as e:
        return (np.full(len(D), -1, np.int32),
                np.zeros((0, D.shape[1]), np.float32),
                {"reason": f"clustering unavailable: {e}"})
    ids = sorted({int(c) for c in lab if c >= 0})
    cent = (np.stack([Z[lab == c].mean(0) for c in ids])
            if ids else np.zeros((0, Z.shape[1]), np.float32))
    remap = {c: i for i, c in enumerate(ids)}
    lab = np.array([remap.get(int(c), -1) for c in lab], np.int32)
    sizes = [int((lab == i).sum()) for i in range(len(ids))]
    return lab, cent.astype(np.float32), {
        "n": int(len(D)), "types": len(ids),
        "unclustered": int((lab < 0).sum()),
        "unclustered_frac": round(float((lab < 0).mean()), 3),
        "sizes": sizes, "min_size": int(min_size),
        "largest_frac": round(max(sizes) / len(D), 3) if sizes else 0.0,
        "mu": mu.tolist(), "sd": sd.tolist()}


def profile(cent, mu, sd):
    """A type's centroid back in physical units, so a human can READ
    what the corpus found without a name being invented for it.

    This is the honest form of naming: report that type 3 moves 4.2 body
    lengths over 1.1 s while enclosure rises 0.6 - and let whoever is
    looking call it whatever their domain calls it.
    """
    mu, sd = np.asarray(mu), np.asarray(sd)
    return [dict(zip(FIELDS, np.round(c * sd + mu, 3).tolist()))
            for c in np.asarray(cent)]


def aperture_descriptor(signed_change, duration_s):
    """A scene-level aperture event, in the SAME descriptor space.

    An aperture opening and a participant moving are different things,
    but both are "something changed, by this much, in this direction,
    over this long", and putting them in one space lets one discovery
    pass type both - and lets the corpus decide whether they are really
    distinct, instead of that being assumed by having two code paths.

    Signed magnitude goes in the direction slot: a corpus whose
    apertures are drawers gets one pair of types, a corpus whose
    apertures are lane gaps gets another, and neither is named here.
    """
    return np.array([abs(signed_change), np.sign(signed_change), 0.0,
                     duration_s, 0.0, signed_change, 1.0, 0.0, 0.0],
                    np.float32)
