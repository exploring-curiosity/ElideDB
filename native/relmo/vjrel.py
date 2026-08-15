"""Graded ground-truth relevance. TRAINING TARGET ONLY, never at serve.

The old target was binary - same group or not. The owner's definition is a
RANKING: a clip matching on the event AND the object AND the kitchen AND the
camera is more relevant than one matching only on the event, and both are far
more relevant than a different verb.

    different EVENT                                  0.00   penalised hard
    same event, different object                     1.00   floor for a match
      + same object                                 +0.50
      + same scene (layout_id, style_id)            +0.25
      + same camera pose                            +0.25
                                                     ----
    identical in every respect                       2.00

"SAME EVENT" IS THE OBJECT GROUP, NOT THE VERB. Owner, 2026-08-14: "a drawer
open is not same as cabinet open but a fridge open is." A hinged panel swinging
about an edge and a drawer sliding on rails are different events that English
happens to call by the same word, so the verb alone cannot define the positive
class. The boundary is vjeval.group_key - verb crossed with how the object
moves (hinged / sliding / basin / vessel / appliance), with PickPlace collapsed
across destinations. An earlier version of this file scored OpenDrawer against
OpenCabinet as 1.00, a positive, which is exactly the error.

The shape matters as much as the values. Gradations among matches are gentle -
1.00 to 2.00 - because a cross-object match of the same event (fridge against
cabinet) is still a good answer and must not be pushed down near a wrong one.
The drop to 0.00 is the only large gap.

WHY SCENE AND CAMERA BELONG IN THE TARGET AND NOT IN THE MODEL. They are
nuisance for the question "what happened", but they are real evidence about how
alike two recordings are, and a ranker that ignores them is being trained to
call two clips equally good when one is plainly closer. They enter the LABEL,
never the input; the model sees only latents and is never asked to predict a
verb, an object, a scene or a camera.

CROSS-VIEW REMAINS BARRED. Two cameras of the SAME rollout are the same
physical event recorded twice, and the owner ruled that out as a training
signal. Those pairs are dropped from training entirely - `same camera` above
compares camera POSE between different rollouts, which is a nuisance-similarity
term, not cross-view matching.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjeval import group_key, parse  # noqa: E402

W_EVENT, W_OBJ, W_SCENE, W_CAM = 1.00, 0.50, 0.25, 0.25
MAX_REL = W_EVENT + W_OBJ + W_SCENE + W_CAM

# DURATION IS CONTENT, and it declines the rank rather than gating it.
# Owner, 2026-08-15: "its not a yes/no. its a passive declination in rank of the
# similiarity. Just the rank goes down as ratio increases." and "reduce the
# ratio further keep it 1.5".
#
# So there is no cut and no positive-class gate. The weight decays smoothly and
# monotonically in the LOG of the duration ratio, by a factor of e for every
# R_SCALE-fold mismatch:
#
#     weight(r) = exp( -ln(r) / ln(R_SCALE) )
#
#     1.00x -> 1.000     1.43x -> 0.396     2.00x -> 0.181
#     1.50x -> 0.368     1.75x -> 0.246     2.86x -> 0.075
#
# A duration-matched instance therefore always outranks a stretched one - "7s
# and 10s linked but not ranked 1" - and a 7 v 20 s pair sinks to 7% of full
# relevance without ever being declared a non-match. NDCG is what reads this;
# precision@support stays event-level, because a binary metric cannot express a
# passive decline.
R_SCALE = 1.5


def dur_ratio(durations):
    """(n,) seconds -> (n,n) max/min ratio, >= 1 everywhere."""
    d = np.maximum(np.asarray(durations, float), 1e-9)
    return np.maximum(d[:, None], d[None, :]) / np.minimum(d[:, None], d[None, :])


def dur_weight(ratio):
    """1.0 at identical duration, falling by 1/e per R_SCALE-fold mismatch."""
    r = np.maximum(np.asarray(ratio, float), 1.0)
    return np.exp(-np.log(r) / np.log(R_SCALE))


def meta_table(dataset="rcasa"):
    """episode id -> dict(event, obj, rollout, scene, camera, dur). GRADING."""
    man = R.read_manifest(dataset)
    fps = float(man.get("fps", 20))
    out = {}
    for e in man["episodes"]:
        m = parse(e["id"])
        cam = e.get("camera", "?")
        # rollout identity: the same demonstration recorded from N cameras
        out[e["id"]] = dict(event=group_key(m), obj=m["obj"],
                            rollout=f"{m['task']}#{m['epnum']}",
                            scene=str(e.get("scene", "?")), camera=cam,
                            # TRUE duration from the manifest, not the trace
                            # length - the GT must not depend on the encoder
                            dur=float(e.get("T", 0)) / float(fps))
    return out


_ALL = {}


def all_meta(datasets=("rcasa", "rcasa_eval")):
    """Merged meta over every corpus an evaluator might mix. Cached."""
    if not _ALL:
        for d in datasets:
            try:
                _ALL.update(meta_table(d))
            except Exception:                                  # noqa: BLE001
                pass
    return _ALL


def relevance(ids, meta):
    """-> (rel (n,n) float32, valid (n,n) bool).

    valid is False on the diagonal and on same-rollout pairs, which are the
    barred cross-view case and must not train anything.
    """
    n = len(ids)
    event = np.array([meta[i]["event"] for i in ids])
    obj = np.array([meta[i]["obj"] for i in ids])
    roll = np.array([meta[i]["rollout"] for i in ids])
    scene = np.array([meta[i]["scene"] for i in ids])
    cam = np.array([meta[i]["camera"] for i in ids])

    ratio = dur_ratio([meta[i]["dur"] for i in ids])
    same = event[:, None] == event[None, :]        # group_key, not the verb
    rel = np.where(same, W_EVENT, 0.0)
    rel = rel + np.where(same & (obj[:, None] == obj[None, :]), W_OBJ, 0.0)
    rel = rel + np.where(same & (scene[:, None] == scene[None, :]),
                         W_SCENE, 0.0)
    rel = rel + np.where(same & (cam[:, None] == cam[None, :]), W_CAM, 0.0)

    # duration scales the WHOLE relevance, not just the event floor - a
    # stretched instance of the same event in the same kitchen is still less
    # like the query than an unstretched one
    rel = rel * dur_weight(ratio)
    valid = roll[:, None] != roll[None, :]        # drops diagonal AND cross-view
    return rel.astype(np.float32), valid


def ndcg(scores, rel, valid, k=None):
    """Mean NDCG@k over queries. The graded metric the target is defined in."""
    n = len(scores)
    out = []
    for q in range(n):
        m = valid[q]
        if m.sum() < 2:
            continue
        s, r = scores[q][m], rel[q][m]
        kk = int(m.sum()) if k is None else min(k, int(m.sum()))
        order = np.argsort(-s)[:kk]
        disc = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float((r[order] * disc).sum())
        ideal = float((np.sort(r)[::-1][:kk] * disc).sum())
        if ideal > 0:
            out.append(dcg / ideal)
    return float(np.mean(out)) if out else float("nan")


def describe(ids, meta):
    rel, valid = relevance(ids, meta)
    v = rel[valid]
    print(f"{len(ids)} episodes, {int(valid.sum())} usable pairs "
          f"(same-rollout cross-view pairs dropped)")
    for lo, hi, name in [(-.1, .1, "0.00  different event"),
                         (.9, 1.1, "1.00  same event, other object"),
                         (1.2, 1.3, "1.25  + scene or camera"),
                         (1.4, 1.6, "1.50  same event + same object"),
                         (1.7, 1.8, "1.75  + scene or camera"),
                         (1.9, 2.1, "2.00  identical in every respect")]:
        c = int(((v > lo) & (v < hi)).sum())
        if c:
            print(f"   {name:42s} {c:7d}  {c/len(v):6.3f}")
