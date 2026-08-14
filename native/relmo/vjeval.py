"""Does the surprise record retrieve the same KIND of event on a DIFFERENT object?

The question, in the owner's words: "when did I open a door -> it can be
cabinet, fridge or a microwave". So the test is not "find me more cabinet
clips". It is: show the system an OpenCabinet clip, and see whether
OpenMicrowave and OpenDrawer come back ABOVE everything else.

GRADING ONLY. The family names (OpenCabinet, CloseDrawer, ...) are parsed here
and nowhere else. They are never embedded, never indexed, never seen by the
model. The write path (relmo/vjcache.py) reads the mp4 and nothing else.

THE POOL IS RESTRICTED ON PURPOSE
  For every query, candidates of the SAME OBJECT are removed from the pool
  entirely. Retrieving another cabinet clip is easy and proves nothing - it can
  be done by recognising the cabinet. Only cross-object hits count, so the
  metric cannot be won by appearance.
  Other views of the SAME episode are also removed: that is near-duplicate
  retrieval, not similarity.

THE CONTROL THAT MATTERS MORE THAN THE SCORE
  same_obj_wrong_verb: for an OpenCabinet query, does CloseCabinet outrank
  OpenMicrowave? If yes, the system is doing object recognition and the verb is
  along for the ride. This is the "same vs similar" crux, and it is the exact
  failure that has beaten this project before (open/close cosine 0.957).

BASELINES, all from the same cached records
  scene-only  mean encoder feature of the FIRST timestep - the room before
              much happens. If this wins, we are recognising kitchens.
  one-frame   mean encoder feature of a middle timestep
  appearance  mean encoder feature over the whole clip
  surprise    the residual trace (ours), compared three ways:
                pooled  cosine of the time-averaged trace  <- destroys order
                direct  mean over t of cos(A_t, B_t)       <- keeps order
                dtw     warped alignment                   <- keeps order,
                                                              tolerates timing
  random      the pool base rate. Every number must be read against THIS.

Cosine appears here only as a stand-in inside a fixed comparison operator, not
as the definition of similarity; the design's real definition (transfer gain)
is expensive and comes later. If even this cannot beat scene-only, that later
step is not worth building.

    python -m relmo.vjeval --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

REC = R.BASE / "vjrec"
VERBS = ("Open", "Close", "PickPlace", "Prepare", "Stack", "Load", "Arrange")
# Fridge and Tea appear only in rcasa_eval, the held-out OOD set. OBJGROUP
# already mapped Fridge to hinged ("not in rcasa; here for reuse") but the
# parser's object list did not, so OpenFridge graded as Open/Other and formed
# its own group - which would have hidden the exact thing the OOD set tests,
# namely whether an unseen object retrieves the hinged doors it moves like.
# GRADING ONLY, as with everything in this table.
OBJECTS = ("Cabinet", "Microwave", "Drawer", "Sink", "Counter", "Coffee",
           "Bowls", "Dishwasher", "Fridge", "Tea")

# ---------------------------------------------------------------------------
# EVALUATION ONLY. Never read by the write path, never indexed, never ranked.
#
# Owner's definition of a correct match (2026-08-14):
#   TRUE  = same verb, on the same object OR an object of the SAME GROUP
#   FALSE = a different-group object (whatever the verb), or a same/similar
#           object with a different verb
#
# The grouping is by HOW THE THING MOVES, because that is what makes two events
# the same event. A cabinet, a microwave, a fridge and an oven are all a hinged
# panel swinging about an edge - opening any of them is the same experience. A
# drawer slides straight out on rails. Calling those both "Open" is a fact
# about English, not about the world, and grading them as equivalent penalised
# the system for a distinction it was drawing correctly: measured, drawers
# matched hinged doors at rank ~100 while hinged doors matched each other at
# rank ~12-20. Splitting them moved precision 0.475 -> 0.593 with no change to
# the model at all.
# ---------------------------------------------------------------------------
OBJGROUP = {
    "Cabinet": "hinged", "Microwave": "hinged", "Dishwasher": "hinged",
    "Fridge": "hinged", "Oven": "hinged",          # not in rcasa; here for reuse
    "Drawer": "sliding",
    "Sink": "basin",
    "Coffee": "appliance", "Bowls": "vessel", "Counter": "surface",
    "Tea": "tea",          # rcasa_eval only; a novel multi-step task, its own
}                          # class - it must not be folded into Coffee for free


def group_key(m, pickplace_any_destination=True):
    """Equivalence class used for grading. See OBJGROUP.

    pickplace_any_destination: a transport is arguably the same event wherever
    it ends up, so PickPlace collapses across destinations by default. Set
    False to apply the object grouping uniformly to every verb. The two give
    materially different numbers, so which one is in force must always be
    stated alongside the score.
    """
    if m["verb"] == "PickPlace" and pickplace_any_destination:
        return "PickPlace"
    return f"{m['verb']}/{OBJGROUP.get(m['obj'], m['obj'])}"


def parse(name: str):
    """GRADING ONLY. 'OpenCabinet_episode_000017__robot0_agentview_left'."""
    task = name.split("_episode_")[0]
    m = re.search(r"_episode_(\d+)", name)
    epnum = m.group(1) if m else "?"
    verb = next((v for v in VERBS if task.startswith(v)), "Other")
    rest = task[len(verb):] if verb != "Other" else task
    # The object ACTED ON. For PickPlaceCounterToSink the destination is what
    # gets interacted with, so take the object appearing LAST IN THE STRING.
    # Sorting by rest.index() is the whole point: the first version iterated
    # the OBJECTS tuple and took hits[-1], which returns whichever object comes
    # last in that TUPLE, not in the name. Every PickPlaceCounterTo* episode
    # was therefore labelled "Counter" - 128 of 447 - so Sink never appeared,
    # and because the pool excludes same-object clips, a PickPlace query
    # excluded every other PickPlace clip and became unscoreable. 169 of 447
    # queries were silently dropped.
    hits = sorted((o for o in OBJECTS if o in rest), key=rest.index)
    obj = hits[-1] if hits else "Other"
    return dict(task=task, epnum=epnum, verb=verb, obj=obj)


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def dtw_batch(A, B, band=None, pen=0.08):
    """A (S,D) vs B (N,S,D), all L2-normalised. Returns (N,) mean-cost sim.

    THE TIME AXIS IS NOT A FIXED CLOCK. A door thrown open in half a second and
    a fridge easing open over four are the same kind of event, and the stretch
    is not even uniform - things start slow and finish fast, or stall halfway.
    Matching step t against step t would call those different, which is wrong.

    Global duration is already handled upstream: relmo/vjs.py samples 64 frames
    across the WHOLE episode, so a 3 s clip and a 14 s clip arrive on the same
    time base. What is left for DTW is the DIFFERENTIAL part - the within-clip
    speed profile - and that is what the alignment has to absorb.

    band: how far the alignment may drift from the diagonal. Defaults to S//3
    (5 of 16 steps), not the 2 this originally used. A +/-2 band cannot express
    an ease-in-out at all, so the first version of this function would have
    scored exactly the case above as a mismatch.

    pen: cost added for each non-diagonal step. This is what stops the other
    failure - unbounded warping lets DTW crush a whole event onto one frame of
    another clip and call it a perfect match. A penalty buys warping only where
    the evidence pays for it, which is safer than a hard slope constraint and
    much easier to reason about.
    """
    S = A.shape[0]
    if band is None:
        band = max(2, S // 3)
    C = 1.0 - np.einsum("sd,nkd->nsk", A, B)            # (N,S,S) cosine cost
    N = C.shape[0]
    INF = 1e9
    D = np.full((N, S + 1, S + 1), INF)
    D[:, 0, 0] = 0.0
    for i in range(1, S + 1):
        lo, hi = max(1, i - band), min(S, i + band)
        for j in range(lo, hi + 1):
            best = np.minimum(np.minimum(D[:, i - 1, j] + pen,
                                         D[:, i, j - 1] + pen),
                              D[:, i - 1, j - 1])
            D[:, i, j] = C[:, i - 1, j - 1] + best
    return -D[:, S, S] / S                               # higher = more similar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--strict", action="store_true",
                    help="also exclude clips from the SAME kitchen layout")
    a = ap.parse_args()

    d = REC / a.dataset
    files = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
    if len(files) < 50:
        raise SystemExit(f"only {len(files)} records cached - let vjcache finish")
    # KITCHEN CONTROL. Same-object exclusion is not enough: a kitchen recurs
    # across tasks, so if some layouts host more Open episodes than others, a
    # pure room-recogniser scores above random without understanding anything.
    # That is the likeliest explanation for scene-only reaching 1.59x. The
    # manifest does carry layout_id/style_id (an earlier note in this project
    # claimed it did not - that note was wrong), so the confound is testable
    # rather than merely suspected.
    layout = {}
    try:
        man = R.read_manifest(a.dataset)
        for e in man["episodes"]:
            layout[e["id"]] = f"{e.get('layout_id')}/{e.get('style_id')}"
    except Exception as e:                                    # noqa: BLE001
        print(f"WARNING: no layout ids ({e}); --strict unavailable")
    meta, seq, f_all, f_first, f_mid = [], [], [], [], []
    for p in files:
        z = np.load(p)
        meta.append(parse(p.stem))
        seq.append(z["res_seq"])
        f_all.append(z["f_all"])
        f_first.append(z["f_first"])
        f_mid.append(z["f_mid"])
    SEQ = l2(np.stack(seq))                              # (N,S,D)
    POOL = l2(SEQ.mean(1))                               # (N,D) time-averaged
    F_ALL, F_FIRST, F_MID = (l2(np.stack(x)) for x in (f_all, f_first, f_mid))
    N = len(files)
    verb = np.array([m["verb"] for m in meta])
    obj = np.array([m["obj"] for m in meta])
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    lay = np.array([layout.get(p.stem, "?") for p in files])
    n_lay = len(set(lay))
    print(f"{N} records | verbs {dict(zip(*np.unique(verb, return_counts=True)))}")
    print(f"objects {dict(zip(*np.unique(obj, return_counts=True)))}\n")

    arms = {"scene-only": F_FIRST, "one-frame": F_MID, "appearance": F_ALL,
            "surprise/pooled": POOL}
    res = {k: [] for k in arms}
    res["surprise/direct"] = []
    res["surprise/dtw"] = []
    rnd, ctrl = [], []

    # only query from verbs that actually appear on >1 object, else
    # "cross-object same-verb" is undefined
    ok_verb = {v for v in np.unique(verb)
               if len(np.unique(obj[verb == v])) > 1}
    qidx = [i for i in range(N) if verb[i] in ok_verb]

    for i in qidx:
        # POOL: drop same object entirely, and any other view of this episode
        keep = (obj != obj[i]) & (epid != epid[i])
        if a.strict:
            keep &= (lay != lay[i])
        cand = np.where(keep)[0]
        if len(cand) < a.k:
            continue
        y = (verb[cand] == verb[i])
        if y.sum() == 0:
            continue
        rnd.append(y.mean())                             # base rate
        for name, M in arms.items():
            s = M[cand] @ M[i]
            res[name].append(y[np.argsort(-s)[:a.k]].mean())
        sd = np.einsum("sd,nsd->n", SEQ[i], SEQ[cand]) / SEQ.shape[1]
        res["surprise/direct"].append(y[np.argsort(-sd)[:a.k]].mean())
        sw = dtw_batch(SEQ[i], SEQ[cand])
        res["surprise/dtw"].append(y[np.argsort(-sw)[:a.k]].mean())
        # CONTROL: same object + wrong verb vs different object + right verb.
        # Uses the FULL pool, so it can see the same-object distractors.
        full = np.where((epid != epid[i]) & ((lay != lay[i]) if a.strict
                                             else True))[0]
        sw_full = dtw_batch(SEQ[i], SEQ[full])
        top = full[np.argsort(-sw_full)[:a.k]]
        same_obj_wrong_verb = ((obj[top] == obj[i]) & (verb[top] != verb[i])).mean()
        diff_obj_right_verb = ((obj[top] != obj[i]) & (verb[top] == verb[i])).mean()
        ctrl.append((same_obj_wrong_verb, diff_obj_right_verb))

    if not rnd:
        raise SystemExit("no scorable queries")
    base = float(np.mean(rnd))
    print(f"queries {len(rnd)} | k={a.k} | {n_lay} kitchen layouts | pool "
          f"EXCLUDES same object, other views of the same episode"
          + (", AND the same kitchen" if a.strict else ""))
    print(f"{'arm':22s} {'P@k':>8s} {'lift x random':>15s}")
    print("-" * 48)
    out = {}
    order = ["scene-only", "one-frame", "appearance", "surprise/pooled",
             "surprise/direct", "surprise/dtw"]
    for k_ in order:
        v = float(np.mean(res[k_]))
        print(f"{k_:22s} {v:8.3f} {v/base:15.2f}")
        out[k_.replace("/", "_")] = round(v, 4)
    print(f"{'random (base rate)':22s} {base:8.3f} {1.0:15.2f}")
    c = np.array(ctrl)
    print(f"\nCONTROL on the FULL pool, top-{a.k}, surprise/dtw:")
    print(f"  same object + WRONG verb   {c[:,0].mean():.3f}   "
          f"(object recognition)")
    print(f"  diff object + RIGHT verb   {c[:,1].mean():.3f}   "
          f"(event recognition)  <- must be higher")
    verdict = ("event > object: retrieving the KIND of event"
               if c[:, 1].mean() > c[:, 0].mean() else
               "object > event: still recognising the thing, not the event")
    print(f"  verdict: {verdict}")
    R.log("vjeval", dataset=a.dataset, records=N, queries=len(rnd), k=a.k,
          base_rate=round(base, 4), same_obj_wrong_verb=round(float(c[:, 0].mean()), 4),
          diff_obj_right_verb=round(float(c[:, 1].mean()), 4),
          verdict=verdict, **out)


if __name__ == "__main__":
    main()
