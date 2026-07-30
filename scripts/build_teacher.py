"""The teacher, all five elements, with events plural and timestamped.

    scene        one gist vector per demo
    agent        the self-moving thing, with its track extent
    participants what the agent contacted, named and verified
    events       TYPED, TIMESTAMPED transitions - contact, release,
                 open, close, put_into, put_on, take_out, vanish
    answer       the ordered rollup: verb sequence + initial->final

The single-verb-per-demo model this replaces could not express a
compositional query, and that is measurable rather than aesthetic:
"pick up a green object from table and put it into the drawer" is four
events (open, grasp, put_into, close), so q00/q01/q02 scored exactly
zero no matter how good the one verb was. Meanwhile the two queries
that ARE single events went 0.19 -> 0.53 and 0.33 -> 0.59 once their
verb was computed correctly. The schema was right; the flattening was
the bug.

Cost note: participant NAMES are reused from the existing answers
table (same boxes, same generator+verifier, already paid for at
3.4s/demo). Everything else here is geometry and costs ~0.3s/demo.

MEASURED NEGATIVE, do not "fix" again without re-measuring: the
articulated box below is the bbox of ALL coherent non-agent motion,
which spans 96% of the frame at the median, so 97% of both origins and
destinations test as "inside" it. That is physically wrong and it was
replaced with the largest CONNECTED component (plus a 40%-of-frame
rejection) - the correct construction. The metric FELL 0.25 -> 0.15:
with a real box, put_on fires on 86% of demos and stops discriminating
anything, and q03 collapsed 0.79 -> 0.26. The whole-frame box was
accidentally acting as a useful prior, which means the 0.25 is partly
luck and the containment test is not yet a real relation detector.
The honest fix is to identify the CONTAINER as an entity - not to
threshold a motion blob - and that is unbuilt.

THREE MORE ROUTES TO THAT SPLIT ARE NOW ALSO MEASURED DEAD, so the
container really is the binding constraint and not a threshold:
  - persistence via state_diff (an object put INTO something stops
    being visible, one put ON something does not): only 1.87 diff
    sites per demo against 1.52 transitions, 3.5% of participant
    origins carry a vanish site, 91% of transitions type as neither.
    Starved, not wrong - there are not enough sites to type with.
  - temporal bracket (containment needs the container OPEN during the
    transfer, and the events are timestamped for exactly this):
    put_into 22.7% bracketed vs adjust 25.4%. Identical rates - no
    discrimination whatsoever.
  - fixing the displacement threshold below and letting the existing
    containment branch sort the survivors: adjust 82% -> 51% as
    intended, but the freed transitions all funnel into put_into
    (321 -> 799) because inside() is 97% trivially true, and the
    benchmark did not move (0.45/0.32 before and after, 481 vs 483
    true documents). The transition ANCHOR then scores put_into at
    0.000 reliability - it cannot retrieve its own members - which is
    the mechanism correctly refusing a label the geometry never
    earned. put_on reached 17 events and take_out 10.
Net: the displacement fix is kept because it corrects a demonstrated
measurement error (see REL_MIN), NOT because it improved retrieval.
It did not.

  python scripts/build_teacher.py [--n 20]   sample, prints scripts
  python scripts/build_teacher.py --all      writes events + answers
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                    # noqa: E402
from extract_events import agent_track, causal_participants  # noqa: E402

NFRAMES = 12
# MOVED, RELATIVE TO THE OBJECT - not to the frame.
#
# DISP_MIN was a fraction of the frame diagonal, and it silently made
# "did this object move" a question about the TRACKER instead. Measured
# on 150 demos / 228 participant transitions: median displacement 0.025
# of the diagonal, so the 0.05 threshold sat at the ~80th percentile and
# 82% of all participant motion was typed "adjust". The reason is
# truncation, not stillness - median track lifespan is 3.5 of 12 sampled
# frames, 35% die at <=2 frames, and corr(lifespan, displacement) =
# +0.487. A two-frame track cannot exhibit displacement, so the object
# was being called static because we stopped watching it.
#
# An object's own size is the scale-free reference: half its own
# diagonal means its finish does not sit on top of its start. That is a
# statement about the object leaving where it was, and it holds for a
# spoon and for a pot. Not tuned to produce a rate - the measured median
# lands at 0.469x, near the cut, which is what "half moved, half did
# not" should look like for robot demos full of reaching and adjusting.
REL_MIN = 0.5
CAV_MIN = 0.015


def cavity_series(frames, box):
    """Dark-fraction inside the articulated box, per frame. The event
    is the TRANSITION in this series, so it carries a timestamp - the
    v1 first-vs-last scalar could say a drawer opened but never when,
    and 'when' is what makes an event an event."""
    import cv2
    if box is None or (box[2] - box[0]) < 8 or (box[3] - box[1]) < 8:
        return None
    crops = [cv2.cvtColor(f[box[1]:box[3], box[0]:box[2]],
                          cv2.COLOR_RGB2GRAY).astype(float)
             for f in frames]
    thr = np.percentile(np.concatenate([c.ravel() for c in crops]), 25)
    return np.array([float((c < thr).mean()) for c in crops])


def articulated_box(frames, agent_union):
    """Largest coherent non-agent moving region across the demo."""
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    best = None
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        m = (mag > max(1.0, 3 * float(np.median(mag)))) & (~agent_union)
        if m.sum() < 800:
            continue
        ys, xs = np.where(m)
        bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if best is None or ((bb[2] - bb[0]) * (bb[3] - bb[1])
                            > (best[2] - best[0]) * (best[3] - best[1])):
            best = bb
    return best


def build(db, frames_tbl, key, names):
    """One demo -> (scene/agent/answer dict, [event dicts])."""
    from elidedb.video import FrameSet
    s, a, b = key
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), s),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                pc.less_equal(frames_tbl.column("ts"), b))))
    n = len(sel)
    if n < 4:
        return None, []
    pick = np.unique(np.linspace(0, n - 1, min(NFRAMES, n))
                     .round().astype(int))
    try:
        dec = sorted(FrameSet(db, "frames", sel.take(pick)).decode())
    except Exception:
        return None, []
    times = [int(t) for t, _ in dec]
    frames = [f for _, f in dec]
    if len(frames) < 4:
        return None, []
    H, W = frames[0].shape[:2]

    tr, span, masks, all_tracks = agent_track(frames)
    agent_union = masks.any(0)
    abox = articulated_box(frames, agent_union)
    ev = []

    def stamp(i):
        return times[min(max(i, 0), len(times) - 1)]

    # AGENT — recorded as an entity with an extent, not just a scalar
    if tr is not None:
        ev.append({"kind": "agent", "role": "agent",
                   "t0": stamp(tr[0][0]), "t1": stamp(tr[-1][0] + 1),
                   "box": list(tr[-1][3]), "name": "", "conf": span})

    # ARTICULATION — transitions in the cavity series, timestamped
    cav = cavity_series(frames, abox)
    if cav is not None and len(cav) >= 4:
        base = float(np.median(cav[:2]))
        run = None
        for i in range(1, len(cav)):
            d = float(cav[i]) - base
            k = "open" if d >= CAV_MIN else ("close" if d <= -CAV_MIN
                                             else None)
            if k and run is None:
                run = (k, i)
            elif run and (k != run[0]):
                ev.append({"kind": run[0], "role": "container",
                           "t0": stamp(run[1] - 1), "t1": stamp(i),
                           "box": list(abox), "name": "",
                           "conf": abs(d)})
                run = (k, i) if k else None
        if run:
            ev.append({"kind": run[0], "role": "container",
                       "t0": stamp(run[1] - 1), "t1": stamp(len(cav) - 1),
                       "box": list(abox), "name": "",
                       "conf": abs(float(cav[-1]) - base)})

    # PARTICIPANTS — contact at motion onset, release at track end, and
    # the relocation between them typed by geometry
    for origin, dest, onset, life, area in causal_participants(
            all_tracks, tr, len(frames) - 1):
        nm_o = names.get((s, a, "origin"), ("", 0.0))
        nm_d = names.get((s, a, "dest"), ("", 0.0))
        ev.append({"kind": "contact", "role": "participant",
                   "t0": stamp(onset - 1), "t1": stamp(onset),
                   "box": list(origin), "name": nm_o[0],
                   "conf": nm_o[1]})
        ev.append({"kind": "release", "role": "participant",
                   "t0": stamp(onset + life - 1),
                   "t1": stamp(onset + life),
                   "box": list(dest), "name": nm_d[0], "conf": nm_d[1]})
        c0 = np.array([(origin[0] + origin[2]) / 2,
                       (origin[1] + origin[3]) / 2])
        c1 = np.array([(dest[0] + dest[2]) / 2,
                       (dest[1] + dest[3]) / 2])
        odiag = float(np.hypot(origin[2] - origin[0],
                               origin[3] - origin[1]))
        disp = float(np.linalg.norm(c1 - c0)) / max(odiag, 1.0)

        def inside(c, rgn):
            return (rgn is not None and rgn[0] <= c[0] <= rgn[2]
                    and rgn[1] <= c[1] <= rgn[3])

        if disp < REL_MIN:
            k = "adjust"
        elif inside(c0, abox) and not inside(c1, abox):
            k = "take_out"
        elif inside(c1, abox):
            k = "put_into"
        else:
            k = "put_on"
        ev.append({"kind": k, "role": "participant",
                   "t0": stamp(onset), "t1": stamp(onset + life),
                   "box": list(dest), "name": nm_o[0] or nm_d[0],
                   "conf": max(nm_o[1], nm_d[1])})

    ev.sort(key=lambda e: e["t0"])
    verbs = [e["kind"] for e in ev
             if e["kind"] not in ("agent", "contact", "release")]
    head = {"stream": s, "ts": a, "t1": b,
            "agent_span": float(span),
            "verb_seq": ">".join(verbs),
            "n_events": len(ev)}
    return head, ev


def main():
    argv = sys.argv
    write_all = "--all" in argv
    n_want = int(argv[argv.index("--n") + 1]) if "--n" in argv else 20

    db = Store.open("lake/bench")
    ep = db.table("episodes").scan().to_pydict()
    keys = list(zip(ep["stream"], (int(v) for v in ep["ts"]),
                    (int(v) for v in ep["t1"])))
    if not write_all:
        rng = np.random.default_rng(0)
        keys = [keys[i] for i in rng.choice(len(keys), n_want,
                                            replace=False)]
    frames_tbl = db.table("frames").scan()

    # reuse the verified names already paid for
    old = db.table("answers").scan().to_pydict()
    names = {}
    for i in range(len(old["ts"])):
        if old["name"][i]:
            names[(old["stream"][i], int(old["ts"][i]),
                   old["kind"][i])] = (old["name"][i], 1.0)

    heads, evs, t0 = [], [], time.time()
    for i, k in enumerate(keys):
        h, e = build(db, frames_tbl, k, names)
        if h is None:
            continue
        heads.append(h)
        evs.extend([dict(x, stream=k[0], ts=k[1], t1e=k[2]) for x in e])
        if not write_all:
            print(f"{k[0].split('/')[-1]} {str(k[1])[-8:]}  "
                  f"agent {h['agent_span']:.0%}  {h['n_events']:2d} ev  "
                  f"{h['verb_seq'] or '(none)'}")
        elif (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(keys)} {el:.0f}s "
                  f"ETA {el / (i + 1) * len(keys) / 60:.0f}min", flush=True)
    dt = (time.time() - t0) / max(len(heads), 1)
    from collections import Counter
    print(f"\n{len(heads)} demos, {len(evs)} events, {dt:.2f}s/demo")
    print("event kinds:", Counter(e["kind"] for e in evs).most_common())
    print("verb seqs:  ", Counter(h["verb_seq"] for h in heads)
          .most_common(6))

    if not write_all:
        return

    # SCENE: the demo's gist, pooled from vectors the store already has
    from elidedb.embeddings import _vec_table
    tb, PV = _vec_table(db, "pe_vectors")
    PV = np.asarray(PV, np.float32)
    rmap = defaultdict(list)
    for i, (s_, a_) in enumerate(zip(tb.column("stream").to_pylist(),
                                     tb.column("ts").to_pylist())):
        rmap[(str(s_), int(a_))].append(i)
    from elidedb.sig2 import _text_vec
    cache, nvecs = {}, []
    for e in evs:
        nm = e["name"]
        if nm and nm not in cache:
            cache[nm] = np.asarray(_text_vec(nm), np.float32)
        nvecs.append(cache.get(nm, np.zeros(1152, np.float32)))
    NVv = np.stack(nvecs) if nvecs else np.zeros((0, 1152), np.float32)

    etbl = pa.table({
        "ts": pa.array([int(e["ts"]) for e in evs], pa.int64()),
        "t1": pa.array([int(e["t1e"]) for e in evs], pa.int64()),
        "stream": pa.array([e["stream"] for e in evs]),
        "kind": pa.array([e["kind"] for e in evs]),
        "role": pa.array([e["role"] for e in evs]),
        "ev_t0": pa.array([int(e["t0"]) for e in evs], pa.int64()),
        "ev_t1": pa.array([int(e["t1"]) for e in evs], pa.int64()),
        "name": pa.array([e["name"] for e in evs]),
        "conf": pa.array([float(e["conf"]) for e in evs], pa.float32()),
        "box": pa.FixedSizeListArray.from_arrays(
            pa.array(np.array([e["box"] for e in evs],
                              np.int32).reshape(-1), pa.int32()), 4),
        "name_vec": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(
                NVv.astype(np.float16)).reshape(-1), pa.float16()), 1152),
    })
    etbl = etbl.take(pc.sort_indices(etbl.column("ts")))
    db.table("events").replace(etbl, kind="events",
                               meta={"extractor": "teacher-v3"})
    print(f"events: {etbl.num_rows} rows")

    SV = np.stack([
        (PV[rmap[(h["stream"], h["ts"])]].mean(0)
         if rmap.get((h["stream"], h["ts"])) is not None
         and len(rmap[(h["stream"], h["ts"])])
         else np.zeros(PV.shape[1], np.float32))
        for h in heads])
    SV /= np.linalg.norm(SV, axis=1, keepdims=True) + 1e-8
    atbl = pa.table({
        "ts": pa.array([h["ts"] for h in heads], pa.int64()),
        "t1": pa.array([h["t1"] for h in heads], pa.int64()),
        "stream": pa.array([h["stream"] for h in heads]),
        "agent_span": pa.array([h["agent_span"] for h in heads],
                               pa.float32()),
        "verb_seq": pa.array([h["verb_seq"] for h in heads]),
        "n_events": pa.array([h["n_events"] for h in heads], pa.int32()),
        "scene_vec": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(
                SV.astype(np.float16)).reshape(-1), pa.float16()),
            SV.shape[1]),
    })
    atbl = atbl.take(pc.sort_indices(atbl.column("ts")))
    db.table("answers2").replace(atbl, kind="events",
                                 meta={"extractor": "teacher-v3"})
    print(f"answers2: {atbl.num_rows} rows")


if __name__ == "__main__":
    main()
