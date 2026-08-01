"""Verbs from GEOMETRY, not from whether naming succeeded.

The v1 topology asked "did we get a vanish-site and an appear-site?"
and called that `move`. The causal extractor emits both ends of every
track it finds, so 864 of 1,122 demos answered yes and 92% of the
corpus carried one verb - an events field with no information in it,
which is exactly what the verb-only ablation measured (0.08).

The verb is a TRANSITION, so it has to be a partition of what the
geometry says happened. Everything below reads columns the answers
table already stores (origin box, dest box, articulated region,
articulation displacement), so this recomputes without re-extracting:

    adjust     the participant was touched but did not relocate
               (displacement under 5% of the frame diagonal)
    open/close the articulated region GAINED or LOST a dark cavity.
               Flow sign was measured useless for this - it produced
               a verb field whose distribution among true demos was
               the corpus prior (q04 "closes the drawer": close 38%
               vs 28% base), and its calibration against the action
               probe found nothing (corr +0.04). The cavity does
               separate: opening +0.039 dark-fraction, positive in
               82% of q05 trues; closing -0.048, positive in 8% of
               q04 trues. A drawer opening reveals an interior; that
               is the event, and it needs no camera convention.
    put_into   it relocated and ended INSIDE the articulated region
    take_out   it STARTED inside the articulated region and left
    put_away   it relocated and no end position survived (occluded or
               contained - the track died)
    put_on     it relocated and ended outside any articulated region

Writes artifacts/verbs_v2.json (demo key -> verb) so the change can be
measured before it is baked into the extractor.

  python scripts/verb_recompute.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402

DISP_MIN = 0.05        # frame diagonals; below this nothing relocated
ART_MIN = 2.5          # px of cumulative articulated displacement
CAV_MIN = 0.015        # dark-fraction change that counts as a cavity


def classify(origin, dest, art_box, art, diag, cav=0.0):
    has_o = origin is not None and (origin[2] - origin[0]) > 0
    has_d = dest is not None and (dest[2] - dest[0]) > 0
    has_art = art_box is not None and (art_box[2] - art_box[0]) > 0

    def ctr(b):
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])

    def inside(b, rgn):
        c = ctr(b)
        return rgn[0] <= c[0] <= rgn[2] and rgn[1] <= c[1] <= rgn[3]

    disp = (float(np.linalg.norm(ctr(dest) - ctr(origin))) / diag
            if (has_o and has_d) else None)

    # the articulated region dominates when nothing else relocated,
    # and the CAVITY - not the flow sign - says which way
    if abs(cav) >= CAV_MIN and (disp is None or disp < DISP_MIN):
        return "open" if cav > 0 else "close"
    if abs(art) >= ART_MIN and (disp is None or disp < DISP_MIN) \
            and abs(cav) < CAV_MIN:
        return "adjust"          # it moved but revealed nothing
    if disp is not None and disp < DISP_MIN:
        return "adjust"
    if has_o and not has_d:
        return "put_away"
    if disp is not None:
        if has_art and inside(origin, art_box) and not inside(dest, art_box):
            return "take_out"
        if has_art and inside(dest, art_box):
            return "put_into"
        return "put_on"
    if abs(cav) >= CAV_MIN:
        return "open" if cav > 0 else "close"
    return "adjust"


def main():
    db = Store.open("lake/bench")
    ans = db.table("answers").scan().to_pydict()
    BX = np.array(ans["box"], np.int32).reshape(-1, 4)
    AB = np.array(ans["art_box"], np.int32).reshape(-1, 4)
    rows = defaultdict(list)
    for i in range(len(ans["ts"])):
        rows[(ans["stream"][i], int(ans["ts"][i]))].append(i)
    # frame diagonal from the widest box seen (640x480 corpus)
    diag = float(np.hypot(640, 480))

    cav = {}
    cp = ROOT / "artifacts/cavity.json"
    if cp.exists():
        cav = json.loads(cp.read_text())
    out, old = {}, {}
    for k, rr in rows.items():
        art = float(ans["articulation"][rr[0]])
        ab = AB[rr[0]]
        origin = dest = None
        for r in rr:
            if ans["kind"][r] in ("origin", "vanish"):
                origin = BX[r]
            elif ans["kind"][r] in ("dest", "appear"):
                dest = BX[r]
        kk = f"{k[0]}|{k[1]}"
        out[kk] = classify(origin, dest, ab, art, diag,
                           float(cav.get(kk, 0.0)))
        old[f"{k[0]}|{k[1]}"] = ans["verb"][rr[0]]

    (ROOT / "artifacts/verbs_v2.json").write_text(json.dumps(out))
    co, cn = Counter(old.values()), Counter(out.values())
    print(f"{'verb':>10} {'v1':>6} {'v2':>6}")
    for v in sorted(set(co) | set(cn)):
        print(f"{v:>10} {co.get(v, 0):>6} {cn.get(v, 0):>6}")
    top = cn.most_common(1)[0]
    print(f"\nv1 largest class {co.most_common(1)[0][1] / len(old):.0%}"
          f" -> v2 {top[1] / len(out):.0%} ({top[0]})")


if __name__ == "__main__":
    main()
