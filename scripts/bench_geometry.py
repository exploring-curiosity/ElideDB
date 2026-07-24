"""Measure the geometric relational verifier on CONCRETE relations.

v1 of this bench queried 'put the object in the drawer' — grounding
"a object", a placeholder that names nothing visual, so SAM 3 (whose
presence token refuses non-concepts) returned zeros/abstains and the
AUC was chance. The verifier is used with concrete nouns; the bench
must be too.

Design: each labeled episode's OWN label is parsed into its relation
('take the eggplant out of the drawer' -> ('an eggplant', 'out',
'a drawer')), and the containment DELTA (late − early IoA, always
inward-signed) is computed with the pluggable detector. Ground truth
direction comes from the label's preposition. AUC between the inward
group's deltas and the outward group's deltas is the verifier's
direction discrimination on real objects; abstain rate is reported
apart. Labels used for EVALUATION only, per the no-metadata rule.
"""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.grounding import (_INWARD, parse_relation,      # noqa: E402
                               relation_margin)


def main():
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [(stream_of.get(int(i)), int(a), int(b), (k or "").lower())
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    # drawer-containment episodes whose label parses into a relation
    # with the drawer as landmark and a CONCRETE (non-placeholder) X
    inward, outward = [], []
    for s, a, b, L in eps:
        if "draw" not in L:
            continue
        rel = parse_relation(L)
        if rel is None or "drawer" not in rel[2]:
            continue
        if "object" in rel[0] or "something" in rel[0]:
            continue                      # placeholder noun — unusable
        row = (s, a, b, rel)
        if rel[1] in _INWARD and ("put" in L or "inside" in L) \
                and "out" not in L:
            inward.append(row)
        elif "out" in L and ("take" in L or "remove" in L
                             or "get" in L):
            outward.append(row)
    inward, outward = inward[:12], outward[:12]
    print(f"inward={len(inward)} outward={len(outward)}")
    for grp, name in ((inward, "IN "), (outward, "OUT")):
        for s, a, b, rel in grp[:3]:
            print(f"  {name} {rel[0]!r} -{rel[1]}- {rel[2]!r}")

    t0 = time.time()

    def deltas(grp):
        out = []
        for s, a, b, (x, prep, y) in grp:
            out.append(relation_margin(db, s, a, b, x, y, inward=True))
        return np.array(out)

    d_in, d_out = deltas(inward), deltas(outward)
    dt = time.time() - t0
    print(f"deltas inward : {np.round(d_in, 3)}")
    print(f"deltas outward: {np.round(d_out, 3)}")
    P, N = d_in[np.isfinite(d_in)], d_out[np.isfinite(d_out)]
    auc = sum((p > n) + 0.5 * (p == n) for p, n in product(P, N)) \
        / max(1, len(P) * len(N))
    n_ep = len(inward) + len(outward)
    print(f"abstain: in {int(np.isnan(d_in).sum())}/{len(d_in)}, "
          f"out {int(np.isnan(d_out).sum())}/{len(d_out)}")
    print(f"geometry AUC inward vs outward: {auc:.3f}  "
          f"({dt:.0f}s, {dt / max(1, n_ep):.1f}s/episode)")
    print(f"sign rule: in>0 {int((P > 0).sum())}/{len(P)}, "
          f"out<0 {int((N < 0).sum())}/{len(N)}")


if __name__ == "__main__":
    main()
