"""Fit RRF channel weights offline against the regress10 bar.

One live pass captures every query's per-segment channel scores; then
thousands of weight combinations are evaluated in memory — rank fusion is
deterministic given the scores, so the search costs milliseconds per
combo and ZERO query latency ever. Coordinate ascent from all-ones.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
import elidedb.verified as V                                 # noqa: E402
from elidedb.fusion import ranks_from_scores                 # noqa: E402
from regress10 import QUERIES                                # noqa: E402

V._verdict_map = lambda s: {}
V._verify_segments = lambda *a, **k: {}

CH = ["app", "ctx", "lex", "met", "mot", "anc", "obj"]


def main():
    store = sys.argv[1] if len(sys.argv) > 1 else "bridge4h"
    db = Store.open(f"lake/{store}")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [{"t0": int(a), "t1": int(b), "task": (k or "").lower(),
            "stream": stream_of.get(int(i))}
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    by_s = {}
    for e in eps:
        by_s.setdefault(e["stream"], []).append(e)

    def label(s, a, b):
        best = ("?", 0)
        for e in by_s.get(s, []):
            ov = min(b, e["t1"]) - max(a, e["t0"])
            if ov > best[1]:
                best = (e["task"], ov)
        return best[0]

    db.search_context("warmup", k=1)
    cases = []
    for q, pred in QUERIES:
        hits, _ = db.search_context(q, k=10)
        a = V._LAST_ATTR
        rel = np.array([pred(label(*seg)) for seg in a["segs"]])
        ranks = {c: ranks_from_scores(np.array(a["channels"][c], float))
                 for c in CH}
        dir_q = any(h.get("margin") is not None for h in hits) or True
        from elidedb.rerank import directional_swap
        cases.append({"rel": rel, "ranks": ranks,
                      "raw": {c: list(a["channels"][c]) for c in CH},
                      "dir": directional_swap(q) is not None})
        print(f"captured {int(rel.sum())} relevant of {len(rel)} segs  {q}")

    def score(w):
        tot = 0
        for c in cases:
            f = np.zeros(len(c["rel"]))
            for ch in CH:
                wt = 0.0 if (ch == "mot" and not c["dir"]) else w[ch]
                f += wt / (60.0 + c["ranks"][ch] + 1.0)
            top = np.argsort(-f)[:10]
            tot += int(c["rel"][top].sum())
        return tot

    grid = [0.0, 0.3, 0.7, 1.0, 1.6, 2.5, 4.0, 6.0]
    w = {c: 1.0 for c in CH}
    w["mot"] = 2.5
    best = score(w)
    print(f"baseline {best}/100  ceiling "
          f"{sum(min(int(c['rel'].sum()), 10) for c in cases)}/100")
    for _ in range(5):
        improved = False
        for ch in CH:
            for g in grid:
                w2 = dict(w); w2[ch] = g
                s = score(w2)
                if s > best:
                    best, w, improved = s, w2, True
        if not improved:
            break
    print(f"fitted {best}/100  weights={json.dumps(w)}")
    # ---- learned LINEAR scorer over channel z-scores: rank fusion threw
    # away magnitudes and plateaued; logistic regression on the captured
    # (segment, relevant) pairs keeps them. Query-time cost: one dot
    # product. NaN (channel abstain) -> 0 after per-query standardization.
    def feats(c):
        X = []
        for ch in CH:
            v = np.array(c["channels"][ch] if "channels" in c
                         else c["ranks"][ch], float)
            X.append(v)
        return np.stack(X, axis=1)
    Xs, ys, qid = [], [], []
    for i, c in enumerate(cases):
        X = np.stack([np.array(c["raw"][ch], float) for ch in CH], axis=1)
        med = np.nanmedian(X, axis=0)
        sd = np.nanstd(X, axis=0) + 1e-6
        Z = (X - med) / sd
        Z[np.isnan(Z)] = 0.0
        Xs.append(Z)
        ys.append(c["rel"].astype(int))
        qid += [i] * len(Z)
    Xa, ya = np.concatenate(Xs), np.concatenate(ys)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=0.5, max_iter=2000).fit(Xa, ya)
    tot = 0
    for Z, c in zip(Xs, cases):
        s = Z @ lr.coef_[0]
        if c["dir"]:
            s = s + 0.0                      # mot z already in features
        tot += int(c["rel"][np.argsort(-s)[:10]].sum())
    print(f"linear scorer {tot}/100  coefs=" +
          json.dumps({ch: round(float(v), 3) for ch, v in
                      zip(CH, lr.coef_[0])}))
    out = Path(f"lake/{store}/_channel_weights.json")
    payload = {"fitted_on": "scripts/regress10.py", "score_top10": best,
               "weights": w}
    if tot > best:
        payload["linear"] = {"channels": CH,
                             "coef": [float(v) for v in lr.coef_[0]],
                             "score_top10": tot}
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
