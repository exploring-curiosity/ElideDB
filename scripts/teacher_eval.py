"""THE TEACHER's pointwise base: fixed recipe, declared before scoring.

Weights are stated here, in code, BEFORE the fullspan artifact was
evaluated - the teacher's number must not be an in-sample claim, so
nothing in this file was chosen by looking at truthset outcomes:

    base = mean_z(cosine channels: pe, sig2, iv2, obj)
         + 1.0 * z(ITM full-span)     in-distribution: the head was
                                      trained on 4-frame whole actions
         + 0.5 * z(ITM 3-window mean) auxiliary visibility probes -
         + 0.5 * z(ITM 3-window max)  each wins queries the other
                                      loses, neither is the primary

Directional contrast channels (act/mot) were tried in the base after
the first evaluation and REVERTED - not only was the net negative
(q04 +0.13, q00 -0.13, q01 -0.18), adding them was recipe tuning
after seeing outcomes, which this file's whole premise forbids. The
declared recipe stands.

Saves artifacts/teacher_base.npz for the pairwise stage, prints the two
product metrics (yield = true/support at k = ceil(1.5 x support),
prec = true/returned) for the base alone.

  python scripts/teacher_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402


def z(x):
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.zeros_like(x)
    out = (x - x[m].mean()) / (x[m].std() + 1e-9)
    out[~m] = 0.0
    return out


def main():
    d3 = np.load(ROOT / "artifacts/itm_scores3.npz", allow_pickle=True)
    df = np.load(ROOT / "artifacts/itm_fullspan.npz", allow_pickle=True)
    S3, qids = d3["S3"], [int(q) for q in d3["qids"]]
    SF = df["S"]
    assert list(df["streams"]) == list(d3["streams"])
    keys = list(zip([str(s) for s in d3["streams"]],
                    [int(v) for v in d3["ts"]],
                    [int(v) for v in d3["t1"]]))
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    db = Store.open("lake/bench")
    from elidedb.context import embed_texts
    from elidedb.iv2 import iv2_lookup
    from elidedb.objects import object_lookup
    from elidedb.pe import pe_lookup
    from elidedb.sig2 import atoms_of, sig2_lookup

    B = np.zeros((len(keys), len(qids)), np.float32)
    print(f"{'q':>4} {'sup':>4} {'K':>4} {'yield':>6} {'prec':>6}"
          f"   true ranks (first 10)")
    ys, ps = [], []
    for j, qi in enumerate(qids):
        text = QUERIES[qi]
        cs = []
        for fn in (pe_lookup, sig2_lookup, iv2_lookup):
            try:
                look, _ = fn(db, text)
                cs.append(z(np.array([look(*k) for k in keys])))
            except Exception:
                pass
        try:
            nps = atoms_of(text.lower())[:2]
            ol = object_lookup(db, embed_texts(nps))
            cs.append(z(np.array([
                (lambda r: r[0] * (1 + r[1]) if r[0] == r[0]
                 else np.nan)(ol(*k)) for k in keys])))
        except Exception:
            pass
        itm = S3[:, j, :].astype(float)
        base = (np.mean(cs, axis=0)
                + 1.0 * z(SF[:, j].astype(float))
                + 0.5 * z(np.nanmean(itm, 1))
                + 0.5 * z(np.nanmax(itm, 1)))
        B[:, j] = base
        if sup.get(qi, 0) == 0:
            print(f"q{qi:02d} {0:>4}    —      —      —   (gate query)")
            continue
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        K = int(np.ceil(sup[qi] * 1.5))
        order = np.argsort(-base)
        rank = np.empty(len(base), int)
        rank[order] = np.arange(1, len(base) + 1)
        rr = sorted(rank[lab == 1])
        tru = int(lab[order[:K]].sum())
        ys.append(tru / sup[qi]); ps.append(tru / K)
        print(f"q{qi:02d} {sup[qi]:>4} {K:>4} {tru / sup[qi]:>6.2f} "
              f"{tru / K:>6.2f}   {rr[:10]}")
    np.savez(ROOT / "artifacts/teacher_base.npz", B=B, qids=np.array(qids),
             streams=d3["streams"], ts=d3["ts"], t1=d3["t1"])
    print(f"\nbase alone: mean yield {np.mean(ys):.2f}  "
          f"mean prec {np.mean(ps):.2f}  -> artifacts/teacher_base.npz")


if __name__ == "__main__":
    main()
