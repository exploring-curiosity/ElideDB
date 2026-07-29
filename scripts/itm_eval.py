"""Evaluate the ITM matrix on the two product metrics. Corpus, not pool.

  yield = true / support        at k = ceil(1.5 x support), k a ceiling
  prec  = true / returned

Three selection rules are reported so the choice is made on evidence:
  rank      return exactly k top-ranked (ceiling fully used)
  thresh    return scores above a fixed P(match) cut (0.5)
  rank+cut  top-k, then drop the tail below the cut - the product
            shape: k caps, confidence decides

Also reports full-corpus AUC per query and the no-match gate behavior
on q06 (support 0: anything returned above the cut is a false alarm).

  python scripts/itm_eval.py [--cut 0.5]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402


def main():
    argv = sys.argv
    cut = float(argv[argv.index("--cut") + 1]) if "--cut" in argv else 0.5
    d = np.load(ROOT / "ml/itm_scores.npz", allow_pickle=True)
    S, qids = d["S"], [int(q) for q in d["qids"]]
    streams, ts = d["streams"], d["ts"]
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    print(f"{S.shape[0]} episodes x {len(qids)} queries, "
          f"threshold {cut}\n")
    print(f"{'q':>4} {'sup':>4} {'AUC':>6} | {'rank: yld':>10} {'prec':>5} "
          f"| {'thr: yld':>9} {'prec':>5} {'ret':>4} "
          f"| {'r+c: yld':>9} {'prec':>5} {'ret':>4}")
    agg = {k: [] for k in ("ry", "rp", "ty", "tp", "cy", "cp")}
    for j, qi in enumerate(qids):
        s = S[:, j]
        ok = np.isfinite(s)
        lab = np.array([1 if truth.get((qi, str(st), int(a))) == 1 else 0
                        for st, a in zip(streams, ts)])
        if sup.get(qi, 0) == 0:
            ret = int(((s > cut) & ok).sum())
            print(f"q{qi:02d} {0:>4} {'—':>6} | {'gate':>10} "
                  f"{'—':>5} | returned above cut: {ret} "
                  f"({'PASS' if ret == 0 else 'FAIL'})")
            continue
        y, x = lab[ok].astype(bool), s[ok]
        r = np.argsort(np.argsort(x)) + 1
        auc = ((r[y].sum() - y.sum() * (y.sum() + 1) / 2)
               / (y.sum() * (~y).sum()))
        K = int(np.ceil(sup[qi] * 1.5))
        order = np.argsort(-x)

        def metrics(sel):
            tru = int(y[sel].sum())
            return (tru / sup[qi], tru / max(len(sel), 1), len(sel))

        ry, rp, _ = metrics(order[:K])
        tsel = np.where(x > cut)[0]
        ty, tp, tn = metrics(tsel)
        csel = order[:K][x[order[:K]] > cut]
        cy, cp, cn = metrics(csel)
        for k, v in zip(("ry", "rp", "ty", "tp", "cy", "cp"),
                        (ry, rp, ty, tp, cy, cp)):
            agg[k].append(v)
        print(f"q{qi:02d} {sup[qi]:>4} {auc:>6.3f} | {ry:>10.2f} {rp:>5.2f} "
              f"| {ty:>9.2f} {tp:>5.2f} {tn:>4d} "
              f"| {cy:>9.2f} {cp:>5.2f} {cn:>4d}")
    print("-" * 88)
    print(f"{'mean':>4} {'':>4} {'':>6} | "
          f"{np.mean(agg['ry']):>10.2f} {np.mean(agg['rp']):>5.2f} "
          f"| {np.mean(agg['ty']):>9.2f} {np.mean(agg['tp']):>5.2f} {'':>4} "
          f"| {np.mean(agg['cy']):>9.2f} {np.mean(agg['cp']):>5.2f}")


if __name__ == "__main__":
    main()
