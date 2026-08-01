"""EVERY MODEL ON CLIP->CLIP RETRIEVAL, its own way, plus the AUC each
would need for the system to reach 90% yield.

Clip retrieval first: if a model cannot find another instance of what a
clip shows, it does not hold the pattern, and asking it to match TEXT to
that pattern is asking for something harder. Text is downstream.

NATIVE OPERATION PER MODEL
  act    174-way SSv2 classifier -> cross-entropy between posteriors.
         Cosine on a simplex measured AUC 0.483 (below chance) where
         cross-entropy measures 0.812 on identical numbers.
  mot    appearance delta        -> cosine. The one channel for which
         cosine was always right; it is a direction in a vector space.
  pe     image-text contrastive  -> cosine after removing the corpus
  sig2                              common component. Not their designed
  iv2                               task (they are text-image / ITM
  xclip                             models) but it is the only operation
  vjepa                             the STORED artefact supports.

WHAT EACH MODEL NEEDS FOR 90%
A channel's AUC and the recall it delivers at a fixed depth are linked.
Under the binormal model - the standard approximation for a two-class
score distribution - AUC = Phi(d'/sqrt2), and recall at a given false
positive rate is Phi(Phi^-1(FPR) + d'). At the operating point the
product fixes (k = 1.5 x support, so FPR = (k - TP) / (N - support)) that
inverts to the AUC a single channel would need to deliver yield 0.90
alone. It is an estimate, not a guarantee: fusion of complementary
channels can beat any single one, and correlated channels do not add.
Read it as "how far is this model from carrying the query by itself".

    python scripts/eval_qbe_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                     # noqa: E402
from elidedb.qbe import spaces, _score                        # noqa: E402
from diag_channels import auc                                 # noqa: E402

SEEDS, REPS = 10, 6


def _ndtri(p):
    """Inverse normal CDF (Acklam), so scipy is not a dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p < 0.02425:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _ndtr(x):
    return 0.5 * (1.0 + __import__("math").erf(x / np.sqrt(2.0)))


def required_auc(sup, n, target=0.90, mult=1.5):
    """AUC a single channel needs to reach `target` yield at k=mult*sup."""
    k = mult * sup
    tp = target * sup
    fp = max(k - tp, 1.0)
    fpr = fp / max(n - sup, 1.0)
    dprime = _ndtri(target) - _ndtri(fpr)
    return _ndtr(dprime / np.sqrt(2.0))


def main():
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    keys, M = spaces(db)
    ep = db.table("episodes").scan()
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G = {(int(q), int(e)): int(v) for q, e, v in
         zip(t["query_id"], t["episode_index"], t["true"])}

    out = {}
    for qi in (3, 4, 5):
        tr = [i for i, e in enumerate(eidx) if G.get((qi, e)) == 1]
        judged = {i: G[(qi, eidx[i])] for i in range(len(eidx))
                  if (qi, eidx[i]) in G}
        sup, n = len(tr), len(eidx)
        need = required_auc(sup, n)
        print(f"\nq{qi:02d}  support {sup} of {n} episodes"
              f"   -> a single channel needs AUC {need:.3f} for yield 0.90")
        print(f"  {'model':<8}{'op':<11}{'AUC':>7}{'gap':>8}{'prec@sup':>10}")
        rows = []
        for c, (A, ok, op) in M.items():
            live = [i for i in tr if ok[i]]
            if len(live) < SEEDS + 2:
                continue
            rs = np.random.RandomState(0)
            aa, pp = [], []
            for _ in range(REPS):
                sd = rs.choice(live, SEEDS, replace=False)
                sc = _score(A, op, sd)
                sc[~ok] = -np.inf
                ji = [i for i in judged if ok[i] and i not in set(sd)]
                if len(ji) > 20:
                    aa.append(auc(sc[ji], np.array([judged[i] for i in ji],
                                                    np.int8)))
                o = np.argsort(-sc)[:sup]
                pp.append(float(np.mean([judged.get(int(i), 0) for i in o])))
            if not aa:
                continue
            a, p = float(np.mean(aa)), float(np.mean(pp))
            rows.append((c, op, a, p))
        for c, op, a, p in sorted(rows, key=lambda r: -r[2]):
            gap = need - a
            flag = "  MEETS" if gap <= 0 else ""
            print(f"  {c:<8}{op:<11}{a:>7.3f}{gap:>+8.3f}{p:>10.3f}{flag}")
        out[f"q{qi:02d}"] = {"support": sup, "required_auc": need,
                             "models": {c: {"op": op, "auc": a,
                                            "prec_at_sup": p,
                                            "gap": need - a}
                                        for c, op, a, p in rows}}
    (ROOT / "bench" / "eval_qbe_models.json").write_text(
        json.dumps(out, indent=1))
    print("\nwrote bench/eval_qbe_models.json")


if __name__ == "__main__":
    main()
