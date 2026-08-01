"""PER-MODEL QbE EVALUATION: which teacher carries example retrieval, and
for the ones that do not, WHY.

Query-by-example asks one thing of a vector space: are two clips of the
same kind closer to each other than to everything else? A channel can
fail that in three distinguishable ways, and the fix differs for each:

  SEPARATION    within-support cosine vs between cosine. If same-task
                clips are no closer than random pairs, the space does not
                encode the task at all. Reported as a margin and as
                Cohen's d, which is scale-free across channels.
  COLLAPSE      effective rank (participation ratio of the eigenvalue
                spectrum). A space using 3 of 512 dimensions cannot
                separate 11 query types no matter how it is scored -
                this is the failure mode a distilled student shows when
                it has learned the mean.
  HUBNESS       fraction of the corpus captured by the top 1% of
                episodes as nearest neighbours. High hubness means a few
                clips are everyone's neighbour, so retrieval returns the
                same set whatever the seed.

Plus the outcome each channel actually delivers alone: AUC and precision
at support, seeded from graded-true clips.

    python scripts/diag_qbe_channels.py
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
from bench_qbe import CHAN, pooled                            # noqa: E402
from diag_channels import auc                                 # noqa: E402


def eff_rank(A):
    """Participation ratio of the covariance spectrum: how many
    dimensions the space actually uses. Collapse detector."""
    X = A - A.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def hubness(A, ok, sample=400, seed=0):
    """Fraction of nearest-neighbour slots taken by the top 1% of
    episodes. 1.0 would mean one clip is everyone's neighbour."""
    rs = np.random.RandomState(seed)
    ix = np.where(ok)[0]
    if len(ix) < 50:
        return float("nan")
    q = rs.choice(ix, min(sample, len(ix)), replace=False)
    S = A[q] @ A[ix].T
    for r, i in enumerate(q):
        S[r, np.where(ix == i)[0]] = -np.inf
    nn = ix[np.argmax(S, 1)]
    cnt = np.bincount(nn, minlength=len(A))
    top = max(1, int(0.01 * len(ix)))
    return float(np.sort(cnt)[-top:].sum() / len(q))


def main():
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    ep = db.table("episodes").scan()
    keys = [(str(s), int(a)) for s, a in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    pos = {k: i for i, k in enumerate(keys)}

    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G = {}
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        G[(int(q), int(ei))] = int(v)

    M = {}
    for c, tab in CHAN.items():
        if tab not in db.tables():
            continue
        p = pooled(db, tab)
        dim = len(next(iter(p.values())))
        A = np.zeros((len(keys), dim), np.float32)
        ok = np.zeros(len(keys), bool)
        for k, v in p.items():
            if k in pos:
                A[pos[k]] = v
                ok[pos[k]] = True
        M[c] = (A, ok)

    # ---- space-level health, query-independent
    print("SPACE HEALTH (query-independent)\n")
    print(f"  {'channel':<8}{'dim':>6}{'eff_rank':>10}{'rank/dim':>10}"
          f"{'hubness':>9}{'cov':>7}")
    health = {}
    for c, (A, ok) in M.items():
        d = A.shape[1]
        er = eff_rank(A[ok])
        hb = hubness(A, ok)
        health[c] = {"dim": d, "eff_rank": er, "ratio": er / d,
                     "hubness": hb, "coverage": float(ok.mean())}
        print(f"  {c:<8}{d:>6}{er:>10.1f}{er/d:>10.3f}{hb:>9.3f}"
              f"{ok.mean():>7.2f}")

    # ---- task separation + delivered accuracy, per query
    print("\n\nPER-QUERY, PER-MODEL (seeded from graded-true clips)\n")
    rows = []
    for qi in (3, 4, 5):
        truths = [i for i, e in enumerate(eidx) if G.get((qi, e)) == 1]
        judged = {i: G[(qi, eidx[i])] for i in range(len(eidx))
                  if (qi, eidx[i]) in G}
        if len(truths) < 10:
            continue
        sup = len(truths)
        print(f"q{qi:02d}  support {sup}")
        print(f"  {'channel':<8}{'AUC':>7}{'prec@sup':>10}{'within':>8}"
              f"{'between':>9}{'margin':>8}{'cohen_d':>9}")
        for c, (A, ok) in M.items():
            tr = [i for i in truths if ok[i]]
            if len(tr) < 10:
                continue
            T = A[tr]
            W = T @ T.T
            iu = np.triu_indices(len(tr), 1)
            within = float(W[iu].mean())
            other = np.where(ok)[0]
            other = np.setdiff1d(other, tr)
            B = T @ A[other].T
            between = float(B.mean())
            sd = float(np.sqrt((W[iu].var() + B.var()) / 2)) or 1e-9
            d = (within - between) / sd
            # delivered: average over 5 seeds
            seeds = [tr[i] for i in np.linspace(0, len(tr) - 1, 5)
                     .round().astype(int)]
            aucs, precs = [], []
            for sd_i in seeds:
                sc = A @ A[sd_i]
                sc[~ok] = -np.inf
                sc[sd_i] = -np.inf
                ji = [i for i in judged if ok[i] and i != sd_i]
                if len(ji) > 20:
                    aucs.append(auc(sc[ji], np.array([judged[i] for i in ji],
                                                     np.int8)))
                o = np.argsort(-sc)[:sup]
                precs.append(float(np.mean([judged.get(i, 0) for i in o])))
            rows.append({"q": qi, "channel": c,
                         "auc": float(np.mean(aucs)) if aucs else None,
                         "prec_at_sup": float(np.mean(precs)),
                         "within": within, "between": between,
                         "margin": within - between, "cohen_d": d})
            a = f"{np.mean(aucs):.3f}" if aucs else "  -  "
            print(f"  {c:<8}{a:>7}{np.mean(precs):>10.3f}{within:>8.3f}"
                  f"{between:>9.3f}{within-between:>8.3f}{d:>9.2f}")
        print()
    (ROOT / "bench" / "diag_qbe_channels.json").write_text(
        json.dumps({"health": health, "per_query": rows}, indent=1))
    print("wrote bench/diag_qbe_channels.json")


if __name__ == "__main__":
    main()
