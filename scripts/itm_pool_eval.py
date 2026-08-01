"""Choose the window pooling by measurement, from one stored artifact.

Max over windows was measured worse than a single window (0.40 ->
0.36): the max hands every negative three draws at a fluke. Mean
punishes true episodes whose object appears in only one window. The
candidates in between (median, top-2 mean, and mean-plus-half-max)
trade those two failure modes; this prints the two product metrics for
each so the choice is a row in a table, not an argument.

  python scripts/itm_pool_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

POOLS = {
    "win0": lambda S3: S3[:, :, 0],
    "max": lambda S3: np.nanmax(S3, 2),
    "mean": lambda S3: np.nanmean(S3, 2),
    "median": lambda S3: np.nanmedian(S3, 2),
    "top2mean": lambda S3: np.nanmean(
        np.sort(np.nan_to_num(S3, nan=-1e9), 2)[:, :, 1:], 2),
    "mean+max/2": lambda S3: np.nanmean(S3, 2) + 0.5 * np.nanmax(S3, 2),
}


def main():
    d = np.load(ROOT / "artifacts/itm_scores3.npz", allow_pickle=True)
    S3, qids = d["S3"], [int(q) for q in d["qids"]]
    streams = [str(s) for s in d["streams"]]
    ts = [int(v) for v in d["ts"]]
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    names = list(POOLS)
    print(f"{'q':>4} {'sup':>4} " + " ".join(f"{n:>10}" for n in names))
    tot = {n: [] for n in names}
    for j, qi in enumerate(qids):
        if sup.get(qi, 0) == 0:
            continue
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a in zip(streams, ts)])
        K = int(np.ceil(sup[qi] * 1.5))
        cells = []
        for n in names:
            x = POOLS[n](S3)[:, j].astype(float)
            x = np.where(np.isfinite(x), x, -np.inf)
            y = lab[np.argsort(-x)[:K]].sum() / sup[qi]
            tot[n].append(y)
            cells.append(f"{y:>10.2f}")
        print(f"q{qi:02d} {sup[qi]:>4} " + " ".join(cells))
    print("-" * (10 + 11 * len(names)))
    print(f"{'mean':>4} {'':>4} " + " ".join(
        f"{np.mean(tot[n]):>10.2f}" for n in names))


if __name__ == "__main__":
    main()
