"""Per-channel, per-query yield@10 — each channel ALONE, full corpus.

The ledger measures the fused system. This measures the parts, in the
only unit that counts: true / min(k, support) for each query, with the
channel used on its own as the ranker. It exists because a channel can
look strong on a small probe pool and vanish at corpus scale — the
track channel ranked true episodes 1st and 8th out of 80, which says
nothing about its behavior against 1,122.

Uses fit_set_weights.capture, so the arrays are exactly the ones the
fitter and the live path see — no reimplementation to drift.

  python scripts/channel_yield.py [--k 10]
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
from fit_set_weights import CH, capture                      # noqa: E402
from elidedb.scenario import _episodes                       # noqa: E402


def main():
    argv = sys.argv
    K = int(argv[argv.index("--k") + 1]) if "--k" in argv else 10
    db = Store.open("lake/bench")
    keys = _episodes(db)
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth, support = {}, {}
    for q, s, t0, v in zip(t["query_id"], t["stream"], t["t0"], t["true"]):
        truth[(int(q), s, int(t0))] = int(v)
        support[int(q)] = support.get(int(q), 0) + int(v)
    qids = sorted(support)

    print(f"corpus {len(keys)} episodes, k={K}, "
          f"yield = true / min(k, support)\n")
    hdr = f"{'query':>5} {'sup':>4} " + " ".join(f"{c:>5}" for c in CH)
    print(hdr)
    print("-" * len(hdr))
    tot = {c: [] for c in CH}
    for qi in qids:
        ch, _ = capture(db, keys, QUERIES[qi])
        # strict, exactly as the ledger grades: ungraded counts false
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        den = min(K, support[qi])
        cells = []
        for c in CH:
            x = np.asarray(ch.get(c, np.full(len(keys), np.nan)), float)
            if not np.isfinite(x).any():
                cells.append("  -  ")
                continue
            x = np.where(np.isfinite(x), x, -np.inf)
            top = np.argsort(-x)[:K]
            y = float(lab[top].sum()) / den
            tot[c].append(y)
            cells.append(f"{y:>5.2f}")
        print(f"q{qi:02d}   {support[qi]:>4} " + " ".join(cells),
              flush=True)
    print("-" * len(hdr))
    print(f"{'mean':>5} {'':>4} " + " ".join(
        f"{np.mean(tot[c]):>5.2f}" if tot[c] else "  -  " for c in CH))


if __name__ == "__main__":
    main()
