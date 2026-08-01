"""The patch channel ALONE, every query, full corpus, one metric.

yield = true / min(k, support), which is the only number being
optimized. Run against all 1,121 episodes — never a sampled pool. A
crop channel looked like a fix on 80 episodes and was worth nothing
against the corpus; that mistake is not repeated here.

  python scripts/patch_yield.py [--k 100]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402
from elidedb.scenario import _episodes                       # noqa: E402


def main():
    argv = sys.argv
    K = int(argv[argv.index("--k") + 1]) if "--k" in argv else 100
    db = Store.open("lake/bench")
    keys = _episodes(db)
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth, sup = {}, {}
    for q, s, t0, v in zip(t["query_id"], t["stream"], t["t0"], t["true"]):
        truth[(int(q), s, int(t0))] = int(v)
        sup[int(q)] = sup.get(int(q), 0) + int(v)

    from elidedb.patches import patch_lookup
    from elidedb.sig2 import atoms_of

    print(f"{len(keys)} episodes, k={K}\n")
    print(f"{'q':>4} {'sup':>4} {'atoms':>36} {'whole':>7} {'atoms':>7} "
          f"{'ms':>6}")
    tot_w, tot_a = [], []
    for qi in sorted(sup):
        text = QUERIES[qi]
        atoms = atoms_of(text.lower())[:2] or [text]
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        den = min(K, sup[qi])
        out = []
        t0 = time.perf_counter()
        for mode in ([text], atoms):
            look = patch_lookup(db, mode)
            sc = np.array([look(*k) for k in keys])
            sc = np.where(np.isfinite(sc), sc, -np.inf)
            out.append(lab[np.argsort(-sc)[:K]].sum() / den)
        ms = (time.perf_counter() - t0) * 1000 / 2
        tot_w.append(out[0]); tot_a.append(out[1])
        print(f"q{qi:02d} {sup[qi]:>4} {str(atoms):>36} "
              f"{out[0]:>7.2f} {out[1]:>7.2f} {ms:>6.0f}")
    print(f"\n{'mean':>4} {'':>4} {'':>36} "
          f"{np.mean(tot_w):>7.2f} {np.mean(tot_a):>7.2f}")


if __name__ == "__main__":
    main()
