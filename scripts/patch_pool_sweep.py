"""MaxSim ranks episodes badly. Which pooling actually ranks?

Within one frame the patch grid separates cleanly - an eggplant frame
scores 0.173 for "an eggplant" and 0.063 for "a banana". Across
episodes the raw max ranks at 0.08 yield, which says the failure is
not the representation but the STATISTIC: a max over 1,024 patches is
an extreme-value draw, and an episode whose patches merely spread
wider wins it regardless of content. The fix has to make scores
comparable BETWEEN episodes, so every candidate here is a
within-episode normalization or a less extreme order statistic.

  python scripts/patch_pool_sweep.py [--k 100]
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

    from elidedb.patches import _load, _tables
    from elidedb.sig2 import _text_vec, atoms_of
    C, idx, mu, R, dim = _load(db)
    nb = dim // 8
    ar = np.arange(nb)

    # every episode's patch scores for one atom, computed once
    def atom_scores(text):
        v = (np.asarray(_text_vec(text), np.float32) - mu) @ R
        v /= np.linalg.norm(v) + 1e-8
        T = _tables(v, dim)
        out = []
        for s, a, _ in keys:
            rows = idx.get((str(s), int(a)))
            if rows is None:
                out.append(None)
                continue
            out.append(T[ar, C[rows].reshape(-1, nb)].sum(1) / np.sqrt(dim))
        return out

    POOLS = {
        "max": lambda x: x.max(),
        "top8mean": lambda x: np.sort(x)[-8:].mean(),
        "top32mean": lambda x: np.sort(x)[-32:].mean(),
        "mean": lambda x: x.mean(),
        # within-episode standardization: how far the best patch sits
        # ABOVE this episode's own patch distribution, which is the
        # only version of the score that is comparable across episodes
        "max-mean": lambda x: x.max() - x.mean(),
        "z(max)": lambda x: (x.max() - x.mean()) / (x.std() + 1e-6),
        "z(top8)": lambda x: (np.sort(x)[-8:].mean() - x.mean())
        / (x.std() + 1e-6),
        "z(top32)": lambda x: (np.sort(x)[-32:].mean() - x.mean())
        / (x.std() + 1e-6),
    }
    names = list(POOLS)
    print(f"{'q':>4} {'sup':>4} " + " ".join(f"{n:>9}" for n in names))
    tot = {n: [] for n in names}
    for qi in sorted(sup):
        atoms = atoms_of(QUERIES[qi].lower())[:2] or [QUERIES[qi]]
        per_atom = [atom_scores(a) for a in atoms]
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        den = min(K, sup[qi])
        cells = []
        for n in names:
            f = POOLS[n]
            sc = np.array([
                min(f(pa[i]) for pa in per_atom) if per_atom[0][i] is not None
                else -np.inf for i in range(len(keys))])
            y = lab[np.argsort(-sc)[:K]].sum() / den
            tot[n].append(y)
            cells.append(f"{y:>9.2f}")
        print(f"q{qi:02d} {sup[qi]:>4} " + " ".join(cells), flush=True)
    print(f"{'mean':>4} {'':>4} " +
          " ".join(f"{np.mean(tot[n]):>9.2f}" for n in names))


if __name__ == "__main__":
    main()
