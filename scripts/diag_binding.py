"""Binding diagnostics: WHERE do the binding queries die?

Per-channel AUC vs the frozen truthset, top-10 label hits, and
per-atom SigLIP2 max-frame scores on TRUE vs RETURNED episodes.
Instrument only — never a shipping path.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python \
     scripts/diag_binding.py [qid ...]     (default: 8 9 10)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402
from fit_set_weights import capture                           # noqa: E402


def auc(sc, lab):
    """Mann-Whitney AUC from 0-indexed ranks:
    (mean rank of positives - (P-1)/2) / N_negative."""
    fin = np.isfinite(sc)
    s, la = sc[fin], lab[fin]
    if la.sum() in (0, len(la)):
        return float("nan")
    r = np.argsort(np.argsort(s))
    return float((r[la == 1].mean() - (la.sum() - 1) / 2)
                 / (len(la) - la.sum()))


def main():
    db = Store.open("lake/bench")
    from elidedb.scenario import _episodes
    from elidedb.sig2 import _frame_scores, _index, atoms_of
    keys = _episodes(db)
    t = pq.read_table("eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(t0)): int(v) for q, s, t0, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    for qi in [int(a) for a in sys.argv[1:]] or [8, 9, 10]:
        text = QUERIES[qi]
        print(f"\n=== q{qi:02d} {text}")
        ch, _ = capture(db, keys, text)
        lab = np.array([truth.get((qi, s, a), -1) for s, a, b in keys])
        lab = np.where(lab == 1, 1, 0) * (lab >= 0)
        print(f"support {int(lab.sum())}")
        for c in sorted(ch):
            v = ch[c]
            if np.isfinite(v).any():
                top = np.argsort(-np.nan_to_num(v, nan=-9e9))[:10]
                print(f"  {c:5s} AUC {auc(v, lab):+.3f} "
                      f"top10 {int(lab[top].sum())}/10")
        atoms = atoms_of(text)
        if len(atoms) < 2:
            print("  (single atom — conj abstains)")
            continue
        idx = _index(db)
        per = {a: _frame_scores(db, a) for a in atoms}
        v = np.nan_to_num(ch["conj"], nan=-9e9)
        groups = (("TRUE", [k for k, la in zip(keys, lab)
                            if la == 1][:5]),
                  ("RET ", [keys[i] for i in np.argsort(-v)[:5]]))
        for tag, group in groups:
            for s, a, b in group:
                lst = idx.get(str(s))
                if not lst:
                    continue
                starts = [x[0] for x in lst]
                j = int(np.searchsorted(starts, a,
                                        side="right")) - 1
                if j < 0 or lst[j][0] != a:
                    continue
                rows = lst[j][1]
                parts = " ".join(
                    f"{at.split()[-1]}={per[at][rows].max():.3f}"
                    for at in atoms)
                print(f"  {tag} {str(s)[-18:]} {a}: {parts}")


if __name__ == "__main__":
    main()
