"""Fuse the ITM matrix with the cosine channels. Independent errors.

ITM alone: AUC 0.85-0.97 corpus-wide but mean yield 0.40 at
k=1.5xsup, because AUC 0.92 against 1,119 negatives still leaves a
true episode near rank 95, and a support-2 query needs rank <= 3. The
false positives of a cross-encoder and the false positives of cosine
retrieval are DIFFERENT episodes; averaging pushes shared truth up
and unshared error down. This measures that, with a weight sweep, on
the two product metrics.

Channels here are the cheap exact lookups (pe, sig2, iv2, obj cosines)
z-normalized per query, plus z(ITM) at weight w.

  python scripts/itm_fuse.py
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


def z(x):
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.zeros_like(x)
    mu, sd = x[m].mean(), x[m].std() + 1e-9
    out = (x - mu) / sd
    out[~m] = 0.0
    return out


def main():
    d = np.load(ROOT / "ml/itm_scores.npz", allow_pickle=True)
    S, qids = d["S"], [int(q) for q in d["qids"]]
    streams = [str(s) for s in d["streams"]]
    ts = [int(v) for v in d["ts"]]
    t1 = [int(v) for v in d["t1"]]
    keys = list(zip(streams, ts, t1))

    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    db = Store.open("lake/bench")
    from elidedb.iv2 import iv2_lookup
    from elidedb.pe import pe_lookup
    from elidedb.sig2 import atoms_of, sig2_lookup
    from elidedb.context import embed_texts
    from elidedb.objects import object_lookup

    print("computing cosine channels…", flush=True)
    CH = {}
    for j, qi in enumerate(qids):
        if sup.get(qi, 0) == 0:
            continue
        text = QUERIES[qi]
        cs = []
        for lookup_fn in (pe_lookup, sig2_lookup, iv2_lookup):
            try:
                look, _ = lookup_fn(db, text)
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
        CH[qi] = (np.mean(cs, axis=0), z(S[:, j].astype(float)))
        print(f"  q{qi:02d} done", flush=True)

    print(f"\n{'w(itm)':>7} {'mean yield':>11} {'mean prec':>10}  per-query yield")
    for w in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        ys, ps, cells = [], [], []
        for qi in sorted(CH):
            cos, itm = CH[qi]
            f = cos + w * itm
            lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                            for s, a, _ in keys])
            K = int(np.ceil(sup[qi] * 1.5))
            top = np.argsort(-f)[:K]
            tru = int(lab[top].sum())
            ys.append(tru / sup[qi]); ps.append(tru / K)
            cells.append(f"q{qi:02d}:{tru / sup[qi]:.2f}")
        print(f"{w:>7.1f} {np.mean(ys):>11.2f} {np.mean(ps):>10.2f}  "
              + " ".join(cells))


if __name__ == "__main__":
    main()
