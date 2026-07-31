"""Does coding a teacher channel make it prunable? Bytes, and recall.

The claim under test: a vector table is unprunable (measured - a full
scan reads 99.2% of the store and elides 0.777%), and storing each
vector's codebook cell as a CLUSTERED int32 column changes that.

Two numbers decide it, and one alone would be dishonest:

    bytes touched   what the probe actually reads, against a full scan
    recall@k        whether the pruned top-k matches the exact top-k

A prune that halves recall to save bytes is not a win, it is a different
and worse index. Both are reported per probe width.

  python scripts/bench_codes.py --store lake/nohw --table frame_vectors
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                   # noqa: E402
from elidedb.teacher import assign, fit_codebook, probe, table, write  # noqa: E402


def main():
    argv = sys.argv
    src = Store.open(argv[argv.index("--store") + 1]
                     if "--store" in argv else "lake/nohw")
    name = argv[argv.index("--table") + 1] if "--table" in argv \
        else "frame_vectors"
    k = int(argv[argv.index("--k") + 1] if "--k" in argv else 20)

    raw = src.table(name).scan()
    V = np.asarray(raw.column("vector").to_pylist(), np.float32)
    n = len(V)
    ts = np.asarray(raw.column("ts").to_pylist(), np.int64)
    t1 = (np.asarray(raw.column("t1").to_pylist(), np.int64)
          if "t1" in raw.column_names else ts)
    stream = [str(s) for s in raw.column("stream").to_pylist()]
    print(f"{name}: {n:,} rows x {V.shape[1]}d")

    out = Path(argv[argv.index("--out") + 1] if "--out" in argv
               else "lake/_coded")
    import shutil
    if out.exists():
        shutil.rmtree(out)
    db = Store.create(out, "coded")

    a = time.time()
    C = fit_codebook(V)
    tbl, C = table(ts, t1, stream, V, C)
    write(db, name, tbl, C)
    t_build = time.time() - a
    st = db.table(name).state()
    print(f"codebook: {len(C)} cells, fitted+written in {t_build:.1f}s, "
          f"{st.meta['row_groups']} row groups, {st.bytes:,} B")

    # exact answer, for recall
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(0)
    qs = rng.choice(n, size=min(50, n), replace=False)

    full = db.table(name).state().bytes
    print(f"\n{'probe':>6} {'cells':>6} {'bytes':>12} {'% of full':>10} "
          f"{'recall@' + str(k):>10} {'ms':>7}")
    for width in (0.0, 0.02, 0.05, 0.12, None):
        touched, rec, ms = [], [], []
        for qi in qs:
            q = Vn[qi]
            exact = set(np.argsort(-(Vn @ q))[:k].tolist())
            a = time.time()
            cells = (list(range(len(C))) if width is None
                     else probe(q, C, min_frac=width))
            tb, stats = db.table(name).scan_values("code", cells)
            b = full if width is None else stats.bytes_touched
            ms.append((time.time() - a) * 1000)
            touched.append(b)
            if len(tb):
                sub = np.asarray(tb.column("vector").to_pylist(), np.float32)
                sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
                loc = np.argsort(-(sub @ q))[:k]
                # map back by (ts, stream) to compare with the exact set
                sts = np.asarray(tb.column("ts").to_pylist(), np.int64)[loc]
                keep = {int(i) for i in np.flatnonzero(np.isin(ts, sts))}
                rec.append(len(exact & keep) / max(len(exact), 1))
            else:
                rec.append(0.0)
        lab = "full" if width is None else f"{width:.2f}"
        nc = len(C) if width is None else np.mean(
            [len(probe(Vn[q], C, min_frac=width)) for q in qs])
        print(f"{lab:>6} {nc:>6.1f} {np.mean(touched):>12,.0f} "
              f"{100 * np.mean(touched) / full:>9.1f}% "
              f"{np.mean(rec):>10.3f} {np.mean(ms):>7.2f}")


if __name__ == "__main__":
    main()
