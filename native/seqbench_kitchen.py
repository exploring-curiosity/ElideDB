"""Kitchen QbE with V-JEPA window sequences - PROTOCOL-EXACT mirror
of scripts/bench_qbe.py (same truthset, same RandomState, same seed
groups, same k) so numbers compare digit-for-digit with the committed
channel-fusion results (q04 0.94 / q05 0.90). Only the scorer differs:
temporal alignment over pretrained window sequences, domain-blind,
identical to the sim run.

    python native/seqbench_kitchen.py [--variant dtwd] [--all]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
from seq import cache_dir                                      # noqa: E402
from chain_channels import dtw_sim, norm                       # noqa: E402


def rep(V, kind):
    if V is None or len(V) < 2:
        return None
    if kind == "pool":
        return norm(V.mean(0))
    if kind == "dtw":
        return V
    if kind == "dtwd":
        if len(V) < 3:
            return norm(V)          # too short for deltas: raw seq
        return norm(V[1:] - V[:-1])
    raise SystemExit(kind)


def sim(a, b, kind):
    if a is None or b is None:
        return -1.0
    if kind == "pool":
        return float(a @ b)
    return dtw_sim(a, b, band=0.5)


def main():
    argv = sys.argv
    kind = argv[argv.index("--variant") + 1] if "--variant" in argv \
        else "dtwd"
    want = list(range(11)) if "--all" in argv else [3, 4, 5]
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    ep = db.table("episodes").scan()
    keys = [(str(s), int(a)) for s, a in
            zip(ep.column("stream").to_pylist(),
                ep.column("ts").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]

    seqs = {}
    for p in sorted(cache_dir("fresh_bench").glob("*.npz")):
        z = np.load(p)
        seqs[int(p.stem)] = z["V"].astype(np.float32)
    print(f"{len(seqs)} episodes embedded, variant {kind}", flush=True)
    reps = {e: rep(V, kind) for e, V in seqs.items()}

    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet") \
        .to_pydict()
    G, sup = {}, {}
    have = set(eidx)
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        G[(int(q), ei)] = int(v)
        if ei in have:
            sup[int(q)] = sup.get(int(q), 0) + int(v)

    ys_all, ps_all = [], []
    for qi in want:
        s = sup.get(qi, 0)
        if not s:
            continue
        k_max = int(np.ceil(s * 1.5))
        truths = [i for i, e in enumerate(eidx)
                  if G.get((qi, e)) == 1]
        rs = np.random.RandomState(0)
        ns = min(5, len(truths))
        if ns < 2:
            continue
        groups = [rs.choice(truths, ns, replace=False)
                  for _ in range(5)]
        runs = []
        for grp in groups:
            sd = sorted(set(int(x) for x in grp))
            seed_reps = [reps.get(eidx[i]) for i in sd]
            sc = np.full(len(eidx), -1e9, np.float32)
            for i, e in enumerate(eidx):
                r = reps.get(e)
                if r is None or i in sd:
                    continue
                sc[i] = max(sim(sr, r, kind) for sr in seed_reps
                            if sr is not None)
            got = list(np.argsort(-sc)[:k_max])
            y = [G.get((qi, eidx[i]), None) for i in got]
            tr = int(sum(1 for v in y if v == 1))
            runs.append((tr / s, tr / max(len(got), 1)))
        my = float(np.mean([r[0] for r in runs]))
        mp = float(np.mean([r[1] for r in runs]))
        ys_all.append(my)
        ps_all.append(mp)
        print(f"  q{qi:02d} support {s:<3} yield {my:.2f} "
              f"(runs {' '.join(f'{r[0]:.2f}' for r in runs)})  "
              f"prec {mp:.2f}", flush=True)
    if ys_all:
        print(f"  MEAN yield {np.mean(ys_all):.3f}  "
              f"prec {np.mean(ps_all):.3f}")


if __name__ == "__main__":
    main()
