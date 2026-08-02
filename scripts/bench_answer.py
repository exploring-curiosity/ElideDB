"""The ANSWER JOIN on the QbE protocol - directly comparable to fusion.

Same truthset, same 5 evenly-spaced support seeds per query, same
k = ceil(1.5 x support), same confidence cut, same metrics. The only
change is the engine: answer_like's logical join over the element
tables instead of RRF over channel cosines. One number decides whether
the element investment expresses as retrieval.

    python scripts/bench_answer.py [--all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb.answer import answer_like                         # noqa: E402
from elidedb.setpath import confidence_cut                     # noqa: E402


def main():
    argv = sys.argv
    db = Store.open(str(ROOT / (argv[argv.index("--store") + 1]
                                if "--store" in argv
                                else "lake/fresh_bench")))
    n_seed = int(argv[argv.index("--seeds") + 1] if "--seeds" in argv else 5)
    want = (list(range(11)) if "--all" in argv else [3, 4, 5])

    ep = db.table("episodes").scan()
    keys = [(str(s), int(a), int(b)) for s, a, b in
            zip(ep.column("stream").to_pylist(),
                ep.column("ts").to_pylist(),
                ep.column("t1").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]

    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G, sup = {}, {}
    have = set(eidx)
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        G[(int(q), ei)] = int(v)
        if ei in have:
            sup[int(q)] = sup.get(int(q), 0) + int(v)

    out = []
    for qi in tqdm(want, desc="answer", unit="q", dynamic_ncols=True):
        s = sup.get(qi, 0)
        if not s:
            continue
        k_max = int(np.ceil(s * 1.5))
        truths = [i for i, e in enumerate(eidx) if G.get((qi, e)) == 1]
        seeds = [truths[i] for i in
                 np.linspace(0, len(truths) - 1, min(n_seed, len(truths)))
                 .round().astype(int)]
        runs = []
        for sd in seeds:
            st, a, b = keys[sd]
            eps, score, shared = answer_like(db, st, a, b)
            assert [(k[0], k[1], k[2]) for k in eps] == keys
            score = score.copy()
            score[sd] = -np.inf                  # never retrieve the seed
            # the event-kind PARTITION: shared-kind episodes rank as a
            # block ahead of the rest, order inside each block by the
            # conjunction score
            part = np.where(shared, 1.0, 0.0)
            part[sd] = -np.inf
            order = np.lexsort((-score, -part))
            # the cut sees the conjunction scores only - adding the
            # partition offset put a 1.0 cliff at the block boundary
            # and the knee fired on it (ret collapsed to 1)
            cut = confidence_cut(score[order], 0.0, k_max)
            chosen = order[:cut]
            y = [G.get((qi, eidx[i]), None) for i in chosen]
            tr = int(sum(1 for v in y if v == 1))
            ng = int(sum(1 for v in y if v is not None))
            runs.append({"ret": len(chosen), "true": tr, "judged": ng,
                         "yield": tr / s, "prec": tr / max(len(chosen), 1),
                         "prec_g": (tr / ng) if ng else None})
        if not runs:
            continue
        out.append({"q": qi, "support": s, "k": k_max, "seeds": len(runs),
                    "runs": runs,
                    "mean": {m: float(np.mean([r[m] for r in runs]))
                             for m in ("ret", "true", "yield", "prec")},
                    "prec_g": float(np.mean([r["prec_g"] for r in runs
                                             if r["prec_g"] is not None]))
                    if any(r["prec_g"] is not None for r in runs) else None})

    print(f"\n{'q':<5}{'sup':>5}{'k':>5}{'seeds':>7}{'ret':>7}{'true':>7}"
          f"{'yield':>8}{'prec':>7}{'prec_g':>8}   yield spread")
    for r in out:
        ys = [x["yield"] for x in r["runs"]]
        pg = f"{r['prec_g']:.2f}" if r["prec_g"] is not None else "  - "
        print(f"q{r['q']:02d}{r['support']:>5}{r['k']:>5}{r['seeds']:>7}"
              f"{r['mean']['ret']:>7.0f}{r['mean']['true']:>7.1f}"
              f"{r['mean']['yield']:>8.2f}{r['mean']['prec']:>7.2f}{pg:>8}"
              f"   {min(ys):.2f}-{max(ys):.2f}")
    (ROOT / "bench/bench_answer.json").write_text(json.dumps(out, indent=1))
    print("\nwrote bench/bench_answer.json")


if __name__ == "__main__":
    main()
