"""QUERY BY EXAMPLE: can a clip retrieve its own kind, with no text?

The direction finding forced this question. Text cannot express direction
on a corpus that names nothing - "opens" vs "closes" differ only in a
verb, and bridging a verb to a nameless transition family needs either a
hand-written table (deleted) or write-time naming (forbidden). A CLIP has
no such problem: it IS an instance of what it shows.

So: take a graded-true episode for a query, use it as the query, and see
whether the rest of that query's support comes back. Same metric as the
text path - yield = true/support, prec = true/returned, k = ceil(1.5 x
support) - so the two are directly comparable.

METHOD. Every channel is a per-episode vector table; the seed's own row
is the query vector and the score is cosine over the corpus. Channels
fuse by RRF exactly as the text path fuses them, and the same
median+3*MAD cut decides where the returned set ends. No text is
embedded anywhere in this path.

HONESTY. The seed is excluded from its own results (it is given, not
retrieved), several seeds are tried per query, and the spread is reported
- one lucky clip is not an answer. `prec_g` (precision among returns a
human judged) is printed beside `prec` for the same reason it is in
bench_truth: the truthset is pool-limited.

    python scripts/bench_qbe.py                 # q03,q04,q05
    python scripts/bench_qbe.py --all --seeds 5
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
from elidedb.embeddings import _vec_table                     # noqa: E402
from elidedb.setpath import confidence_cut                    # noqa: E402

# per-episode vector tables; pe/sig2 carry 8 rows per episode and are
# pooled, the rest are already one row per episode
CHAN = {"iv2": "iv2_vectors", "xclip": "xclip_vectors",
        "vjepa": "vjepa_vectors", "pe": "pe_vectors",
        "sig2": "sig2_vectors", "act": "action_probs",
        "mot": "motion_vectors"}


def pooled(db, table):
    """{(stream, ts) -> unit vector}, mean-pooled over rows per episode."""
    tb, V = _vec_table(db, table)
    V = np.asarray(V, np.float32)
    ss = tb.column("stream").to_pylist()
    ts = [int(v) for v in tb.column("ts").to_pylist()]
    acc = {}
    for s, a, v in zip(ss, ts, V):
        acc.setdefault((str(s), int(a)), []).append(v)
    out = {}
    for k, vs in acc.items():
        m = np.mean(vs, 0)
        n = np.linalg.norm(m)
        out[k] = (m / n) if n > 0 else m
    return out


def rrf(rank_lists, k=60.0):
    """Reciprocal-rank fusion, the same combiner the text path uses."""
    tot = None
    for r in rank_lists:
        s = 1.0 / (k + r)
        tot = s if tot is None else tot + s
    return tot


def main():
    argv = sys.argv
    db = Store.open(str(ROOT / (argv[argv.index("--store") + 1]
                                if "--store" in argv else "lake/fresh_bench")))
    n_seed = int(argv[argv.index("--seeds") + 1] if "--seeds" in argv else 5)
    want = (list(range(11)) if "--all" in argv else [3, 4, 5])

    ep = db.table("episodes").scan()
    keys = [(str(s), int(a)) for s, a in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    idx_of = dict(zip(keys, eidx))
    pos = {k: i for i, k in enumerate(keys)}

    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G, sup = {}, {}
    have = set(eidx)
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        G[(int(q), ei)] = int(v)
        if ei in have:
            sup[int(q)] = sup.get(int(q), 0) + int(v)

    # one score matrix per channel, aligned with `keys`
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

    from tqdm import tqdm
    out = []
    for qi in tqdm(want, desc="qbe", unit="q", dynamic_ncols=True):
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
            ranks = []
            for c, (A, ok) in M.items():
                if not ok[sd]:
                    continue
                sc = A @ A[sd]
                sc[~ok] = -np.inf
                sc[sd] = -np.inf                     # never retrieve the seed
                r = np.empty(len(sc))
                r[np.argsort(-sc)] = np.arange(len(sc))
                ranks.append(r)
            if not ranks:
                continue
            fused = rrf(ranks)
            order = np.argsort(-fused)
            cut = confidence_cut(fused[order], 0.0, k_max)
            chosen = order[:cut]
            y = np.array([G.get((qi, eidx[i]), None) for i in chosen])
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
    (ROOT / "bench" / "bench_qbe.json").write_text(json.dumps(out, indent=1))
    print("\nwrote bench/bench_qbe.json")


if __name__ == "__main__":
    main()
