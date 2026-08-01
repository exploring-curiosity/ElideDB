"""PER-CHANNEL DIAGNOSTIC: which input is carrying the query, and which
is noise.

"The ranking is weak" is not actionable - the fusion has seven inputs
that fail independently. This scores EACH channel separately against the
truthset and reports:

  AUC      does this channel rank true above false at all (0.5 = chance)
  prec@sup precision if this channel alone chose the top `support` clips
  fused    the same numbers for the combination actually shipped

A channel below ~0.55 AUC is contributing noise to the fusion and its
weight is buying nothing. A channel well above the fused AUC is being
diluted by the others. Both are fixable; "the ranking is bad" is not.

    python scripts/diag_channels.py            # q03,q04,q05
    python scripts/diag_channels.py --all
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
from elidedb.scenario import search_set                       # noqa: E402
from _common import queries                                   # noqa: E402


def auc(scores, labels):
    """Rank AUC, ties averaged. No sklearn dependency."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, np.int8)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties so a constant channel scores exactly 0.5
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * nneg))


def main():
    store = ROOT / (sys.argv[sys.argv.index("--store") + 1]
                    if "--store" in sys.argv else "lake/fresh_bench")
    want = ([int(x) for x in sys.argv[sys.argv.index("--q") + 1].split(",")]
            if "--q" in sys.argv else
            list(range(11)) if "--all" in sys.argv else [3, 4, 5])
    db = Store.open(str(store))
    QS = queries()

    ept = db.table("episodes").scan()
    idx_of = {(s, int(a)): int(i) for s, a, i in
              zip(ept.column("stream").to_pylist(),
                  ept.column("ts").to_pylist(),
                  ept.column("episode_index").to_pylist())}
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    truth, support = {}, {}
    in_store = set(idx_of.values())
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        truth[(int(q), ei)] = int(v)
        if ei in in_store:
            support[int(q)] = support.get(int(q), 0) + int(v)

    from tqdm import tqdm
    out = []
    for qi in tqdm(want, desc="channels", unit="q", dynamic_ncols=True):
        sup = support.get(qi, 0)
        if not sup or qi >= len(QS):
            continue
        r = search_set(db, QS[qi], purity="fast",
                       k_max=int(np.ceil(sup * 1.5)), return_ranking=True)
        rank = r.get("ranking") or []
        cs = r.get("channel_scores") or {}
        y = []
        for s_, a_, _ in rank:
            ei = idx_of.get((s_, int(a_)))
            y.append(0 if ei is None else truth.get((qi, ei), 0))
        y = np.asarray(y, np.int8)
        if not len(y) or not y.sum():
            continue
        fused = np.asarray([sc for _, _, sc in rank], float)
        row = {"q": qi, "support": sup, "n": len(y),
               "chance": round(float(y.mean()), 3),
               "weights": r.get("channel_weights", {}),
               "fused": {"auc": round(auc(fused, y), 3),
                         "prec_at_sup": round(
                             float(y[np.argsort(-fused)[:sup]].mean()), 3)},
               "channels": {}}
        for c, v in cs.items():
            v = np.asarray(v, float)
            if len(v) != len(y):
                continue
            row["channels"][c] = {
                "auc": round(auc(v, y), 3),
                "prec_at_sup": round(
                    float(y[np.argsort(-v)[:sup]].mean()), 3)}
        row["failed"] = sorted(r.get("channels_failed", {}))
        out.append(row)

    for r in out:
        print("\n" + "=" * 72)
        print(f"q{r['q']:02d}  support {r['support']}  of {r['n']} scored"
              f"   chance {r['chance']:.3f}")
        print(f"  {'channel':<12}{'weight':>8}{'AUC':>8}{'prec@sup':>10}")
        print(f"  {'FUSED':<12}{'':>8}{r['fused']['auc']:>8.3f}"
              f"{r['fused']['prec_at_sup']:>10.3f}")
        for c, v in sorted(r["channels"].items(),
                           key=lambda kv: -(kv[1]["auc"] or 0)):
            w = r["weights"].get(c, 1.0)
            flag = "   <- noise" if v["auc"] < 0.55 else (
                "   <- beats fused" if v["auc"] > r["fused"]["auc"] else "")
            print(f"  {c:<12}{w:>8.2f}{v['auc']:>8.3f}"
                  f"{v['prec_at_sup']:>10.3f}{flag}")
        if r["failed"]:
            print(f"  dead: {r['failed']}")
    (ROOT / "bench" / "diag_channels.json").write_text(json.dumps(out, indent=1))
    print("\nwrote bench/diag_channels.json")


if __name__ == "__main__":
    main()
