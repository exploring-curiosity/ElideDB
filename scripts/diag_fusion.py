"""HEADROOM: what fusion could deliver, and whether an UNSUPERVISED rule
gets there.

diag_channels showed the shipped fusion scoring BELOW its own best input
on all three high-support queries, because every channel votes with the
same weight whether it measures 0.94 AUC or 0.33. The obvious fix - weight
by measured quality - is not available: measured quality needs the
truthset, and the truthset is eval-only. Fitting weights on it would make
every number afterwards meaningless.

So this compares three things:

  ORACLE    weights from the truthset. NOT SHIPPABLE. It exists only to
            say how much headroom weighting has at all - if the oracle
            cannot reach 0.8 then no weighting rule can, and the answer
            has to come from a better channel instead.
  UNSUP     the candidate shippable rule: weights from properties of the
            SCORES THEMSELVES, no labels. See _unsup_weights.
  SHIPPED   what runs today.

Read UNSUP against ORACLE, not against SHIPPED: ORACLE is the ceiling of
this whole approach.

    python scripts/diag_fusion.py
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
from diag_channels import auc                                 # noqa: E402


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / (sd + 1e-9) if sd > 0 else np.zeros_like(v)


def _unsup_weights(cs):
    """Per-query channel weights from the scores alone - no labels.

    Two signals, both properties of a score vector:

    CONCENTRATION. A channel that separates something puts a little mass
    far above its own bulk; a channel with nothing to say is flat. Excess
    kurtosis of the score distribution measures exactly that, and it is
    scale-free, so channels with different score ranges compare.

    CONSENSUS. A channel anti-correlated with the pooled opinion of the
    others is either uniquely right or inverted, and at fusion time we
    cannot tell which - but we can refuse to let it vote against everyone
    at full strength. Correlation to the leave-one-out mean of the z-scored
    others, clipped at 0, does that: q05's act channel (AUC 0.330, i.e.
    inverted) gets weight 0 without anyone labelling it.

    Neither term looks at a truthset, a query type, or a channel name.
    """
    names = list(cs)
    Z = {c: _z(cs[c]) for c in names}
    w = {}
    for c in names:
        z = Z[c]
        others = [Z[o] for o in names if o != c]
        if others:
            pool = np.mean(others, axis=0)
            agree = float(np.corrcoef(z, pool)[0, 1]) if pool.std() > 0 else 0.0
        else:
            agree = 1.0
        agree = max(agree, 0.0)
        kurt = float(((z ** 4).mean() - 3.0))
        conc = np.log1p(max(kurt, 0.0))
        w[c] = agree * (1.0 + conc)
    tot = sum(w.values()) or 1.0
    return {c: v / tot * len(names) for c, v in w.items()}


def _prf_weights(cs, frac=0.05, power=2.0):
    """Weights by PSEUDO-RELEVANCE self-calibration - still no labels.

    Standard IR move, adapted: the pooled opinion of all channels is a
    noisy but non-random relevance signal, so treat its extremes as
    pseudo-labels and ask each channel how well it reproduces them. A
    channel that ranks the consensus head above the consensus tail is
    measuring the same thing the corpus agrees on; one that inverts it
    (q05's act, AUC 0.330 against real labels) scores below 0.5 here too
    and is removed - discovered, not declared.

    The kurtosis rule this replaces was too flat: it spread weight almost
    evenly (iv2 1.24 of ~7) when iv2 alone was carrying the query at AUC
    0.94. `power` sharpens the contrast so an informative channel
    actually dominates instead of being outvoted by five mediocre ones.
    """
    names = list(cs)
    Z = {c: _z(cs[c]) for c in names}
    n = len(next(iter(Z.values())))
    m = max(int(n * frac), 5)
    pool = np.mean([Z[c] for c in names], axis=0)
    o = np.argsort(-pool)
    pos, neg = o[:m], o[-m:]
    w = {}
    for c in names:
        z = Z[c]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[z[pos], z[neg]]
        a = auc(s, y.astype(np.int8))
        w[c] = max(a - 0.5, 0.0) ** power if a == a else 0.0
    tot = sum(w.values()) or 1.0
    return {c: v / tot * len(names) for c, v in w.items()}


def _fuse(cs, w):
    return sum(w.get(c, 0.0) * _z(v) for c, v in cs.items())


def main():
    store = ROOT / "lake/fresh_bench"
    want = ([int(x) for x in sys.argv[sys.argv.index("--q") + 1].split(",")]
            if "--q" in sys.argv else [3, 4, 5])
    db = Store.open(str(store))
    QS = queries()
    ept = db.table("episodes").scan()
    idx_of = {(s, int(a)): int(i) for s, a, i in
              zip(ept.column("stream").to_pylist(),
                  ept.column("ts").to_pylist(),
                  ept.column("episode_index").to_pylist())}
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    truth, support = {}, {}
    have = set(idx_of.values())
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei); truth[(int(q), ei)] = int(v)
        if ei in have:
            support[int(q)] = support.get(int(q), 0) + int(v)

    from tqdm import tqdm
    rows = []
    for qi in tqdm(want, desc="fusion", unit="q", dynamic_ncols=True):
        sup = support.get(qi, 0)
        if not sup:
            continue
        r = search_set(db, QS[qi], purity="fast",
                       k_max=int(np.ceil(sup * 1.5)), return_ranking=True)
        rank = r.get("ranking") or []
        cs = {c: np.asarray(v, float)
              for c, v in (r.get("channel_scores") or {}).items()}
        y = np.asarray([truth.get((qi, idx_of.get((s_, int(a_)), -1)), 0)
                        for s_, a_, _ in rank], np.int8)
        if not len(y) or not y.sum() or not cs:
            continue
        shipped = np.asarray([sc for _, _, sc in rank], float)

        def report(v):
            o = np.argsort(-np.asarray(v, float))
            return {"auc": round(auc(v, y), 3),
                    "prec_at_sup": round(float(y[o[:sup]].mean()), 3),
                    "yield_at_sup": round(float(y[o[:sup]].sum() / sup), 3)}

        # ORACLE: weight each channel by (AUC - 0.5), clipped. Truthset-
        # derived, NOT shippable - the ceiling only.
        ow = {c: max(auc(v, y) - 0.5, 0.0) for c, v in cs.items()}
        uw = _unsup_weights(cs)
        pw = _prf_weights(cs)
        rows.append({
            "q": qi, "support": sup,
            "shipped": report(shipped),
            "unsup": report(_fuse(cs, uw)),
            "prf": report(_fuse(cs, pw)),
            "best_alone": report(cs[max(cs, key=lambda c: 0)]),
            "oracle": report(_fuse(cs, ow)),
            "best_single": max(((c, auc(v, y)) for c, v in cs.items()),
                               key=lambda kv: kv[1]),
            "unsup_weights": {c: round(v, 2) for c, v in
                              sorted(uw.items(), key=lambda kv: -kv[1])},
            "prf_weights": {c: round(v, 2) for c, v in
                            sorted(pw.items(), key=lambda kv: -kv[1])},
        })

    for r in rows:
        print("\n" + "=" * 66)
        print(f"q{r['q']:02d}  support {r['support']}")
        print(f"  {'variant':<10}{'AUC':>8}{'prec@sup':>10}{'yield@sup':>11}")
        for k in ("shipped", "unsup", "prf", "oracle"):
            v = r[k]
            print(f"  {k:<10}{v['auc']:>8.3f}{v['prec_at_sup']:>10.3f}"
                  f"{v['yield_at_sup']:>11.3f}")
        bc, ba = r["best_single"]
        print(f"  best single channel: {bc} AUC {ba:.3f}")
        print(f"  prf weights:   {r['prf_weights']}")
    (ROOT / "bench" / "diag_fusion.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote bench/diag_fusion.json")


if __name__ == "__main__":
    main()
