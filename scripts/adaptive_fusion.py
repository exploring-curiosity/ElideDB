"""Per-query channel weights, computed from the scores themselves.

The fused system reaches 0.66 mean yield at k=100 while the BEST
SINGLE CHANNEL per query averages 0.73 — fusion loses to its own
parts on 8 of 10 queries. One global weight vector has to be right for
"put the eggplant into the drawer" (won by appearance, 1.00) and for
"pick up a vessel and put it on the stove" (won by the video-native
encoder, 0.77) at the same time, and it cannot be.

The weights therefore have to come from the query, and they cannot
come from labels — the truthset is eval-only. They come from the shape
of each channel's own score distribution over the corpus: a channel
that knows something about this query concentrates its mass in a few
episodes, and one that does not spreads flat. That is measurable
without knowing which episodes are right, and it is a corpus property,
not a dataset prior, so it transfers to any store.

This script measures candidate statistics against the metric and
against the oracle, so the choice is made on evidence.

  python scripts/adaptive_fusion.py [--k 100]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402
from elidedb.fusion import rrf                               # noqa: E402
from elidedb.scenario import _episodes                       # noqa: E402
from fit_set_weights import CH, capture                      # noqa: E402


def stats(s, K):
    """Label-free descriptors of one channel's score vector."""
    x = s[np.isfinite(s)]
    if len(x) < 8 or x.std() < 1e-9:
        return None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    top = np.sort(x)[-K:]
    return {
        # how far the head sits above the bulk, in robust units — a
        # channel with a real answer has an outlier head
        "tail": float((top.mean() - med) / (1.4826 * mad)),
        # peakedness of the whole distribution
        "kurt": float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-12)),
        # how much of the head's mass is above the bulk, scale-free
        "gap": float((top.mean() - x.mean()) / (x.std() + 1e-9)),
        # fraction of the corpus the channel is willing to score at all
        "cover": float(len(x) / len(s)),
    }


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
    import json
    cfg = json.loads((Path("lake/bench") / "_set_weights.json").read_text())

    cases = []
    for qi in sorted(sup):
        ch, isdir = capture(db, keys, QUERIES[qi])
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        cases.append((qi, ch, lab, isdir))
        print(f"captured q{qi:02d}", flush=True)

    def yld(qi, lab, fused):
        return lab[np.argsort(-fused)[:K]].sum() / min(K, sup[qi])

    def live(ch, w):
        use = {c: v for c, v in ch.items()
               if np.isfinite(v).any() and w.get(c, 0) > 0}
        return rrf(use, weights=w) if use else np.zeros(len(keys))

    rows = []
    for qi, ch, lab, isdir in cases:
        sfx = "_dir" if isdir else ""
        gw = {c: float(cfg[f"set_weights{sfx}"].get(c, 1.0)) for c in CH}
        st = {c: stats(np.asarray(ch[c], float), K) for c in CH}
        singles = {}
        for c in CH:
            v = np.asarray(ch[c], float)
            if np.isfinite(v).any():
                singles[c] = yld(qi, lab, np.where(np.isfinite(v), v,
                                                   -np.inf))
        row = {"q": qi, "global": yld(qi, lab, live(ch, gw)),
               "oracle": max(singles.values()),
               "best_ch": max(singles, key=singles.get)}
        for key in ("tail", "kurt", "gap"):
            w = {}
            for c in CH:
                w[c] = 0.0 if st[c] is None else max(st[c][key], 0.0)
            m = max(w.values()) or 1.0
            # scale to the same 0..6 range the fitted weights use
            w = {c: 6.0 * (v / m) ** 2 for c, v in w.items()}
            row[key] = yld(qi, lab, live(ch, w))
        # global weights MULTIPLIED by per-query informativeness: keep
        # what the fit learned about channels in general, modulate it
        # by what this query's score shapes say
        for key in ("tail", "gap"):
            w = {}
            for c in CH:
                s_ = 0.0 if st[c] is None else max(st[c][key], 0.0)
                w[c] = gw[c] * s_
            m = max(w.values()) or 1.0
            w = {c: 6.0 * v / m for c, v in w.items()}
            row["g*" + key] = yld(qi, lab, live(ch, w))
        # SPECTRAL META-LEARNER (Parisi et al., PNAS 2014: "Ranking and
        # combining multiple predictors without labeled data"). Treat
        # each channel's top-K set as a binary predictor. If the
        # channels err independently, the off-diagonal covariance
        # matrix of their predictions is rank-one, and its leading
        # eigenvector's entries are each predictor's balanced accuracy
        # up to scale — reliability estimated from AGREEMENT, with no
        # labels anywhere. This is the right shape of tool for the
        # problem: the truthset is eval-only, so the only evidence
        # about which channel is right for THIS query is how the
        # channels relate to each other on it.
        avail = [c for c in CH if np.isfinite(np.asarray(ch[c],
                                                         float)).any()]
        P = []
        for c in avail:
            v = np.asarray(ch[c], float)
            v = np.where(np.isfinite(v), v, -np.inf)
            b = np.zeros(len(keys))
            b[np.argsort(-v)[:K]] = 1.0
            P.append(2 * b - 1)                    # +-1 predictions
        P = np.stack(P)
        Q = np.cov(P)
        np.fill_diagonal(Q, 0.0)
        ev = np.linalg.eigh(Q)[1][:, -1]
        if ev.sum() < 0:
            ev = -ev
        w = {c: 0.0 for c in CH}
        m = float(np.max(ev)) or 1.0
        for c, e in zip(avail, ev):
            w[c] = 6.0 * max(float(e), 0.0) / m
        row["sml"] = yld(qi, lab, live(ch, w))
        # and the same reliability applied on top of the fitted weights
        w2 = {c: gw[c] * w[c] for c in CH}
        m2 = max(w2.values()) or 1.0
        row["g*sml"] = yld(qi, lab, live(ch, {c: 6.0 * v / m2
                                              for c, v in w2.items()}))
        base = live(ch, gw)
        # TEMPORAL SMOOTHING. Episodes are windows cut from continuous
        # recordings, so a true episode's neighbours in the same stream
        # are usually true too — a property of video, not of this
        # corpus. A retrieved episode therefore vouches for its
        # neighbours, which is exactly what recall@100 needs.
        sid = np.array([k[0] for k in keys])
        pos = np.array([k[1] for k in keys])
        order_t = np.lexsort((pos, sid))
        for alpha in (0.3, 0.5):
            sm = base.copy()
            b = base[order_t]
            s_ = sid[order_t]
            for d in (1, 2):
                nb = np.full(len(b), -np.inf)
                nb[d:] = np.where(s_[d:] == s_[:-d], b[:-d], -np.inf)
                nb2 = np.full(len(b), -np.inf)
                nb2[:-d] = np.where(s_[:-d] == s_[d:], b[d:], -np.inf)
                b = np.maximum(b, alpha * np.maximum(nb, nb2))
            sm[order_t] = b
            row[f"smooth{alpha}"] = yld(qi, lab, sm)
        # PSEUDO-RELEVANCE FEEDBACK (Rocchio). The top of the fused list
        # is the best available description of what the user meant; its
        # centroid in appearance space re-scores the corpus and enters
        # the fusion as one more voter. Classic IR, no labels, and it
        # targets recall specifically.
        try:
            from elidedb.embeddings import _vec_table
            tb, V = _vec_table(db, "pe_vectors")
            V = np.asarray(V, np.float32)
            rmap = {}
            for i, (s_, a_) in enumerate(zip(tb.column("stream").to_pylist(),
                                             tb.column("ts").to_pylist())):
                rmap.setdefault((str(s_), int(a_)), []).append(i)
            ep_vec = np.stack([
                V[rmap[(s_, a_)]].mean(0) if (s_, a_) in rmap
                else np.zeros(V.shape[1], np.float32)
                for s_, a_, _ in keys])
            ep_vec /= np.linalg.norm(ep_vec, axis=1, keepdims=True) + 1e-8
            for n in (10, 25):
                seed = np.argsort(-base)[:n]
                c = ep_vec[seed].mean(0)
                c /= np.linalg.norm(c) + 1e-8
                ch2 = dict(ch); ch2["prf_q"] = ep_vec @ c
                w2 = dict(gw); w2["prf_q"] = 3.0
                row[f"prf{n}"] = yld(qi, lab, live(ch2, w2))
        except Exception as e:
            row["prf10"] = row["prf25"] = float("nan")
            print("prf failed:", e)
        rows.append(row)

    cols = ["global", "sml", "smooth0.3", "smooth0.5", "prf10", "prf25",
            "oracle"]
    print("\n" + f"{'q':>4} " + " ".join(f"{c:>8}" for c in cols) +
          "   best channel")
    for r in rows:
        print(f"q{r['q']:02d} " + " ".join(f"{r[c]:>8.2f}" for c in cols) +
              f"   {r['best_ch']}")
    print(f"{'mean':>4} " + " ".join(
        f"{np.mean([r[c] for r in rows]):>8.2f}" for c in cols))


if __name__ == "__main__":
    main()
