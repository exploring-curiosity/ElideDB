"""Fit the SET-PATH channel roles from the truthset — no hand rules.

Every hand role rule of the acceptance sprint (act filters, motion
never orders, obj boost) becomes a FITTED artifact: ordering weights
for all channels (contrast included) plus the contrast-filter quantile
are searched by coordinate ascent on the truthset, scored
leave-one-query-out so 10 queries cannot be memorized. The result is a
per-store learned file — data-derived, corpus-specific, exactly the
self-recognized prior the no-hardwire rule demands. Unseen corpora
start at neutral defaults until fitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402
from elidedb.fusion import rrf, variant_max                  # noqa: E402
from elidedb.setpath import filter_mask                      # noqa: E402

CH = ["pe", "act", "vid", "obj", "mot", "prf", "sig2", "conj"]
K = 10


def capture(db, keys, text):
    from elidedb.action_channel import act_lookup
    from elidedb.context import embed_texts
    from elidedb.grounding import parse_relation
    from elidedb.motion import motion_lookup
    from elidedb.objects import object_lookup
    from elidedb.pe import pe_lookup
    from elidedb.prf import prf_contrast
    from elidedb.rerank import directional_swap
    from elidedb.vid import vid_lookup
    from elidedb.vocab import corpus_variants
    variants = corpus_variants(db, text)
    out = {}
    vs = []
    for vt in variants:
        look, _ = pe_lookup(db, vt)
        vs.append(np.array([look(*k) for k in keys]))
    out["pe"] = variant_max(vs)
    look, _ = act_lookup(db, text)
    out["act"] = np.array([look(*k) for k in keys])
    vs = []
    for vt in variants:
        look, _ = vid_lookup(db, vt)
        vs.append(np.array([look(*k) for k in keys]))
    out["vid"] = variant_max(vs)
    from elidedb.sig2 import conj_lookup, sig2_lookup
    vs = []
    for vt in variants:
        look, _ = sig2_lookup(db, vt)
        vs.append(np.array([look(*k) for k in keys]))
    out["sig2"] = variant_max(vs)
    cl = conj_lookup(db, text)
    out["conj"] = (np.array([cl(*k) for k in keys]) if cl is not None
                   else np.full(len(keys), np.nan))
    rel = parse_relation(text)
    nps = [p for p in ((rel[0], rel[2]) if rel else ())
           if p and "object" not in p]
    if not nps:
        # mirror the live path's fallback (scenario.search_set): fit
        # must see the same obj arrays the query executes, or the
        # toggle search could grant filter authority on arrays that
        # do not exist live
        import re as _re
        nps = [m.group(0) for m in _re.finditer(
            r"\b(?:a|an|the)\s+(?:\w+\s+){0,2}\w+", text.lower())][:2]
    if nps:
        ol = object_lookup(db, embed_texts(nps))
        out["obj"] = np.array(
            [(lambda r: r[0] * (1 + r[1]) if r[0] == r[0]
              else np.nan)(ol(*k)) for k in keys])
    else:
        out["obj"] = np.full(len(keys), np.nan)
    sq = directional_swap(text)
    if sq:
        qv2 = embed_texts([text, sq])
        ml = motion_lookup(db, qv2[0], qv2[1])
        out["mot"] = np.array([ml(*k) for k in keys])
        look, _ = pe_lookup(db, sq)
        pes = np.array([look(*k) for k in keys])
        out["prf"] = prf_contrast(db, keys, out["pe"], pes)
    else:
        out["mot"] = np.full(len(keys), np.nan)
        out["prf"] = np.full(len(keys), np.nan)
    return out, sq is not None


def score_query(case, w, fq, fc):
    if case.get("gated"):
        # live returns empty for gated queries regardless of weights;
        # a constant removes them from the ascent so weights are never
        # tuned to please episodes the gate will kill
        return 0.0
    ch = {c: case["ch"][c] for c in CH
          if np.isfinite(case["ch"][c]).any() and w.get(c, 0) > 0}
    if not ch:
        return 0.0
    fused = rrf(ch, weights=w)
    alive = (filter_mask(case["ch"], fc, fq) if fc and fq > 0
             else np.ones(len(fused), bool))
    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])][:K]
    lab = case["lab"][order]
    n = len(order)
    if n == 0:
        return 0.0
    tru = int((lab == 1).sum())
    prec = tru / n
    yld = tru / min(K, case["sup"]) if case["sup"] else 0.0
    return prec + 0.5 * yld


def main():
    db = Store.open("lake/bench")
    from elidedb.scenario import _episodes
    keys = _episodes(db)
    t = pq.read_table("eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(t0)): int(v) for q, s, t0, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    qids = sorted({int(q) for q in t["query_id"]})
    cases = []
    from elidedb.scenario import _auto_action_support
    for qi in qids:
        text = QUERIES[qi]
        ch, isdir = capture(db, keys, text)
        lab = np.array([truth.get((qi, s, a), -1) for s, a, b in keys])
        # strict: ungraded counts false, matching the ledger
        lab = np.where(lab == 1, 1, 0) * (lab >= 0)
        g = _auto_action_support(db, text)
        gated = bool(g is not None and g["max_p"] < 0.05)
        cases.append({"ch": ch, "lab": lab, "dir": isdir, "gated":
                      gated, "sup": int((lab == 1).sum()), "q": text})
        print(f"captured q{qi:02d} sup={int((lab == 1).sum())}"
              f"{' GATED' if gated else ''}", flush=True)

    grid = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
    fqs = [0.0, 0.25, 1 / 3, 0.5]

    def fit(subset, fc0):
        w = {c: 1.0 for c in CH}
        fq = 1 / 3
        fc = list(fc0)
        best = sum(score_query(c, w, fq, fc) for c in subset)
        for _ in range(4):
            improved = False
            for c in CH:
                for g in grid:
                    w2 = dict(w); w2[c] = g
                    s = sum(score_query(x, w2, fq, fc) for x in subset)
                    if s > best + 1e-9:
                        best, w, improved = s, w2, True
            for f2 in fqs:
                s = sum(score_query(x, w, f2, fc) for x in subset)
                if s > best + 1e-9:
                    best, fq, improved = s, f2, True
            # membership toggle: any channel may join or leave the
            # filter set — the fitter, not code, decides which
            # channels have veto authority for this query type
            # NOTE: greedy, fixed visitation order, 4-round budget —
            # a rejection here means "no marginal gain from this
            # start", not a categorical falsification of the channel
            for c in CH:
                fc2 = ([x for x in fc if x != c] if c in fc
                       else fc + [c])
                s = sum(score_query(x, w, fq, fc2) for x in subset)
                if s > best + 1e-9:
                    best, fc, improved = s, fc2, True
            if not improved:
                break
        return w, fq, fc

    # starts = today's live behavior, so the fitted result can only
    # move away from it by measured improvement
    DIR0 = ["mot", "act", "prf"]
    CON0 = []

    # leave-one-query-out: the honest generalization estimate.
    # Weights AND filter membership fitted PER QUERY TYPE (directional
    # vs not) — routing is a lexicon property (swap exists?), roles
    # are data.
    loqo = []
    for i in range(len(cases)):
        train = [c for j, c in enumerate(cases) if j != i]
        same = [c for c in train if c["dir"] == cases[i]["dir"]]
        w, fq, fc = fit(same or train,
                        DIR0 if cases[i]["dir"] else CON0)
        loqo.append(score_query(cases[i], w, fq, fc))
        print(f"LOQO holdout {cases[i]['q'][:44]:44s} "
              f"score {loqo[-1]:.2f}", flush=True)
    print(f"LOQO mean objective: {np.mean(loqo):.3f}")

    w_dir, fq_dir, fc_dir = fit([c for c in cases if c["dir"]]
                                or cases, DIR0)
    w_con, fq_con, fc_con = fit([c for c in cases if not c["dir"]]
                                or cases, CON0)
    out = {"set_weights_dir": w_dir, "filter_quantile_dir": fq_dir,
           "filter_channels_dir": fc_dir,
           "set_weights": w_con, "filter_quantile": fq_con,
           "filter_channels": fc_con,
           "fitted_on": "eval/truthsets/bridge4h.parquet",
           "loqo_mean": round(float(np.mean(loqo)), 3)}
    p = Path("lake/bench/_set_weights.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"dir {json.dumps(w_dir)} fq={fq_dir:.2f} fc={fc_dir}")
    print(f"con {json.dumps(w_con)} fq={fq_con:.2f} fc={fc_con}")


if __name__ == "__main__":
    main()
