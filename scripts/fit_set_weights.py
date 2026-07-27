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
    from elidedb.scenario import _HYPONYMS
    variants = [text]
    for w_, syns in _HYPONYMS.items():
        if w_ in text.lower():
            variants = [text] + [text.lower().replace(w_, s_)
                                 for s_ in syns]
            break
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


def rankfrac(v):
    r = np.full(len(v), 0.5)
    fin = np.isfinite(v)
    if fin.sum() > 1:
        r[fin] = np.argsort(np.argsort(v[fin])) / (fin.sum() - 1)
    return r


def score_query(case, w, fq):
    ch = {c: case["ch"][c] for c in CH
          if np.isfinite(case["ch"][c]).any() and w.get(c, 0) > 0}
    if not ch:
        return 0.0
    fused = rrf(ch, weights=w)
    alive = np.ones(len(fused), bool)
    if case["dir"] and fq > 0:
        con = [rankfrac(case["ch"][c]) for c in ("mot", "act", "prf")
               if np.isfinite(case["ch"][c]).any()]
        if con:
            alive &= ~(np.median(np.stack(con), 0) < fq)
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
    for qi in qids:
        text = QUERIES[qi]
        ch, isdir = capture(db, keys, text)
        lab = np.array([truth.get((qi, s, a), -1) for s, a, b in keys])
        # strict: ungraded counts false, matching the ledger
        lab = np.where(lab == 1, 1, 0) * (lab >= 0)
        cases.append({"ch": ch, "lab": lab, "dir": isdir,
                      "sup": int((lab == 1).sum()), "q": text})
        print(f"captured q{qi:02d} sup={int((lab == 1).sum())}",
              flush=True)

    grid = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
    fqs = [0.0, 0.25, 1 / 3, 0.5]

    def fit(subset):
        w = {c: 1.0 for c in CH}
        fq = 1 / 3
        best = sum(score_query(c, w, fq) for c in subset)
        for _ in range(4):
            improved = False
            for c in CH:
                for g in grid:
                    w2 = dict(w); w2[c] = g
                    s = sum(score_query(x, w2, fq) for x in subset)
                    if s > best + 1e-9:
                        best, w, improved = s, w2, True
            for f2 in fqs:
                s = sum(score_query(x, w, f2) for x in subset)
                if s > best + 1e-9:
                    best, fq, improved = s, f2, True
            if not improved:
                break
        return w, fq

    # leave-one-query-out: the honest generalization estimate.
    # Weights fitted PER QUERY TYPE (directional vs not) — routing is
    # a lexicon property (swap exists?), weights are data.
    loqo = []
    for i in range(len(cases)):
        train = [c for j, c in enumerate(cases) if j != i]
        w, fq = fit([c for c in train
                     if c["dir"] == cases[i]["dir"]] or train)
        loqo.append(score_query(cases[i], w, fq))
        print(f"LOQO holdout {cases[i]['q'][:44]:44s} "
              f"score {loqo[-1]:.2f}", flush=True)
    print(f"LOQO mean objective: {np.mean(loqo):.3f}")

    w_dir, fq_dir = fit([c for c in cases if c["dir"]] or cases)
    w_con, fq_con = fit([c for c in cases if not c["dir"]] or cases)
    out = {"set_weights_dir": w_dir, "filter_quantile_dir": fq_dir,
           "set_weights": w_con, "filter_quantile": fq_con,
           "fitted_on": "eval/truthsets/bridge4h.parquet",
           "loqo_mean": round(float(np.mean(loqo)), 3)}
    p = Path("lake/bench/_set_weights.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"dir {json.dumps(w_dir)} fq={fq_dir:.2f}")
    print(f"con {json.dumps(w_con)} fq={fq_con:.2f} -> {p}")


if __name__ == "__main__":
    main()
