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
from elidedb.setpath import (confidence_cut, event_positions,  # noqa: E402
                             filter_mask, nms_keep)

CH = ["pe", "act", "vid", "obj", "mot", "prf", "sig2", "conj", "iv2",
      "prf_q"]
# THE OPERATING POINT, and it must match the one being evaluated. The
# fitted artifact was tuned at K=10 and then read at k=100, where its
# choices are actively wrong: a filter that vetoes 60% of candidates
# costs nothing when only 10 slots exist and caps recall hard when 100
# do. Same env var as the bench so the two cannot drift apart.
K = int(__import__("os").environ.get("ELIDEDB_BENCH_K", "10"))


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
        # must see the same obj arrays the query executes — both now
        # decompose via atoms_of's closed-class boundaries
        from elidedb.sig2 import atoms_of
        nps = atoms_of(text.lower())[:2]
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
    try:
        # NaN abstention when the store has no iv2_vectors — matches
        # live's missing-channel behavior (unlike vocab, iv2 may
        # legitimately be absent on an uningested store)
        from elidedb.iv2 import iv2_lookup
        vs = []
        for vt in variants:
            look, _ = iv2_lookup(db, vt)
            vs.append(np.array([look(*k) for k in keys]))
        out["iv2"] = variant_max(vs)
    except Exception:
        out["iv2"] = np.full(len(keys), np.nan)
    out["prf_q"] = np.full(len(keys), np.nan)   # computed in-loop
    return out, sq is not None


_SP = {}      # (sid, pos) store geometry, set once in main()


def score_query(case, w, fq, fc, al=0.0, r=0):
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
    # PRF, identical to the live path (scenario.search_set): the fit
    # must score the same list the query returns, or the artifact is
    # tuned for a system that does not exist.
    if _SP.get("ev") is not None and w.get("prf_q", 0) > 0:
        ev = _SP["ev"]
        c = ev[np.argsort(-fused)[:25]].mean(0)
        c /= np.linalg.norm(c) + 1e-8
        ch = dict(ch); ch["prf_q"] = ev @ c
        fused = rrf(ch, weights=w)
    alive = (filter_mask(case["ch"], fc, fq) if fc and fq > 0
             else np.ones(len(fused), bool))
    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])]
    # temporal NMS before the K cap: suppressed duplicates of an
    # event free slots that refill with the next-ranked DISTINCT
    # events (fitted radius; measured: eggplant's true moment sat at
    # rank ~15 behind five windows of one junk event)
    order = nms_keep(order, _SP["sid"], _SP["pos"], r)[:K]
    # the returned set ends where confidence does (fitted alpha), not
    # at a fixed K — the product ratio true/returned is the objective
    order = order[:confidence_cut(fused[order], al, K)]
    lab = case["lab"][order]
    n = len(order)
    if n == 0:
        return 0.0
    tru = int((lab == 1).sum())
    prec = tru / n
    yld = tru / min(K, case["sup"]) if case["sup"] else 0.0
    # YIELD IS THE PRODUCT METRIC: true / min(k, support), stated as the
    # only one that counts. It led at 0.5 weight while precision led at
    # 1.0, which is why the fit kept buying precision with recall - at
    # k=100 that trade costs whole queries (q04 returned 49 of 165
    # available). Precision stays as a tiebreaker so a query that can
    # be answered with 12 results is not padded to 100 for free.
    return yld + 0.25 * prec


def main():
    db = Store.open("lake/bench")
    from elidedb.scenario import _episodes
    keys = _episodes(db)
    _SP["sid"], _SP["pos"] = event_positions(keys)
    # episode-level appearance centroids, for the PRF round
    from elidedb.embeddings import _vec_table
    _tb, _V = _vec_table(db, "pe_vectors")
    _V = np.asarray(_V, np.float32)
    _rmap = {}
    for _i, (_s, _a) in enumerate(zip(_tb.column("stream").to_pylist(),
                                      _tb.column("ts").to_pylist())):
        _rmap.setdefault((str(_s), int(_a)), []).append(_i)
    _ev = np.stack([_V[_rmap[(s_, a_)]].mean(0) if (s_, a_) in _rmap
                    else np.zeros(_V.shape[1], np.float32)
                    for s_, a_, _b in keys])
    _SP["ev"] = _ev / (np.linalg.norm(_ev, axis=1, keepdims=True) + 1e-8)
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
    # confidence-cut alphas: 0 = always fill to K (pre-cut behavior);
    # higher = the set ends where fused confidence falls below
    # alpha x the query's own top mass. The product goal is
    # true/returned -> 1 BEFORE growing toward full support, and the
    # objective (prec + 0.5*yield) already prices that ordering.
    als = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    rs = [0, 1, 2, 3]

    def ascend(subset, w, fq, fc, al, r):
        best = sum(score_query(c, w, fq, fc, al, r) for c in subset)
        for _ in range(4):
            improved = False
            for c in CH:
                for g in grid:
                    w2 = dict(w); w2[c] = g
                    s = sum(score_query(x, w2, fq, fc, al, r)
                            for x in subset)
                    if s > best + 1e-9:
                        best, w, improved = s, w2, True
            for f2 in fqs:
                s = sum(score_query(x, w, f2, fc, al, r)
                        for x in subset)
                if s > best + 1e-9:
                    best, fq, improved = s, f2, True
            for a2 in als:
                s = sum(score_query(x, w, fq, fc, a2, r)
                        for x in subset)
                if s > best + 1e-9:
                    best, al, improved = s, a2, True
            for r2 in rs:
                s = sum(score_query(x, w, fq, fc, al, r2)
                        for x in subset)
                if s > best + 1e-9:
                    best, r, improved = s, r2, True
            # membership toggle: any channel may join or leave the
            # filter set — the fitter, not code, decides which
            # channels have veto authority for this query type
            # NOTE: greedy, fixed visitation order, 4-round budget —
            # a rejection here means "no marginal gain from this
            # start", not a categorical falsification of the channel
            for c in CH:
                fc2 = ([x for x in fc if x != c] if c in fc
                       else fc + [c])
                s = sum(score_query(x, w, fq, fc2, al, r)
                        for x in subset)
                if s > best + 1e-9:
                    best, fc, improved = s, fc2, True
            if not improved:
                break
        return best, w, fq, fc, al, r

    def fit(subset, fc0, starts=()):
        """Best of coordinate ascent from the neutral start AND any
        warm starts (the previous fitted artifacts). Greedy ascent is
        path-dependent: a single channel change once collapsed the
        dir config (pe 6->0.5, cut and NMS discarded) purely by
        landing in a different local optimum — warm starts make the
        in-sample objective monotone across refits."""
        cands = [({c: 1.0 for c in CH}, 1 / 3, list(fc0), 0.0, 0)]
        cands += list(starts)
        # DELIBERATELY A WEAK SEARCH — measured, not assumed. Greedy
        # ascent from one start is path-dependent, and that path
        # dependence is what lost the shipped config when trk entered
        # CH. The obvious fix, 12 seeded random restarts, was tried:
        # it found a HIGHER in-sample optimum and the leave-one-query-
        # out estimate FELL, 0.265 -> 0.151. With ten queries the
        # search itself is the overfitting, so the limited search is
        # the regularizer and stays. Recoverability comes from the
        # backup rotation below instead.
        out = None
        for w0, fq0, fcs, al0, r0 in cands:
            got = ascend(subset, dict(w0), fq0, list(fcs), al0, r0)
            if out is None or got[0] > out[0] + 1e-9:
                out = got
        return out[1:]

    # starts = today's live behavior, so the fitted result can only
    # move away from it by measured improvement
    DIR0 = ["mot", "act", "prf"]
    CON0 = []

    def _starts(prefix):
        """Warm starts from the current + previous fitted artifacts.
        FINAL fits only — LOQO stays cold-start (an artifact fitted
        on all queries has seen the holdout; warming folds with it
        would leak)."""
        outs = []
        for pth in (Path("lake/bench/_set_weights.json"),
                    Path("lake/bench/_set_weights.prev.json")):
            try:
                c = json.loads(pth.read_text())
                wk = f"set_weights{prefix}"
                if wk in c:
                    outs.append((
                        {x: float(c[wk].get(x, 1.0)) for x in CH},
                        float(c.get(f"filter_quantile{prefix}",
                                    1 / 3)),
                        list(c.get(f"filter_channels{prefix}", [])),
                        float(c.get(f"cut_alpha{prefix}", 0.0)),
                        int(c.get(f"nms_r{prefix}", 0))))
            except Exception:
                pass
        return outs

    # leave-one-query-out: the honest generalization estimate of the
    # COLD-START procedure. Weights AND filter membership fitted PER
    # QUERY TYPE (directional vs not) — routing is a lexicon property
    # (swap exists?), roles are data.
    loqo = []
    folds = {}          # query text -> the config fitted WITHOUT it
    for i in range(len(cases)):
        train = [c for j, c in enumerate(cases) if j != i]
        same = [c for c in train if c["dir"] == cases[i]["dir"]]
        w, fq, fc, al, r = fit(same or train,
                               DIR0 if cases[i]["dir"] else CON0)
        # Keeping ONLY the score made the honest estimate unusable: the
        # benchmark had no way to evaluate a query with weights that had
        # not seen it, so every reported number was in-sample. Persist
        # the fold so the evaluation can actually be run that way.
        suffix = "_dir" if cases[i]["dir"] else ""
        folds[cases[i]["q"]] = {
            f"set_weights{suffix}": w, f"filter_quantile{suffix}": fq,
            f"filter_channels{suffix}": fc, f"cut_alpha{suffix}": al,
            f"nms_r{suffix}": r}
        loqo.append(score_query(cases[i], w, fq, fc, al, r))
        print(f"LOQO holdout {cases[i]['q'][:44]:44s} "
              f"score {loqo[-1]:.2f}", flush=True)
    print(f"LOQO mean objective: {np.mean(loqo):.3f}")

    w_dir, fq_dir, fc_dir, al_dir, r_dir = fit(
        [c for c in cases if c["dir"]] or cases, DIR0,
        _starts("_dir"))
    w_con, fq_con, fc_con, al_con, r_con = fit(
        [c for c in cases if not c["dir"]] or cases, CON0,
        _starts(""))
    out = {"set_weights_dir": w_dir, "filter_quantile_dir": fq_dir,
           "filter_channels_dir": fc_dir, "cut_alpha_dir": al_dir,
           "nms_r_dir": r_dir,
           "set_weights": w_con, "filter_quantile": fq_con,
           "filter_channels": fc_con, "cut_alpha": al_con,
           "nms_r": r_con,
           "fitted_on": "eval/truthsets/bridge4h.parquet",
           "loqo_mean": round(float(np.mean(loqo)), 3)}
    p = Path("lake/bench/_set_weights.json")
    # ROTATE BEFORE OVERWRITING. _starts() has always read
    # _set_weights.prev.json as a warm start, but nothing ever wrote
    # it, so the safety net was decorative: adding the trk channel
    # sent greedy ascent down a different path, the fit discarded the
    # confidence cut (alpha 0.85 -> 0.0), the bench fell 0.38 -> 0.22,
    # and the only copy of the good config was the one just
    # overwritten (lake/ is gitignored; it came back off HF). A fit is
    # a lossy write to an untracked artifact — it keeps a predecessor.
    if p.exists():
        Path("lake/bench/_set_weights.prev.json").write_text(p.read_text())
    p.write_text(json.dumps(out, indent=1))
    Path("lake/bench/_set_weights.loqo.json").write_text(json.dumps(folds, indent=1))
    print("wrote per-fold configs -> _set_weights.loqo.json "
          "(each fitted WITHOUT the query it scores)")
    print(f"dir {json.dumps(w_dir)} fq={fq_dir:.2f} fc={fc_dir} "
          f"al={al_dir:.2f} r={r_dir}")
    print(f"con {json.dumps(w_con)} fq={fq_con:.2f} fc={fc_con} "
          f"al={al_con:.2f} r={r_con}")


if __name__ == "__main__":
    main()
