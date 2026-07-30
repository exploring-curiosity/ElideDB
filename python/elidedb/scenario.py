"""SET RETRIEVAL — the robotics query model, seconds-scale, VLM-free.

The product truth (user-stated): a robotics team doesn't want the one
best clip, they want ALL clips matching a scenario, clean enough to
retrain on. Precision of the DELIVERED SET is king; latency budget is
seconds.

v2 after the visual calibration (2026-07-24) tore v1 down:
  - Scene-clustered IVF cells hid 45/46 true "lid" episodes from the
    opened-6 — actions do not live in scene cells. At this corpus size
    (thousands of episodes) a flat exact scan is sub-millisecond, so
    query-time cell pruning bought nothing and cost nearly all recall.
    Cells remain as the browsable scenario MAP (build_scenarios) and
    as the pruning layout for a future 100k+ corpus.
  - Single-encoder scoring cannot rank relational actions deep into a
    list (PE full-scan R@257 = 9/46 on "lid"). The set path now fuses
    the channels that each own a dimension: PE (appearance-text),
    ACT (SSv2 action posteriors — put-in vs take-out AUC 0.889),
    VID (X-CLIP), MOT (delta-appearance contrast, close/open AUC 0.98).
  - Direction is a HARD FILTER, not a rerank: an episode both
    direction-aware channels score negative is dropped, not demoted —
    junk excluded from the delivery, per the set-purity mandate.
  - The VLM sample audit is GONE (user directive: no LLM/VLM judges;
    calibration showed 7B est 1.0 on visually ~10%-pure sets). The
    audited tier is now GEOMETRY: SAM 3 grounds the query's noun
    phrases and the containment change over time verifies the
    relation. Deterministic, explainable, abstains honestly.
"""
from __future__ import annotations

import json
import time

import numpy as np

_CELLS = {}


def _pool_recordings(store):
    """(keys, matrix) — one pooled PE vector per recording."""
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "pe_vectors")
    ss = np.asarray(tbl.column("stream").to_pylist())
    sa = np.asarray([int(v) for v in tbl.column("ts").to_pylist()])
    sb = np.asarray([int(v) for v in tbl.column("t1").to_pylist()])
    order = np.lexsort((sa, ss))
    keys, mats, rowsets = [], [], []
    i = 0
    o = order
    while i < len(o):
        j = i
        k = (str(ss[o[i]]), int(sa[o[i]]), int(sb[o[i]]))
        while j < len(o) and (str(ss[o[j]]), int(sa[o[j]]),
                              int(sb[o[j]])) == k:
            j += 1
        rows = o[i:j]
        v = np.asarray(vecs[np.sort(rows)]).mean(0)
        v /= np.linalg.norm(v) + 1e-8
        keys.append(k)
        mats.append(v.astype(np.float32))
        rowsets.append(np.sort(rows))
        i = j
    return keys, np.stack(mats), rowsets


def build_scenarios(store, min_cluster_size=8):
    """HDBSCAN scenario groups (the human-browsable map) + KMeans-IVF
    cells (the physical layout for 100k+ scale), persisted beside
    pe_vectors. NOT used for query-time pruning at this corpus size —
    measured hiding 45/46 relational positives."""
    t0 = time.time()
    keys, M, rowsets = _pool_recordings(store)
    try:
        from sklearn.cluster import HDBSCAN
        groups = HDBSCAN(min_cluster_size=min_cluster_size,
                         metric="euclidean", copy=True).fit_predict(M)
    except Exception:
        groups = np.zeros(len(M), np.int32)
    from sklearn.cluster import KMeans
    k = int(np.clip(len(M) // 32, 16, 256))
    lab = KMeans(n_clusters=k, random_state=0,
                 n_init=4).fit_predict(M)
    cids = sorted(set(int(c) for c in lab))
    cents = np.stack([M[lab == c].mean(0) for c in cids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8
    cell_dir = store.table("pe_vectors").dir / "_cache"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.save(cell_dir / "cell_centroids.npy", cents)
    np.save(cell_dir / "cell_labels.npy", np.asarray(lab, np.int32))
    np.save(cell_dir / "cell_matrix.npy", M)
    np.save(cell_dir / "scenario_groups.npy",
            np.asarray(groups, np.int32))
    (cell_dir / "cells.json").write_text(json.dumps({
        "version": store.table("pe_vectors").state().version,
        "keys": [[k[0], k[1], k[2]] for k in keys],
        "cells": int(len(cents)),
        "scenario_groups": int(len(set(int(g) for g in groups
                                       if g >= 0))),
    }))
    return {"recordings": len(keys), "cells": int(len(cents)),
            "scenario_groups": int(len(set(int(g) for g in groups
                                           if g >= 0))),
            "seconds": round(time.time() - t0, 1)}


def _episodes(store):
    ep = store.table("episodes").scan()
    return list(zip((str(s) for s in ep.column("stream").to_pylist()),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))


def _auto_action_support(store, text):
    """Self-recognized no-match signal: map the query onto the action
    probe's OWN class vocabulary by embedding similarity (no coded
    verb list — the vocabulary is a model property, the mapping is
    computed) and report the corpus's best posterior for the top
    matched classes. Only gates when the mapping is confident: a
    query far from every class name (a domain this probe does not
    cover) must never be silenced by it."""
    from .action_probe import query_class_weights, ssv2_classes
    from .embeddings import _vec_table
    from .pe import _text_vec
    names = ssv2_classes()
    qv = _text_vec(text)
    sims = np.array([float(_text_vec(c.lower()) @ qv) for c in names])
    top = np.argsort(-sims)[:3]
    # SCALE-FREE mapping confidence: is the best class an outlier of
    # the similarity distribution, or just the least-bad of a flat
    # field? (An absolute cosine threshold silently disabled the gate
    # — ledger-caught: fold gate FAIL.)
    z = float((sims[top[0]] - sims.mean()) / (sims.std() + 1e-9))
    if z < 3.0:
        return None                    # vocabulary doesn't cover this
    # CONJUNCTIVE gate, every term scale-free (single-class posterior
    # could not separate 'fold' 0.008 from 'red-out' 0.010 — measured;
    # half the vocabulary is below 0.05 on this corpus):
    #   1. an outlier class exists (z >= 3, above)
    #   2. those outlier classes are posterior-DEAD here (bottom third
    #      of the class-max distribution)
    #   3. EXPECTED support under the full similarity distribution is
    #      below the corpus mean — a query whose sim mass spreads over
    #      living classes (red-out: picking/putting) never gates; one
    #      whose mass concentrates on a dead class (fold) does
    zs = (sims - sims.mean()) / (sims.std() + 1e-9)
    top = np.where(zs >= 3.0)[0]
    _, probs = _vec_table(store, "action_probs")
    cmax = np.asarray(probs).max(0)
    mp = float(cmax[top].max())
    dead = mp < float(np.percentile(cmax, 33))
    w = np.exp((sims - sims.max()) / 0.05)
    w /= w.sum()
    ratio = float(w @ cmax) / (float(cmax.mean()) + 1e-9)
    if not (dead and ratio < 1.0):
        return None
    return {"classes": [names[int(i)] for i in top],
            "z": round(z, 2), "max_p": round(mp, 4),
            "support_ratio": round(ratio, 2)}


def _knee(sorted_desc):
    """Set boundary on the sorted fused curve. The raw largest-drop
    knee cut 183-episode classes to 4 (RRF consensus gives the top few
    an outsized gap): the boundary is now the last position whose score
    keeps a fixed fraction of the top-5 mean — scale-stable under RRF
    (scores are bounded sums of w/(60+rank)) — with the largest-drop
    knee only allowed to TIGHTEN it, never to cut inside the top-8."""
    s = np.asarray(sorted_desc, float)
    if len(s) < 2:
        return len(s)
    top = float(np.mean(s[:min(5, len(s))]))
    floor_cut = int(np.searchsorted(-s, -0.62 * top, side="right"))
    d = s[:-1] - s[1:]
    knee = int(np.argmax(d[2:])) + 2 + 1 if len(d) > 2 else len(s)
    # NO minimum set size: the user's contract is "up to K, and
    # everything returned is true" — a forced floor delivers junk
    return min(floor_cut, knee)


def search_set(store, text, purity="fast", k_max=400, audit_n=12,
               return_ranking=False, cfg_override=None):
    """The robotics query: ALL matching clips, purity-first, VLM-free.

    purity="fast"    exact fused scan, direction filter, knee cut
    purity="audited" + geometric relational audit (SAM 3 boxes over
                     time) on a stratified sample of the set; pruned at
                     the last geometry-positive sample. Only fires when
                     the query parses as a relation; abstains otherwise.
    """
    from .fusion import rrf, variant_max
    from .grounding import parse_relation
    from .rerank import directional_swap
    t0 = time.perf_counter()
    keys = _episodes(store)
    # LEXICON ONLY beyond this point (no-hardwire rule): the antonym
    # swap and the parsed relation are dictionary knowledge; every
    # dataset-facing decision below is derived from the corpus at
    # query time.
    sq = directional_swap(text)
    rel = parse_relation(text)
    tl = text.lower()

    # CATEGORY-WORD EXPANSION, corpus-attested: WordNet supplies the
    # candidate hyponyms (dictionary), the store's SigLIP2 frame
    # space decides which exist HERE (data) — "vessel" starves the
    # text channels (measured: sup 247, prec 0.25) and SAM cannot
    # ground it; the specific attested terms can.
    try:
        from .vocab import corpus_variants
        variants = corpus_variants(store, text)
    except Exception:
        variants = [text]

    ch = {}
    # CHANNEL DEATH MUST BE LOUD (2026-07-28). Every block below used
    # to swallow its exception, so a broken dependency degraded search
    # silently: a transformers upgrade killed iv2 and the frozen bench
    # fell 0.38 -> 0.13 with no error anywhere, diagnosable only by
    # bisecting the environment. Failures are now recorded per channel
    # and returned in the result; the caller decides whether a
    # degraded answer is acceptable (bench_truth refuses to record a
    # ledger row, the Desk shows a warning).
    failed = {}

    def _fail(name, exc):
        failed[name] = f"{type(exc).__name__}: {exc}"[:200]

    try:
        from .pe import pe_lookup
        vs = []
        for vtext in variants:
            look, _ = pe_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["pe"] = variant_max(vs)
    except Exception as e:
        _fail("pe", e)
    try:
        # act: text-mapped class weights onto the probe's own
        # vocabulary (contrast against the swap when one exists) —
        # embedding-derived, no coded class lists
        from .action_channel import act_lookup
        look, _ = act_lookup(store, text)
        ch["act"] = np.array([look(*k) for k in keys])
    except Exception as e:
        _fail('act', e)
    try:
        from .sig2 import conj_lookup
        # sig2 (SigLIP2 appearance) is NOT scored: leave-one-out on the
        # frozen truthset measured its contribution at exactly zero
        # (35 true with it, 35 without), while pe and obj answer the same
        # "what does it look like" question. conj stays and is worth -5,
        # and it reads sig2_vectors, so the SigLIP2 pass is still paid at
        # INGEST - dropping the channel saves query work, not ingest.
        cl = conj_lookup(store, text)
        if cl is not None:
            ch["conj"] = np.array([cl(*k) for k in keys])
    except Exception as e:
        _fail('sig2', e)
    try:
        # iv2: VIDEO-native text alignment (InternVideo2-Stage2 1B,
        # temporal modeling the frame-pooled channels lack) — the
        # 4 frames pass through the encoder together
        from .iv2 import iv2_lookup
        vs = []
        for vtext in variants:
            look, _ = iv2_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["iv2"] = variant_max(vs)
    except Exception as e:
        _fail('iv2', e)
    try:
        from .vid import vid_lookup
        vs = []
        for vtext in variants:
            look, _ = vid_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["vid"] = variant_max(vs)
    except Exception as e:
        _fail('vid', e)
    # OBJ channel — FastSAM crops matched against the query's noun
    # phrases (SigLIP space): the color/attribute binding the product
    # bench showed missing from the set path
    try:
        from .context import embed_texts
        from .objects import object_lookup
        nps = [p for p in ((rel[0], rel[2]) if rel else ())
               if p and "object" not in p]
        if not nps:
            # ONE decomposition for the whole system: atoms_of's
            # closed-class boundaries (the raw regex here had the
            # same swallowed-preposition bug atoms_of was fixed for)
            from .sig2 import atoms_of
            nps = atoms_of(tl)[:2]
        if nps:
            olook = object_lookup(store, embed_texts(nps))
            def _obj(k):
                osc, omo = olook(*k)
                return osc * (1.0 + omo) if osc == osc else float("nan")
            ch["obj"] = np.array([_obj(k) for k in keys])
    except Exception as e:
        _fail('obj', e)
    # NOT a channel: late interaction over the patch grid. Within a
    # frame the grid separates cleanly (an eggplant frame scores 0.173
    # for "an eggplant" against 0.063 for "a banana", where the pooled
    # vector manages 0.107 vs 0.071) and across episodes it ranks at
    # 0.08 standalone yield, last of ten, against pooled SigLIP2's 0.53.
    # MaxSim over 1,024 patches is an extreme-value draw: the episode
    # with the widest patch spread wins it whatever it contains. Eight
    # poolings were measured (max, top-k means, within-episode z) and
    # the best reached 0.15. scripts/patch_ingest.py + patches.py stay
    # as the reproduction.
    # NOT a channel: region identity from crops, measured dead at corpus
    # scale. The hypothesis was that obj failed only because its crops
    # were cut out of DOWNSCALED decodes. Recut at native 640x480 along
    # common-fate tracks (scripts/track_ingest.py, 25,335 crops) it looked
    # strong on an 80-episode pool — yield@10 of 1.00/1.00/0.75/0.50 on
    # q10/q09/q00/q08 — and collapsed against the real 1,121: banana's two
    # true episodes rank 280th and 534th, spoon's 21/38/49, standalone
    # mean yield 0.08 against obj's 0.14 and iv2's 0.24. The pool was 14x
    # easier and the separation was its artifact. Resolution was not the
    # blocker; a crop asks a small patch what it is with the context that
    # would answer the question cropped away.
    # CONTRAST channels — direction EVIDENCE, computed only when the
    # lexicon yields a swap. Architectural principle replacing every
    # hand routing rule (ledger-derived, now task-free): contrast
    # channels FILTER, content channels ORDER — a contrast score says
    # "more like the query than its opposite", never "relevant".
    contrast_ch = {}
    if sq is not None:
        if "act" in ch:
            contrast_ch["act"] = ch["act"]
        try:
            from .context import embed_texts
            from .motion import motion_lookup
            qv2 = embed_texts([text, sq])
            mlook = motion_lookup(store, qv2[0], qv2[1])
            contrast_ch["mot"] = np.array([mlook(*k) for k in keys])
            ch["mot"] = contrast_ch["mot"]
        except Exception as e:
            _fail('mot', e)
        try:
            # SELF-RECOGNIZED contrast: Rocchio anchors in the
            # domain-general video-native space — the corpus itself
            # defines what this direction looks like here. No class
            # names, works unchanged on any domain.
            from .pe import pe_lookup
            from .prf import prf_contrast
            look, _ = pe_lookup(store, sq)
            pe_swap = np.array([look(*k) for k in keys])
            contrast_ch["prf"] = prf_contrast(
                store, keys, ch.get("pe"), pe_swap)
            ch["prf"] = contrast_ch["prf"]
        except Exception as e:
            _fail('prf', e)

    # ABLATION HOOK: drop channels by name to measure what each one is
    # actually worth. Reads the env so the live path is untouched when
    # unset, and so an ablation runs through the SAME code as production
    # rather than a reimplementation of it.
    import os as _os
    _drop = {c.strip() for c in _os.environ.get("ELIDEDB_DROP_CHANNELS", "").split(",") if c.strip()}
    if _drop:
        for c in _drop:
            ch.pop(c, None)
            contrast_ch.pop(c, None)

    directional = sq is not None
    weights = {c: 1.0 for c in ch}
    filter_q = 1 / 3
    fnames = None       # None => legacy: every contrast channel
    cut_alpha = 0.0     # 0 => fill to k_max (legacy, pre-cut)
    nms_r = 0           # 0 => no temporal event dedup (legacy)
    prf_n, prf_w = 25, 3.0     # feedback depth and voice, both fitted
    from pathlib import Path
    sw = Path(store.dir) / "_set_weights.json"
    try:
        if cfg_override is not None:
            # honest evaluation: weights fitted WITHOUT this query
            cfg = dict(cfg_override)
        elif sw.exists():
            # FITTED roles (scripts/fit_set_weights.py): ordering
            # weights for every channel INCLUDING contrasts, plus the
            # filter quantile — coordinate ascent on the truthset,
            # LOQO-validated. Data-derived per store; the no-hardwire
            # rule's answer to hand role rules (the fit independently
            # rediscovered mot=0-in-ordering).
            cfg = json.loads(sw.read_text())
        if cfg is not None:
            wk = ("set_weights_dir" if directional and
                  "set_weights_dir" in cfg else "set_weights")
            fk = ("filter_quantile_dir" if directional and
                  "filter_quantile_dir" in cfg else "filter_quantile")
            weights = {c: float(cfg[wk].get(c, 1.0)) for c in ch}
            filter_q = float(cfg.get(fk, 1 / 3))
            ck = ("filter_channels_dir" if directional and
                  "filter_channels_dir" in cfg else "filter_channels")
            if ck in cfg:
                fnames = list(cfg[ck])
            ak = ("cut_alpha_dir" if directional and
                  "cut_alpha_dir" in cfg else "cut_alpha")
            cut_alpha = float(cfg.get(ak, 0.0))
            rk = ("nms_r_dir" if directional and
                  "nms_r_dir" in cfg else "nms_r")
            nms_r = int(cfg.get(rk, 0))
        else:
            cfg = json.loads((Path(store.dir)
                              / "_channel_weights.json").read_text())
            learned = (cfg.get("weights_dir", cfg.get("weights", {}))
                       if directional else cfg.get("weights", {}))
            weights = {c: float(learned.get(c, 1.0)) for c in ch}
    except Exception:
        pass
    fused = rrf(ch, weights=weights)

    # PSEUDO-RELEVANCE FEEDBACK (Rocchio, and it is measured, not
    # assumed). The top of the first fused list is the best available
    # description of what the user actually meant; its centroid in
    # appearance space re-scores the corpus and rejoins the fusion as
    # one more voter. Two rounds of channel work bought nothing here -
    # per-query weights from score-distribution shape (0.59 vs 0.60
    # global) and the spectral meta-learner's label-free reliability
    # estimate (0.56) both LOST - while this, the oldest trick in IR,
    # is the only thing that moved the metric: mean yield 0.60 -> 0.61
    # at k=100, carried by the queries with real support (q03 0.60 ->
    # 0.67, q07 0.83 -> 0.92, q08 0.50 -> 0.56). It targets recall,
    # which is what true/min(k, support) rewards.
    try:
        from .embeddings import _vec_table
        _tb, _V = _vec_table(store, "pe_vectors")
        _V = np.asarray(_V, np.float32)
        _rmap = {}
        for _i, (_s, _a) in enumerate(zip(_tb.column("stream").to_pylist(),
                                          _tb.column("ts").to_pylist())):
            _rmap.setdefault((str(_s), int(_a)), []).append(_i)
        _ev = np.stack([_V[_rmap[(s_, a_)]].mean(0) if (s_, a_) in _rmap
                        else np.zeros(_V.shape[1], np.float32)
                        for s_, a_, _b in keys])
        _ev /= np.linalg.norm(_ev, axis=1, keepdims=True) + 1e-8
        _seed = np.argsort(-fused)[:prf_n]
        _c = _ev[_seed].mean(0)
        _c /= np.linalg.norm(_c) + 1e-8
        ch["prf_q"] = _ev @ _c
        weights["prf_q"] = float(weights.get("prf_q", prf_w))
        fused = rrf(ch, weights=weights)
    except Exception as e:
        _fail('prf_q', e)

    # ITM CASCADE — the cross-encoder rerank, cost-gated by depth.
    # Measured at k=1.5xsupport: shipped RRF 0.29/0.23, this store's
    # cosine ensemble under z-fusion 0.32/0.21, ITM alone 0.38/0.25,
    # cosine+ITM 0.40/0.27. Reranking the top-N of the cheap ranking
    # reaches the full-scan number exactly (N=500 -> 0.40/0.27), so ITM
    # enters as EVIDENCE INSIDE A CANDIDATE SET (L7), never a corpus
    # scan: its vision pass is 0.4s/episode and its tokens are 3.24 GB
    # corpus-wide (unpoolable - 4x reduction drops rank correlation to
    # 0.18), so a scan is neither affordable nor storable.
    # Off by default: ELIDEDB_ITM=1 enables, because the cost is real.
    import os as _os2
    if _os2.environ.get("ELIDEDB_ITM") == "1":
        try:
            from .itm import itm_scores, rerank_depth
            n_re = rerank_depth(k_max, len(keys))
            cand = np.argsort(-fused)[:n_re]
            sc = itm_scores(store, text, [keys[i] for i in cand])
            if np.isfinite(sc).any():
                zc = np.zeros(len(keys))
                ok = np.isfinite(sc)
                v = sc[ok]
                zc[cand[ok]] = (v - v.mean()) / (v.std() + 1e-9)
                # SCORE-DISTRIBUTION-PRESERVING RESCORE. Writing
                # z-scores into `fused` broke the confidence cut, which
                # is fitted against RRF's own scale (bounded sums of
                # w/(60+rank)): q03 returned 83 of a 371 ceiling,
                # yield 0.87 -> 0.28. The cut is downstream and must
                # keep seeing the distribution it was fitted on, so the
                # rerank PERMUTES the candidates and hands back the
                # same sorted score values in the new order. Order
                # changes, scale does not.
                zf = fused[cand]
                zf = (zf - zf.mean()) / (zf.std() + 1e-9)
                new = np.argsort(-(zf + zc[cand]))
                f = fused.copy()
                f[cand[new]] = np.sort(fused[cand])[::-1]
                fused = f
                ch["itm"] = zc
        except Exception as e:
            _fail('itm', e)

    # NO-MATCH GATE, self-recognized: map the query onto the action
    # probe's OWN vocabulary by embedding similarity (no hand verb
    # list) and ask whether ANY episode in this corpus expresses those
    # classes above noise. Measured separation on bridge4h: absent
    # actions max 0.008-0.021 (fold/tear/throw) vs present 0.12-0.93;
    # threshold 0.05 sits in the gap. (A PE-cosine z-gate could not
    # separate — fold z 2.2 ranked ABOVE lid z 1.9.)
    gate = _auto_action_support(store, text)
    if gate is not None and gate["max_p"] < 0.05:
        ms = (time.perf_counter() - t0) * 1e3
        return {"clips": [], "borderline": [], "audit": None,
                "no_match": True, "reason": gate,
                "direction_filtered": 0, "channels": sorted(ch),
                "channels_failed": failed,
                "degraded": sorted(failed),
                "scored": len(keys), "ms": round(ms, 1)}

    # DIRECTION HARD FILTER — QUANTILE, NOT SIGN (AUC-validated
    # channels have uncalibrated zero points; a sign test executed
    # 130/183 true closes). Shared with the fitter via setpath.py: the
    # knee/obj-boost divergences of the acceptance sprint (fit LOQO
    # 0.21 vs live 0.16) were measured regressions from the live path
    # reshaping what the fit optimized — one code path kills the class.
    # FITTED VETO AUTHORITY: which channels filter is a per-store
    # learned artifact, not code. For binding queries this lets conj
    # act as a hard constraint (each query atom must find its own
    # frame evidence) instead of a drowned RRF vote — consensus
    # fusion structurally outvotes a decisive minority channel
    # (Cormack et al. 2009), and bag-of-concepts encoders cannot
    # rank binding (Winoground/ARO), so the constraint must prune.
    from .setpath import (confidence_cut, event_positions, filter_mask,
                          nms_keep)
    fsrc = dict(ch)
    fsrc.update(contrast_ch)
    if fnames is None:
        fnames = list(contrast_ch)
    alive = (filter_mask(fsrc, fnames, filter_q)
             if fnames else np.ones(len(keys), bool))
    dropped = int((~alive).sum())

    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])]
    if nms_r > 0:
        # temporal event dedup (fitted radius, shared with the fit):
        # duplicates of one event give way to the next-ranked
        # DISTINCT events — the product wants each true event once
        sid, pos = event_positions(keys)
        order = nms_keep(order, sid, pos, nms_r)
    # when roles are FITTED, the boundary is part of the fitted
    # configuration: the set ends where fused confidence drops below
    # the fitted alpha x the query's own top mass (setpath.
    # confidence_cut, shared with the fit — the hand knee here was a
    # measured fit-live divergence: fit LOQO 0.21 vs live 0.16). The
    # product ratio is true/returned -> returned/support; padding to
    # k_max buys yield with junk, and the fitted alpha prices that
    # trade on the truthset instead of a fixed count.
    cut = (confidence_cut(fused[order], cut_alpha, k_max)
           if sw.exists() else min(_knee(fused[order]), k_max))
    chosen = order[:cut]
    borderline = order[cut:cut + 20]

    audit = None
    # The audit stays TIER-2 (opt-in): running it by default was
    # measured 2026-07-27 — prec +0.01, yield -0.07, with the damage
    # concentrated where X is a category word ("vessel") whose
    # corpus-attested variants ground spurious objects and the
    # geometry kills then execute true clips. With the quorum rule
    # and clean relation phrases it is far safer than before, but the
    # fast tier's fused index is the better default by the ledger.
    if purity == "audited" and len(chosen) > 0 and rel is not None:
        try:
            audit, keep_mask = _binding_audit(store, rel,
                                              [keys[i] for i in chosen])
        except Exception as e:
            # deployments without the tracker stack keep the fused set
            # and SAY so rather than failing the query
            audit = {"unavailable": f"{type(e).__name__}"}
            keep_mask = None
        if keep_mask is not None:
            killed = chosen[~keep_mask]
            borderline = np.concatenate([killed, borderline])
            chosen = chosen[keep_mask]

    ms = (time.perf_counter() - t0) * 1e3
    return {
        "clips": [{"stream": keys[i][0], "t0": keys[i][1],
                   "t1": keys[i][2], "score": float(fused[i])}
                  for i in chosen],
        "borderline": [{"stream": keys[i][0], "t0": keys[i][1],
                        "t1": keys[i][2]} for i in borderline],
        "audit": audit,
        "direction_filtered": dropped,
        "channels": sorted(ch),
        # a channel the fitted configuration gives ordering or filter
        # authority to, that failed to compute: the answer is degraded
        # and the caller must be able to see it (see `failed` above)
        "channels_failed": failed,
        "degraded": sorted(c for c in failed
                           if abs(weights.get(c, 1.0)) > 0
                           or c in (fnames or ())),
        "scored": len(keys),
        "ms": round(ms, 1),
        # full fused ordering, for diagnosis: it separates "the ranking
        # never found the true episodes" from "it found them and the cut
        # refused to return them" - two failures with opposite fixes.
        **({"ranking": [(keys[i][0], keys[i][1], float(fused[i]))
                        for i in order]} if return_ranking else {}),
    }


# closed-class color words -> OpenCV hue bands (H in 0..180); S/V
# floors exclude gray/white. Deterministic pixel evidence from the
# tracked masklet — no model, no metadata, fully explainable.
_HUE = {"red": [(0, 10), (170, 180)], "orange": [(10, 20)],
        "yellow": [(20, 33)], "green": [(35, 85)],
        "blue": [(95, 130)], "purple": [(130, 165)],
        "pink": [(150, 175)]}


def _mask_color_frac(store, stream, t0, t1, tr_x, color):
    """Fraction of the tracked X masklet's pixels in the color band,
    measured on the MOVER identity at its most confident frame (the
    object that acted — a static, genuinely-green bystander must not
    vouch for a clip where the yellow cheese did the moving)."""
    import cv2
    import pyarrow.compute as pc

    from .video import FrameSet
    if tr_x.get("mover_mask") is not None:
        best_f = int(tr_x["mover_frame"])
        m0 = tr_x["mover_mask"]
    else:
        best_f = int(np.argmax(tr_x["presence"]))
        m0 = tr_x["masks"][best_f]
    if m0 is None:
        return None
    frames_tbl = store.table("frames").scan()
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), stream),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), t0),
                pc.less_equal(frames_tbl.column("ts"), t1))))
    if len(sel) < 4:
        return None
    n = len(tr_x["presence"])
    pick = np.linspace(0, len(sel) - 1, n).round().astype(int)
    dec = FrameSet(store, "frames",
                   sel.take(pick[best_f:best_f + 1])).decode(width=480)
    if not dec:
        return None
    img = sorted(dec)[0][1]
    m = m0
    if m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1],
                                            img.shape[0])) > 0
    if m.sum() < 20:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0][m], hsv[..., 1][m], hsv[..., 2][m]
    ok = np.zeros(len(h), bool)
    for lo, hi in _HUE[color]:
        ok |= (h >= lo) & (h <= hi)
    ok &= (s > 60) & (v > 50)
    return float(ok.mean())


def _binding_audit(store, rel, clip_keys):
    """SAM 3.1 tracker BINDING audit on every returned clip: does the
    queried OBJECT actually appear, and does it engage the LANDMARK?

    Division of labor fixed by measurement: DIRECTION belongs to the
    index channels (mot 0.98 open/close, act 0.889 put/take, free);
    the tracker's occlusion-persistent memory made it a poor direction
    instrument (AUC 0.643) but a robust IDENTITY one — exactly the
    binding failures the product bench showed (wrong colors, wrong
    objects). A clip is killed only on positive evidence of absence:
    the X masklet never appears, or X and Y masklets never come near
    each other. Tracker failure on a clip = abstain = keep.
    """
    from .grounding import _ioa
    from .sam3x import track_concepts

    def _concrete(p):
        """None for placeholder-only phrases ('the object'); an
        ATTRIBUTE-bearing phrase ('a green object') is groundable —
        the blanket 'object'-in-phrase test silently skipped the whole
        X audit on every attribute query (debug-caught: any_moved and
        color fracs were real, the audit just never asked)."""
        if not p:
            return None
        words = [w for w in p.split()
                 if w not in ("a", "an", "the", "object", "objects",
                              "something", "thing")]
        return p if words else None
    x, _, y = rel
    x, y = _concrete(x), _concrete(y)
    if x is None and y is None:
        return None, None
    def _variants(p):
        if p is None:
            return [None]
        try:
            from .vocab import corpus_variants
            return corpus_variants(store, p)
        except Exception:
            return [p]

    keep = np.ones(len(clip_keys), bool)
    checked = killed_absent = killed_disjoint = abstained = 0
    killed_static = killed_wrong_color = 0
    absent_idx = []
    for i, (s, a, b) in enumerate(clip_keys):
        # try phrase variants until X grounds (category words like
        # "vessel" ground as pot/pan/bowl); first grounding wins
        tr = None
        for xv in _variants(x):
            phrases = [p for p in (xv, y) if p]
            try:
                trv = track_concepts(store, s, a, b, phrases,
                                     n_frames=8)
            except Exception:
                trv = None
            if trv is None:
                continue
            if tr is None:
                tr, xg = trv, xv
            if xv is None or max(trv[xv]["presence"]) >= 0.5:
                tr, xg = trv, xv
                break
        if tr is None:
            abstained += 1
            continue
        # xk = the phrase key that grounded for THIS clip (x itself
        # stays loop-invariant — an earlier version mutated it and
        # corrupted later iterations' variant lists)
        xk = xg if x is not None else None
        checked += 1
        if xk is not None and max(tr[xk]["presence"]) < 0.5:
            keep[i] = False
            killed_absent += 1
            absent_idx.append(i)
            continue
        # THE MANIPULATED-OBJECT TEST: SOME instance of the queried X
        # must MOVE. "a green object" grounds on any green thing in
        # the scene (audit-bench-caught: zero kills on the green/
        # yellow sets); the query is about the object being ACTED ON,
        # and that one travels. any_moved spans ALL tracked identities
        # so a second, static instance never executes a true clip.
        if xk is not None and not tr[xk].get("any_moved", True):
            keep[i] = False
            killed_static += 1
            continue
        # COLOR CHECK, pixel-level: SAM 3's presence token accepts a
        # yellow-green cheese for "a green object" (audit-bench-caught,
        # zero kills on attribute queries) — but the masklet hands us
        # the object's PIXELS, and color words are closed-class. The
        # object claimed as <color> must actually be <color>.
        color = next((c for c in _HUE if xk and c in xk), None)
        if color is not None:
            frac = _mask_color_frac(store, s, a, b, tr[xk], color)
            if frac is not None and frac < 0.25:
                keep[i] = False
                killed_wrong_color += 1
                continue
        if xk is not None and y is not None \
                and max(tr[y]["presence"]) >= 0.5:
            near = False
            for f in range(len(tr[xk]["boxes"])):
                bx, by = tr[xk]["boxes"][f], tr[y]["boxes"][f]
                if bx is None or by is None:
                    continue
                # engagement: overlap, or gap under half of X's size
                if _ioa(bx, by) > 0.02:
                    near = True
                    break
                gap = max(by[0] - bx[2], bx[0] - by[2],
                          by[1] - bx[3], bx[1] - by[3])
                if gap < 0.5 * max(bx[2] - bx[0], bx[3] - bx[1]):
                    near = True
                    break
            if not near:
                keep[i] = False
                killed_disjoint += 1
    # GROUNDING-RELIABILITY QUORUM (generalizes the old 100%-absent
    # rule): absence is only evidence when the phrase grounds in at
    # least half the checked clips. A phrase the detector cannot find
    # ("a vessel": grounded 1/10, one spurious static bottle — the
    # single grounding defeated the 100% rule and the audit executed
    # a 9/10-true set, measured) indicts the GROUNDING, not the
    # clips: revert only the absent kills. Kills where grounding
    # SUCCEEDED (static/color/disjoint) always stand.
    grounded = checked - killed_absent
    ungroundable = (checked > 0 and grounded < killed_absent)
    if ungroundable:
        for j in absent_idx:
            keep[j] = True
        killed_absent = 0
    return ({"checked": checked, "killed_absent": killed_absent,
             "killed_disjoint": killed_disjoint,
             "killed_static": killed_static,
             "killed_wrong_color": killed_wrong_color,
             "abstained": abstained,
             "ungroundable": ungroundable,
             "x": x, "y": y}, keep)
