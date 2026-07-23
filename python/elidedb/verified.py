"""Verified search: any query, action or not, on any store.

WHY THIS SHAPE (from the measured root cause)
---------------------------------------------
Verbs are erased at both ends of embedding retrieval: the text tower scores
cos("close the drawer","open the drawer")=0.951, and even the full teacher's
frame embeddings separate close/open/put-in at 0.60 — a coin flip. So no
amount of embedding work can enforce "the green object is the SUBJECT of the
action" or "and close it". The only generic component in the system that can
read an EVENT is a VLM shown frames in time order.

The database answer is the same as ever: cheap operators propose, the
expensive operator disposes, and its work is cached.

    RECALL  (cheap, high recall on purpose)
        union of appearance rankings — the full query plus each comma/AND
        atom ("green object", "the drawer"), so a clip that is strong on any
        ingredient reaches the pool. If VLM captions exist in the store, the
        lexical ranking joins the union (verbs live in words).
        The union fixes what killed sharp search: one weak ranker never
        gates the candidate set.

    VERIFY  (expensive, precise, LAST)
        4 frames spanning each candidate segment, in time order, judged by
        the VLM with the whole sentence: logP(yes)-logP(no). Subject-of-
        action and sequential clauses ("...and close it") are enforced here
        — the one place they exist.

    CACHE   (cracking)
        every verdict is written to `vlm_verdicts` keyed by (window, query
        hash). Repeats and refinements of hot queries get cheaper; cold
        corpus costs nothing.

Nothing here is dataset-specific: no object lists, no robot fields, no
task vocabulary. The prompts are the user's own words.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

_VERDICTS: dict = {}
# Background verification: one worker at a time, one job per (store, query).
# The QUERY never waits for a model — verification is index MAINTENANCE that
# happens to be triggered by traffic (cracking without query-time latency).
_BG_LOCK = threading.Lock()
_BG_INFLIGHT: set = set()


def _qhash(text: str) -> int:
    return int.from_bytes(
        hashlib.sha1(text.strip().lower().encode()).digest()[:8],
        "little", signed=True)


def _verdict_map(store):
    t = store.table("vlm_verdicts")
    st = t.state()
    key = (str(store.dir), st.version)
    if key not in _VERDICTS:
        m = {}
        if st.files:
            tb = t.scan()
            for s, a, q, v in zip(tb.column("stream").to_pylist(),
                                  tb.column("ts").to_pylist(),
                                  tb.column("qhash").to_pylist(),
                                  tb.column("margin").to_pylist()):
                m[(s, int(a), int(q))] = float(v)
        _VERDICTS.clear()
        _VERDICTS[key] = m
    return _VERDICTS[key]


def _atoms(text: str):
    """The query plus its ingredients. 'put a green object in the drawer and
    close it' -> the full sentence, 'put a green object in the drawer',
    'close it'. Each is embedded separately for RECALL only — precision is
    the verifier's job, so over-splitting costs nothing but candidates."""
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r",|\band then\b|\bthen\b|\band\b|\bAND\b", t)
    atoms = [p.strip() for p in parts if len(p.strip().split()) >= 2]
    return [t] + [a for a in atoms if a.lower() != t.lower()]


_CAP_CACHE = {}


def _caption_candidates(store, text, n):
    """Lexical recall over VLM captions, when the store has them. Verbs live
    in words, so this is the ranker most likely to surface action matches.

    The fitted TF-IDF + document matrix are cached per table VERSION —
    refitting on every query was 50 ms of an 81 ms query (measured), i.e.
    the entire latency gate spent recomputing something that only changes
    when captions are appended. A query now pays one sparse transform."""
    try:
        ver = store.table("context_captions").state().version
    except Exception:
        return []
    key = (str(store.dir), ver)
    if key not in _CAP_CACHE:
        caps = store.table("context_captions").scan()
        if len(caps) == 0:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = caps.column("caption").to_pylist()
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        wins = list(zip(caps.column("stream").to_pylist(),
                        (int(v) for v in caps.column("ts").to_pylist()),
                        (int(v) for v in caps.column("t1").to_pylist())))
        if len(_CAP_CACHE) > 8:
            _CAP_CACHE.clear()
        _CAP_CACHE[key] = (vec, X, wins)
    vec, X, wins = _CAP_CACHE[key]
    q = vec.transform([text])
    sc = np.asarray((X @ q.T).todense()).ravel()
    order = np.argsort(-sc)[:n]
    return [(wins[int(i)], float(sc[i])) for i in order if sc[i] > 0]


_ADAPTER_CACHE = {}


def _ctx_event_scores(store, qv):
    """(stream, t0, t1, ctx-cosine) for every event in the V2 index.

    The adapter is the query-time half of stage B's contrastive pair: `qv`
    is the SigLIP text embedding the appearance ranker ALREADY computed
    (verified identical to the tower stage B trained with, cos 1.0); the
    adapter maps it to the 256-d ctx space. Loaded once per store, then a
    query costs one 2-layer forward + one matmul — no second tower pass.
    """
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "context_events")
    key = str(store.dir)
    if key not in _ADAPTER_CACHE:
        import mlx.core as mx
        from .fdnnv2 import TextAdapter
        from mlx.utils import tree_unflatten
        z = np.load(Path(store.dir) / "models" / "fdnnv2" / "adapter.npz")
        adapter = TextAdapter()
        adapter.update(tree_unflatten([(k_, mx.array(z[k_]))
                                       for k_ in z.files]))
        mx.eval(adapter.parameters())
        _ADAPTER_CACHE[key] = adapter
    import mlx.core as mx
    qc = np.array(_ADAPTER_CACHE[key](mx.array(qv)[None]))[0]
    sc = vecs @ qc
    ss = tbl.column("stream").to_pylist()
    sa = tbl.column("ts").to_pylist()
    sb = tbl.column("t1").to_pylist()
    return [((s, int(a), int(b)), float(c))
            for s, a, b, c in zip(ss, sa, sb, sc)]


def search_verified(store, text, k=8, pool=48, frames_per_clip=2,
                    pad_s=2.0, deep=0, t0=None, t1=None, streams=None,
                    verify="async", verbose=False):
    """Union recall -> segment building -> multi-frame VLM verification.

    Returns (hits, stats); each hit carries `margin` (the VLM's yes/no
    log-odds for the WHOLE event) and `cached` (whether the verdict was
    already in the store).
    """
    from PIL import Image

    from .embeddings import _vec_table
    from .rerank import (DEEP_VLM, as_change_question, as_clip_question,
                         score_clip_sequences)
    from .video import FrameSet
    t_start = time.perf_counter()

    tbl, vecs = _vec_table(store, "embeddings")
    w_t0 = tbl.column("ts").to_numpy()
    w_t1 = tbl.column("t1").to_numpy()
    w_s = tbl.column("stream").to_numpy(zero_copy_only=False)

    # hybrid predicates pushed INTO recall, as everywhere else in this store
    pred = np.ones(len(w_t0), bool)
    if t0 is not None:
        pred &= w_t1 >= t0
    if t1 is not None:
        pred &= w_t0 <= t1
    if streams:
        pred &= np.isin(w_s, list(streams))
    idx_all = np.where(pred)[0]
    if len(idx_all) == 0:
        return [], {"method": "verified", "candidates": 0, "segments": 0,
                    "ms": 0.0}

    # ---- RECALL: union over atoms + captions + context vectors ------------
    atoms = _atoms(text)
    per = max(pool // len(atoms), 12)
    cand = {}
    # ONE tower pass for the query and all its atoms: per-atom embed_text
    # calls cost ~21 ms EACH, which put every compound query over the
    # latency gate (measured 84-106 ms); a batch of 3 costs the same as 1.
    from .context import embed_texts
    qvs = embed_texts(atoms)
    q_full = qvs[0]
    app_all = vecs[idx_all] @ q_full          # index signal, reused below
    app_of = {int(i): float(sc) for i, sc in zip(idx_all, app_all)}
    for a, qv in zip(atoms, qvs):
        sub = idx_all[np.argsort(-(vecs[idx_all] @ qv))[:per]]
        for i in sub:
            cand.setdefault((str(w_s[i]), int(w_t0[i]), int(w_t1[i])),
                            app_of.get(int(i), 0.0))
    # lexical scores are KEPT, not just used for recall: a caption that
    # says the verb is the strongest index evidence this store has for an
    # action query, and dropping its score buried caption hits at the
    # bottom of the unverified ordering (measured on the verb battery).
    lex_of = {}
    for w, sc in _caption_candidates(store, text, per):
        if streams and w[0] not in streams:
            continue
        cand.setdefault(w, 0.0)
        lex_of[w] = max(lex_of.get(w, 0.0), sc)
    # stores with the full context tier contribute their caption-LSA ranking
    try:
        from .context import _ctx_matrix, caption_space
        if store.table("context").state().files:
            ctx = store.table("context").scan()
            cv = _ctx_matrix(store)
            qc = caption_space(store).transform([text])[0]
            cs = ctx.column("stream").to_pylist()
            ca = ctx.column("ts").to_pylist()
            cb = ctx.column("t1").to_pylist()
            for i in np.argsort(-(cv @ qc))[:per]:
                if streams and cs[i] not in streams:
                    continue
                cand.setdefault((cs[i], int(ca[i]), int(cb[i])), 0.0)
    except Exception:
        pass
    # stores with a V2 context-event index contribute VERB-AWARE ranking:
    # adapter(text) and novelty-pooled event vectors share the 256-d space
    # stage B trained, where "close" and "open" are different directions —
    # the one thing appearance cosines cannot express. Index-only: one
    # matmul plus a 2-layer adapter (~0.1 ms), no VLM.
    ctx_of = {}
    try:
        ev_scores = _ctx_event_scores(store, q_full)
        for (s_, a, b), sc in ev_scores:
            if streams and s_ not in streams:
                continue
            if t0 is not None and b < t0:
                continue
            if t1 is not None and a > t1:
                continue
            ctx_of[(s_, a, b)] = sc
        for w, sc in sorted(ctx_of.items(), key=lambda kv: -kv[1])[:per]:
            cand.setdefault(w, 0.0)
    except Exception:
        pass

    # ---- SEGMENTS: pad each window so the verifier sees the WHOLE event
    # (the close happens after the put; a bare window may hold only one) ----
    pad = int(pad_s * 1e9)
    by_stream = {}
    for (s, a, b) in cand:
        by_stream.setdefault(s, []).append((a - pad, b + pad))
    segs = []
    for s, spans in by_stream.items():
        spans.sort()
        cur = list(spans[0])
        for a, b in spans[1:]:
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                segs.append((s, cur[0], cur[1]))
                cur = [a, b]
        segs.append((s, cur[0], cur[1]))
    segs = segs[:pool]

    # ---- segment index score: best member window's appearance cosine,
    # plus (when the store has a V2 event index) best member ctx cosine.
    # NaN = the ctx ranker ABSTAINS on that segment; RRF gives it the
    # median rank rather than the bottom (see fusion.py on why).
    seg_score, seg_ctx, seg_lex = {}, {}, {}
    for (s_, a, b), sc in cand.items():
        for seg in segs:
            if seg[0] == s_ and seg[1] <= a and b <= seg[2]:
                seg_score[seg] = max(seg_score.get(seg, -1.0), sc)
                if (s_, a, b) in ctx_of:
                    seg_ctx[seg] = max(seg_ctx.get(seg, -2.0),
                                       ctx_of[(s_, a, b)])
                if (s_, a, b) in lex_of:
                    seg_lex[seg] = max(seg_lex.get(seg, 0.0),
                                       lex_of[(s_, a, b)])
                break
    for seg in segs:
        seg_score.setdefault(seg, 0.0)
        seg_ctx.setdefault(seg, float("nan"))
        seg_lex.setdefault(seg, float("nan"))

    qh = _qhash(text)
    vmap = _verdict_map(store)
    cached_m = {seg: vmap[(seg[0], seg[1], qh)] for seg in segs
                if (seg[0], seg[1], qh) in vmap}
    fresh = [seg for seg in segs if seg not in cached_m]

    if verify == "sync":
        # the old blocking path — scripts and evals that WANT to wait
        margins = dict(cached_m)
        margins.update(_verify_segments(store, fresh, text, qh,
                                        frames_per_clip))
        hits = [{"stream": s_, "t0": a, "t1": b,
                 "margin": round(margins[(s_, a, b)], 3),
                 "score": margins[(s_, a, b)],
                 "verified": True,
                 "cached": (s_, a, b) in cached_m}
                for (s_, a, b) in margins]
        hits.sort(key=lambda h: -h["score"])
        if deep and hits:
            hits = _deep_rerank(store, hits, text, deep, vmap)
        stats = {"method": "verified", "verify": "sync", "atoms": atoms,
                 "candidates": len(cand), "segments": len(segs),
                 "verified_fresh": len(fresh),
                 "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
        return hits[:k], stats

    # ---- ASYNC (default): the query is INDEX-ONLY — no model call ever ----
    # Ranking rule, deliberately explainable:
    #   1. verified positives, by the VLM's cached margin
    #   2. unverified candidates, by index score (they are what the
    #      background worker is judging right now)
    #   3. verified negatives last — the VLM looked and said no
    pos = [seg for seg, m in cached_m.items() if m >= 0]
    neg = [seg for seg, m in cached_m.items() if m < 0]
    pos.sort(key=lambda g: -cached_m[g])
    # unverified order: appearance, ctx, and caption-lexical fused by RRF —
    # scale-free, and a segment several rankers like beats one a single
    # ranker likes. NaN = that ranker abstains (median rank, no veto).
    if fresh:
        from .fusion import rrf
        fused = rrf({"app": np.array([seg_score[g] for g in fresh]),
                     "ctx": np.array([seg_ctx[g] for g in fresh]),
                     "lex": np.array([seg_lex[g] for g in fresh])})
        fresh = [g for _, g in sorted(zip(-fused, fresh))]
    neg.sort(key=lambda g: -cached_m[g])

    def _hit(seg, verified):
        return {"stream": seg[0], "t0": seg[1], "t1": seg[2],
                "margin": round(cached_m[seg], 3) if verified else None,
                "score": cached_m.get(seg, seg_score[seg]),
                "verified": verified, "cached": verified}
    hits = ([_hit(g, True) for g in pos]
            + [_hit(g, False) for g in fresh]
            + [_hit(g, True) for g in neg])

    launched = False
    if fresh:
        key = (str(store.dir), qh)
        with _BG_LOCK:
            if key not in _BG_INFLIGHT:
                _BG_INFLIGHT.add(key)
                launched = True
        if launched:
            todo = fresh[:pool]

            def worker():
                try:
                    _verify_segments(store, todo, text, qh, frames_per_clip)
                except Exception:
                    pass
                finally:
                    with _BG_LOCK:
                        _BG_INFLIGHT.discard(key)
            threading.Thread(target=worker, daemon=True).start()

    stats = {"method": "verified", "verify": "async", "atoms": atoms,
             "candidates": len(cand), "segments": len(segs),
             "verified_cached": len(cached_m),
             "verifying_in_background": len(fresh) if launched else 0,
             "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    return hits[:k], stats


def _verify_segments(store, segs, text, qh, frames_per_clip=2):
    """Decode start/end frames, score with the 2B before/after question,
    append verdicts to the store. Used sync by evals, async by queries."""
    from PIL import Image

    from .rerank import as_change_question, as_clip_question, \
        score_clip_sequences
    from .video import FrameSet
    if not segs:
        return {}
    question = (as_change_question(text) if frames_per_clip == 2
                else as_clip_question(text))
    frames_tbl = store.table("frames").scan()
    rot = store.meta.get("display", {}).get("rotate", 0)
    need, clips = [], []
    for seg in segs:
        s, a, b = seg
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            continue
        pick = np.linspace(0, len(sel) - 1,
                           min(frames_per_clip, len(sel))).round().astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=448)
        if len(dec) < 2:
            continue
        imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
        if rot:
            imgs = [im.rotate(rot, expand=True) for im in imgs]
        need.append(seg)
        clips.append(imgs)
    if not clips:
        return {}
    margins = {}
    for seg, m in zip(need, score_clip_sequences(clips, question)):
        margins[seg] = float(m)
    # `query` is stored as TEXT, not only qhash: verdicts are the one verb
    # supervision source that survived measurement (yes/no margins, AUC
    # 0.75-0.82, where free-form captions failed at 1-8/24), and as
    # (text, segment, ±margin) triples they can retrain the ctx adapter.
    # A hash can rank cached answers; only the text can teach.
    vt = pa.table({
        "ts": pa.array([g[1] for g in need], pa.int64()),
        "t1": pa.array([g[2] for g in need], pa.int64()),
        "stream": pa.array([g[0] for g in need]),
        "qhash": pa.array([qh] * len(need), pa.int64()),
        "margin": pa.array([margins[g] for g in need], pa.float32()),
        "query": pa.array([text] * len(need)),
    })
    try:
        store.table("vlm_verdicts").append(
            vt, kind="timeseries", evolve=True,
            meta={"written_by": "verified search", "query": text[:120]})
    except Exception:
        pass                                   # concurrent commit: retryable
    vmap = _verdict_map(store)
    for g in need:
        vmap[(g[0], g[1], qh)] = margins[g]
    return margins


def _deep_rerank(store, hits, text, deep, vmap):
    """7B judge over the top hits — sync callers only."""
    from PIL import Image

    from .rerank import DEEP_VLM, as_clip_question, score_clip_sequences
    from .video import FrameSet
    frames_tbl = store.table("frames").scan()
    dqh = _qhash("deep:" + text)
    head = hits[:deep]
    keep, clips7 = [], []
    for h in head:
        key = (h["stream"], h["t0"], dqh)
        if key in vmap:
            h["deep_margin"] = round(vmap[key], 3)
            h["score"] = vmap[key]
            continue
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), h["stream"]),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), h["t0"]),
                    pc.less_equal(frames_tbl.column("ts"), h["t1"]))))
        if len(sel) < 2:
            continue
        pick = np.linspace(0, len(sel) - 1, min(4, len(sel))).round() \
            .astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=448)
        if len(dec) < 2:
            continue
        clips7.append([Image.fromarray(d[1]) for d in sorted(dec)])
        keep.append(h)
    if clips7:
        deep_m = score_clip_sequences(clips7, as_clip_question(text),
                                      model_id=DEEP_VLM)
        rows = []
        for h, m in zip(keep, deep_m):
            h["deep_margin"] = round(float(m), 3)
            h["score"] = float(m)
            vmap[(h["stream"], h["t0"], dqh)] = float(m)
            rows.append(h)
        if rows:
            try:
                store.table("vlm_verdicts").append(pa.table({
                    "ts": pa.array([h["t0"] for h in rows], pa.int64()),
                    "t1": pa.array([h["t1"] for h in rows], pa.int64()),
                    "stream": pa.array([h["stream"] for h in rows]),
                    "qhash": pa.array([dqh] * len(rows), pa.int64()),
                    "margin": pa.array([h["score"] for h in rows],
                                       pa.float32()),
                }), kind="timeseries", meta={"written_by": "deep verify",
                                             "query": text[:120]})
            except Exception:
                pass
    head.sort(key=lambda h: -h["score"])
    return head + hits[deep:]