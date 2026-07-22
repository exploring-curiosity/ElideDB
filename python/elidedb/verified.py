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
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

_VERDICTS: dict = {}


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


def _caption_candidates(store, text, n):
    """Lexical recall over VLM captions, when the store has them. Verbs live
    in words, so this is the ranker most likely to surface action matches."""
    try:
        caps = store.table("context_captions").scan()
    except Exception:
        return []
    if len(caps) == 0:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = caps.column("caption").to_pylist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    q = vec.transform([text])
    sc = np.asarray((X @ q.T).todense()).ravel()
    order = np.argsort(-sc)[:n]
    return [(caps.column("stream")[int(i)].as_py(),
             int(caps.column("ts")[int(i)].as_py()),
             int(caps.column("t1")[int(i)].as_py()))
            for i in order if sc[i] > 0]


def search_verified(store, text, k=8, pool=48, frames_per_clip=2,
                    pad_s=2.0, deep=0, verbose=False):
    """Union recall -> segment building -> multi-frame VLM verification.

    Returns (hits, stats); each hit carries `margin` (the VLM's yes/no
    log-odds for the WHOLE event) and `cached` (whether the verdict was
    already in the store).
    """
    from PIL import Image

    from .embeddings import _vec_table, embed_text
    from .rerank import (DEEP_VLM, as_change_question, as_clip_question,
                         score_clip_sequences)
    from .video import FrameSet
    t_start = time.perf_counter()

    tbl, vecs = _vec_table(store, "embeddings")
    w_t0 = tbl.column("ts").to_numpy()
    w_t1 = tbl.column("t1").to_numpy()
    w_s = tbl.column("stream").to_numpy(zero_copy_only=False)

    # ---- RECALL: union over atoms + captions ------------------------------
    atoms = _atoms(text)
    per = max(pool // len(atoms), 12)
    cand = {}
    for a in atoms:
        qv = embed_text(a)
        for i in np.argsort(-(vecs @ qv))[:per]:
            cand.setdefault((str(w_s[i]), int(w_t0[i]), int(w_t1[i])),
                            0.0)
    for w in _caption_candidates(store, text, per):
        cand.setdefault(w, 0.0)

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

    # ---- VERIFY: 4 ordered frames per segment, verdicts cached ------------
    qh = _qhash(text)
    vmap = _verdict_map(store)
    # default verifier: 2B on (first, last) frame — AUC 0.75 at 0.5 s/clip.
    # frames_per_clip > 2 switches to the time-order question (for the deep
    # tier's 7B, where it measured AUC 0.82).
    question = (as_change_question(text) if frames_per_clip == 2
                else as_clip_question(text))
    frames_tbl = store.table("frames").scan()
    need, clips, margins, cached = [], [], {}, {}
    for seg in segs:
        key = (seg[0], seg[1], qh)
        if key in vmap:
            margins[seg] = vmap[key]
            cached[seg] = True
            continue
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
        rot = store.meta.get("display", {}).get("rotate", 0)
        imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
        if rot:
            imgs = [im.rotate(rot, expand=True) for im in imgs]
        need.append(seg)
        clips.append(imgs)

    t_vlm = time.perf_counter()
    if clips:
        for seg, m in zip(need, score_clip_sequences(clips, question)):
            margins[seg] = float(m)
            cached[seg] = False
        vt = pa.table({
            "ts": pa.array([s[1] for s in need], pa.int64()),
            "t1": pa.array([s[2] for s in need], pa.int64()),
            "stream": pa.array([s[0] for s in need]),
            "qhash": pa.array([qh] * len(need), pa.int64()),
            "margin": pa.array([margins[s] for s in need], pa.float32()),
        })
        store.table("vlm_verdicts").append(
            vt, kind="timeseries",
            meta={"written_by": "verified search", "query": text[:120]})
        for seg in need:
            vmap[(seg[0], seg[1], qh)] = margins[seg]
    vlm_ms = (time.perf_counter() - t_vlm) * 1e3

    hits = [{"stream": s, "t0": a, "t1": b, "margin": round(margins[(s, a, b)], 3),
             "score": margins[(s, a, b)], "cached": cached.get((s, a, b), True)}
            for (s, a, b) in margins]
    hits.sort(key=lambda h: -h["margin"])

    # ---- optional DEEP tier: 7B, 4 ordered frames, top candidates only ----
    if deep and hits:
        head = hits[:deep]
        clips7 = []
        keep = []
        for h in head:
            s7, a7, b7 = h["stream"], h["t0"], h["t1"]
            sel = frames_tbl.filter(pc.and_(
                pc.equal(frames_tbl.column("stream"), s7),
                pc.and_(pc.greater_equal(frames_tbl.column("ts"), a7),
                        pc.less_equal(frames_tbl.column("ts"), b7))))
            if len(sel) < 2:
                continue
            pick = np.linspace(0, len(sel) - 1,
                               min(4, len(sel))).round().astype(int)
            dec = FrameSet(store, "frames", sel.take(pick)).decode(width=448)
            if len(dec) < 2:
                continue
            clips7.append([Image.fromarray(d[1]) for d in sorted(dec)])
            keep.append(h)
        # deep verdicts cache under their own key space, so a warm repeat
        # skips the 7B entirely
        dqh = _qhash("deep:" + text)
        fresh7, fresh_clips = [], []
        for h, c in zip(keep, clips7):
            key = (h["stream"], h["t0"], dqh)
            if key in vmap:
                h["deep_margin"] = round(vmap[key], 3)
                h["score"] = vmap[key]
            else:
                fresh7.append(h)
                fresh_clips.append(c)
        if fresh_clips:
            deep_m = score_clip_sequences(fresh_clips, as_clip_question(text),
                                          model_id=DEEP_VLM)
            for h, m in zip(fresh7, deep_m):
                h["deep_margin"] = round(float(m), 3)
                h["score"] = float(m)
                vmap[(h["stream"], h["t0"], dqh)] = float(m)
            store.table("vlm_verdicts").append(pa.table({
                "ts": pa.array([h["t0"] for h in fresh7], pa.int64()),
                "t1": pa.array([h["t1"] for h in fresh7], pa.int64()),
                "stream": pa.array([h["stream"] for h in fresh7]),
                "qhash": pa.array([dqh] * len(fresh7), pa.int64()),
                "margin": pa.array([h["score"] for h in fresh7],
                                   pa.float32()),
            }), kind="timeseries", meta={"written_by": "deep verify",
                                         "query": text[:120]})
        keep.sort(key=lambda h: -h["score"])
        hits = keep + hits[deep:]
    stats = {"method": "verified", "atoms": atoms,
             "candidates": len(cand), "segments": len(segs),
             "verified_fresh": len(need),
             "verified_cached": sum(1 for h in hits if h["cached"]),
             "vlm_ms": round(vlm_ms, 1),
             "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    return hits[:k], stats
