"""SET RETRIEVAL — the robotics query model, seconds-scale.

The product truth (user-stated): a robotics team doesn't want the one
best clip, they want ALL clips matching a scenario, clean enough to
retrain on. That flips the design: precision of the DELIVERED SET is
king, recall is negotiable, latency budget is seconds (it is a database,
not a batch job).

Architecture (the IVF idea with learned cells, finally made physical):
  INGEST  per-recording PE vector (top-frame pooled, best measured
          encoder) -> HDBSCAN clusters = SCENARIO CELLS. Centroids +
          member permutation live as derived artifacts. Clusters double
          as a browsable "scenario map" of the corpus.
  QUERY   text -> PE text vector -> score ~K centroids (microseconds)
          -> open only the matching cells (elision: untouched cells are
          never read) -> exact scoring inside -> distribution knee sets
          the set boundary -> optional 2B SAMPLE AUDIT (~8 stratified
          clips, ~4 s) estimates purity WITHOUT labels and prunes the
          set to the requested purity.
  OUTPUT  {clips, scores, est_purity, borderline, cells} — counts and
          confidences, and an honest borderline pile, never junk mixed
          silently into the delivery.

Torn out per measurement: caption-trained ctx and subject-anchor
channels play no part here (measured harmful in composites); this path
is PE-primary with motion contrast for directional queries.
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
    """HDBSCAN over recording vectors -> scenario cells, persisted as a
    derived artifact beside the pe_vectors table."""
    t0 = time.time()
    keys, M, rowsets = _pool_recordings(store)
    # TWO clusterings, two jobs (HDBSCAN alone gave 4 density groups on
    # this space — a fine scenario MAP, a useless INDEX):
    #   scenario groups (HDBSCAN): the human-browsable map of what the
    #     corpus contains — density-true, few, nameable
    #   index cells (KMeans-IVF): the physical pruning layout — enough
    #     cells that opening a handful elides most of the corpus
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


def _load_cells(store):
    ver = store.table("pe_vectors").state().version
    key = (str(store.dir), ver)
    if key in _CELLS:
        return _CELLS[key]
    cell_dir = store.table("pe_vectors").dir / "_cache"
    meta = json.loads((cell_dir / "cells.json").read_text())
    if meta["version"] != ver:
        raise RuntimeError("scenario cells stale — run build_scenarios()")
    cents = np.load(cell_dir / "cell_centroids.npy")
    lab = np.load(cell_dir / "cell_labels.npy")
    M = np.load(cell_dir / "cell_matrix.npy", mmap_mode="r")
    keys = [(k[0], int(k[1]), int(k[2])) for k in meta["keys"]]
    by_cell = {}
    for i, c in enumerate(lab):
        by_cell.setdefault(int(c), []).append(i)
    if len(_CELLS) > 4:
        _CELLS.clear()
    _CELLS[key] = (keys, M, cents, lab, by_cell)
    return _CELLS[key]


def _knee(sorted_desc):
    """Largest relative drop in the sorted score curve = set boundary.
    Generic — no labels, no tuned constants beyond a minimum set size."""
    s = np.asarray(sorted_desc, float)
    if len(s) < 6:
        return len(s)
    d = s[:-1] - s[1:]
    j = int(np.argmax(d[3:])) + 3          # never cut inside the top-3
    return j + 1


def search_set(store, text, purity="audited", k_max=400, cells_top=6):
    """The robotics query: ALL matching clips, purity-first.

    purity="fast"    index-only (~0.1-0.3 s warm)
    purity="audited" + ~8-clip 2B sample audit (~4-6 s) -> est_purity,
                     and the set is pruned to the audited boundary
    """
    from .pe import _text_vec
    t0 = time.perf_counter()
    keys, M, cents, lab, by_cell = _load_cells(store)
    qv = _text_vec(text)

    # 1. cells: open only the scenario cells the query points at
    csc = cents @ qv
    open_cells = np.argsort(-csc)[:cells_top]
    members = []
    for c in open_cells:
        members += by_cell.get(int(c), [])
    members = np.asarray(sorted(set(members)))

    # 2. exact scoring inside the opened cells only (elision)
    sc = np.asarray(M[members]) @ qv
    order = np.argsort(-sc)
    cut = min(_knee(sc[order]), k_max)
    chosen = members[order[:cut]]
    chosen_sc = sc[order[:cut]]
    borderline = members[order[cut:min(cut + 20, len(order))]]

    # 3. directional queries: motion contrast re-ranks within the set
    from .rerank import directional_swap
    sq = directional_swap(text)
    if sq is not None:
        try:
            from .motion import motion_lookup
            from .context import embed_texts
            qv2 = embed_texts([text, sq])
            mlook = motion_lookup(store, qv2[0], qv2[1])
            m = np.array([mlook(*keys[i]) for i in chosen])
            m = np.nan_to_num(m, nan=np.nanmedian(m) if
                              np.isfinite(m).any() else 0.0)
            reorder = np.argsort(-(0.5 * (chosen_sc / (np.abs(chosen_sc)
                                  .max() + 1e-8)) + 0.5 * (m / (np.abs(m)
                                  .max() + 1e-8))))
            chosen = chosen[reorder]
            chosen_sc = chosen_sc[reorder]
        except Exception:
            pass

    est_purity = None
    if purity in ("audited", "deep") and len(chosen) >= 4:
        est_purity, keep = _sample_audit(store, text,
                                         [keys[i] for i in chosen],
                                         n_sample=6 if purity == "deep"
                                         else 8,
                                         deep=(purity == "deep"))
        if keep is not None:
            borderline = np.concatenate([chosen[keep:], borderline])
            chosen = chosen[:keep]
            chosen_sc = chosen_sc[:keep]

    ms = (time.perf_counter() - t0) * 1e3
    return {
        "clips": [{"stream": keys[i][0], "t0": keys[i][1],
                   "t1": keys[i][2], "score": float(s)}
                  for i, s in zip(chosen, chosen_sc)],
        "borderline": [{"stream": keys[i][0], "t0": keys[i][1],
                        "t1": keys[i][2]} for i in borderline],
        "est_purity": est_purity,
        "cells_opened": [int(c) for c in open_cells],
        "cells_total": int(len(cents)),
        "ms": round(ms, 1),
    }


def _sample_audit(store, text, clip_keys, n_sample=8, deep=False):
    """Judge a STRATIFIED sample with the 2B (before/after, swap-contrast
    when directional) -> purity estimate + a prune point. No labels; the
    judge looks at pixels. ~0.5 s per sampled clip."""
    import pyarrow.compute as pc
    from PIL import Image

    from .rerank import (as_change_question, directional_swap,
                         score_clip_sequences)
    from .video import FrameSet
    n = len(clip_keys)
    idx = sorted(set(np.linspace(0, n - 1, min(n_sample, n))
                     .round().astype(int)))
    frames_tbl = store.table("frames").scan()
    clips, pos = [], []
    for i in idx:
        s, a, b = clip_keys[i]
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            continue
        dec = FrameSet(store, "frames", sel.take(
            np.array([0, len(sel) - 1]))).decode(width=448)
        if len(dec) < 2:
            continue
        clips.append([Image.fromarray(d[1]) for d in sorted(dec)])
        pos.append(i)
    if not clips:
        return None, None
    from .rerank import DEEP_VLM
    kw = {"model_id": DEEP_VLM} if deep else {}
    q = as_change_question(text)
    m = np.array(score_clip_sequences(clips, q, **kw), float)
    sq = directional_swap(text)
    if sq:
        m = m - np.array(score_clip_sequences(
            clips, as_change_question(sq), **kw), float)
    ok = m > 0
    est = float(ok.mean())
    # prune to the last sampled position that still judged positive
    keep = None
    if not ok.all():
        good = [p for p, o in zip(pos, ok) if o]
        keep = (max(good) + 1) if good else 0
    return round(est, 2), keep
