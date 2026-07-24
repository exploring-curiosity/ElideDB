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


def _knee(sorted_desc):
    """Set boundary on the sorted fused curve. The raw largest-drop
    knee cut 183-episode classes to 4 (RRF consensus gives the top few
    an outsized gap): the boundary is now the last position whose score
    keeps a fixed fraction of the top-5 mean — scale-stable under RRF
    (scores are bounded sums of w/(60+rank)) — with the largest-drop
    knee only allowed to TIGHTEN it, never to cut inside the top-8."""
    s = np.asarray(sorted_desc, float)
    if len(s) < 6:
        return len(s)
    top = float(np.mean(s[:5]))
    floor_cut = int(np.searchsorted(-s, -0.62 * top, side="right"))
    d = s[:-1] - s[1:]
    knee = int(np.argmax(d[8:])) + 8 + 1 if len(d) > 8 else len(s)
    return max(min(floor_cut, knee), 8)


def search_set(store, text, purity="fast", k_max=400, audit_n=12):
    """The robotics query: ALL matching clips, purity-first, VLM-free.

    purity="fast"    exact fused scan, direction filter, knee cut
    purity="audited" + geometric relational audit (SAM 3 boxes over
                     time) on a stratified sample of the set; pruned at
                     the last geometry-positive sample. Only fires when
                     the query parses as a relation; abstains otherwise.
    """
    from .fusion import rrf
    from .rerank import directional_swap
    t0 = time.perf_counter()
    keys = _episodes(store)
    sq = directional_swap(text)

    ch = {}
    try:
        from .pe import pe_lookup
        look, _ = pe_lookup(store, text)
        ch["pe"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    try:
        from .action_channel import act_lookup
        look, _ = act_lookup(store, text)
        ch["act"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    try:
        from .vid import vid_lookup
        look, _ = vid_lookup(store, text)
        ch["vid"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    mot = None
    if sq is not None:
        try:
            from .context import embed_texts
            from .motion import motion_lookup
            qv2 = embed_texts([text, sq])
            mlook = motion_lookup(store, qv2[0], qv2[1])
            mot = np.array([mlook(*k) for k in keys])
            ch["mot"] = mot
        except Exception:
            pass

    # learned per-store weights, routed by query type (same artifact
    # the ranked path uses)
    weights = {c: 1.0 for c in ch}
    try:
        from pathlib import Path
        cfg = json.loads((Path(store.dir) / "_channel_weights.json")
                         .read_text())
        learned = (cfg.get("weights_dir", cfg.get("weights", {}))
                   if sq is not None else cfg.get("weights", {}))
        weights = {c: float(learned.get(c, 1.0)) for c in ch}
        if sq is not None and "mot" in weights:
            weights["mot"] = max(weights["mot"], 1.0)
    except Exception:
        pass

    fused = rrf(ch, weights=weights)

    # DIRECTION HARD FILTER — QUANTILE, NOT SIGN. Both direction
    # channels were validated by AUC (an ORDERING property); their zero
    # point is uncalibrated — a sign test executed 130/183 true closes
    # (measured) because most true closes score mildly negative on both.
    # Rank properties get rank thresholds: drop only episodes that BOTH
    # channels place in the bottom third of their contrast orderings.
    # NaN = neutral (median), abstain never kills.
    dropped = 0
    alive = np.ones(len(keys), bool)
    if sq is not None:
        act = ch.get("act")
        if mot is not None and act is not None:
            def _rankfrac(v):
                r = np.full(len(v), 0.5)
                fin = np.isfinite(v)
                if fin.sum() > 1:
                    order = np.argsort(np.argsort(v[fin]))
                    r[fin] = order / (fin.sum() - 1)
                return r
            bad = (_rankfrac(mot) < 1 / 3) & (_rankfrac(act) < 1 / 3)
            alive &= ~bad
            dropped = int(bad.sum())

    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])]
    cut = min(_knee(fused[order]), k_max)
    chosen = order[:cut]
    borderline = order[cut:cut + 20]

    audit = None
    if purity == "audited" and len(chosen) >= 4:
        audit, keep = _geometry_audit(store, text,
                                      [keys[i] for i in chosen],
                                      n_sample=audit_n)
        if keep is not None:
            borderline = np.concatenate([chosen[keep:], borderline])
            chosen = chosen[:keep]

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
        "scored": len(keys),
        "ms": round(ms, 1),
    }


def _geometry_audit(store, text, clip_keys, n_sample=12):
    """SAM 3 containment-change audit on a stratified sample. Returns
    ({checked, judged, positive, abstained, pass_rate},
    prune_point|None); (None, None) when the query has no parseable
    relation — geometry only ever claims what geometry can see."""
    from .grounding import parse_relation, verify_relation
    if parse_relation(text) is None:
        return None, None
    n = len(clip_keys)
    pos_idx = sorted(set(np.linspace(0, n - 1, min(n_sample, n))
                         .round().astype(int)))
    sample = [clip_keys[i] for i in pos_idx]
    margins = verify_relation(store, text, sample)
    if margins is None:
        return None, None
    abstained = int(np.isnan(margins).sum())
    judged = int(np.isfinite(margins).sum())
    passed = int((margins[np.isfinite(margins)] > 0).sum())
    keep = None
    fin = np.isfinite(margins)
    if judged and not (margins[fin] > 0).all():
        good = [p for p, m in zip(pos_idx, margins)
                if np.isfinite(m) and m > 0]
        keep = (max(good) + 1) if good else 0
    return ({"checked": len(sample), "judged": judged,
             "positive": passed, "abstained": abstained,
             "pass_rate": round(passed / judged, 2) if judged
             else None}, keep)
