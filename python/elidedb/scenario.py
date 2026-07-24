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


# generic English verb -> SSv2 class family (model vocabulary, nothing
# per-dataset). Only verbs with an unambiguous class mapping gate.
_VERB_CLASSES = {
    "fold": ("Folding something",),
    "unfold": ("Unfolding something",),
    "tear": ("Tearing something into two pieces",
             "Tearing something just a little bit"),
    "throw": ("Throwing something",),
    "pour": ("Pouring something into something",),
    "squeeze": ("Squeezing something",),
    "stack": ("Stacking number of something",),
}


def _action_support(store, text_lower):
    """{verb, classes, max_p} when the query names a gateable action;
    None otherwise. max_p = the corpus's best SSv2 posterior for the
    family — an index-only 'does this action happen here at all'."""
    hit = next((v for v in _VERB_CLASSES if v in text_lower), None)
    if hit is None:
        return None
    from .action_probe import ssv2_classes
    from .embeddings import _vec_table
    _, probs = _vec_table(store, "action_probs")
    ci = {c: i for i, c in enumerate(ssv2_classes())}
    cols = [ci[c] for c in _VERB_CLASSES[hit] if c in ci]
    mp = float(np.asarray(probs)[:, cols].max()) if cols else 1.0
    return {"verb": hit, "classes": list(_VERB_CLASSES[hit]),
            "max_p": round(mp, 4)}


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


def search_set(store, text, purity="fast", k_max=400, audit_n=12):
    """The robotics query: ALL matching clips, purity-first, VLM-free.

    purity="fast"    exact fused scan, direction filter, knee cut
    purity="audited" + geometric relational audit (SAM 3 boxes over
                     time) on a stratified sample of the set; pruned at
                     the last geometry-positive sample. Only fires when
                     the query parses as a relation; abstains otherwise.
    """
    from .fusion import rrf
    from .grounding import _INWARD, parse_relation
    from .rerank import directional_swap
    t0 = time.perf_counter()
    keys = _episodes(store)
    sq = directional_swap(text)

    # QUERY UNDERSTANDING, all mechanical: the parsed relation names
    # the containment direction relative to the landmark; open/close
    # verbs name the articulation direction. These select CANONICAL
    # literal-class contrasts for the act channel — the text-mapped
    # weights sent "open" onto the pulling-out family, which fires on
    # take-out clips (product-bench-caught).
    rel = parse_relation(text)
    tl = text.lower()
    verb_dir = ("close" if any(w in tl for w in
                               ("close", "closes", "closing", "shut"))
                else "open" if "open" in tl else None)
    containment = None
    if rel is not None and verb_dir is None:
        containment = "inward" if rel[1] in _INWARD else "outward"

    ch = {}
    try:
        from .pe import pe_lookup
        look, _ = pe_lookup(store, text)
        ch["pe"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    try:
        from .action_channel import act_lookup, canonical_contrast
        contrast = (canonical_contrast(containment or verb_dir)
                    if (containment or verb_dir) else None)
        look, _ = act_lookup(store, text, contrast=contrast)
        ch["act"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    try:
        from .vid import vid_lookup
        look, _ = vid_lookup(store, text)
        ch["vid"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    # OBJ channel — FastSAM crops matched against the query's noun
    # phrases (SigLIP space): the color/attribute binding the product
    # bench showed missing from the set path
    try:
        from .context import embed_texts
        from .objects import object_lookup
        nps = [p for p in ((rel[0], rel[2]) if rel else ())
               if p and "object" not in p]
        if not nps:
            import re as _re
            nps = [m.group(0) for m in _re.finditer(
                r"\b(?:a|an|the)\s+(?:\w+\s+){0,2}\w+", tl)][:2]
        if nps:
            olook = object_lookup(store, embed_texts(nps))
            def _obj(k):
                osc, omo = olook(*k)
                return osc * (1.0 + omo) if osc == osc else float("nan")
            ch["obj"] = np.array([_obj(k) for k in keys])
    except Exception:
        pass
    mot = None
    if sq is not None or verb_dir is not None:
        try:
            from .context import embed_texts
            from .motion import motion_lookup
            msq = sq or directional_swap(
                "opening the drawer" if verb_dir == "open"
                else "closing the drawer")
            if msq:
                qv2 = embed_texts([text, msq])
                mlook = motion_lookup(store, qv2[0], qv2[1])
                mot = np.array([mlook(*k) for k in keys])
                ch["mot"] = mot
        except Exception:
            pass

    # weights: learned per-store artifact, routed; for OPEN/CLOSE the
    # motion channel is the AUTHORITY (0.98 AUC, measured — the weight
    # fit diluted it chasing containment gains, breaking open 0/4)
    directional = sq is not None or verb_dir is not None
    weights = {c: 1.0 for c in ch}
    try:
        from pathlib import Path
        cfg = json.loads((Path(store.dir) / "_channel_weights.json")
                         .read_text())
        learned = (cfg.get("weights_dir", cfg.get("weights", {}))
                   if directional else cfg.get("weights", {}))
        weights = {c: float(learned.get(c, 1.0)) for c in ch}
    except Exception:
        pass
    if verb_dir is not None and "mot" in ch:
        weights["mot"] = max(weights.get("mot", 1.0), 8.0)
    elif directional and "mot" in weights:
        weights["mot"] = max(weights["mot"], 1.0)

    fused = rrf(ch, weights=weights)

    # NO-MATCH GATE, video-native: when the query names an ACTION and
    # the corpus's maximum SSv2 posterior for that action family is at
    # noise level, the action does not happen anywhere in this store —
    # the honest answer is the EMPTY set. Measured separation on
    # bridge4h: absent actions max 0.008-0.021 (fold/tear/throw) vs
    # present ones 0.12-0.93 (close/open/put-in); threshold 0.05 sits
    # in the gap. (A PE-cosine z-gate was tried first and could not
    # separate — fold z 2.2 ranked ABOVE lid z 1.9; cosine spread
    # compresses on compositional phrasings.)
    gate = _action_support(store, tl)
    if gate is not None and gate["max_p"] < 0.05:
        ms = (time.perf_counter() - t0) * 1e3
        return {"clips": [], "borderline": [], "audit": None,
                "no_match": True, "reason": gate,
                "direction_filtered": 0, "channels": sorted(ch),
                "scored": len(keys), "ms": round(ms, 1)}

    # DIRECTION HARD FILTER — QUANTILE, NOT SIGN (AUC-validated
    # channels have uncalibrated zero points; a sign test executed
    # 130/183 true closes). For open/close, motion ALONE decides
    # (bottom half dropped); for containment, both direction channels
    # must agree (bottom third AND).
    def _rankfrac(v):
        r = np.full(len(v), 0.5)
        fin = np.isfinite(v)
        if fin.sum() > 1:
            order = np.argsort(np.argsort(v[fin]))
            r[fin] = order / (fin.sum() - 1)
        return r

    dropped = 0
    alive = np.ones(len(keys), bool)
    act = ch.get("act")
    if verb_dir is not None and mot is not None:
        # bottom THIRD only: motion's 0.98 AUC is close-vs-OPEN
        # (pairwise); against the whole corpus true closes sit
        # mid-distribution and a half-cut executed them (bench-caught,
        # close 4/6 -> 2/7)
        bad = _rankfrac(mot) < 1 / 3
        alive &= ~bad
        dropped = int(bad.sum())
    elif containment is not None and act is not None:
        af = _rankfrac(act)
        if mot is not None:
            bad = (af < 1 / 3) & (_rankfrac(mot) < 1 / 3)
        else:
            bad = af < 1 / 4
        alive &= ~bad
        dropped = int(bad.sum())

    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])]
    cut = min(_knee(fused[order]), k_max)
    chosen = order[:cut]
    borderline = order[cut:cut + 20]

    audit = None
    if purity == "audited" and len(chosen) > 0 and rel is not None:
        audit, keep_mask = _binding_audit(store, rel,
                                          [keys[i] for i in chosen])
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
        "scored": len(keys),
        "ms": round(ms, 1),
    }


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
    x, _, y = rel
    x = None if (not x or "object" in x or "something" in x) else x
    y = None if (not y or "object" in y or "something" in y) else y
    if x is None and y is None:
        return None, None
    phrases = [p for p in (x, y) if p]
    keep = np.ones(len(clip_keys), bool)
    checked = killed_absent = killed_disjoint = abstained = 0
    killed_static = 0
    for i, (s, a, b) in enumerate(clip_keys):
        try:
            tr = track_concepts(store, s, a, b, phrases, n_frames=8)
        except Exception:
            tr = None
        if tr is None:
            abstained += 1
            continue
        checked += 1
        if x is not None and max(tr[x]["presence"]) < 0.5:
            keep[i] = False
            killed_absent += 1
            continue
        # THE MANIPULATED-OBJECT TEST: SOME instance of the queried X
        # must MOVE. "a green object" grounds on any green thing in
        # the scene (audit-bench-caught: zero kills on the green/
        # yellow sets); the query is about the object being ACTED ON,
        # and that one travels. any_moved spans ALL tracked identities
        # so a second, static instance never executes a true clip.
        if x is not None and not tr[x].get("any_moved", True):
            keep[i] = False
            killed_static += 1
            continue
        if x is not None and y is not None \
                and max(tr[y]["presence"]) >= 0.5:
            near = False
            for f in range(len(tr[x]["boxes"])):
                bx, by = tr[x]["boxes"][f], tr[y]["boxes"][f]
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
    # 100% absent = the PHRASE does not ground in this detector's
    # vocabulary ("a vessel" — audit-bench-caught killing the two true
    # pot-lifts along with everything else). Phrase failure is not clip
    # evidence: keep everything, report it.
    ungroundable = (checked > 0 and killed_absent == checked)
    if ungroundable:
        keep[:] = True
        killed_absent = 0
    return ({"checked": checked, "killed_absent": killed_absent,
             "killed_disjoint": killed_disjoint,
             "killed_static": killed_static,
             "abstained": abstained,
             "ungroundable": ungroundable,
             "x": x, "y": y}, keep)
