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


def search_set(store, text, purity="fast", k_max=400, audit_n=12):
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
    try:
        from .pe import pe_lookup
        vs = []
        for vtext in variants:
            look, _ = pe_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["pe"] = variant_max(vs)
    except Exception:
        pass
    try:
        # act: text-mapped class weights onto the probe's own
        # vocabulary (contrast against the swap when one exists) —
        # embedding-derived, no coded class lists
        from .action_channel import act_lookup
        look, _ = act_lookup(store, text)
        ch["act"] = np.array([look(*k) for k in keys])
    except Exception:
        pass
    try:
        from .sig2 import conj_lookup, sig2_lookup
        vs = []
        for vtext in variants:
            look, _ = sig2_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["sig2"] = variant_max(vs)
        cl = conj_lookup(store, text)
        if cl is not None:
            ch["conj"] = np.array([cl(*k) for k in keys])
    except Exception:
        pass
    try:
        from .vid import vid_lookup
        vs = []
        for vtext in variants:
            look, _ = vid_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["vid"] = variant_max(vs)
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
        except Exception:
            pass
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
        except Exception:
            pass

    directional = sq is not None
    weights = {c: 1.0 for c in ch}
    filter_q = 1 / 3
    fnames = None       # None => legacy: every contrast channel
    cut_alpha = 0.0     # 0 => fill to k_max (legacy, pre-cut)
    from pathlib import Path
    sw = Path(store.dir) / "_set_weights.json"
    try:
        if sw.exists():
            # FITTED roles (scripts/fit_set_weights.py): ordering
            # weights for every channel INCLUDING contrasts, plus the
            # filter quantile — coordinate ascent on the truthset,
            # LOQO-validated. Data-derived per store; the no-hardwire
            # rule's answer to hand role rules (the fit independently
            # rediscovered mot=0-in-ordering).
            cfg = json.loads(sw.read_text())
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
        else:
            cfg = json.loads((Path(store.dir)
                              / "_channel_weights.json").read_text())
            learned = (cfg.get("weights_dir", cfg.get("weights", {}))
                       if directional else cfg.get("weights", {}))
            weights = {c: float(learned.get(c, 1.0)) for c in ch}
    except Exception:
        pass
    fused = rrf(ch, weights=weights)

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
    from .setpath import confidence_cut, filter_mask
    fsrc = dict(ch)
    fsrc.update(contrast_ch)
    if fnames is None:
        fnames = list(contrast_ch)
    alive = (filter_mask(fsrc, fnames, filter_q)
             if fnames else np.ones(len(keys), bool))
    dropped = int((~alive).sum())

    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])]
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
             "killed_wrong_color": killed_wrong_color,
             "abstained": abstained,
             "ungroundable": ungroundable,
             "x": x, "y": y}, keep)
