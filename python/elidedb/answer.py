"""ANSWER: the join of scene, agent, participants and events. No text.

The user's definition, verbatim: "answer is just a join... everything is
just logical." So this module contains no text tower, no fitted weight,
no learned combiner - only the element tables and three logical
operations:

    AND   min over percentile ranks (scale-free conjunction)
    OR    max over a candidate episode's objects
    BIND  the AND is evaluated PER CANDIDATE OBJECT - "an object that
          looks like X and moves like X" must be satisfied by one
          object, not by one object that looks right and a different
          one that moves right. This binding is the entire difference
          between a join and a channel soup, and it is what "object1
          (and similar) undergoing action1 (and similar)" means.

Percentile ranks, not raw cosines, because the terms live on different
scales (DINOv3 identity cosine, path-delta cosine, V-JEPA cosine) and a
min over raw scales lets the tightest-distributed term govern
everything. Rank-normalising is order statistics, not a weight.

Element terms per query participant, each one a table:

    looks   object_vectors   DINOv3 track descriptor cosine
    moves   trajectories     signed, speed-normalised (dx,dy,dz) path
                             deltas resampled to N steps - direction
                             survives, which is the thing text erases
    acts    vjepa_part       physics tubelet cosine, where both sides
                             have one

Episode-level terms:

    scene   scene_vectors    pooled window cosine, one more rank in the
                             conjunction
    events  kind overlap     a PARTITION, not a score: episodes sharing
                             at least one event kind with the query rank
                             ahead of episodes sharing none. Logical
                             precedence, no number invented.
"""
from __future__ import annotations

import numpy as np

_C = {}

NPATH = 16          # path resample steps; 15 deltas x 3 dims


def _ranks(x):
    """Percentile rank in [0,1] per entry; NaN stays NaN (missing)."""
    x = np.asarray(x, np.float64)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.sum() > 1:
        r = x[ok].argsort().argsort()
        out[ok] = r / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 1.0
    return out


def _path_desc(ts, px, py, pz, diag):
    """Signed, speed-normalised path deltas. Translation-invariant by
    construction (deltas), scale-normalised by the object's own box
    diagonal, direction preserved (open vs close differ by SIGN, the
    one thing appearance embeddings collapse - cos 0.957 measured)."""
    if len(ts) < 3:
        return None
    o = np.argsort(ts)
    t = np.linspace(0, len(o) - 1, NPATH).round().astype(int)
    x, y, z = (np.asarray(v, np.float64)[o][t] for v in (px, py, pz))
    d = np.stack([np.diff(x) / max(diag, 1.0),
                  np.diff(y) / max(diag, 1.0),
                  np.diff(z) * 4.0], 1).ravel()
    n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 1e-6 else None


def _elements(store):
    """Every element table, loaded once per store version, joined on
    the track key (stream, track_ts, t1, object_id)."""
    ver = store.table("trajectories").state().version
    key = (str(store.dir), ver)
    if key in _C:
        return _C[key]
    E = {}
    tr = store.table("trajectories").scan().to_pydict()
    by = {}
    for i in range(len(tr["ts"])):
        k = (str(tr["stream"][i]), int(tr["track_ts"][i]),
             int(tr["t1"][i]), int(tr["object_id"][i]))
        by.setdefault(k, []).append(i)
    paths, agents = {}, {}
    for k, idx in by.items():
        bx = [(tr["x1"][i] - tr["x0"][i], tr["y1"][i] - tr["y0"][i])
              for i in idx]
        diag = float(np.median([np.hypot(w, h) for w, h in bx]))
        d = _path_desc([tr["ts"][i] for i in idx],
                       [tr["px"][i] for i in idx],
                       [tr["py"][i] for i in idx],
                       [tr["pz"][i] for i in idx], diag)
        if d is not None:
            paths[k] = d
        if any(tr["is_agent"][i] for i in idx):
            agents[k] = True
    E["paths"] = paths

    # key -> ROW index; double-detection twins share a key, so the
    # dict is smaller than the table and the reshape must use the ROW
    # count - keying by len(dict) sheared the matrix off by 2,086 rows
    ov = store.table("object_vectors").scan().to_pydict()
    E["obj"] = {(str(s), int(a), int(b), int(o)): i for i, (s, a, b, o)
                in enumerate(zip(ov["stream"], ov["ts"], ov["t1"],
                                 ov["object_id"]))}
    V = np.asarray(ov["vector"], np.float32).reshape(len(ov["ts"]), -1)
    E["objV"] = V / np.maximum(
        np.linalg.norm(V, axis=1, keepdims=True), 1e-8)

    vp = store.table("vjepa_part_vectors").scan().to_pydict()
    E["phys"] = {(str(s), int(a), int(b), int(o)): i for i, (s, a, b, o)
                 in enumerate(zip(vp["stream"], vp["ts"], vp["t1"],
                                  vp["object_id"]))}
    Vp = np.asarray(vp["vector"], np.float32).reshape(len(vp["ts"]), -1)
    E["physV"] = Vp / np.maximum(
        np.linalg.norm(Vp, axis=1, keepdims=True), 1e-8)

    sc = store.table("scene_vectors").scan().to_pydict()
    E["scene"] = (sc["stream"], np.asarray(sc["ts"], np.int64),
                  np.asarray(sc["vector"], np.float32)
                  .reshape(len(sc["ts"]), -1))

    ev = store.table("events").scan().to_pydict()
    E["events"] = ev
    E["agents"] = agents

    # EVENT-LEVEL join arrays: each event row -> its motion vector, its
    # bound object's DINOv3 descriptor row, its physics row, its
    # episode. The image-plane path descriptor above measured 0.08
    # yield on the direction queries - 2D geometry does not survive a
    # camera change, and the supports span four cameras. The per-event
    # delta-appearance vector does (it is appearance change, not
    # coordinates), so it is the join's moves-term; geometry remains
    # the fallback where a window has no bound events.
    mv = store.table("motion_vectors").scan().to_pydict()
    mrow = {(str(s), int(a), int(b)): i for i, (s, a, b) in
            enumerate(zip(mv["stream"], mv["ts"], mv["t1"]))}
    MV = np.asarray(mv["vector"], np.float32).reshape(len(mv["ts"]), -1)
    MV /= np.maximum(np.linalg.norm(MV, axis=1, keepdims=True), 1e-8)
    tracks_of = {}
    for k in paths:
        tracks_of.setdefault((k[0], k[3]), []).append(k)
    n_ev = len(ev["ts"])
    e_m = np.full(n_ev, -1)
    e_obj = np.full(n_ev, -1)
    e_phys = np.full(n_ev, -1)
    for i in range(n_ev):
        s, a, b = str(ev["stream"][i]), int(ev["ts"][i]), int(ev["t1"][i])
        e_m[i] = mrow.get((s, a, b), -1)
        oid = int(ev["object_id"][i])
        if oid >= 0:
            for k in tracks_of.get((s, oid), ()):
                if k[1] <= b and a <= k[2]:
                    e_obj[i] = E["obj"].get(k, -1)
                    e_phys[i] = E["phys"].get(k, -1)
                    break
    E["e_m"], E["e_obj"], E["e_phys"], E["MV"] = e_m, e_obj, e_phys, MV

    ep = store.table("episodes").scan().to_pydict()
    E["episodes"] = [(str(s), int(a), int(b)) for s, a, b in
                     zip(ep["stream"], ep["ts"], ep["t1"])]
    if len(_C) > 4:
        _C.clear()
    _C[key] = E
    return E


def _tracks_in(E, stream, t0, t1):
    return [k for k in E["paths"]
            if k[0] == stream and k[1] <= t1 and t0 <= k[2]]


def _scene_vec(E, stream, t0, t1):
    ss, ts, V = E["scene"]
    m = np.array([s == stream and t0 <= t <= t1
                  for s, t in zip(ss, ts)])
    if not m.any():
        return None
    v = V[m].mean(0)
    return v / (np.linalg.norm(v) + 1e-8)


def answer_like(store, stream, t0, t1):
    """Episodes telling the same story as the query window.

    Returns (keys, score, shared_kind) - score is the conjunction rank
    (higher = better), shared_kind marks the event-kind partition.
    """
    E = _elements(store)
    eps = E["episodes"]
    n = len(eps)

    # ---- the query's elements -----------------------------------
    q_tracks = _tracks_in(E, stream, t0, t1)
    ev = E["events"]
    q_kinds = {ev["kind"][i] for i in range(len(ev["ts"]))
               if str(ev["stream"][i]) == stream
               and int(ev["ts"][i]) >= t0 and int(ev["t1"][i]) <= t1
               and ev["kind"][i]}
    ep_of = {}
    for i, (s, a, b) in enumerate(eps):
        ep_of.setdefault(s, []).append((a, b, i))

    # events of the query window, and each event's episode index
    q_ev = [i for i in range(len(ev["ts"]))
            if str(ev["stream"][i]) == stream
            and int(ev["ts"][i]) >= t0 and int(ev["t1"][i]) <= t1]
    n_ev = len(ev["ts"])
    ev_ep = np.full(n_ev, -1)
    for i in range(n_ev):
        s, a = str(ev["stream"][i]), int(ev["ts"][i])
        for ea, eb, j in ep_of.get(s, ()):
            if a >= ea and a <= eb:
                ev_ep[i] = j
                break

    # ---- EVENT-LEVEL join (primary) -----------------------------
    # "the same transition happening to the same kind of thing": per
    # query event, every candidate event is scored on moves (delta-
    # appearance of the transition - the view-tolerant motion term;
    # image-plane geometry measured 0.08 on the direction queries
    # because the supports span four cameras) AND looks (DINOv3 of the
    # object each event BINDS to) AND acts (its physics tubelet). One
    # candidate event must satisfy all of it - that is the join.
    q_bound = [i for i in q_ev if E["e_m"][i] >= 0]
    per_q = []
    for qi in q_bound:
        terms = []
        terms.append(_ranks(E["MV"] @ E["MV"][E["e_m"][qi]]))
        if E["e_obj"][qi] >= 0:
            qv = E["objV"][E["e_obj"][qi]]
            sims = np.full(n_ev, np.nan)
            has = E["e_obj"] >= 0
            sims[has] = E["objV"][E["e_obj"][has]] @ qv
            terms.append(_ranks(sims))
        if E["e_phys"][qi] >= 0:
            qv = E["physV"][E["e_phys"][qi]]
            sims = np.full(n_ev, np.nan)
            has = E["e_phys"] >= 0
            sims[has] = E["physV"][E["e_phys"][has]] @ qv
            terms.append(_ranks(sims))
        with np.errstate(invalid="ignore"):
            bound = np.nanmin(np.stack(terms), 0)
        row = np.full(n, np.nan)
        for i in range(n_ev):
            e = ev_ep[i]
            if e >= 0 and np.isfinite(bound[i]):
                row[e] = bound[i] if np.isnan(row[e]) \
                    else max(row[e], bound[i])
        per_q.append(row)

    if per_q:
        with np.errstate(invalid="ignore"):
            score = np.nanmin(np.stack(per_q), 0)  # AND over query events
        score = np.where(np.isnan(score), 0.0, score)
    else:
        # ---- track-path fallback: a window with no bound events ----
        all_keys = list(E["paths"].keys())
        P = np.stack([E["paths"][k] for k in all_keys])
        track_ep = np.full(len(all_keys), -1)
        for i, k in enumerate(all_keys):
            for a, b, j in ep_of.get(k[0], ()):
                if k[1] >= a and k[1] <= b:
                    track_ep[i] = j
                    break
        q_parts = [k for k in q_tracks if E["agents"].get(k)] or q_tracks
        if not q_parts:
            return eps, np.zeros(n, np.float32), np.zeros(n, bool)
        per = []
        for qk in q_parts:
            bound = _ranks(P @ E["paths"][qk])
            row = np.full(n, np.nan)
            for i, e in enumerate(track_ep):
                if e >= 0 and np.isfinite(bound[i]):
                    row[e] = bound[i] if np.isnan(row[e]) \
                        else max(row[e], bound[i])
            per.append(row)
        with np.errstate(invalid="ignore"):
            score = np.nanmin(np.stack(per), 0)
        score = np.where(np.isnan(score), 0.0, score)

    qs = _scene_vec(E, stream, t0, t1)
    if qs is not None:
        ssims = np.full(n, np.nan)
        for i, (s, a, b) in enumerate(eps):
            v = _scene_vec(E, s, a, b)
            if v is not None:
                ssims[i] = float(v @ qs)
        sr = _ranks(ssims)
        with np.errstate(invalid="ignore"):
            score = np.fmin(score, np.where(np.isnan(sr), score, sr))

    # ---- event-kind PARTITION -----------------------------------
    shared = np.zeros(n, bool)
    if q_kinds:
        for i in range(len(ev["ts"])):
            k = ev["kind"][i]
            if k and k in q_kinds:
                s = str(ev["stream"][i])
                a = int(ev["ts"][i])
                for ea, eb, j in ep_of.get(s, ()):
                    if a >= ea and a <= eb:
                        shared[j] = True
                        break
    return eps, score.astype(np.float32), shared
