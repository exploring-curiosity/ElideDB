#!/usr/bin/env python3
"""ElideDB Desk — local database browser (the Atlas/Compass role).

Zero-dependency server: stdlib http.server + the elidedb package. All state
lives in the stores themselves; Desk only reads. Thumbnails are decoded on
demand through the same byte-range path queries use — nothing is pre-baked,
so every embedded window can always show its frame.

  elidedb desk [--root lake] [--port 8787] [--open]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path.cwd()

import numpy as np  # noqa: E402
from elidedb import Store  # noqa: E402

STORES: dict[str, Store] = {}
LAKE = ROOT / "lake"
_CACHE: dict = {}
import os  # noqa: E402
READONLY = os.environ.get("DESK_READONLY", "") == "1"


def discover():
    STORES.clear()
    if LAKE.is_dir():
        for p in sorted(LAKE.iterdir()):
            if (p / "_store.json").exists():
                try:
                    STORES[p.name] = Store.open(p)
                except Exception:
                    pass


def _covered_seconds(db, desc):
    """Recorded CONTENT time: the union of row-group ts ranges of the
    frame tables, from parquet footers only. Extent (max minus min) lies
    on sparse stores; a store holding three hours of episodes spread
    over three days of wall clock covers three hours, not three days."""
    import pyarrow.parquet as pq
    frame_tabs = [d["table"] for d in desc
                  if d["kind"] == "frame_index" and d["rows"]]
    if not frame_tabs:
        frame_tabs = [d["table"] for d in desc
                      if d["table"] == "frames" and d["rows"]]
    # exact when available: every ingest commit recorded its fps, and
    # frame count over fps is the recorded duration regardless of how
    # the rows pack into row groups
    total_s = 0.0
    exact = False
    for name in frame_tabs:
        try:
            for c in db.table(name).history():
                m = c.get("meta") or {}
                fps = m.get("fps")
                if fps and c.get("added_rows"):
                    total_s += c["added_rows"] / float(fps)
                    exact = True
        except Exception:
            continue
    if exact:
        return total_s
    # episode-organized stores: the episodes table IS the content list
    if any(d["table"] == "episodes" and d["rows"] for d in desc):
        try:
            t = db.table("episodes").scan(columns=["ts", "t1"])
            a = t.column("ts").to_numpy()
            b = t.column("t1").to_numpy()
            return float((b - a).sum() / 1e9)
        except Exception:
            pass
    spans = []
    for name in frame_tabs:
        try:
            st = db.table(name).state()
            for f in st.files:
                pf = pq.ParquetFile(db.dir / "tables" / name / f.path)
                md = pf.metadata
                names = md.schema.names
                if "ts" not in names:
                    continue
                ti = names.index("ts")
                for g in range(md.num_row_groups):
                    s = md.row_group(g).column(ti).statistics
                    if s and s.min is not None:
                        spans.append((int(s.min), int(s.max)))
        except Exception:
            continue
    if not spans:
        return None
    spans.sort()
    total, cur_a, cur_b = 0, spans[0][0], spans[0][1]
    for a, b in spans[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += cur_b - cur_a
    return total / 1e9


def _raw_source_bytes(db, desc):
    """Bytes of the ORIGINAL ingested sources, from the ingest metadata
    each append recorded. This is the honest numerator of the
    compression story; a missing original is reported as unknown, not
    guessed."""
    originals = set()
    for d in desc:
        if d["kind"] != "frame_index" and d["table"] != "frames":
            continue
        try:
            for c in db.table(d["table"]).history():
                o = (c.get("meta") or {}).get("original")
                if o:
                    originals.add(o)
        except Exception:
            continue
    known = missing = 0
    total = 0
    for o in originals:
        p = Path(o)
        if p.exists():
            total += p.stat().st_size
            known += 1
        else:
            missing += 1
    return {"bytes": total, "files": known, "missing": missing}


def store_summary(key: str):
    db = STORES[key]
    desc = db.describe()
    ck = ("summary", key,
          tuple(sorted((d["table"], d["version"]) for d in desc)))
    if ck in _CACHE:
        return _CACHE[ck]
    total_rows = sum(d["rows"] for d in desc)
    total_bytes = sum(d["bytes"] for d in desc)
    span_tabs = [d for d in desc if d["rows"] and d["table"] != "centroids"]
    lo = min((d["min_ts"] for d in span_tabs), default=0)
    hi = max((d["max_ts"] for d in span_tabs), default=0)
    emb = next((d for d in desc if d["table"] == "embeddings"), None)
    # PHYSICAL directory bytes (lstat): a symlink is a path entry, not
    # the target's bytes. Media reached through symlinks is counted as
    # linked_bytes instead, because the store STOPS WORKING if those
    # targets go away, and the standalone claim must be earned.
    media_bytes = linked_bytes = 0
    mdir = db.dir / "media"
    if mdir.is_dir():
        for p in mdir.glob("*"):
            if p.is_symlink():
                try:
                    linked_bytes += p.stat().st_size
                except OSError:
                    pass
            elif p.is_file():
                media_bytes += p.lstat().st_size
    db_bytes = sum(p.lstat().st_size for p in db.dir.rglob("*")
                   if p.is_file() and not p.is_symlink())
    external_bytes = 0
    for d in desc:
        if d["kind"] != "frame_index":
            continue
        t = db.table(d["table"]).scan(columns=["source"])
        for s in set(t.column("source").to_pylist()):
            if not s.startswith("@") and Path(s).exists():
                external_bytes += Path(s).stat().st_size
    raw = _raw_source_bytes(db, desc)
    vec_rows = sum(d["rows"] for d in desc if d["kind"] == "embeddings")
    vec_tables = sum(1 for d in desc
                     if d["kind"] == "embeddings" and d["rows"])
    out = {
        "key": key, "name": db.name, "path": str(db.dir),
        "tables": desc, "rows": total_rows, "bytes": total_bytes,
        "db_bytes": db_bytes, "media_bytes": media_bytes,
        "emb_bytes": emb["bytes"] if emb else 0,
        "external_bytes": external_bytes,
        "linked_bytes": linked_bytes,
        "standalone": external_bytes == 0 and linked_bytes == 0,
        "raw_bytes": raw["bytes"], "raw_files": raw["files"],
        "raw_missing": raw["missing"],
        "covered_s": _covered_seconds(db, desc),
        "min_ts": lo, "max_ts": hi,
        "windows": emb["rows"] if emb else 0,
        "vec_rows": vec_rows, "vec_tables": vec_tables,
        "model": (emb or {}).get("meta", {}).get("model", ""),
        "display": db.meta.get("display", {}),
    }
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[ck] = out
    return out


def api_storage(key: str, table: str):
    """The Parquet format, made visible: this table's commit log plus every
    active file's row-group layout straight from the Parquet footers."""
    import pyarrow.parquet as pq
    db = STORES[key]
    tab = db.table(table)
    st = tab.state()
    files = []
    for f in st.files:
        pf = pq.ParquetFile(db.dir / "tables" / table / f.path)
        md = pf.metadata
        names = md.schema.names
        ts_i = names.index("ts") if "ts" in names else 0
        rgs = []
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            s = rg.column(ts_i).statistics
            rgs.append({"rows": rg.num_rows,
                        "bytes": sum(rg.column(c).total_compressed_size
                                     for c in range(rg.num_columns)),
                        "min_ts": s.min if s else None,
                        "max_ts": s.max if s else None})
        files.append({"name": f.path, "bytes": f.bytes, "rows": f.rows,
                      "min_ts": f.min_ts, "max_ts": f.max_ts,
                      "footer_bytes": md.serialized_size,
                      "columns": names, "row_groups": rgs})
    return {"table": table, "kind": st.kind, "version": st.version,
            "schema": st.schema, "history": tab.history(), "files": files}


MAP_MAX_POINTS = 6000


def api_map(key: str):
    """2D layout of the embeddings table. UMAP over a bounded SAMPLE,
    cached beside the log (a derived view; never used for retrieval).

    The unsampled version killed the app at pilot scale, three ways at
    once (measured on the 100 h store): to_pylist() over 180k x 1152
    vectors is ~25 GB of Python floats (4 -> 28 GB RSS), UMAP over 180k
    points runs for minutes, and a 180k-point JSON payload crushes the
    WebView. A map is an OVERVIEW: an even time-stride sample of a few
    thousand windows shows the same structure, reads from the mmap
    sidecar, and stays bounded no matter how large the corpus grows."""
    ck = ("map", key)
    if ck in _CACHE:
        return _CACHE[ck]
    db = STORES[key]
    from elidedb.embeddings import _vec_table
    try:
        t, vecs = _vec_table(db, "embeddings")
    except Exception:
        return {"points": []}
    n = len(t)
    if n == 0:
        return {"points": []}
    stride = max(1, n // MAP_MAX_POINTS)
    idx = np.arange(0, n, stride)
    sample = np.asarray(vecs[idx], np.float32)

    cache_file = db.dir / "tables" / "embeddings" / "_desk_umap.json"
    st = db.table("embeddings").state()
    xy = None
    if cache_file.exists():
        c = json.loads(cache_file.read_text())
        if c.get("version") == st.version and len(c["xy"]) == len(idx):
            xy = np.array(c["xy"], np.float32)
    if xy is None:
        try:
            import umap
            from sklearn.decomposition import PCA
            red = PCA(n_components=min(50, len(sample), sample.shape[1]),
                      random_state=0).fit_transform(sample)
            xy = umap.UMAP(n_components=2, random_state=0,
                           low_memory=True).fit_transform(red)
        except Exception:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=0).fit_transform(sample)
        cache_file.write_text(json.dumps(
            {"version": st.version, "xy": np.round(xy, 3).tolist()}))
    labels = (t.column("cluster").to_pylist()
              if "cluster" in t.column_names else None)
    # NO cluster column: every point got c=0 and the map was one flat
    # colour with a legend reading "cluster 0 · 3891" - information-free.
    # Cluster the 2D LAYOUT for display, cached beside the UMAP cache.
    # DISPLAY ONLY, and the UI says so: UMAP preserves neighbourhoods,
    # not distances, so these coordinates never touch a retrieval
    # decision - that rule is why the store has no cluster column in the
    # first place.
    kmeans_display = False
    if labels is None and len(xy) >= 24:
        kc = db.dir / "tables" / "embeddings" / "_desk_kmeans.json"
        got = None
        if kc.exists():
            try:
                j = json.loads(kc.read_text())
                if j.get("version") == st.version and len(j["lab"]) == n:
                    got = j["lab"]
            except Exception:
                got = None
        if got is None:
            P = np.asarray(xy, np.float32)
            K = min(10, max(2, len(P) // 40))
            rng = np.random.default_rng(0)
            C = P[rng.choice(len(P), K, replace=False)]
            for _ in range(25):                     # Lloyd, few rounds
                d2 = ((P[:, None, :] - C[None]) ** 2).sum(-1)
                a = d2.argmin(1)
                for k in range(K):
                    m = a == k
                    if m.any():
                        C[k] = P[m].mean(0)
            sub = a.astype(int).tolist()
            got = [0] * n
            for j, i in enumerate(idx):
                got[i] = sub[j]
            try:
                kc.write_text(json.dumps({"version": st.version,
                                          "lab": got}))
            except Exception:
                pass
        labels = got
        kmeans_display = True
    ss = t.column("stream").to_pylist()
    ta = t.column("ts").to_pylist()
    tb = t.column("t1").to_pylist()
    out = {"points": [
        {"x": float(xy[j][0]), "y": float(xy[j][1]),
         "c": int(labels[i]) if labels else 0,
         "s": ss[i], "t0": ta[i], "t1": tb[i]}
        for j, i in enumerate(idx)],
        "sampled_of": n, "stride": int(stride),
        "cluster_source": ("stored cluster column" if not kmeans_display
                           else "k-means over the 2D layout — DISPLAY "
                                "ONLY, never a retrieval decision")}
    _CACHE[ck] = out
    return out


def api_architecture(key: str):
    """The store as a living schematic: every node and edge derived from
    what is ACTUALLY on disk — parquet footers for schemas, _meta.json for
    state, table meta for lineage, _cache for mmap sidecars. Nothing here
    is drawn from documentation; a table that vanished vanishes from the
    drawing, a channel appears the moment its table exists."""
    import pyarrow.parquet as _pq
    db = STORES[key]
    nodes, edges = [], []

    def table_node(name):
        tab = db.table(name)
        try:
            st = tab.state()
        except Exception:
            return None
        if not st.files:
            return None
        meta_json = {}
        mf = tab.dir / "_meta.json"
        if mf.exists():
            try:
                meta_json = json.loads(mf.read_text())
            except Exception:
                pass
        fields = []
        try:
            sch = _pq.ParquetFile(tab.dir / st.files[0].path).schema_arrow
            for f in sch:
                t = str(f.type)
                t = t.replace("fixed_size_list<item: float>", "f32vec")
                fields.append({"name": f.name, "type": t})
        except Exception:
            pass
        sidecars = sorted(p.name for p in (tab.dir / "_cache").glob("*.npy")) \
            if (tab.dir / "_cache").is_dir() else []
        rows = sum(f.rows for f in st.files)
        return {"id": name, "kind": st.kind, "rows": rows,
                "bytes": sum(f.bytes for f in st.files),
                "files": len(st.files), "version": st.version,
                "min_ts": min((f.min_ts for f in st.files), default=None),
                "max_ts": max((f.max_ts for f in st.files), default=None),
                "fields": fields, "meta": st.meta or {},
                "meta_json": bool(meta_json), "sidecars": sidecars}

    names = db.tables()
    for n in names:
        nd = table_node(n)
        if nd:
            nodes.append(nd)
    have = {n["id"] for n in nodes}

    # media + models are first-class citizens of the drawing
    mdir = db.dir / "media"
    if mdir.is_dir():
        fs = list(mdir.glob("*"))
        nodes.append({"id": "media", "kind": "media",
                      "rows": len(fs),
                      "bytes": sum(f.stat().st_size for f in fs
                                   if f.is_file()),
                      "files": len(fs), "fields": [], "meta": {},
                      "note": "transcoded H.264, byte-range decoded"})
    for m in sorted((db.dir / "models").glob("*")) \
            if (db.dir / "models").is_dir() else []:
        if m.is_dir():
            nodes.append({"id": f"model:{m.name}", "kind": "model",
                          "bytes": sum(f.stat().st_size
                                       for f in m.rglob("*") if f.is_file()),
                          "fields": [], "meta": {},
                          "rows": None, "files": None})

    # lineage: explicit table meta first, then structural conventions
    def edge(a, b, label):
        if a in have or a in ("media", "SOURCE", "QUERIES") or \
                a.startswith("model:"):
            edges.append({"from": a, "to": b, "label": label})
    for n in nodes:
        meta, nid = n.get("meta", {}), n["id"]
        if meta.get("source_table"):
            edge(meta["source_table"], nid,
                 meta.get("built_by", "derived"))
        if meta.get("events_table"):
            edge(meta["events_table"], nid, "spans")
        if meta.get("teacher"):
            edge("frames", nid, meta.get("model", "encoder"))
    conventions = {
        "frame_vectors": ("frames", "encoder, every frame"),
        "object_vectors": ("frames", "FastSAM regions + crops"),
        "context_captions": ("frames", "VLM captions"),
        "context_events": ("frames", "gate segmentation"),
        "vlm_verdicts": ("frames", "2B/7B judgments"),
        "frames": ("media", "frame index"),
    }
    done = {(e["from"], e["to"]) for e in edges}
    for nid, (src, lab) in conventions.items():
        if nid in have and (src, nid) not in done and \
                (src in have or src == "media"):
            edges.append({"from": src, "to": nid, "label": lab})

    # query channels: present iff their table exists on disk. This list
    # is the CURRENT set path, nothing else; a channel appears the
    # moment its vectors are ingested and vanishes with them.
    channels = []
    def chan(cid, label, need, note):
        ok = need() if callable(need) else need in have
        if ok:
            channels.append({"id": cid, "label": label, "note": note})
    chan("app", "appearance", "embeddings", "window vectors, exact scan")
    chan("pe", "perception", "pe_vectors", "PE-Core text to frame")
    chan("sig2", "fine-grained", "sig2_vectors", "SigLIP 2 frame space")
    chan("conj", "conjunction", "sig2_vectors",
         "every noun phrase must find its own frame")
    chan("iv2", "video-text", "iv2_vectors",
         "InternVideo2, 4 frames encoded together")
    chan("act", "action", "action_probs", "V-JEPA 2 verb posteriors")
    chan("vid", "clip-text", "xclip_vectors", "X-CLIP pooled clips")
    chan("obj", "objects", "object_vectors", "region crops, conjunctive")
    chan("mot", "motion", "motion_vectors",
         "delta appearance against the antonym")
    chan("prf", "feedback", "vjepa_vectors",
         "Rocchio anchors mined from this corpus")
    chan("itm", "cross-encoder", lambda: (
        os.environ.get("ELIDEDB_ITM") == "1"
        and (db.dir / "_cache/itm_tokens").exists()),
        "InternVideo2 1B, rerank STAGE over the top-N - never a channel "
        "(as a weighted voter it cost 0.38 -> 0.27, RRF discards the "
        "margin's scale). Cost-gated: 0.4s/episode")
    chan("anchor", "transition anchor", "motion_vectors",
         "the query names a transition, the corpus defines its DIRECTION "
         "in motion space. Text cannot ask for it: cos(opens, closes) = "
         "0.957. Motion separates them 0.983 held-out")

    # selection pipeline: the fitted stages every result passes
    # through, in order. Fitted values come from the per-store
    # artifact; stages render even unfitted (neutral defaults).
    fitted = {}
    swp = db.dir / "_set_weights.json"
    if swp.exists():
        try:
            fitted = json.loads(swp.read_text())
        except Exception:
            pass
    pipeline = [
        {"id": "fuse", "label": "Fitted fusion",
         "note": "weighted rank consensus, per-store weights"},
        {"id": "prf", "label": "Pseudo-relevance",
         "note": "the head of pass 1 re-queries the corpus"},
        {"id": "itm", "label": "Cross-encoder cascade",
         "note": "distribution-preserving: it PERMUTES the candidates "
                 "and returns the same sorted scores in the new order, "
                 "because the cut downstream is fitted to RRF's scale"},
        {"id": "anchor", "label": "Transition anchor",
         "note": "direction from the corpus, weighted by a reliability "
                 "each kind earns unsupervised (close 0.52, open 0.35, "
                 "put_on 0.00 - so it cannot damage what it cannot help)"},
        {"id": "evk", "label": "Event corroboration",
         "note": "does this demo carry the asked-for transition at all"},
        {"id": "dens", "label": "Motion density",
         "note": "15-NN agreement in motion space"},
        {"id": "gate", "label": "No-match gate",
         "note": "abstains when the corpus lacks the action"},
        {"id": "filter", "label": "Contrast filter",
         "note": "direction evidence, fitted quantile"},
        {"id": "nms", "label": "Event dedup",
         "note": "one clip per event, fitted radius"},
        {"id": "cut", "label": "Confidence cut",
         "note": "the set ends where confidence does"},
        {"id": "audit", "label": "Geometry audit",
         "note": "SAM 3 tracker verification, opt-in tier"},
    ]

    # THE FIVE ELEMENTS — what a clip is decomposed into. Each renders
    # only if the table that carries it exists in THIS store, so an
    # un-elemented store shows an empty list rather than a promise.
    def _rows(t):
        try:
            return db.table(t).scan().num_rows if t in have else 0
        except Exception:
            return 0

    ev_t = "events" if "events" in have else (
        "events_s" if "events_s" in have else None)
    elements = []
    if ev_t:
        import collections as _c
        kinds = _c.Counter(db.table(ev_t).scan().column("kind").to_pylist())
        elements = [
            {"id": "scene", "label": "scene", "note": "demo gist vector",
             "n": _rows("answers2")},
            {"id": "agent", "label": "agent",
             "note": "the self-moving thing, from flow",
             "n": kinds.get("agent", 0)},
            {"id": "participants", "label": "participants",
             "note": "what the agent contacts, in order - by CAUSALITY "
                     "(motion onset adjacent to the agent), not pixel "
                     "change", "n": kinds.get("contact", 0)},
            {"id": "events", "label": "events",
             "note": "typed, TIMESTAMPED transitions: " + ", ".join(
                 f"{k} {v}" for k, v in kinds.most_common()
                 if k not in ("agent", "contact", "release")),
             "n": sum(v for k, v in kinds.items()
                      if k not in ("agent", "contact", "release"))},
            {"id": "answer", "label": "answer",
             "note": "initial -> final diff, attributed to the agent",
             "n": _rows("answers2")},
        ]

    # MODELS — the manifest is the authority, and it records the env
    # needed to reproduce its own numbers (see teacher_v2.json).
    models = []
    for tag in ("teacher_v2", "teacher_v1"):
        p = Path(__file__).resolve().parents[2] / f"models/{tag}.json"
        if p.exists():
            try:
                m = json.loads(p.read_text())
                models.append({
                    "id": tag, "kind": "teacher",
                    "yield": m["metric"].get("mean_yield"),
                    "prec": m["metric"].get("mean_prec"),
                    "env": m.get("env", {}),
                    "reproduce": m.get("reproduce", ""),
                    "note": "cosine channels -> RRF -> PRF -> ITM cascade "
                            "-> anchor + event gate + density -> cut"})
                break
            except Exception:
                pass
    sp = Path(__file__).resolve().parents[2] / "models/student_v1/meta.json"
    if sp.exists():
        try:
            sm = json.loads(sp.read_text())
            models.append({
                "id": "student_v1", "kind": "student",
                "params_M": round(sm.get("params", 0) / 1e6, 2),
                "read_ms": 27, "write_s_per_demo": 0.85,
                "note": "two-tower bi-encoder + listwise rerank head. "
                        "Produces the five elements ITSELF; stage order "
                        "PRF -> gate -> cascade was measured, not "
                        "inherited (the teacher's order scored worse)"})
        except Exception:
            pass

    # ARTIFACTS lane: the model files and fitted state this store's query
    # path actually loads. It was empty because nothing ever emitted a
    # node of kind "model" - the lane existed with nothing to put in it.
    RT = Path(__file__).resolve().parents[2]

    def _sz(p):
        p = Path(p)
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return p.stat().st_size if p.exists() else 0

    for path, label, note in (
        ("models/teacher_v2.json", "teacher_v2", "manifest: tables, "
         "digests, stages, env, measured yield/prec"),
        ("models/student_v1", "student_v1", "student.pt (two-tower + "
         "rerank head), episode_emb.npz, namer.pt"),
        ("models/iv2_stage2_1b", "iv2_stage2_1b", "InternVideo2-Stage2 "
         "1B — the ITM cross-encoder, cost-gated"),
        ("models/fdnnv", "fdnnv", "FDNN-V encoder — embeds every frame "
         "at ingest, 1,109 fps"),
        ("artifacts/verbs_v2.json", "verbs_v2", "geometry verb partition"),
        ("artifacts/cavity.json", "cavity", "cavity/articulation thresholds"),
    ):
        b = _sz(RT / path)
        if b:
            nodes.append({"id": label, "kind": "model", "rows": None,
                          "bytes": b, "note": note, "path": path})
    for a, note in (("_set_weights.json", "fitted selection weights "
                     "(+ .prev rotation, .loqo holdout)"),
                    ("_channel_weights.json", "per-channel fusion weights"),
                    ("_vocab.json", "corpus-attested vocabulary cache")):
        b = _sz(db.dir / a)
        if b:
            nodes.append({"id": a, "kind": "model", "rows": None,
                          "bytes": b, "note": note, "path": a})

    # THE TRAINING / RETRAINING LOOP — how more data becomes a better
    # model. This is a closed loop and the store is inside it.
    training = [
        {"id": "ingest", "label": "1 · INGEST",
         "note": "raw video → frames (byte-range index) + FDNN-V vectors "
                 "for EVERY frame. 63.5 s for 3.91 h, 1,109 fps. No "
                 "labels, no metadata: the store never ingests task "
                 "strings."},
        {"id": "elements", "label": "2 · WRITE-PATH ELEMENTS",
         "note": "geometry over the frames produces the five elements "
                 "per demo — agent from flow, participants by causality, "
                 "events from cavity + displacement, answer rollup. The "
                 "student's Namer head predicts the participant NAME "
                 "VECTOR the 7B VLM would have produced."},
        {"id": "teacher", "label": "3 · TEACHER RANKS",
         "note": "the expensive path (channels → RRF → PRF → ITM 1B "
                 "cross-encoder → anchor + gate) ranks the corpus for "
                 "generated queries. 2–75 s/query, so it is never the "
                 "serving path — it exists to produce labels."},
        {"id": "distill", "label": "4 · DISTILL",
         "note": "teacher rankings over 200 corpus-vocabulary queries "
                 "become listwise targets for the student's two-tower "
                 "+ rerank head. 645 s to label, 8 s to train, 3.28M "
                 "params. The eval truthset is NEVER trained on."},
        {"id": "serve", "label": "5 · SERVE",
         "note": "student answers in 27 ms: one text forward + one "
                 "matmul over precomputed episode vectors, then the "
                 "structural gate reads columns the write path already "
                 "produced."},
        {"id": "grow", "label": "6 · MORE DATA → BETTER MODEL",
         "note": "adding episodes widens every corpus statistic the "
                 "system is built on: the transition anchors get more "
                 "attesting episodes so their earned reliability rises, "
                 "PRF anchors sharpen, the attested vocabulary grows, "
                 "and the teacher has more to label. Re-running step 3–4 "
                 "is the retrain; nothing here needs human annotation."},
    ]

    total_rows = sum(n.get("rows") or 0 for n in nodes)
    total_bytes = sum(n.get("bytes") or 0 for n in nodes)
    return {"store": db.name, "key": key, "nodes": nodes, "edges": edges,
            "channels": channels, "pipeline": pipeline,
            "elements": elements, "models": models, "training": training,
            "fitted": {k: fitted[k] for k in
                       ("set_weights_dir", "set_weights", "cut_alpha_dir",
                        "cut_alpha", "nms_r_dir", "nms_r", "loqo_mean")
                       if k in fitted},
            "totals": {"rows": total_rows, "bytes": total_bytes,
                       "tables": len([n for n in nodes
                                      if n["kind"] not in
                                      ("media", "model")])}}


def api_bytes(key: str):
    """WHERE THE BYTES ACTUALLY ARE, and how much compression bought.

    `du` on this store says 3.9 GB while the tables hold 249 MB, and the
    gap is not a compression failure - it is three different things that
    a single directory size silently adds together:

      live      the parquet the LOG currently points at. The database.
      orphaned  parquet superseded by a replace/compact commit. Removed
                from the active set, still on disk until vacuum.
      media     managed video renditions, byte-range decoded.
      cache     _cache/ - the ITM cross-encoder's vision tokens, 3 GB
                of it. DISPOSABLE: deleting it costs recompute time and
                never correctness, and it is not the database.

    Models are NOT in the store. They live in models/ at the repo root
    (iv2_stage2_1b alone is 2.8 GB), which is why "the store" and "what
    this system needs on disk" are different questions.
    """
    import pyarrow.parquet as pq
    db = STORES[key]
    live = orphan = on_disk = 0
    rows = []
    for t in sorted(db.tables()):
        st = db.table(t).state()
        keep = {f.path for f in st.files}
        c = u = 0
        encs = set()
        comp = set()
        d = db.dir / "tables" / t
        for f in d.glob("*.parquet"):
            b = f.stat().st_size
            on_disk += b
            if f.name not in keep:
                orphan += b
                continue
            try:
                md = pq.ParquetFile(f).metadata
                for g in range(md.num_row_groups):
                    rg = md.row_group(g)
                    for j in range(rg.num_columns):
                        col = rg.column(j)
                        c += col.total_compressed_size
                        u += col.total_uncompressed_size
                        for e in (col.encodings or ()):
                            encs.add(str(e))
                        if col.compression:
                            comp.add(str(col.compression))
            except Exception:
                pass
        live += st.bytes
        rows.append({"table": t, "rows": st.rows, "compressed": c,
                     "uncompressed": u,
                     "ratio": round(u / max(c, 1), 2),
                     "encodings": sorted(encs), "codec": sorted(comp),
                     "files": len(st.files), "version": st.version})
    rows.sort(key=lambda r: -r["compressed"])
    media = sum(p.lstat().st_size for p in (db.dir / "media").glob("*")
                if p.is_file()) if (db.dir / "media").is_dir() else 0
    cdir = db.dir / "_cache"
    cache = sum(f.stat().st_size for f in cdir.rglob("*")
                if f.is_file()) if cdir.is_dir() else 0
    cparts = ([{"name": p.name,
                "bytes": sum(f.stat().st_size for f in p.rglob("*")
                             if f.is_file())}
               for p in cdir.iterdir() if p.is_dir()] if cdir.is_dir()
              else [])
    tc = sum(r["compressed"] for r in rows)
    tu = sum(r["uncompressed"] for r in rows)
    raw = _raw_source_bytes(db, db.describe())
    return {"store": db.name, "key": key, "tables": rows,
            "live": live, "orphaned": orphan, "on_disk": on_disk,
            "media": media, "cache": cache, "cache_parts": cparts,
            "total": on_disk + media + cache,
            "compressed": tc, "uncompressed": tu,
            "ratio": round(tu / max(tc, 1), 2),
            "raw_source": raw["bytes"], "raw_files": raw["files"]}


def api_dbinternals(key: str, table: str | None = None):
    """The storage engine, as it actually is on disk.

    Not a diagram: every number here is read from the transaction log
    and the Parquet footers of this store, right now. Four things a
    storage engine has to be able to show:

      1. the LOG   — the table is the fold of an append-only list of
                     JSON commits, each carrying its own file list
      2. FILES     — with the zone map (min_ts/max_ts) that lets a
                     query drop a whole file without opening it
      3. PAGES     — row groups inside a file, each with its own ts
                     statistics and per-column chunk sizes
      4. PRUNING   — what a real window query touches, in bytes,
                     across both layers
    """
    import pyarrow.parquet as pq
    db = STORES[key]
    names = sorted(db.tables())
    table = table if table in names else (
        "frames" if "frames" in names else names[0])
    t = db.table(table)
    st = t.state()

    # ---- 1. the log: manifest commits, newest last
    # read the RAW commit files: Table.history() summarises and drops the
    # removal count, which is the whole point of an op=replace entry.
    log = []
    ldir = db.dir / "tables" / table / "_log"
    for f in sorted(ldir.glob("*.json")) if ldir.is_dir() else []:
        try:
            e = json.loads(f.read_text())
        except Exception:
            continue
        add = e.get("add") or []
        log.append({
            "version": int(f.stem), "op": e.get("op"), "kind": e.get("kind"),
            "added": len(add),
            "removed": len(e.get("remove") or []),
            "added_rows": sum(a.get("rows", 0) for a in add),
            "added_bytes": sum(a.get("bytes", 0) for a in add),
            "ts": e.get("ts_utc"),
            "meta": {k: v for k, v in (e.get("meta") or {}).items()
                     if not isinstance(v, (list, dict))},
        })

    # ---- 2/3. files and their row groups (pages)
    files, pages = [], []
    total_rg = 0
    for f in st.files:
        p = db.dir / "tables" / table / f.path
        row = {"path": f.path, "rows": f.rows, "bytes": f.bytes,
               "min_ts": f.min_ts, "max_ts": f.max_ts, "row_groups": None,
               "footer_bytes": None}
        try:
            pf = pq.ParquetFile(p)
            md = pf.metadata
            row["row_groups"] = md.num_row_groups
            row["footer_bytes"] = md.serialized_size
            total_rg += md.num_row_groups
            ts_i = (md.schema.names.index("ts")
                    if "ts" in md.schema.names else 0)
            if len(pages) < 24:            # a readable sample, not all
                for g in range(min(md.num_row_groups, 8)):
                    rg = md.row_group(g)
                    s = rg.column(ts_i).statistics
                    cols = []
                    for c in range(rg.num_columns):
                        col = rg.column(c)
                        cols.append({
                            "name": (md.schema.names[c]
                                     if c < len(md.schema.names) else "?"),
                            "compressed": col.total_compressed_size,
                            "uncompressed": col.total_uncompressed_size,
                            "encodings": [str(e) for e in
                                          (col.encodings or [])][:3],
                        })
                    cols.sort(key=lambda x: -x["compressed"])
                    pages.append({
                        "file": f.path[:18], "group": g,
                        "rows": rg.num_rows, "bytes": rg.total_byte_size,
                        "min_ts": getattr(s, "min", None),
                        "max_ts": getattr(s, "max", None),
                        "columns": cols[:6],
                        "n_columns": rg.num_columns})
        except Exception:
            pass
        files.append(row)

    # ---- 4. pruning, executed for real on a 2% slice of the span
    prune = None
    if st.files and st.min_ts is not None:
        # ANCHOR THE WINDOW ON REAL DATA. Taking the midpoint of
        # min_ts..max_ts lands in one of the 60 s gaps this store puts
        # between demos and returns zero rows, which measures nothing.
        # A row group's own ts statistics are, by construction, a range
        # that contains rows - so the demo window is the middle row
        # group of the middle file.
        t0, t1 = st.min_ts, st.min_ts + max(
            (st.max_ts - st.min_ts) // 50, 1)
        try:
            mid = st.files[len(st.files) // 2]
            pf = pq.ParquetFile(db.dir / "tables" / table / mid.path)
            md = pf.metadata
            ts_i = (md.schema.names.index("ts")
                    if "ts" in md.schema.names else 0)
            g = md.row_group(md.num_row_groups // 2)
            s = g.column(ts_i).statistics
            if s is not None and s.min is not None:
                t0, t1 = int(s.min), int(s.max)
        except Exception:
            pass
        from .store import QueryStats
        qs = QueryStats()
        try:
            got = t.scan(t0, t1, stats=qs)
            prune = {
                "window_ns": int(t1 - t0),
                "files_total": qs.files_total,
                "files_touched": qs.files_touched,
                "corpus_bytes": qs.corpus_bytes,
                "bytes_touched": qs.bytes_touched,
                "rows_returned": len(got),
                "elided_pct": (round(100.0 * (qs.corpus_bytes
                                              - qs.bytes_touched)
                                     / max(qs.corpus_bytes, 1), 2)),
            }
        except Exception as e:
            prune = {"error": f"{type(e).__name__}: {e}"[:120]}

    # ---- THE LADDER: the containment hierarchy, one rung per level,
    # with this store's real numbers on each. Ordered smallest first.
    lake_dir = db.dir.parent
    n_stores = sum(1 for p in lake_dir.iterdir()
                   if p.is_dir() and (p / "_store.json").exists())
    f0 = files[0] if files else None    # the dict built above
    rg0 = pages[0] if pages else None
    ch0 = (rg0 or {}).get("columns", [{}])[0] if rg0 else {}
    PAGE_TARGET = 1 << 20                       # parquet default 1 MiB
    est_pages = max(1, round((ch0.get("compressed") or 0) / PAGE_TARGET)) \
        if ch0 else None
    ladder = [
        {"id": "value", "label": "VALUE / ROW",
         "n": f"{st.rows:,} rows",
         "sub": f"{(rg0 or {}).get('n_columns', 0)} columns",
         "meta": "—",
         "why": "One cell. Rows are never stored contiguously: inside a "
                "row group the data is laid out COLUMN BY COLUMN, which "
                "is what lets a query read one column and skip the rest."},
        {"id": "page", "label": "PAGE",
         "n": (f"~{est_pages} per column chunk" if est_pages else "—"),
         "sub": "~1 MiB target",
         "meta": "page header: encoding, value count, (optional) stats",
         "why": "The atomic unit of compression and decode. You cannot "
                "read half a page - it is decompressed whole - so page "
                "size is the floor on random-access cost."},
        {"id": "chunk", "label": "COLUMN CHUNK",
         "n": f"{(rg0 or {}).get('n_columns', 0)} per row group",
         "sub": (f"{ch0.get('name','')} "
                 f"{(ch0.get('encodings') or [None])[0] or ''}"),
         "meta": "offset, size, encodings, compression, min/max/nulls",
         "why": "All the pages of ONE column within ONE row group, "
                "contiguous on disk. This is the unit PROJECTION "
                "pushdown skips: ask for 2 of 11 columns and the other "
                "9 chunks are never read."},
        {"id": "rowgroup", "label": "ROW GROUP",
         "n": f"{total_rg} in this table",
         "sub": (f"{rg0['rows']:,} rows · {rg0['bytes']:,} B"
                 if rg0 else "—"),
         "meta": "per-column statistics: min, max, null_count",
         "why": "A horizontal slice of rows holding every column's "
                "chunk. Its ts statistics are what PREDICATE pushdown "
                "tests, so a non-overlapping group is never "
                "decompressed. Sized by BYTES (8 MB target), not rows."},
        {"id": "footer", "label": "FOOTER  (FileMetaData)",
         "n": (f"{f0.get('footer_bytes') or 0:,} B" if f0 else "—"),
         "sub": "at the END of the file",
         "meta": "THE schema + every row group's metadata + offsets",
         "why": "Written last so the file streams out in one pass, read "
                "first so one seek reveals the whole layout. NOTE: row "
                "group metadata lives HERE, inside the same file - not "
                "in a separate meta file. Separate meta files start one "
                "level up."},
        {"id": "file", "label": "PARQUET FILE",
         "n": f"{len(st.files)} in this table",
         "sub": (f"{f0['rows']:,} rows · {f0['bytes']:,} B" if f0 else "—"),
         "meta": "immutable; never edited in place",
         "why": "The unit of atomic addition and removal. Rewriting is "
                "how you 'edit', which is what makes snapshots cheap."},
        {"id": "commit", "label": "COMMIT  (manifest file)",
         "n": f"{len(log)} in tables/{table}/_log/",
         "sub": (f"latest: v{log[-1]['version']} {log[-1]['op']} "
                 f"+{log[-1]['added']}"
                 f"{' −' + str(log[-1]['removed']) if log[-1]['removed'] else ''}"
                 if log else "—"),
         "meta": "op, schema, file list WITH zone maps (min_ts/max_ts)",
         "why": "THE separate meta file. One JSON per commit, listing "
                "the files this version contains and each file's time "
                "range - so a window query drops whole files here, "
                "before opening a single footer. Iceberg calls this a "
                "manifest; Delta calls it a log entry."},
        {"id": "log", "label": "LOG  →  TABLE STATE",
         "n": f"version {st.version}",
         "sub": f"state = fold of {len(log)} commits",
         "meta": "the fold: adds minus removes, in order",
         "why": "A table is NOT the files in its directory - it is the "
                "result of replaying this log. That is what buys "
                "snapshot isolation (v{N} is immutable forever), atomic "
                "multi-file commits, and time travel."},
        {"id": "table", "label": "TABLE  (+ schema)",
         "n": f"{len(names)} in this store",
         "sub": f"{table}: {st.rows:,} rows · {st.bytes:,} B",
         "meta": "schema travels with each commit (schema-on-log)",
         "why": "Schema evolution is an append, never a rewrite. Every "
                "table must carry ts (int64 ns) sorted within a file - "
                "the one schema law, and why time is the primary axis."},
        {"id": "store", "label": "STORE",
         "n": f"{len(names)} tables",
         "sub": f"{db.name} · {sum((db.table(x).state().bytes) for x in names):,} B",
         "meta": "_store.json + fitted artifacts + caches + media/",
         "why": "A directory of tables plus sidecar state. Sidecars are "
                "graded: authoritative (_store.json), fitted "
                "(_set_weights.json), cache (_vocab.json), disposable "
                "(_cache/). Raw media is referenced in place, never "
                "copied in."},
        {"id": "lake", "label": "LAKE",
         "n": f"{n_stores} store{'' if n_stores == 1 else 's'}",
         "sub": "lake/  (" + lake_dir.name + ")",
         "meta": "plain directories — no catalog service",
         "why": "Many stores side by side. Nothing above this is "
                "needed: the format is open, so any engine (DuckDB, "
                "Spark, pandas) reads these files directly without "
                "going through us."},
    ]

    return {
        "store": db.name, "key": key, "table": table, "tables": names,
        "version": st.version, "kind": st.kind, "rows": st.rows,
        "bytes": st.bytes, "n_files": len(st.files),
        "n_row_groups": total_rg, "ladder": ladder,
        "schema": [str(x) for x in str(st.schema).split("\n") if x][:24],
        "log": log[-12:], "log_total": len(log),
        "files": files[:12], "pages": pages, "prune": prune,
        "log_dir": f"tables/{table}/_log/",
        "store_files": sorted(
            p.name for p in db.dir.glob("*")
            if p.is_file())[:20],
    }


def api_analytics(key: str):
    """Operational analytics, general to ANY store: everything here is
    derived from transaction logs, parquet footers, and table meta.
    No data pages are read and nothing is dataset-specific; a store of
    factory video, dashcam runs, or plain sensor CSVs renders the same
    panels."""
    import pyarrow.parquet as pq
    ck = ("analytics", key,
          tuple(sorted((d["table"], d["version"])
                       for d in STORES[key].describe())))
    if ck in _CACHE:
        return _CACHE[ck]
    db = STORES[key]
    desc = db.describe()
    lo = min((d["min_ts"] for d in desc
              if d["rows"] and d.get("min_ts")), default=0)
    hi = max((d["max_ts"] for d in desc
              if d["rows"] and d.get("max_ts")), default=lo + 1)
    span = max(hi - lo, 1)

    # write history straight off the transaction logs
    commits = []
    for d in desc:
        try:
            for c in db.table(d["table"]).history():
                commits.append({"table": d["table"],
                                "version": c.get("version"),
                                "op": c.get("op", ""),
                                "rows": c.get("added_rows", 0),
                                "ts_utc": c.get("ts_utc", "")})
        except Exception:
            pass
    commits.sort(key=lambda c: c["ts_utc"])

    # temporal density from ROW-GROUP footer stats only: rows per time
    # bucket per table. The row group is the pruning unit, so this is
    # literally the elision map a range query sees.
    buckets_n = 64
    density = {}
    for d in desc:
        if not d["rows"] or not d.get("min_ts"):
            continue
        st = db.table(d["table"]).state()
        buckets = [0.0] * buckets_n
        try:
            for f in st.files:
                pf = pq.ParquetFile(db.dir / "tables" / d["table"]
                                    / f.path)
                md = pf.metadata
                names = md.schema.names
                if "ts" not in names:
                    continue
                ti = names.index("ts")
                for g in range(md.num_row_groups):
                    rg = md.row_group(g)
                    s = rg.column(ti).statistics
                    if not s or s.min is None:
                        continue
                    a = int((s.min - lo) * buckets_n // span)
                    b = int((s.max - lo) * buckets_n // span)
                    a = min(max(a, 0), buckets_n - 1)
                    b = min(max(b, a), buckets_n - 1)
                    per = rg.num_rows / (b - a + 1)
                    for i in range(a, b + 1):
                        buckets[i] += per
        except Exception:
            continue
        if sum(buckets) > 0:
            density[d["table"]] = [int(round(x)) for x in buckets]

    # vector inventory: every embeddings-kind table, with coverage
    # against the store's episode base when one exists
    base_rows = next((d["rows"] for d in desc
                      if d["table"] == "episodes" and d["rows"]), None)
    vectors = []
    for d in desc:
        if d["kind"] != "embeddings" or not d["rows"]:
            continue
        meta = d.get("meta") or {}
        vectors.append({
            "table": d["table"], "rows": d["rows"], "bytes": d["bytes"],
            "dim": meta.get("dim"),
            "model": str(meta.get("model", ""))[:60],
            "per_base": (round(d["rows"] / base_rows, 2)
                         if base_rows else None)})

    fitted = None
    swp = db.dir / "_set_weights.json"
    if swp.exists():
        try:
            fitted = json.loads(swp.read_text())
        except Exception:
            pass

    ix = api_indexes(key)
    out = {
        "span": {"lo": lo, "hi": hi},
        "tables": [{"table": d["table"], "kind": d["kind"],
                    "rows": d["rows"], "bytes": d["bytes"],
                    "files": d["files"], "version": d["version"],
                    "bpr": (round(d["bytes"] / d["rows"], 1)
                            if d["rows"] else None)} for d in desc],
        "commits": commits[-48:],
        "commit_total": len(commits),
        "density": density, "buckets": buckets_n,
        "vectors": vectors, "base_rows": base_rows,
        "fitted": fitted,
        "indexes": {"bptree": ix["bptree"], "ann": ix["ann"]},
    }
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[ck] = out
    return out


def api_geo(key: str):
    """Generic geo panel: any timeseries table with latitude+longitude."""
    db = STORES[key]
    for d in db.describe():
        if d["kind"] != "timeseries":
            continue
        cols = db.table(d["table"]).scan(columns=None)
        if {"latitude", "longitude"} <= set(cols.column_names):
            la = cols.column("latitude").to_numpy()
            lo = cols.column("longitude").to_numpy()
            ts = cols.column("ts").to_numpy()
            step = max(1, len(la) // 2500)
            return {"table": d["table"],
                    "points": [{"la": float(la[i]), "lo": float(lo[i]),
                                "t": int(ts[i])}
                               for i in range(0, len(la), step)]}
    return {"points": []}


def _as_frameset(db, fs):
    """Tolerate frame tables whose registered kind is not frame_index
    (seen on stores assembled by filtering another store): window()
    then returns a plain Arrow table, which still carries the frame
    index columns and decodes fine once wrapped."""
    if fs is None or hasattr(fs, "decode"):
        return fs
    try:
        from .video import FrameSet
        return FrameSet(db, "frames", fs)
    except Exception:
        return None


def api_thumb(key: str, stream: str, t: int, width: int = 360):
    from PIL import Image
    db = STORES[key]
    win, _ = db.window(t - 2_000_000_000, t + 2_000_000_000, tables=["frames"])
    fs = _as_frameset(db, win.get("frames"))
    if fs is None or len(fs) == 0:
        return None
    decoded = fs.decode(stream=stream or None, width=width, limit=1)
    if not decoded:
        return None
    img = Image.fromarray(decoded[0][1])
    rot = db.meta.get("display", {}).get("rotate", 0)
    if rot:
        img = img.rotate(rot, expand=True)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _audio_for_stream(db, stream: str, t0: int, t1: int):
    """Find the sensor's audio table (e.g. 'Sensor 108/cam0' →
    sensor_108_audio), return mono WAV bytes for the window, or None."""
    import struct as _struct
    prefix = stream.split("/")[0].replace("/", "_").replace(" ", "_").lower()
    cand = f"{prefix}_audio"
    if cand not in db.tables():
        return None
    t = db.table(cand).scan(t0, t1, columns=["ts", "ch0"])
    if len(t) < 100:
        return None
    ts = t.column("ts").to_numpy()
    rate = int(round((len(ts) - 1) * 1e9 / max(int(ts[-1] - ts[0]), 1)))
    pcm = t.column("ch0").to_numpy().astype("<i2").tobytes()
    hdr = b"RIFF" + _struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + \
        _struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + \
        b"data" + _struct.pack("<I", len(pcm))
    return hdr + pcm


def api_clip(key: str, stream: str, t0: int, t1: int, width: int = 640):
    """Playable clip: byte-range decode of the window's frames → H.264 MP4
    (plus the sensor's microphone track when the store has one). Cached by
    parameters; the source media is only ever read, never touched."""
    import hashlib
    import subprocess
    import tempfile
    # Merged segments can be minutes long; the player previews the first 30 s
    # rather than refusing (full-range export belongs to the Python API).
    t1 = min(t1, t0 + 30_000_000_000)
    cache_dir = Path(tempfile.gettempdir()) / "elidedb_clips"
    cache_dir.mkdir(exist_ok=True)
    ck = hashlib.sha1(f"{key}|{stream}|{t0}|{t1}|{width}".encode()).hexdigest()
    out_path = cache_dir / f"{ck}.mp4"
    if out_path.exists():
        return out_path.read_bytes()

    from PIL import Image
    db = STORES[key]
    win, _ = db.window(t0, t1, tables=["frames"])
    fs = _as_frameset(db, win.get("frames"))
    # Every failure below names itself. A <video> element cannot render an
    # error body, so the player fetches the clip and shows these strings —
    # "could not build a clip" with no reason is not a diagnosis.
    if fs is None or len(fs) == 0:
        return {"error": "no frames indexed in this window",
                "detail": f"{stream or 'all streams'} "
                          f"{(t1 - t0) / 1e9:.2f}s window"}
    have = fs.streams()
    if stream and stream not in have:
        return {"error": f"stream '{stream}' has no frames here",
                "detail": f"streams present in this window: "
                          f"{', '.join(have) or 'none'}"}
    try:
        decoded = fs.decode(stream=stream or None, width=width)
    except Exception as e:
        return {"error": f"decode failed: {type(e).__name__}", "detail": str(e)}
    if len(decoded) < 2:
        return {"error": "not enough decodable frames for a clip",
                "detail": f"{len(decoded)} frame(s) decoded from "
                          f"{len(fs)} indexed"}
    rot = db.meta.get("display", {}).get("rotate", 0)
    span_s = max((decoded[-1][0] - decoded[0][0]) / 1e9, 0.1)
    fps = max(round((len(decoded) - 1) / span_s, 2), 1)

    wav = _audio_for_stream(db, stream, decoded[0][0], decoded[-1][0])
    wav_path = None
    if wav:
        wav_path = cache_dir / f"{ck}.wav"
        wav_path.write_bytes(wav)
    from .fftools import find
    cmd = [find("ffmpeg"), "-v", "error", "-y",
           "-f", "image2pipe", "-framerate", str(fps), "-i", "-"]
    if wav_path:
        cmd += ["-i", str(wav_path)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        for (_ts, arr) in decoded:
            img = Image.fromarray(arr)
            if rot:
                img = img.rotate(rot, expand=True)
            # libx264 yuv420 needs even dimensions
            if img.width % 2 or img.height % 2:
                img = img.crop((0, 0, img.width & ~1, img.height & ~1))
            img.save(proc.stdin, "JPEG", quality=90)
        proc.stdin.close()
    except BrokenPipeError:
        pass  # ffmpeg died early; the stderr below explains why
    err = proc.stderr.read().decode(errors="replace").strip()
    proc.wait()
    if wav_path:
        wav_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_path.exists():
        return {"error": "ffmpeg could not mux the clip",
                "detail": err or "no output produced"}
    return out_path.read_bytes()


def build_id() -> str:
    """Identity of the code this server is actually running.

    The launcher reuses whatever already listens on the port, so a server
    started from an older checkout keeps serving stale code forever — Python
    caches modules at import, so editing files changes nothing until restart.
    That is invisible from the browser and produces bug reports about
    behaviour that no longer exists in the source. The launcher now compares
    this against the on-disk files and restarts on a mismatch.
    """
    import hashlib
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")) + \
            [Path(__file__).parent / "desk_ui.html"]:
        if f.exists():
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def api_schema(key: str, table: str | None = None):
    """What the data actually IS: columns, types, and real sample rows.
    The first thing anyone opening a database wants to see."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    db = STORES[key]
    names = [table] if table else db.tables()
    out = []
    for name in names:
        st = db.table(name).state()
        if not st.files:
            continue
        pf = pq.ParquetFile(db.dir / "tables" / name / st.files[0].path)
        schema = pf.schema_arrow
        cols = []
        for f in schema:
            t = str(f.type)
            if pa.types.is_fixed_size_list(f.type):
                t = f"vector[{f.type.list_size}]"
            cols.append({"name": f.name, "type": t})
        sample = []
        if table:  # only materialise rows for the focused table
            head = db.table(name).scan(columns=[c["name"] for c in cols
                                                if not c["type"].startswith("vector")]
                                       ).slice(0, 8).to_pylist()
            for row in head:
                sample.append({k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in row.items()})
        out.append({"table": name, "kind": st.kind, "rows": st.rows,
                    "bytes": st.bytes, "version": st.version,
                    "min_ts": st.min_ts, "max_ts": st.max_ts,
                    "columns": cols, "sample": sample})
    return out


def api_indexes(key: str):
    """Every index in the store: B+ trees per table + ANN tiers on
    embeddings, with size and the data version each was built for."""
    db = STORES[key]
    out = {"bptree": [], "ann": []}
    for name in db.tables():
        ixdir = db.dir / "tables" / name / "_index"
        if not ixdir.is_dir():
            continue
        for p in ixdir.glob("*.bpt"):
            col, ver = p.stem.rsplit(".v", 1)
            out["bptree"].append({"table": name, "column": col,
                                  "version": int(ver),
                                  "bytes": p.stat().st_size})
        for p in list(ixdir.glob("hnsw.v*.bin")) + list(ixdir.glob("ivfpq.v*.npz")):
            kind = "hnsw" if p.name.startswith("hnsw") else "ivfpq"
            ver = int(p.name.split(".v")[1].split(".")[0])
            cur = db.table("embeddings").state().version
            out["ann"].append({"kind": kind, "version": ver,
                               "current": cur, "stale": ver != cur,
                               "bytes": p.stat().st_size})
    # candidate numeric columns for B+ indexing
    out["indexable"] = {}
    for name in db.tables():
        st = db.table(name).state()
        if st.kind != "timeseries" or not st.files:
            continue
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(db.dir / "tables" / name / st.files[0].path) \
            .schema_arrow
        import pyarrow as pa
        cols = [f.name for f in schema
                if f.name != "ts" and (pa.types.is_integer(f.type)
                                       or pa.types.is_floating(f.type))]
        if cols:
            out["indexable"][name] = cols
    out["has_embeddings"] = "embeddings" in db.tables() and \
        db.table("embeddings").state().rows > 0
    return out


def api_build_index(key: str, body: dict):
    db = STORES[key]
    t0 = time.perf_counter()
    if body.get("ann"):
        from . import ann
        r = (ann.build_hnsw(db) if body["ann"] == "hnsw"
             else ann.build_ivfpq(db))
        r["ms"] = round((time.perf_counter() - t0) * 1e3)
        return r
    r = db.table(body["table"]).create_index(body["column"])
    r["ms"] = round((time.perf_counter() - t0) * 1e3)
    return r


def api_maintenance(key: str, body: dict):
    db = STORES[key]
    op = body["op"]
    t0 = time.perf_counter()
    if op == "compact":
        r = db.table(body["table"]).compact()
    elif op == "vacuum":
        r = db.vacuum(retain_versions=int(body.get("retain", 3)),
                      dry_run=bool(body.get("dry_run", False)))
    elif op == "delete_range":
        r = db.table(body["table"]).delete_range(int(body["t0"]),
                                                 int(body["t1"]))
    else:
        raise ValueError(f"unknown maintenance op {op!r}")
    r["ms"] = round((time.perf_counter() - t0) * 1e3)
    return r


def api_query(key: str, body: dict):
    db = STORES[key]
    kind = body.get("type")
    t_start = time.perf_counter()
    if kind == "sql":
        df = db.sql(body["sql"]).head(200)
        return {"columns": list(df.columns),
                "rows": json.loads(df.to_json(orient="values")),
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "text":
        floor = body.get("floor")
        kw = {}
        if body.get("floor_mode") == "percentile" and floor is not None:
            kw["percentile"] = float(floor)
        elif body.get("floor_mode") == "min_score" and floor is not None:
            kw["min_score"] = float(floor)
        hits, stats = db.search(
            body["text"], k=int(body.get("k", 8)),
            nprobe=int(body.get("nprobe", 3)),
            method=body.get("method", "auto"),
            neg_weight=float(body.get("neg_weight", 0.5)),
            t0=body.get("t0"), t1=body.get("t1"),
            streams=body.get("streams") or None,
            rerank=bool(body.get("rerank")),
            rerank_top=int(body.get("rerank_top", 10)), **kw)
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "context":
        # PRODUCT SURFACE = MEASURED SURFACE: this is the exact
        # search_set the ledger benchmarks (stable-benchmark
        # directive — the old captioned search_context served here
        # while acceptance was measured elsewhere; never again).
        from elidedb.scenario import search_set
        want = int(body.get("k", 10))
        scoped = bool(body.get("streams") or body.get("t0") is not None
                      or body.get("t1") is not None)
        # scoped queries must rank a large pool BEFORE the scope
        # filters below — filtering the unscoped top-k starves the
        # result. Cap sizing differs by tier: audit_n (search_set's
        # "stratified sample" doc) is DEAD — never passed to
        # _binding_audit, which loops the SAM-3.1 tracker over every
        # element of `chosen` (scenario.py:390-392, 509-512), so audit
        # cost grows with k_max, not a fixed sample. A scoped+audited
        # query therefore gets a small pool; scoped+fast can afford 400.
        pool = (400 if not body.get("rerank") else max(4 * want, 40)) \
            if scoped else want
        # ENGINE. The Desk called search_set unconditionally, which is
        # the TEACHER: 59 s cold, 8.4 s for a warm NEW query, because it
        # loads PE + SigLIP2 + IV2 + V-JEPA + X-CLIP and runs every
        # channel over the corpus. The teacher is a LABELLER, not a
        # serving path. The student answers the same question in ~27 ms
        # from precomputed columns and is now the default.
        from .student import available as _st_ok, search_student
        engine = body.get("engine") or ("student" if _st_ok()
                                        else "teacher")
        if engine == "student" and _st_ok() and not body.get("rerank"):
            r = search_student(db, body["text"], k_max=pool)
        else:
            engine = "teacher"
            r = search_set(db, body["text"],
                           purity="audited" if body.get("rerank") else "fast",
                           k_max=pool)
        hits = [{"stream": c["stream"], "t0": c["t0"], "t1": c["t1"],
                 "score": c["score"]} for c in r["clips"]]
        if body.get("streams"):
            hits = [h for h in hits if h["stream"] in body["streams"]]
        lo, hi = body.get("t0"), body.get("t1")
        if lo is not None:
            hits = [h for h in hits if h["t1"] >= int(lo)]
        if hi is not None:
            hits = [h for h in hits if h["t0"] <= int(hi)]
        hits = hits[:want]
        stats = {"channels": r.get("channels", []),
                 "scored": r.get("scored", 0),
                 "direction_filtered": r.get("direction_filtered", 0),
                 "no_match": bool(r.get("no_match")),
                 "engine": engine,
                 "set_ms": r.get("ms")}
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "predicate":
        from elidedb.store import QueryStats
        qs = QueryStats()
        tab = db.table(body["table"])
        vals = [float(body["value"])] if body.get("value2") in (None, "") \
            else [float(body["value"]), float(body["value2"])]
        out = tab.where(body["column"], body["op"], *vals, stats=qs).to_pandas()
        return {"columns": list(out.columns[:8]),
                "rows": json.loads(out.head(50).iloc[:, :8].to_json(
                    orient="values")),
                "count": len(out),
                "stats": {"files": f"{qs.files_touched}/{qs.files_total}",
                          "bytes_touched": qs.bytes_touched,
                          "corpus_bytes": qs.corpus_bytes,
                          "elided_pct": round(qs.elided_pct, 3)},
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "clip":
        hits, stats = db.search_clip(body["stream"], int(body["t0"]),
                                     int(body["t1"]), k=int(body.get("k", 8)),
                                     method=body.get("method", "auto"))
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    if kind == "window":
        w, stats = db.window(int(body["t0"]), int(body["t1"]),
                             tables=body.get("tables") or None)
        out = {"tables": {}, "ms": round(stats.wall_ms, 1),
               "stats": {"files": f"{stats.files_touched}/{stats.files_total}",
                         "bytes_touched": stats.bytes_touched,
                         "corpus_bytes": stats.corpus_bytes,
                         "elided_pct": round(stats.elided_pct, 3),
                         "rows": stats.rows_returned}}
        from elidedb.video import FrameSet
        for name, v in w.items():
            if isinstance(v, FrameSet):
                out["tables"][name] = {"kind": "frames", "count": len(v),
                                       "streams": v.streams()}
            else:
                df = v.to_pandas().head(6)
                out["tables"][name] = {
                    "kind": "rows", "count": len(v),
                    "columns": list(df.columns),
                    "head": json.loads(df.to_json(orient="values"))}
        return out
    raise ValueError(f"unknown query type {kind!r}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if READONLY:
            # the public demo is pinged cross-origin by the landing
            # page to tell "awake" from "cold starting"
            self.send_header("Access-Control-Allow-Origin", "*")
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, body, ctype):
        """Range-aware send: <video> elements (Safari especially) seek via
        byte-range requests; a byte-range database ought to honor them."""
        rng = self.headers.get("Range")
        total = len(body)
        if rng and rng.startswith("bytes="):
            spec = rng[6:].split("-")
            a = int(spec[0]) if spec[0] else 0
            b = int(spec[1]) if len(spec) > 1 and spec[1] else total - 1
            b = min(b, total - 1)
            chunk = body[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {a}-{b}/{total}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/" or u.path == "/index.html":
                html = (Path(__file__).parent / "desk_ui.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/api/stores":
                discover()
                return self._json([store_summary(k) for k in STORES])
            if u.path == "/api/architecture":
                return self._json(api_architecture(q["store"]))
            if u.path == "/api/bytes":
                return self._json(api_bytes(q["store"]))
            if u.path == "/api/dbinternals":
                return self._json(api_dbinternals(q["store"],
                                                  q.get("table")))
            if u.path == "/api/map":
                return self._json(api_map(q["store"]))
            if u.path == "/api/geo":
                return self._json(api_geo(q["store"]))
            if u.path == "/api/analytics":
                return self._json(api_analytics(q["store"]))
            if u.path == "/api/history":
                db = STORES[q["store"]]
                return self._json(db.table(q["table"]).history())
            if u.path == "/api/storage":
                return self._json(api_storage(q["store"], q["table"]))
            if u.path == "/api/indexes":
                return self._json(api_indexes(q["store"]))
            if u.path == "/api/schema":
                return self._json(api_schema(q["store"], q.get("table")))
            if u.path == "/api/thumb":
                jpg = api_thumb(q["store"], q.get("stream", ""),
                                int(q["t"]), int(q.get("w", "360")))
                if jpg is None:
                    return self._json({"error": "no frame"}, 404)
                return self._send(200, jpg, "image/jpeg", cache=True)
            if u.path == "/api/version":
                return self._json({"build": build_id(),
                                   "readonly": READONLY})
            if u.path == "/api/clip":
                mp4 = api_clip(q["store"], q.get("stream", ""),
                               int(q["t0"]), int(q["t1"]),
                               int(q.get("w", "640")))
                if isinstance(mp4, dict):        # structured failure
                    return self._json(mp4, 422)
                if mp4 is None:
                    return self._json({"error": "no frames in window"}, 404)
                return self._send_media(mp4, "video/mp4")
            return self._json({"error": "not found"}, 404)
        except Exception as e:  # surface, don't die
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/query":
                return self._json(api_query(body["store"], body))
            # mutating operations: refused in the public demo. Queries
            # stay open; the stores stay exactly as shipped.
            if READONLY and u.path in ("/api/build_index",
                                       "/api/maintenance"):
                return self._json(
                    {"error": "this deployment is read only"}, 403)
            if u.path == "/api/build_index":
                return self._json(api_build_index(body["store"], body))
            if u.path == "/api/maintenance":
                return self._json(api_maintenance(body["store"], body))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def _warm():
    """Pre-build the per-store matrix caches AND load the text tower so the
    FIRST query of a session is as fast as the hundredth. Off the request
    path; measured: an unwarmed first query pays ~1 s of model load."""
    try:
        from elidedb.embeddings import embed_text
        embed_text("warmup")
    except Exception:
        pass
    # the STUDENT is the serving path, so warm what it needs: the PE
    # text tower and its own weights. Measured cold 10.2 s (almost all
    # of it the text encoder), warm 25-41 ms - so this thread is the
    # difference between the first UI query feeling broken and feeling
    # instant.
    try:
        from elidedb.student import available, search_student
        if available():
            for _k, _db in list(STORES.items()):
                search_student(_db, "the robot opens the drawer", k_max=5)
                break
    except Exception:
        pass
    for key, db in list(STORES.items()):
        try:
            from elidedb.embeddings import _vec_table
            _vec_table(db, "embeddings")
            from elidedb.verified import _recording_spans, _verdict_map
            _verdict_map(db)
            _recording_spans(db)
        except Exception:
            pass
        try:
            _vec_table(db, "motion_vectors")   # builds the mmap sidecar
        except Exception:
            pass
        try:
            from elidedb.subjects import subject_prefixes
            subject_prefixes(db)               # mines on first run, cached
        except Exception:
            pass


def main():
    global LAKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DESK_ROOT",
                                                     str(LAKE)))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8787")))
    ap.add_argument("--host", default=os.environ.get("DESK_HOST",
                                                     "127.0.0.1"))
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    LAKE = Path(args.root).resolve()
    discover()
    threading.Thread(target=_warm, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ElideDB Desk: http://{args.host}:{args.port}  "
          f"({len(STORES)} stores under {LAKE})"
          f"{'  [read only]' if READONLY else ''}")
    if args.open:
        import subprocess
        threading.Timer(0.4, lambda: subprocess.run(
            ["open", f"http://localhost:{args.port}"])).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
