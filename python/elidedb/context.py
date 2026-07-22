"""Context retrieval: the ingest-side half.

WHY THIS EXISTS
---------------
`embeddings` holds one mean-pooled SigLIP vector per window. Mean pooling is
order-blind: reverse the frames and the vector is identical. So the index can
answer "is there a car and a person here" but never "is the person walking
*toward* the car" — appearance, not context. Reranking with a VLM fixes the
ranking but costs seconds per query, which is the wrong place to spend time in
a database.

The fix is the oldest trick a database has: precompute the expensive operator
into an index and make the query a lookup. Concretely,

    ingest (once, offline)      query (every time, hot)
    ------------------------    -----------------------
    VLM reads 3 frames of the   text -> SigLIP text vector
    window and writes a
    relational caption          ONE matmul against ctx vectors
        |
    SigLIP text tower           (no VLM in the loop, ever)
        |
    target vector  -----------> train a small temporal tower to
                                predict it from cheap frame vectors

The VLM only ever labels a subset. The tower generalises the label to every
window, including windows ingested later, so a new day of footage costs a
forward pass over frame vectors instead of a day of VLM time. This is
pseudo-labelling in the sense of "Distilling Vision-Language Models on
Millions of Videos" (arXiv 2401.06129), applied at ingest instead of at
pretraining scale.

Three tables come out of this module, all ordinary Parquet:

    frame_vectors     ts, stream, vector[d]              one row per FRAME
    context_captions  ts, t1, stream, caption, vector[d] teacher labels
    context           ts, t1, stream, vector[d]          student output
"""
from __future__ import annotations

import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .embeddings import DEFAULT_MODEL, _embed_images, _load_model

# The teacher is asked for RELATIONS and CHANGE, not for a list of objects.
# An object list is exactly what SigLIP already encodes, so a caption that
# reads "a car, a road, a building" teaches the student nothing it does not
# already know. Everything interesting is in the verbs.
CAPTION_PROMPT = (
    "These frames are in time order from one short video clip. "
    "Reply with ONE sentence of at most 25 words describing what is "
    "happening: who or what is present, what each is doing, how they are "
    "positioned relative to each other, and how the scene moves or changes. "
    "Be concrete and literal. Do not say 'frame', 'image', or 'video'."
)

# SigLIP's text tower truncates at 64 tokens, so a rambling caption is
# silently cut mid-clause and the tail is lost anyway. Trim deliberately
# instead: drop the VLM's framing preamble, keep whole sentences.
#
# The preamble pattern is deliberately narrow — a leading PREPOSITIONAL
# phrase only ("In the first frame,", "Across this sequence,"). An earlier
# looser version also matched "The video shows a street with cars," and
# amputated the subject, leaving captions that began "and various buildings".
# Requiring a leading preposition and at most two filler words makes that
# impossible.
_PREAMBLE = re.compile(
    r"^(?:in|across|throughout|over|during)\s+(?:the|this|these)\s+"
    r"(?:[\w-]+\s+){0,2}?(?:frames?|images?|pictures?|sequence|clip|video)\s*,\s*",
    re.I)


def tidy_caption(text: str, max_words: int = 32) -> str:
    t = " ".join(text.strip().split())
    t = _PREAMBLE.sub("", t)
    parts = re.split(r"(?<=[.!?])\s+", t)
    # A generation cut off at max_tokens ends mid-clause. Keep only sentences
    # that actually terminate, unless that would leave nothing at all.
    whole = [p for p in parts if p.rstrip().endswith((".", "!", "?"))]
    parts = whole or parts[:1]
    out = []
    for p in parts:
        if out and len(" ".join(out + [p]).split()) > max_words:
            break
        out.append(p)
    t = " ".join(out).strip()
    return (t[:1].upper() + t[1:]) if t else " ".join(text.split())[:200]


# ---------------------------------------------------------------------------
# 1. Per-frame vectors — the sequence a temporal model needs
# ---------------------------------------------------------------------------
def embed_frames(store, frame_table="frames", model=None, width=512,
                 batch=32, incremental=True, verbose=True):
    """One SigLIP vector per frame → `frame_vectors`.

    Deliberately NOT pooled. Pooling is the student's job, and pooling here
    would throw away the only signal that distinguishes context from
    appearance. Decode goes through the same byte-range path queries use, so
    this costs the frames it reads and nothing else.
    """
    from PIL import Image

    from .video import FrameSet
    model = model or DEFAULT_MODEL
    frames = store.table(frame_table).scan()
    streams = sorted(set(frames.column("stream").to_pylist()))

    done = {}
    if incremental:
        try:
            prev = store.table("frame_vectors").scan()
            for s_, t_ in zip(prev.column("stream").to_pylist(),
                              prev.column("ts").to_pylist()):
                done[s_] = max(done.get(s_, -1), t_)
        except Exception:
            pass

    t_start = time.time()
    rows_ts, rows_stream, rows_vec = [], [], []
    for s in streams:
        sel = frames.filter(pc.equal(frames.column("stream"), s))
        if s in done:
            sel = sel.filter(pc.greater(sel.column("ts"), done[s]))
        if len(sel) == 0:
            continue
        decoded = FrameSet(store, frame_table, sel).decode(width=width)
        if verbose:
            print(f"  {s}: {len(decoded)} frames decoded", flush=True)
        for i in range(0, len(decoded), batch):
            chunk = decoded[i:i + batch]
            vecs = _embed_images([Image.fromarray(a) for _, a in chunk], model)
            for (ts, _), v in zip(chunk, vecs):
                rows_ts.append(int(ts))
                rows_stream.append(s)
                rows_vec.append(v)
    if not rows_ts:
        return {"frames": 0, "note": "nothing new (incremental)"}

    dim = len(rows_vec[0])
    tbl = pa.table({
        "ts": pa.array(rows_ts, pa.int64()),
        "stream": pa.array(rows_stream),
        "vector": pa.array([v.tolist() for v in rows_vec],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("frame_vectors").append(
        tbl, kind="embeddings",
        meta={"model": model, "dim": dim, "decode_width": width,
              "source_table": frame_table})
    return {"frames": len(tbl), "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 1)}


# ---------------------------------------------------------------------------
# 2. Window plan — shared by teacher and student so labels line up exactly
# ---------------------------------------------------------------------------
def plan_windows(store, window_s=2.0, stride_s=0.5, table="frame_vectors",
                 min_frames=4):
    """Sliding windows over each stream's timeline.

    Stride < window on purpose: overlapping windows are how a *sliding* index
    avoids the boundary problem where an event straddles two tumbling windows
    and lands strongly in neither. Merging overlaps back into one answer is
    already handled downstream by the segment merger.
    """
    fv = store.table(table).scan()
    win = int(window_s * 1e9)
    stride = int(stride_s * 1e9)
    out = []
    for s in sorted(set(fv.column("stream").to_pylist())):
        rows = fv.filter(pc.equal(fv.column("stream"), s))
        ts = np.sort(rows.column("ts").to_numpy())
        if len(ts) == 0:
            continue
        t = int(ts[0])
        end = int(ts[-1])
        while t <= end - win // 2:
            lo, hi = np.searchsorted(ts, [t, t + win])
            if hi - lo >= min_frames:
                out.append((s, t, t + win - 1))
            t += stride
    return out


def window_sequences(store, windows, table="frame_vectors", max_len=32):
    """(stream, t0, t1) → (T, d) float32 stack of that window's frame vectors.

    Subsampled to `max_len` evenly. A 2 s window at 16 Hz is 32 frames; the
    cap keeps the tower's cost independent of frame rate, which is what makes
    the same model valid across a 10 Hz LiDAR-synced camera and a 60 Hz one.
    """
    fv = store.table(table).scan()
    by_stream = {}
    for s in sorted(set(fv.column("stream").to_pylist())):
        rows = fv.filter(pc.equal(fv.column("stream"), s))
        ts = rows.column("ts").to_numpy()
        order = np.argsort(ts)
        vecs = np.asarray(rows.column("vector").to_pylist(), dtype=np.float32)
        by_stream[s] = (ts[order], vecs[order])
    seqs = []
    for (s, t0, t1) in windows:
        ts, vecs = by_stream[s]
        lo, hi = np.searchsorted(ts, [t0, t1 + 1])
        idx = np.arange(lo, hi)
        if len(idx) > max_len:
            idx = idx[np.linspace(0, len(idx) - 1, max_len).round().astype(int)]
        seqs.append(vecs[idx])
    return seqs


# ---------------------------------------------------------------------------
# 3. The teacher — a VLM that actually reads the pixels, run ONCE per window
# ---------------------------------------------------------------------------
def caption_windows(store, windows, frames_per_window=3, model_id=None,
                    max_tokens=64, width=448, verbose=True, limit=None):
    """VLM captions for `windows` → `context_captions` table.

    The VLM is shown several frames of the SAME window in order, so the
    caption can describe motion. A single-frame caption would be another
    appearance label and the student would learn nothing a mean-pool cannot
    already produce.
    """
    import tempfile
    from pathlib import Path

    from PIL import Image

    from .rerank import DEFAULT_VLM, _load
    from .video import FrameSet
    model_id = model_id or DEFAULT_VLM
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    vlm, processor, cfg, _, _ = _load(model_id)
    rot = store.meta.get("display", {}).get("rotate", 0)
    frames = store.table("frames").scan()
    tmpdir = Path(tempfile.mkdtemp(prefix="elidedb_ctx_"))

    if limit:
        windows = windows[:limit]
    prompt = apply_chat_template(processor, cfg, CAPTION_PROMPT,
                                 num_images=frames_per_window)

    recs = {"ts": [], "t1": [], "stream": [], "caption": []}
    t_start = time.time()
    for n, (s, t0, t1) in enumerate(windows):
        sel = frames.filter(pc.and_(
            pc.equal(frames.column("stream"), s),
            pc.and_(pc.greater_equal(frames.column("ts"), t0),
                    pc.less_equal(frames.column("ts"), t1))))
        fs = FrameSet(store, "frames", sel)
        dec = fs.decode(width=width)
        if len(dec) < 1:
            continue
        picks = np.linspace(0, len(dec) - 1,
                            min(frames_per_window, len(dec))).round().astype(int)
        paths = []
        for j, p in enumerate(picks):
            im = Image.fromarray(dec[p][1])
            if rot:
                im = im.rotate(rot, expand=True)
            fp = tmpdir / f"w{n}_{j}.jpg"
            im.save(fp, "JPEG", quality=85)
            paths.append(str(fp))
        while len(paths) < frames_per_window:      # pad short windows
            paths.append(paths[-1])
        r = generate(vlm, processor, prompt, image=paths,
                     max_tokens=max_tokens, verbose=False)
        cap = tidy_caption(r.text if hasattr(r, "text") else str(r))
        recs["ts"].append(t0)
        recs["t1"].append(t1)
        recs["stream"].append(s)
        recs["caption"].append(cap)
        if verbose and n % 10 == 0:
            el = time.time() - t_start
            print(f"  [{n + 1}/{len(windows)}] {el:.0f}s  {s} +"
                  f"{(t0 - int(frames.column('ts')[0].as_py())) / 1e9:.1f}s :: "
                  f"{cap[:90]}", flush=True)
    if not recs["ts"]:
        return {"captions": 0}

    # Caption → frozen SigLIP TEXT tower. This is the whole point of using
    # SigLIP as the teacher's codec: the target lands in the SAME space a
    # user's query lands in, so the student is learning to be the image side
    # of a dual encoder — not to regress an arbitrary embedding.
    vecs = embed_texts(recs["caption"])
    dim = vecs.shape[1]
    tbl = pa.table({
        "ts": pa.array(recs["ts"], pa.int64()),
        "t1": pa.array(recs["t1"], pa.int64()),
        "stream": pa.array(recs["stream"]),
        "caption": pa.array(recs["caption"]),
        "vector": pa.array([v.tolist() for v in vecs],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("context_captions").append(
        tbl, kind="embeddings",
        meta={"teacher": model_id, "text_model": DEFAULT_MODEL, "dim": dim,
              "frames_per_window": frames_per_window,
              "seconds": round(time.time() - t_start, 1)})
    return {"captions": len(tbl), "version": version,
            "seconds": round(time.time() - t_start, 1)}


def embed_texts(texts, model_id=DEFAULT_MODEL, batch=32):
    """Batched SigLIP text tower. Same normalisation as image vectors so the
    two are directly comparable by dot product."""
    import mlx.core as mx
    model, processor = _load_model(model_id)
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else "a scene" for t in texts[i:i + batch]]
        ti = processor(text=chunk, padding="max_length", max_length=64,
                       truncation=True, return_tensors="np")
        v = np.array(model.get_text_features(mx.array(ti["input_ids"])),
                     dtype=np.float32)
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 4. CaptionSpace — the output space, chosen by measurement
# ---------------------------------------------------------------------------
# Three candidate spaces were benchmarked against a VLM judge on windows the
# tower never trained on (mean yes/no logprob margin over each method's top-5,
# six relational queries, higher is better):
#
#     appearance only  (SigLIP image-text)                +0.276
#     caption EMBEDDING (SigLIP text tower, oracle)       +0.218   <- worse
#     VLM rerank at query time (2.1 s/query)              +0.314
#     caption TEXT, LSA-48                                +0.339   <- winner
#     caption TEXT, raw TF-IDF, fused with appearance     +0.345
#
# The embedding route loses because SigLIP's text tower is trained to sit
# near IMAGES, not near other text; comparing a query embedding to a caption
# embedding uses a geometry the model was never optimised for. Matching the
# caption as TEXT sidesteps that entirely.
#
# LSA-48 is chosen over raw TF-IDF despite scoring 0.006 lower: it is dense
# and fixed-width, so (a) it is a target a small tower can actually regress,
# which is what lets unlabelled windows get a context vector at all, and
# (b) it is one more fixed_size_list column, so every index already in the
# store consumes it unchanged.


class CaptionSpace:
    """TF-IDF + LSA over the caption corpus. Queries and captions share it.

    Not persisted as a pickle. The captions themselves are the durable
    artifact — they live in `context_captions` as ordinary Parquet — and the
    lexical index is refit from them on load (a few ms for this corpus) and
    cached. That keeps the index unconditionally consistent with the data and
    free of any sklearn version pinning. At corpus sizes where refitting
    stops being free, persist the vocabulary and the SVD basis; the interface
    does not change.
    """

    def __init__(self, vec, svd):
        self.vec, self.svd = vec, svd
        self.dim = svd.n_components

    @staticmethod
    def fit(texts, dim=48):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              sublinear_tf=True).fit(texts)
        X = vec.transform(texts)
        dim = int(min(dim, X.shape[1] - 1, len(texts) - 1))
        svd = TruncatedSVD(n_components=dim, random_state=0).fit(X)
        return CaptionSpace(vec, svd)

    def transform(self, texts):
        v = self.svd.transform(self.vec.transform(list(texts)))
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


_SPACE_CACHE: dict = {}


def caption_space(store, dim=48):
    t = store.table("context_captions")
    key = (str(store.dir), t.state().version, dim)
    if key not in _SPACE_CACHE:
        _SPACE_CACHE.clear()
        _SPACE_CACHE[key] = CaptionSpace.fit(
            t.scan().column("caption").to_pylist(), dim=dim)
    return _SPACE_CACHE[key]


# ---------------------------------------------------------------------------
# 5. The student — train the tower, then materialise context vectors
# ---------------------------------------------------------------------------
def _labelled(store, windows):
    """Join the caption table onto the window plan by (stream, ts)."""
    caps = store.table("context_captions").scan()
    key = {(s, t): i for i, (s, t) in enumerate(
        zip(caps.column("stream").to_pylist(), caps.column("ts").to_pylist()))}
    vecs = np.asarray(caps.column("vector").to_pylist(), dtype=np.float32)
    texts = caps.column("caption").to_pylist()
    keep, tv, tt = [], [], []
    for w in windows:
        i = key.get((w[0], w[1]))
        if i is not None:
            keep.append(w)
            tv.append(vecs[i])
            tt.append(texts[i])
    return keep, np.asarray(tv, dtype=np.float32), tt


def train_context(store, window_s=2.0, stride_s=0.5, val_frac=0.3,
                  d_in=48, d_out=48, epochs=400, cfg=None, verbose=True,
                  seed=0):
    """Fit the input PCA + tower to predict caption-LSA coordinates.

    The validation split is by TIME, never at random: windows slide with 75%
    overlap, so a random split puts near-duplicate windows on both sides and
    reports a score that measures nothing.
    """
    from .ctxtower import ContextCodec, retrieval_r1, save_tower, train_tower

    windows = plan_windows(store, window_s, stride_s)
    windows, _, cap_text = _labelled(store, windows)
    if len(windows) < 16:
        raise RuntimeError(f"only {len(windows)} captioned windows — run "
                           "caption_windows() first")

    fv = store.table("frame_vectors").scan()
    frame_mat = np.asarray(fv.column("vector").to_pylist(), dtype=np.float32)
    codec = ContextCodec.fit(frame_mat, d_in=d_in)
    space = caption_space(store, dim=d_out)

    seqs = [codec.encode(s) for s in window_sequences(store, windows)]
    targets = space.transform(cap_text)

    t0s = np.array([w[1] for w in windows])
    cut = np.quantile(t0s, 1.0 - val_frac)
    val_idx = np.where(t0s >= cut)[0]
    if verbose:
        print(f"  {len(windows)} labelled windows | train "
              f"{len(windows) - len(val_idx)} | val {len(val_idx)} "
              f"(time split at +{(cut - t0s.min()) / 1e9:.1f}s) | "
              f"target dim {space.dim}", flush=True)

    cfg = {**dict(d_in=codec.P_in.shape[0], d_out=space.dim), **(cfg or {})}
    model, info = train_tower(seqs, targets, windows, val_idx, cfg=cfg,
                              epochs=epochs, verbose=verbose, seed=seed)

    import mlx.core as mx
    val = np.zeros(len(seqs), bool)
    val[val_idx] = True
    Xva = _pad_stack([s for s, m in zip(seqs, val) if m])
    pred = np.array(model(mx.array(Xva)))
    pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
    tgt_va = targets[val]

    # Baseline: predict the TRAIN-set mean target for every window. That is
    # the best a model can do while knowing nothing about the specific window,
    # so any lift over it is genuine per-window information and not the corpus
    # prior leaking through.
    prior = targets[~val].mean(axis=0)
    prior /= np.linalg.norm(prior) + 1e-8
    prior = np.repeat(prior[None, :], len(tgt_va), axis=0)

    metrics = {
        "labelled_windows": len(windows),
        "train": int((~val).sum()), "val": int(val.sum()),
        "target_dim": space.dim,
        "val_R@1_tower": retrieval_r1(pred, tgt_va),
        "val_cos_tower": float((pred * tgt_va).sum(1).mean()),
        "val_cos_prior": float((prior * tgt_va).sum(1).mean()),
        "params": int(sum(v.size for v in _flat(model).values())),
        **{k: v for k, v in info.items() if k != "history"},
    }
    save_tower(model, codec,
               {"window_s": window_s, "stride_s": stride_s, "d_out": space.dim,
                "metrics": metrics, "history": info["history"]},
               store.dir / "models" / "context")
    return model, codec, metrics


def _flat(model):
    from mlx.utils import tree_flatten
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def _pad_stack(seqs):
    T = max(s.shape[0] for s in seqs)
    X = np.zeros((len(seqs), T, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        if len(s) < T:
            X[i, len(s):] = s[-1]
    return X


def build_context(store, window_s=None, stride_s=None, batch=64, verbose=True):
    """Materialise the `context` table: one context vector per window.

    Where a window has a teacher caption, its EXACT caption-LSA vector is
    stored. Where it does not, the tower's prediction is stored and the row is
    flagged `estimated`. This is the ordinary database distinction between a
    materialised value and an estimated one, and it is the point of having a
    student at all: the VLM labels what you can afford, the tower covers the
    rest, and the query does not care which it got.
    """
    import time

    import mlx.core as mx

    from .ctxtower import load_tower
    model, codec, meta = load_tower(store.dir / "models" / "context")
    window_s = window_s or meta.get("window_s", 2.0)
    stride_s = stride_s or meta.get("stride_s", 0.5)
    space = caption_space(store, dim=meta.get("d_out", 48))

    windows = plan_windows(store, window_s, stride_s)
    raw = window_sequences(store, windows)
    seqs = [codec.encode(s) for s in raw]
    # The mean-pooled appearance vector for the SAME window, stored alongside.
    # Fusing appearance with context otherwise needs a join between two tables
    # built on different window plans; keeping both columns in one row makes
    # the fused query two matmuls over one Parquet scan and no join at all.
    appear = np.stack([s.mean(axis=0) for s in raw])
    appear /= np.linalg.norm(appear, axis=1, keepdims=True) + 1e-8

    t_start = time.time()
    out = []
    for i in range(0, len(seqs), batch):
        out.append(np.array(model(mx.array(_pad_stack(seqs[i:i + batch])))))
    z = np.concatenate(out, axis=0)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    infer_s = time.time() - t_start

    caps = store.table("context_captions").scan()
    known = {(a, b): c for a, b, c in zip(caps.column("stream").to_pylist(),
                                          caps.column("ts").to_pylist(),
                                          caps.column("caption").to_pylist())}
    have = [(i, known[(w[0], w[1])]) for i, w in enumerate(windows)
            if (w[0], w[1]) in known]
    estimated = np.ones(len(windows), bool)
    if have:
        exact = space.transform([c for _, c in have])
        for (i, _), v in zip(have, exact):
            z[i] = v
            estimated[i] = False

    dim, adim = z.shape[1], appear.shape[1]
    tbl = pa.table({
        "ts": pa.array([w[1] for w in windows], pa.int64()),
        "t1": pa.array([w[2] for w in windows], pa.int64()),
        "stream": pa.array([w[0] for w in windows]),
        "vector": pa.array([v.tolist() for v in z],
                           pa.list_(pa.float32(), dim)),
        "appearance": pa.array([v.tolist() for v in appear],
                               pa.list_(pa.float32(), adim)),
        "estimated": pa.array(estimated.tolist(), pa.bool_()),
    })
    st = store.table("context").state()
    meta_out = {"dim": dim, "window_s": window_s, "stride_s": stride_s,
                "estimated_rows": int(estimated.sum()),
                "exact_rows": int((~estimated).sum())}
    if st.files:                                 # replace: one atomic commit
        import uuid as _uuid

        from .log import FileEntry
        from .store import write_parquet
        fname = f"part-{_uuid.uuid4().hex[:12]}.parquet"
        p = store.dir / "tables" / "context" / fname
        tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
        write_parquet(tbl, p)
        tsv = tbl.column("ts").to_numpy()
        version = store.table("context").log.commit(
            op="replace", kind="embeddings", schema=str(tbl.schema),
            add=[FileEntry(fname, len(tbl), p.stat().st_size,
                           int(tsv.min()), int(tsv.max()))],
            remove=[f.path for f in st.files], meta=meta_out)
    else:
        version = store.table("context").append(tbl, kind="embeddings",
                                                meta=meta_out)
    if verbose:
        print(f"  {len(tbl)} context vectors ({int((~estimated).sum())} exact, "
              f"{int(estimated.sum())} estimated) | tower inference "
              f"{infer_s * 1000:.0f} ms "
              f"({infer_s / len(tbl) * 1e6:.0f} us/window)")
    return {"windows": len(tbl), "dim": dim, "version": version,
            "inference_s": round(infer_s, 3),
            "us_per_window": round(infer_s / len(tbl) * 1e6, 1), **meta_out}


# ---------------------------------------------------------------------------
# 6. Post-training cellular turnover — capacity the data cannot support is
#    both latency and overfitting, so apoptosis pays twice.
# ---------------------------------------------------------------------------
def _prepare(store, window_s, stride_s, codec, val_frac, d_out):
    windows, _, texts = _labelled(store, plan_windows(store, window_s,
                                                      stride_s))
    seqs = [codec.encode(s) for s in window_sequences(store, windows)]
    targets = caption_space(store, dim=d_out).transform(texts)
    t0s = np.array([w[1] for w in windows])
    val = t0s >= np.quantile(t0s, 1.0 - val_frac)
    return windows, seqs, targets, val


def prune_context(store, val_frac=0.3, ppo_iters=40, sparsity_coef=0.15,
                  finetune_epochs=300, rebirth_fraction=0.5,
                  select_tolerance=0.03, verbose=True, seed=0):
    """apoptosis → re-settle → neurogenesis → re-settle, then COMPACT so the
    surviving channels are the only ones that cost anything."""
    import time

    import mlx.core as mx
    import mlx.nn as nn

    from .ctxprune import compact, get_channel_mask, run_pruning_cycle
    from .ctxtower import (load_tower, overlap_mask, retrieval_r1,
                           save_tower, siglip_loss)

    model, codec, meta = load_tower(store.dir / "models" / "context")
    windows, seqs, targets, val = _prepare(
        store, meta["window_s"], meta["stride_s"], codec, val_frac,
        meta.get("d_out", 48))

    Xtr = mx.array(_pad_stack([s for s, m in zip(seqs, ~val) if m]))
    Xva = mx.array(_pad_stack([s for s, m in zip(seqs, val) if m]))
    Ytr, Yva = mx.array(targets[~val]), mx.array(targets[val])
    ig_tr = mx.array(overlap_mask([w for w, m in zip(windows, ~val) if m]))
    ig_va = mx.array(overlap_mask([w for w, m in zip(windows, val) if m]))
    log_t = mx.array(np.float32(meta["metrics"]["log_t"]))
    bias = mx.array(np.float32(meta["metrics"]["bias"]))

    def _norm(a):
        return a * mx.rsqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-8)

    def _loss(m, x, y, ig):
        v = _norm(m(x))
        u = _norm(y)
        return siglip_loss(v, u, ignore=ig, log_t=log_t, bias=bias) \
            + 0.3 * mx.mean(1.0 - mx.sum(v * u, axis=-1))

    def loss_of(m):
        """The number PPO is scored against: VALIDATION retrieval loss. Using
        train loss here would reward a policy for keeping memorisers."""
        m.set_training(False)
        return float(_loss(m, Xva, Yva, ig_va).item())

    lg = nn.value_and_grad(model, lambda m: _loss(m, Xtr, Ytr, ig_tr))

    def r1(m):
        m.set_training(False)
        return retrieval_r1(np.array(_norm(m(Xva))), np.array(_norm(Yva)))

    def latency(m, reps=20):
        m.set_training(False)
        mx.eval(m(Xva))
        t = time.perf_counter()
        for _ in range(reps):
            mx.eval(m(Xva))
        return (time.perf_counter() - t) / reps / Xva.shape[0] * 1e6

    before = {"channels": int(get_channel_mask(model).sum()),
              "val_loss": loss_of(model), "val_R@1": r1(model),
              "us_per_window": latency(model),
              "params": int(sum(v.size for v in _flat(model).values()))}

    rec = run_pruning_cycle(model, np.array(Xtr), loss_of, lg,
                            ppo_iters=ppo_iters, sparsity_coef=sparsity_coef,
                            finetune_epochs=finetune_epochs,
                            rebirth_fraction=rebirth_fraction,
                            select_tolerance=select_tolerance, seed=seed,
                            verbose=verbose)

    # Compaction must be a no-op numerically — it only deletes channels the
    # mask already zeroed. Measure both sides and say so if they disagree,
    # rather than quietly shipping a tower that differs from the one the
    # search selected.
    restored = loss_of(model)
    model, kept = compact(model)
    after = {"channels": int(len(kept)), "val_loss": loss_of(model),
             "loss_before_compaction": restored, "val_R@1": r1(model),
             "us_per_window": latency(model),
             "params": int(sum(v.size for v in _flat(model).values()))}
    rec["before"], rec["after"] = before, after
    drift = abs(after["val_loss"] - restored)
    if drift > 1e-3:
        print(f"  WARNING: compaction changed val loss by {drift:.4f} "
              f"({restored:.4f} -> {after['val_loss']:.4f}) — expected ~0")
    if verbose:
        print(f"\n  channels {before['channels']} -> {after['channels']} | "
              f"params {before['params']:,} -> {after['params']:,} | "
              f"{before['us_per_window']:.1f} -> {after['us_per_window']:.1f} "
              f"us/window | val loss {before['val_loss']:.4f} -> "
              f"{after['val_loss']:.4f}")
    save_tower(model, codec,
               {**{k: v for k, v in meta.items() if k != "history"},
                "pruned": {"before": before, "after": after,
                           "ppo_history": rec.get("ppo_history"),
                           "keep_probs": rec.get("keep_probs"),
                           "reverse_attention": rec.get("reverse_attention"),
                           "selected": rec.get("selected"),
                           "stages": [{k: v for k, v in s.items()
                                       if k != "mask"} for s in rec["stages"]]}},
               store.dir / "models" / "context")
    return model, rec


# ---------------------------------------------------------------------------
# 7. The query path — two matmuls, no VLM, no decode
# ---------------------------------------------------------------------------
_CTX_CACHE: dict = {}


def _ctx_matrix(store):
    t = store.table("context")
    key = (str(store.dir), t.state().version)
    if key not in _CTX_CACHE:
        _CTX_CACHE.clear()          # one version at a time; it is a cache
        _CTX_CACHE[key] = np.stack([
            np.asarray(v, dtype=np.float32)
            for v in t.scan().column("vector").to_pylist()])
    return _CTX_CACHE[key]


def search(store, text, k=10, alpha=0.6, merge=True, t0=None, t1=None,
           streams=None, min_score=None, percentile=None, neg_weight=0.5):
    """Contextual search over the `context` table.

    `alpha` mixes two indexes held as columns of the same row, so this stays
    one scan and no join:
        alpha = 0.0  pure appearance — what plain semantic search does
        alpha = 1.0  pure context — the caption space, relations and change
        alpha = 0.6  default. Appearance anchors WHAT is in frame; context
                     decides whether it is doing the queried thing.

    Compositional grammar (AND / NOT / -term) works as in `search_text`;
    the appearance side scores every term, the context side scores the query
    as prose, which is where a relation survives.
    """
    from .embeddings import _parse_query, _rank, embed_text
    pos, neg = _parse_query(text)
    pos_app = np.stack([embed_text(p) for p in pos])
    neg_app = np.stack([embed_text(n) for n in neg]) if neg else None
    q_app = pos_app.mean(axis=0)
    q_app /= np.linalg.norm(q_app) + 1e-8

    q_ctx = caption_space(store).transform([" ".join(pos)])[0]
    ctx = {"vecs": _ctx_matrix(store), "q": q_ctx, "alpha": alpha}

    hits, stats = _rank(store, q_app, k, nprobe=0, merge=merge, t0=t0, t1=t1,
                        streams=streams, pos_vecs=pos_app, neg_vecs=neg_app,
                        neg_weight=neg_weight, min_score=min_score,
                        percentile=percentile, table="context",
                        column="appearance", ctx=ctx)
    stats["alpha"] = alpha
    stats["index"] = "context"
    return hits, stats


def explain(store, t0, t1, stream=None):
    """What the database believes is happening in a window — the teacher's own
    words. Lets a result be checked rather than trusted."""
    caps = store.table("context_captions").scan()
    out = []
    for s, a, b, c in zip(caps.column("stream").to_pylist(),
                          caps.column("ts").to_pylist(),
                          caps.column("t1").to_pylist(),
                          caps.column("caption").to_pylist()):
        if stream and s != stream:
            continue
        if b >= t0 and a <= t1:
            out.append({"stream": s, "t0": a, "t1": b, "caption": c})
    return out
