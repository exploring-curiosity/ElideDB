"""ITM cross-encoder rerank — the discarded head, wired for live queries.

InternVideo2's checkpoint ships an `itm_head` that every load discarded
("Some weights ... were not used"). It is the cross-encoder half of the
model's own retrieval recipe (BLIP-2 arXiv 2301.12597): text tokens
cross-attend to video tokens, a 2-class head reads the fused CLS. It
out-ranks everything else in this store — measured at k=1.5xsupport,
ITM alone 0.38/0.25 against the cosine ensemble's 0.32/0.21 and the
shipped RRF path's 0.29/0.23.

Two measurements decide the shape of this module:

  NO POOLING. Vision tokens are 1,025 x 1,408 per episode = 3.24 GB
  corpus-wide, four times the raw source, so storing them in the store
  is out. Pooling them is worse than out: 16x16 -> 8x8 per frame drops
  Spearman against the full-token score to 0.18, and to NEGATIVE on two
  of three probe queries. Cross-attention does not tolerate a reduced
  key set. So the tokens are a disposable CACHE outside the store's
  byte budget, never a table.

  CASCADE, NOT SCAN. Reranking the top-N of the cheap fused ranking
  reaches the full-scan number exactly: N=500 gives 0.40/0.27, N=150
  already gives 0.38/0.25, and every query with support under ~200 is
  saturated at N=150. So N scales with the ceiling k rather than the
  corpus, which is L7 ("ITM is evidence inside a candidate set, not
  authority over the corpus") arrived at from the cost side.

ITM IS A STAGE, NOT A CHANNEL — and that distinction was measured, not
chosen. Making it a weighted RRF voter and refitting cost 0.38 -> 0.27
for two compounding reasons: RRF converts every channel to RANKS, which
throws away the logit margin's scale, and that scale is where the
cross-encoder's separation lives (the 0.40 offline number came from
adding z-scores, not ranks); and the k-ladder refit then optimized the
ladder mean away from the 1.5x-support operating point the product
metric uses. So ITM stays a rerank STAGE alongside NMS and the
confidence cut - stages the fitter switches on and off rather than
weights - and the fit models it as such.

The cache is content-addressed by (episode, model) and capped; deleting
it costs time, never correctness.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_S: dict = {}
# 3.24 GB holds the whole 1,122-episode corpus. This is a DISPOSABLE
# cache under the store's _cache/, not a table: it never enters the
# store's byte ledger, and deleting it costs recompute time, never
# correctness. Override with ELIDEDB_ITM_CACHE_GB.
CACHE_CAP_GB = float(__import__("os").environ.get(
    "ELIDEDB_ITM_CACHE_GB", "4.0"))
NF = 4


def _head():
    """The checkpoint's own 2-class matcher — AutoModel drops it because
    the vendored class never declares the attribute."""
    if "head" in _S:
        return _S["head"], _S["m"], _S["dev"]
    import torch

    from .iv2 import MDIR, load_model
    sd = torch.load(f"{MDIR}/pytorch_model.bin", map_location="cpu",
                    weights_only=True, mmap=True)
    m, dev = load_model()
    h = torch.nn.Linear(sd["itm_head.weight"].shape[1], 2)
    h.load_state_dict({"weight": sd["itm_head.weight"],
                       "bias": sd["itm_head.bias"]})
    h = h.to(dev, m.dtype).eval()
    _S["head"], _S["m"], _S["dev"] = h, m, dev
    return h, m, dev


def _cache_dir(store):
    d = Path(store.dir) / "_cache" / "itm_tokens"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(store, stream, ts):
    key = hashlib.sha1(f"{stream}|{ts}|iv2-1b|{NF}".encode()).hexdigest()
    return _cache_dir(store) / f"{key}.npy"


def _evict(store):
    """Keep the disposable cache under CACHE_CAP_GB, oldest first."""
    d = _cache_dir(store)
    files = sorted(d.glob("*.npy"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    cap = CACHE_CAP_GB * 1e9
    while total > cap and files:
        p = files.pop(0)
        total -= p.stat().st_size
        try:
            p.unlink()
        except OSError:
            pass


def _vision_tokens(store, frames_tbl, key):
    """Cached (1, T, C) vision tokens for one episode."""
    import cv2
    import torch

    from .iv2 import V_MEAN, V_STD
    from .video import FrameSet
    s, a, b = key
    p = _cache_path(store, s, a)
    _, m, dev = _head()
    if p.exists():
        arr = np.load(p)
        return torch.from_numpy(arr).to(dev, m.dtype)
    import pyarrow.compute as pc
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), s),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                pc.less_equal(frames_tbl.column("ts"), b))))
    if len(sel) < 2:
        return None
    pi = np.linspace(0, len(sel) - 1, min(NF, len(sel))).round().astype(int)
    try:
        dec = FrameSet(store, "frames", sel.take(pi)).decode(width=224)
    except Exception:
        return None
    fr = [f for _, f in sorted(dec)]
    if len(fr) < 2:
        return None
    fs = [cv2.resize(f, (224, 224)) for f in fr]
    x = (np.stack(fs).astype(np.float32) / 255.0 - V_MEAN) / V_STD
    px = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(dev, m.dtype)
    with torch.no_grad():
        vis, _ = m.encode_vision(px, test=True)
    np.save(p, vis.cpu().numpy())
    _evict(store)
    return vis


def itm_scores(store, text, keys):
    """Logit margin P(match) - P(no match) for each candidate episode.

    Returns np.array aligned with `keys`; NaN where frames are
    undecodable. The vision pass dominates and is query-independent, so
    a repeated query over the same candidates is nearly free."""
    import torch
    head, m, dev = _head()
    frames_tbl = store.table("frames").scan()
    tok = m.tokenizer(text, padding="max_length", truncation=True,
                      max_length=m._config.max_txt_l,
                      return_tensors="pt").to(dev)
    out = np.full(len(keys), np.nan, np.float32)
    for i, k in enumerate(keys):
        vis = _vision_tokens(store, frames_tbl, k)
        if vis is None:
            continue
        with torch.no_grad():
            vam = torch.ones(vis.shape[:2], dtype=torch.long, device=dev)
            o = m.get_text_encoder()(
                tok.input_ids, attention_mask=tok.attention_mask,
                encoder_hidden_states=vis, encoder_attention_mask=vam,
                return_dict=True, mode="multi_modal")
            lg = head(o.last_hidden_state[:, 0]).float()[0]
        out[i] = float(lg[1] - lg[0])
    return out


def rerank_depth(k_max, n_total):
    """Candidates to rerank. Measured: N=150 saturates every query with
    support under ~200; only the 247-support query needed 500. Scaling
    with the ceiling rather than the corpus keeps the cost proportional
    to what the caller actually asked for."""
    return int(min(max(150, 2 * k_max), n_total))
