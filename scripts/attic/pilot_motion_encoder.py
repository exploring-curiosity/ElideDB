"""Pilot: does a video-NATIVE encoder beat SigLIP appearance on verb queries?

The Marengo lesson (multi-vector: motion must be its own channel) says a
video encoder with temporal attention should rank action queries better
than a frame-appearance model. Measured here per verb class on truth
episodes: X-CLIP (cross-frame attention, video-text contrastive) vs the
store's SigLIP appearance embeddings, same episodes, same queries.
AUC(own class vs rest) per query. Direction (close vs open) reported
separately — literature says even video models fail it; swap-contrast
verification covers that case regardless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

CLASSES = {
    "close": lambda k: "close" in k and "put" not in k and "take" not in k,
    "open": lambda k: "open" in k and "put" not in k and "take" not in k,
    "putin": lambda k: ("put" in k or "place" in k) and (" in" in k or "into" in k or "inside" in k),
    "takeout": lambda k: "take" in k or "remove" in k or "out of" in k,
    "move": lambda k: "move" in k or "push" in k or "slide" in k,
}
QUERIES = {
    "close": "a robot arm closing a drawer",
    "open": "a robot arm opening a drawer",
    "putin": "a robot arm putting an object into a drawer",
    "takeout": "a robot arm taking an object out of a container",
    "move": "a robot arm pushing an object across the table",
}
N_PER = 12


def episodes_by_class(db):
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    out = {c: [] for c in CLASSES}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        k = (k or "").lower()
        s = stream_of.get(int(i))
        if not s:
            continue
        for c, f in CLASSES.items():
            if f(k) and len(out[c]) < N_PER:
                out[c].append((s, int(a), int(b)))
                break
    return out


def decode8(db, frames_tbl, s, a, b, width=224):
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), s),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                pc.less_equal(frames_tbl.column("ts"), b))))
    if len(sel) < 8:
        return None
    pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
    dec = FrameSet(db, "frames", sel.take(pick)).decode(width=width)
    if len(dec) < 8:
        return None
    return [d[1] for d in sorted(dec)]


def auc(pos, neg):
    return float(np.mean([[p > n for n in neg] for p in pos]))


def main():
    import torch
    from transformers import AutoProcessor, XCLIPModel

    db = Store.open("lake/bridge4h")
    frames_tbl = db.table("frames").scan()
    eps = episodes_by_class(db)
    print({c: len(v) for c, v in eps.items()})

    # ---- X-CLIP video features --------------------------------------------
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "microsoft/xclip-base-patch16"
    model = XCLIPModel.from_pretrained(model_id).to(dev).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    vids, labels, keys = [], [], []
    for c, lst in eps.items():
        for s, a, b in lst:
            f = decode8(db, frames_tbl, s, a, b)
            if f is not None:
                vids.append(f)
                labels.append(c)
                keys.append((s, a, b))
    labels = np.array(labels)
    print(f"{len(vids)} episodes decoded")

    import time
    xv, tq = [], None
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(vids), 4):
            batch = vids[i:i + 4]
            inp = proc(text=list(QUERIES.values()),
                       videos=[[fr for fr in v] for v in batch],
                       return_tensors="pt", padding=True)
            out = model(**{k: v.to(dev) for k, v in inp.items()})
            xv.append(out.video_embeds.cpu().float().numpy())
            if tq is None:
                # unconditioned dual-encoder text side (index-only law):
                # X-CLIP's forward conditions text on each video, which a
                # precomputed index cannot use
                ti = {k: v.to(dev) for k, v in inp.items()
                      if k in ("input_ids", "attention_mask")}
                t = model.get_text_features(**ti)
                if not torch.is_tensor(t):
                    t = model.text_projection(t.pooler_output)
                tq = t.cpu().float().numpy()
    xv = np.concatenate(xv)
    xv /= np.linalg.norm(xv, axis=1, keepdims=True) + 1e-8
    ms = (time.time() - t0) / len(vids) * 1e3
    print(f"x-clip video: {ms:.0f} ms/episode on {dev}")
    tq /= np.linalg.norm(tq, axis=1, keepdims=True) + 1e-8

    # ---- SigLIP appearance baseline: pooled store embeddings per episode --
    from elidedb.embeddings import _vec_table
    from elidedb.context import embed_texts
    tbl, vecs = _vec_table(db, "embeddings")
    ws = tbl.column("stream").to_pylist()
    wa = np.array([int(v) for v in tbl.column("ts").to_pylist()])
    wb = np.array([int(v) for v in tbl.column("t1").to_pylist()])
    ws = np.array(ws)
    sv = []
    for (s, a, b) in keys:
        m = (ws == s) & (wb >= a) & (wa <= b)
        v = vecs[m].mean(0) if m.any() else np.zeros(vecs.shape[1])
        sv.append(v / (np.linalg.norm(v) + 1e-8))
    sv = np.stack(sv)
    sq = embed_texts(list(QUERIES.values()))

    # ---- per-class AUC ----------------------------------------------------
    print(f"\n{'query':8s} {'X-CLIP(motion)':>14s} {'SigLIP(appear)':>14s}")
    for qi, c in enumerate(QUERIES):
        for name, ep_v, q_v in (("x", xv, tq), ("s", sv, sq)):
            sc = ep_v @ q_v[qi]
            a = auc(sc[labels == c], sc[labels != c])
            if name == "x":
                ax = a
            else:
                print(f"{c:8s} {ax:14.2f} {a:14.2f}")
    # direction specifically
    for name, ep_v, q_v in (("X-CLIP", xv, tq), ("SigLIP", sv, sq)):
        sc, so = ep_v @ q_v[0], ep_v @ q_v[1]
        d = auc((sc - so)[labels == "close"], (sc - so)[labels == "open"])
        a = auc(sc[labels == "close"], sc[labels == "open"])
        print(f"direction {name}: close-vs-open abs {a:.2f} | "
              f"query-contrast {d:.2f}")


if __name__ == "__main__":
    main()
