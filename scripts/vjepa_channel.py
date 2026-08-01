"""VIDEO-NATIVE channel: V-JEPA 2 clip embeddings, no captions anywhere.

User directive: video->text->text-retrieval is the expensive, lossy
detour; the model must be video-native. Pipeline, measured before wired:

1. INGEST: every recording -> one V-JEPA 2 clip embedding (16 frames,
   ViT-L, mean-pooled tokens). ~291 ms/clip on MPS, background.
2. TEXT BRIDGE, pixels-only: ridge regression from the recording's mean
   SigLIP frame embedding -> its V-JEPA embedding, trained on OUR OWN
   corpus pairs (no captions, no labels — the two encoders watched the
   same pixels, the map between their spaces is self-supervised). A text
   query rides SigLIP's pretrained text-image alignment, then crosses
   the bridge into V-JEPA space where MOTION structure lives.
3. EVAL before wiring: close/open direction AUC + query-class ranking vs
   the SigLIP appearance baseline, same recordings, same grading.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402

MID = "facebook/vjepa2-vitl-fpc64-256"


def ingest(db):
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    from elidedb.video import FrameSet
    proc = AutoVideoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=torch.float16) \
        .to("mps").eval()
    ep = db.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = db.table("frames").scan()
    rows_s, rows_a, rows_b, vecs = [], [], [], []
    t0 = time.time()
    # PER-ITERATION progress. The old "every 200th recording"
    # print moved a bar three times over a 600-item run, which
    # tells you nothing about whether it is alive between them.
    from tqdm import tqdm
    _bar = tqdm(total=len(recs), desc="vjepa", unit="rec",
                dynamic_ncols=True, mininterval=0.3)
    for ri, (s, a, b) in enumerate(recs):
        _bar.update(1)
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 4:
            continue
        pick = np.linspace(0, len(sel) - 1, 16).round().astype(int)
        dec = FrameSet(db, "frames",
                       sel.take(np.unique(pick))).decode(width=256)
        if len(dec) < 4:
            continue
        arr = [d[1] for d in sorted(dec)]
        while len(arr) < 16:
            arr.append(arr[-1])
        inp = proc(arr[:16], return_tensors="pt")
        pv = inp["pixel_values_videos"].to("mps", torch.float16)
        with torch.no_grad():
            out = model(pixel_values_videos=pv)
        v = out.last_hidden_state[0].mean(0).float().cpu().numpy()
        v /= np.linalg.norm(v) + 1e-8
        rows_s.append(s); rows_a.append(a); rows_b.append(b)
        vecs.append(v.astype(np.float32))
    _bar.close()
    V = np.stack(vecs)
    dim = V.shape[1]
    flat = np.ascontiguousarray(V).reshape(-1)
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(flat), dim),
    })
    order = pc.sort_indices(tbl.column("ts"))
    tbl = tbl.take(order)
    db.table("vjepa_vectors").append(tbl, kind="embeddings",
                                     meta={"model": MID, "dim": dim,
                                           "frames_per_clip": 16})
    return {"recordings": len(tbl), "dim": dim,
            "seconds": round(time.time() - t0, 1)}


def bridge_and_eval(db):
    from elidedb.embeddings import _vec_table
    from elidedb.context import embed_texts
    jt, J = _vec_table(db, "vjepa_vectors")
    js = np.array(jt.column("stream").to_pylist())
    ja = np.array([int(v) for v in jt.column("ts").to_pylist()])
    jb = np.array([int(v) for v in jt.column("t1").to_pylist()])
    ft, F = _vec_table(db, "frame_vectors")
    fs = np.array(ft.column("stream").to_pylist())
    fts = np.array([int(v) for v in ft.column("ts").to_pylist()])
    # mean SigLIP frame vec per recording, matched row-for-row with J
    S = np.zeros((len(J), F.shape[1]), np.float32)
    for i in range(len(J)):
        m = (fs == js[i]) & (fts >= ja[i]) & (fts <= jb[i])
        if m.any():
            v = np.asarray(F[np.where(m)[0]]).mean(0)
            S[i] = v / (np.linalg.norm(v) + 1e-8)
    # ridge: SigLIP-space -> V-JEPA-space, pixels-only supervision
    lam = 1e-2
    W = np.linalg.solve(S.T @ S + lam * np.eye(S.shape[1]),
                        S.T @ np.asarray(J)).astype(np.float32)
    # fit quality (held-out 20%)
    n = len(S); cut = int(n * 0.8)
    Wtr = np.linalg.solve(S[:cut].T @ S[:cut] + lam * np.eye(S.shape[1]),
                          S[:cut].T @ np.asarray(J[:cut])).astype(np.float32)
    P = S[cut:] @ Wtr
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    fid = float((P * np.asarray(J[cut:])).sum(1).mean())

    # eval: direction + class ranking, episode labels (eval-only)
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    lab = {}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        if k:
            lab[(stream_of.get(int(i)), int(a))] = k.lower()
    L = np.array([lab.get((js[i], int(ja[i])), "") for i in range(len(J))])

    def q2j(q):
        z = embed_texts([q])[0] @ W
        return z / (np.linalg.norm(z) + 1e-8)

    def auc(pos, neg):
        return float(np.mean([[p > n_ for n_ in neg] for p in pos]))

    rep = {"bridge_fid_heldout": round(fid, 3)}
    close = np.array(["close" in x and "drawer" in x for x in L])
    opn = np.array(["open" in x and "drawer" in x and "put" not in x
                    for x in L])
    for name, space, tq in (("vjepa", np.asarray(J), q2j),
                            ("siglip", S, lambda q: embed_texts([q])[0])):
        qc, qo = tq("closing the drawer"), tq("opening the drawer")
        d = space @ (qc - qo)
        rep[f"{name}_close_vs_open_auc"] = round(
            auc(d[close], d[opn]), 3)
        sc = space @ qc
        top = np.argsort(-sc)[:10]
        rep[f"{name}_close_top10"] = int(close[top].sum())
    # class ranking sample: colored-object queries
    for cname, pred, q in (
            ("yellow", lambda x: "yellow" in x, "picking up a yellow object"),
            ("pot", lambda x: "pot" in x or "burner" in x,
             "moving the pot to the burner")):
        mask = np.array([pred(x) for x in L])
        for name, space, tq in (("vjepa", np.asarray(J), q2j),
                                ("siglip", S,
                                 lambda q_: embed_texts([q_])[0])):
            sc = space @ tq(q)
            rep[f"{name}_{cname}_top10"] = int(
                mask[np.argsort(-sc)[:10]].sum())
    np.save(db.dir / "models" / "vjepa_bridge_W.npy", W)
    return rep


def main():
    db = Store.open("lake/bridge4h")
    try:
        have = len(db.table("vjepa_vectors").scan())
    except Exception:
        have = 0
    out = {}
    if not have:
        out["ingest"] = ingest(db)
    out["eval"] = bridge_and_eval(db)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
