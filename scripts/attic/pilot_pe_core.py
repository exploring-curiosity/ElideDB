"""Base-encoder upgrade evaluation: Meta Perception Encoder (PE-Core-L),
the current published SOTA for zero-shot video-text retrieval — open
weights, CLIP-style, via open_clip. Same balanced sample, both metrics
(precision@10 AND published-style Recall@K), vs the shipping fusion."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elidedb import Store                                    # noqa: E402
from regress10 import QUERIES                                # noqa: E402


def main():
    import torch
    import open_clip
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(
        "PE-Core-L-14-336", pretrained="meta")
    tok = open_clip.get_tokenizer("PE-Core-L-14-336")
    model = model.to(dev).eval()

    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [(stream_of.get(int(i)), int(a), int(b), (k or "").lower())
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    rng = np.random.default_rng(0)
    chosen = {}
    for q, pred in QUERIES:
        rel = [e for e in eps if pred(e[3])]
        for e in (rel if len(rel) <= 25 else
                  [rel[i] for i in rng.choice(len(rel), 25,
                                              replace=False)]):
            chosen[(e[0], e[1])] = e
    dist = [e for e in eps if (e[0], e[1]) not in chosen]
    for e in [dist[i] for i in rng.choice(len(dist),
                                          min(150, len(dist)),
                                          replace=False)]:
        chosen[(e[0], e[1])] = e
    sample = list(chosen.values())

    from elidedb.video import FrameSet
    frames_tbl = db.table("frames").scan()
    feats, labels = [], []
    t0 = time.time()
    for i, (s, a, b, lab) in enumerate(sample):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 8:
            continue
        pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
        dec = FrameSet(db, "frames", sel.take(pick)).decode(width=336)
        if len(dec) < 8:
            continue
        imgs = torch.stack([pre(Image.fromarray(d[1]))
                            for d in sorted(dec)]).to(dev)
        with torch.no_grad():
            f = model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float().numpy())
        labels.append(lab)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample)} (t={time.time() - t0:.0f}s)",
                  flush=True)
    ms = (time.time() - t0) / max(len(feats), 1) * 1e3
    print(f"embedded {len(feats)} recordings ({ms:.0f} ms/recording,"
          f" 8 frames each)", flush=True)

    with torch.no_grad():
        tt = model.encode_text(tok([q for q, _ in QUERIES]).to(dev))
    T = (tt / tt.norm(dim=-1, keepdim=True)).cpu().float().numpy()

    p10 = 0
    r1 = r5 = r10 = 0
    for qi, (q, pred) in enumerate(QUERIES):
        sc = np.array([np.sort(f @ T[qi])[-5:].mean() for f in feats])
        order = np.argsort(-sc)
        g = sum(pred(labels[i]) for i in order[:10])
        p10 += g
        hr = [i for i, idx in enumerate(order) if pred(labels[idx])]
        fr = hr[0] if hr else 999
        r1 += fr < 1; r5 += fr < 5; r10 += fr < 10
        print(f"{g:2d}/10  {q}")
    n = len(QUERIES)
    print(f"\n== PE-Core-L: precision@10 {p10}/140 | "
          f"R@1 {100*r1/n:.0f}% R@5 {100*r5/n:.0f}% R@10 {100*r10/n:.0f}% "
          f"({ms:.0f} ms/recording) ==")


if __name__ == "__main__":
    main()
