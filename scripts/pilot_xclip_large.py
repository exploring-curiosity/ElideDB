"""Stock-component evaluation, attempt 2: X-CLIP LARGE (natively
co-trained video-text, transformers-native — no env fork needed).
LanguageBind is unrunnable against transformers 5.x (KeyError inside its
bundled vision model); the base X-CLIP failed earlier — large is the
fair test of the family. Same balanced sample, same 14-query grading.
"""
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

MID = "microsoft/xclip-large-patch14"


def main():
    import torch
    from transformers import AutoProcessor, XCLIPModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = XCLIPModel.from_pretrained(MID).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MID)
    # the large checkpoint ships an incomplete preprocessor config that
    # silently DROPS videos (measured: only text keys came back); the
    # base checkpoint's video pipeline is the identical CLIP transform
    vproc = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")

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
    print(f"{len(sample)} recordings in sample", flush=True)

    from elidedb.video import FrameSet
    frames_tbl = db.table("frames").scan()
    vids, labels = [], []
    for s, a, b, lab in sample:
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 8:
            continue
        pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
        dec = FrameSet(db, "frames", sel.take(pick)).decode(width=224)
        if len(dec) < 8:
            continue
        vids.append([d[1] for d in sorted(dec)])
        labels.append(lab)
    print(f"{len(vids)} clips decoded", flush=True)

    t0 = time.time()
    V, tq = [], None
    with torch.no_grad():
        for i in range(0, len(vids), 4):
            batch = vids[i:i + 4]
            inp = proc(text=[q for q, _ in QUERIES],
                       return_tensors="pt", padding=True)
            pv = torch.cat([vproc(images=list(v),
                                  return_tensors="pt")["pixel_values"]
                            for v in batch])
            inp["pixel_values"] = pv
            out = model(**{k: v.to(dev) for k, v in inp.items()})
            V.append(out.video_embeds.cpu().float().numpy())
            if tq is None:
                ti = {k: v.to(dev) for k, v in inp.items()
                      if k in ("input_ids", "attention_mask")}
                tt = model.get_text_features(**ti)
                if not torch.is_tensor(tt):
                    tt = model.text_projection(tt.pooler_output)
                tq = tt.cpu().float().numpy()
    V = np.concatenate(V)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    tq /= np.linalg.norm(tq, axis=1, keepdims=True) + 1e-8
    ms = (time.time() - t0) / len(V) * 1e3
    print(f"embedded {len(V)} clips, {ms:.0f} ms/clip", flush=True)

    total = 0
    for qi, (q, pred) in enumerate(QUERIES):
        sc = V @ tq[qi]
        top = np.argsort(-sc)[:10]
        n = sum(pred(labels[i]) for i in top)
        total += n
        print(f"{n:2d}/10  {q}")
    print(f"\n== X-CLIP-large: {total}/140 balanced sample "
          f"({ms:.0f} ms/clip) ==")


if __name__ == "__main__":
    main()
