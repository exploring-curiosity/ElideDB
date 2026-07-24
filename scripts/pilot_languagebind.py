"""Stock-component evaluation: LanguageBind_Video_FT as the general
video-text encoder — NATIVELY co-trained on generic video-text pairs
(nothing of ours), the candidate to raise day-one quality on unseen data.

Graded on the same 14-query bench as the shipping system, top-10 against
episode labels, on a balanced subsample of recordings. Adopt only if it
beats the appearance channel where the system is measured weakest
(relational queries), at acceptable ingest cost.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# torchvision removed functional_tensor; pytorchvideo still imports it
import torchvision.transforms.functional as _F
sys.modules["torchvision.transforms.functional_tensor"] = _F

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elidedb import Store                                    # noqa: E402
from regress10 import QUERIES                                # noqa: E402


def main():
    import torch
    from languagebind import (LanguageBind, to_device,
                              transform_dict, LanguageBindImageTokenizer)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip_type = {"video": "LanguageBind_Video_FT"}
    model = LanguageBind(clip_type=clip_type, cache_dir="./.cache_lb")
    model = model.to(device).eval()
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        "LanguageBind/LanguageBind_Image", cache_dir="./.cache_lb")
    video_transform = transform_dict["video"](model.modality_config["video"])

    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [(stream_of.get(int(i)), int(a), int(b), (k or "").lower())
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    # balanced subsample: for each bench query take up to 25 relevant
    # episodes + shared distractors, so every query CAN reach 10/10
    rng = np.random.default_rng(0)
    chosen = {}
    for q, pred in QUERIES:
        rel = [e for e in eps if pred(e[3])]
        for e in (rel if len(rel) <= 25 else
                  [rel[i] for i in rng.choice(len(rel), 25, replace=False)]):
            chosen[(e[0], e[1])] = e
    dist = [e for e in eps if (e[0], e[1]) not in chosen]
    for e in [dist[i] for i in rng.choice(len(dist),
                                          min(150, len(dist)),
                                          replace=False)]:
        chosen[(e[0], e[1])] = e
    sample = list(chosen.values())
    print(f"{len(sample)} recordings in the evaluation sample", flush=True)

    from elidedb.video import FrameSet
    frames_tbl = db.table("frames").scan()
    import tempfile
    import subprocess
    from elidedb.fftools import find as _find
    tmp = Path(tempfile.mkdtemp(prefix="lb_"))

    vecs, labels = [], []
    t0 = time.time()
    for i, (s, a, b, lab) in enumerate(sample):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 4:
            continue
        pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
        dec = FrameSet(db, "frames",
                       sel.take(np.unique(pick))).decode(width=224)
        if len(dec) < 4:
            continue
        # LanguageBind's video pipeline reads FILES via decord — write a
        # tiny mp4 of the 8 keyframes
        import imageio.v3 as iio
        fp = tmp / f"c{i}.mp4"
        iio.imwrite(fp, np.stack([d[1] for d in sorted(dec)]), fps=2,
                    codec="libx264")
        try:
            vt = video_transform([str(fp)])
            inp = {"video": to_device(vt, device)}
            with torch.no_grad():
                out = model(inp)
            v = out["video"][0].float().cpu().numpy()
        except Exception as e:
            print("skip:", type(e).__name__, str(e)[:80])
            continue
        vecs.append(v / (np.linalg.norm(v) + 1e-8))
        labels.append(lab)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)} (t={time.time() - t0:.0f}s)",
                  flush=True)
    V = np.stack(vecs)
    L = labels
    ms = (time.time() - t0) / max(len(V), 1) * 1e3
    print(f"embedded {len(V)} clips, {ms:.0f} ms/clip", flush=True)

    # text side: the model's OWN tower
    qs = [q for q, _ in QUERIES]
    tok = tokenizer(qs, max_length=77, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        tout = model({"language": to_device(tok, device)})
    T = tout["language"].float().cpu().numpy()
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8

    total = 0
    for qi, (q, pred) in enumerate(QUERIES):
        sc = V @ T[qi]
        top = np.argsort(-sc)[:10]
        n = sum(pred(L[i]) for i in top)
        total += n
        print(f"{n:2d}/10  {q}")
    print(f"\n== LanguageBind: {total}/140 on the balanced sample "
          f"({ms:.0f} ms/clip ingest) ==")


if __name__ == "__main__":
    main()
