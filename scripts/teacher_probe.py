"""CAN an expensive judge reach 0.90? Measure before building on it.

The plan is a costly teacher that hits the product metrics, then an
FDNN-principled student distilled from it. The whole plan rests on the
teacher being right, and there is a standing measurement saying it may
not be: the 7B judge once returned ~1.0 confidence on sets that were
visually ~10% pure, which is why VLM judges were removed from every
shipping path. Being wrong-but-confident is fatal for a TEACHER in a
way it is not for a channel - a student distilled from a bad teacher
learns the teacher's errors and cannot be debugged afterwards.

So this asks the question directly, small, on a stratified slice with
known truth, across three support sizes (2, 18, 165):

  AUC        does the margin separate true from false AT ALL
  yield/prec the product metrics, on the sample
  agreement  how often the sign of the margin matches the truthset

The teacher sees FRAMES ONLY - the query text and pixels, nothing about
the dataset. Sampling is stratified rather than random so the slice
contains positives at all; the numbers are a calibration diagnostic,
not a corpus claim, and a sample this size flatters ranking metrics
(the crop channel scored 1.00 on 80 episodes and 0.00 on 1,121).

  python scripts/teacher_probe.py [--q 9,8,4] [--neg 40] [--model 7b]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402

NF = 4


def clip_margin(images, question, model_id):
    """P(yes) - P(no) for a CLIP: all frames in one forward pass, so the
    judge sees the action rather than a still."""
    import tempfile

    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    from elidedb.rerank import _load
    model, processor, cfg, yes_ids, no_ids = _load(model_id)
    prompt = apply_chat_template(processor, cfg, question,
                                 num_images=len(images))
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, im in enumerate(images):
            p = f"{td}/f{i}.jpg"
            im.save(p, "JPEG", quality=90)
            paths.append(p)
        r = generate(model, processor, prompt, image=paths,
                     max_tokens=1, verbose=False)
    if r.logprobs is None:
        return 0.0
    a = mx.array(r.logprobs).reshape(-1)
    return (max(float(a[i]) for i in yes_ids)
            - max(float(a[i]) for i in no_ids))


def main():
    from PIL import Image

    from elidedb.video import FrameSet

    argv = sys.argv
    qs = [int(x) for x in (argv[argv.index("--q") + 1].split(",")
                           if "--q" in argv else ["9", "8", "4"])]
    nneg = int(argv[argv.index("--neg") + 1]) if "--neg" in argv else 40
    tag = argv[argv.index("--model") + 1] if "--model" in argv else "7b"
    from elidedb.rerank import DEEP_VLM, DEFAULT_VLM
    model_id = DEEP_VLM if tag == "7b" else DEFAULT_VLM

    db = Store.open("lake/bench")
    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = db.table("frames").scan()
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    def frames_of(s, a, b):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            return []
        pi = np.linspace(0, len(sel) - 1, min(NF, len(sel))).round().astype(int)
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode(width=336)
        except Exception:
            return []
        return [Image.fromarray(f) for _, f in sorted(dec)]

    print(f"judge = {model_id}\n")
    rng = np.random.default_rng(0)
    for qi in qs:
        text = QUERIES[qi]
        q = (f"Here are {NF} frames in time order from one video clip. "
             f"Does the clip show: {text}? Answer only yes or no.")
        pos = [k for k in keys if truth.get((qi, k[0], k[1])) == 1][:20]
        rest = [k for k in keys if truth.get((qi, k[0], k[1])) != 1]
        neg = [rest[i] for i in rng.choice(len(rest),
                                           min(nneg, len(rest)),
                                           replace=False)]
        sample = [(k, 1) for k in pos] + [(k, 0) for k in neg]
        rows, t0 = [], time.time()
        for k, lab in sample:
            ims = frames_of(*k)
            if len(ims) < 2:
                continue
            try:
                m = clip_margin(ims, q, model_id)
            except Exception as e:
                print("judge failed:", type(e).__name__, e)
                return
            rows.append((lab, m))
        A = np.array(rows, float)
        y, m = A[:, 0].astype(bool), A[:, 1]
        # AUC by rank statistic
        r = np.argsort(np.argsort(m)) + 1
        auc = ((r[y].sum() - y.sum() * (y.sum() + 1) / 2)
               / (y.sum() * (~y).sum()))
        # yield is computed against the positives PRESENT IN THE SAMPLE,
        # not the corpus support: positives are capped at 20, so
        # dividing by sup would report 0.12 for a query whose sample
        # only ever contained 20 of its 165 true episodes and make a
        # working judge look broken.
        n_pos = int(y.sum())
        K = max(1, int(np.ceil(n_pos * 1.5)))
        top = np.argsort(-m)[:K]
        tru = int(y[top].sum())
        agree = float(((m > 0) == y).mean())
        print(f"q{qi:02d} sup {sup[qi]:3d}  sample {len(A):3d} "
              f"({n_pos} true)  AUC {auc:.3f}  "
              f"yield {tru / n_pos:.2f}  prec {tru / len(top):.2f}  "
              f"sign-agree {agree:.2f}  "
              f"{(time.time() - t0) / len(A):.1f}s/clip", flush=True)
        print(f"      margin  true {m[y].mean():+.3f}  "
              f"false {m[~y].mean():+.3f}", flush=True)


if __name__ == "__main__":
    main()
