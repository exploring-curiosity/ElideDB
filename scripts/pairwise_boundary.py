"""Pairwise VLM comparisons at the k-boundary. The teacher's last stage.

Pointwise scoring is done: the ITM logit matrix ranks the corpus, and
its residual failure is INTERLEAVING - true episodes sit at ranks
5-35 with false positives woven between them (q00: 6 of 8 true inside
top-31, yield still 0.38 because the boundary cuts through the weave).
Absolute scores cannot fix their own ties. The ranking literature's
answer (Pairwise Ranking Prompting, arXiv 2306.17563; Vote-in-Context,
arXiv 2511.01617) is to stop asking "how good is this" and ask "which
of these two", which VLMs answer far more reliably than they calibrate.

So: a sliding bubble pass over the boundary region [1 .. K+W]. Each
comparison shows the judge BOTH clips (4 frames each, labelled) and
asks which shows the query; the winner moves up. Two passes move a
true episode past at most 2xpasses false positives - exactly the
interleaving depth observed. Cost is O(boundary), not O(n^2).

Judge: Qwen2.5-VL-7B (4-bit MLX) - the 2025 generation, not the
Qwen2-VL that measured AUC 0.31-0.89 pointwise. Pairwise is also the
form the PRP result says these models are good at.

  python scripts/pairwise_boundary.py [--q 0,9,10] [--passes 2] [--w 15]
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

JUDGE = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
NF = 4


def compare(frames_a, frames_b, text):
    """1 if clip A shows the query better, -1 if B, 0 on abstain.
    Order-debiased: asked both ways, disagreement = abstain."""
    import tempfile

    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    from elidedb.rerank import _load
    model, processor, cfg, _, _ = _load(JUDGE)
    tok = processor.tokenizer
    a_ids = sorted({tok.encode(s)[0] for s in ("A", " A")})
    b_ids = sorted({tok.encode(s)[0] for s in ("B", " B")})

    def ask(fa, fb):
        n = len(fa) + len(fb)
        q = (f"The first {len(fa)} images are frames in time order from "
             f"clip A. The next {len(fb)} images are frames from clip B. "
             f"Which clip shows: {text}? Answer only A or B.")
        prompt = apply_chat_template(processor, cfg, q, num_images=n)
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, im in enumerate(fa + fb):
                p = f"{td}/f{i}.jpg"
                im.save(p, "JPEG", quality=90)
                paths.append(p)
            r = generate(model, processor, prompt, image=paths,
                         max_tokens=1, verbose=False)
        if r.logprobs is None:
            return 0
        lg = mx.array(r.logprobs).reshape(-1)
        a = max(float(lg[i]) for i in a_ids)
        b = max(float(lg[i]) for i in b_ids)
        return 1 if a > b else -1

    first = ask(frames_a, frames_b)
    second = -ask(frames_b, frames_a)       # swapped, sign-corrected
    return first if first == second else 0


def main():
    from PIL import Image

    from elidedb.video import FrameSet

    argv = sys.argv
    qsel = ([int(x) for x in argv[argv.index("--q") + 1].split(",")]
            if "--q" in argv else None)
    passes = int(argv[argv.index("--passes") + 1]) if "--passes" in argv \
        else 2
    W = int(argv[argv.index("--w") + 1]) if "--w" in argv else 15

    d = np.load(ROOT / "ml/teacher_base.npz", allow_pickle=True)
    B, qids = d["B"], [int(q) for q in d["qids"]]
    keys = list(zip([str(s) for s in d["streams"]],
                    [int(v) for v in d["ts"]], [int(v) for v in d["t1"]]))
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    db = Store.open("lake/bench")
    frames_tbl = db.table("frames").scan()

    fcache: dict = {}

    def frames_of(i):
        if i in fcache:
            return fcache[i]
        s, a, b = keys[i]
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        pi = np.linspace(0, len(sel) - 1,
                         min(NF, len(sel))).round().astype(int)
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode(width=336)
        except Exception:
            fcache[i] = []
            return []
        fcache[i] = [Image.fromarray(f) for _, f in sorted(dec)]
        return fcache[i]

    out = {}
    for j, qi in enumerate(qids):
        if sup.get(qi, 0) == 0 or (qsel and qi not in qsel):
            continue
        text = QUERIES[qi]
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        x = B[:, j].astype(float)
        order = list(np.argsort(-np.where(np.isfinite(x), x, -np.inf)))
        K = int(np.ceil(sup[qi] * 1.5))
        # BUDGET policy, not a tuned constant: a boundary of K+W
        # explores almost nothing when K is 3, so small-K queries get
        # depth in place of breadth - examine at least 60, at most
        # 160, else K+W. The judge itself never sees the truthset.
        depth = min(max(K + W, 60), 160) if K < 60 else K + W
        lo, hi = 0, min(depth, len(order))

        def yld(o):
            return int(lab[o[:K]].sum()) / sup[qi]

        y0 = yld(order)
        t0, ncmp = time.time(), 0
        for _ in range(passes):
            # bubble upward: a winner climbs through the boundary
            for p in range(hi - 2, lo - 1, -1):
                fa, fb = frames_of(order[p]), frames_of(order[p + 1])
                if not fa or not fb:
                    continue
                ncmp += 1
                if compare(fa, fb, text) < 0:
                    order[p], order[p + 1] = order[p + 1], order[p]
        y1 = yld(order)
        out[qi] = (y0, y1)
        print(f"q{qi:02d} sup {sup[qi]:3d} K {K:3d}  yield {y0:.2f} -> "
              f"{y1:.2f}  ({ncmp} comparisons, "
              f"{(time.time() - t0) / max(ncmp, 1):.1f}s each)", flush=True)
        fcache.clear()

    if out:
        print(f"\nmean yield over reranked queries: "
              f"{np.mean([v[0] for v in out.values()]):.2f} -> "
              f"{np.mean([v[1] for v in out.values()]):.2f}")


if __name__ == "__main__":
    main()
