"""Context retrieval: does it work, and how fast?

JUDGE
-----
An independent VLM verdict per (window, query): the yes/no logprob margin for
"Does this image show: <query>?". The judge never sees a ranking, and is asked
a different question than the teacher was, so it cannot favour a method. It
shares a model family with the teacher — stated here rather than hidden; a
fully independent judge would need a second VLM.

SPLIT
-----
Everything is measured on the HELD-OUT TIME RANGE: the last 30% of the
timeline, which the tower never trained on. Scoring on fitted windows would
measure memorisation.

METHODS
-------
    appearance        SigLIP image-text cosine — plain semantic search today
    context_exact     caption-LSA, captions materialised by the VLM at ingest
    context_student   caption-LSA PREDICTED by the tower from frame vectors
                      (what an unlabelled window gets — the honest test of
                      whether distillation carries any signal)
    fused_*           0.6*context + 0.4*appearance, standardised per query
    vlm_rerank        retrieve by appearance, then run the VLM at QUERY time
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctx_eval                                              # noqa: E402

from elidedb import Store                                    # noqa: E402
from elidedb import context as C                             # noqa: E402
from elidedb import rerank                                   # noqa: E402
from elidedb.embeddings import embed_text                    # noqa: E402


def main():
    store = Store.open("lake/oxford")
    meta = json.loads((store.dir / "models" / "context" /
                       "tower.json").read_text())
    d_out = meta.get("d_out", 48)

    windows = C.plan_windows(store, meta["window_s"], meta["stride_s"])
    windows, _, texts = C._labelled(store, windows)
    t0s = np.array([w[1] for w in windows])
    cut = np.quantile(t0s, 0.7)
    held = t0s >= cut
    Wh = [w for w, h in zip(windows, held) if h]
    print(f"held-out: {len(Wh)} windows from +{(cut - t0s.min())/1e9:.1f}s, "
          f"never trained on\n")

    space = C.caption_space(store, dim=d_out)
    exact = space.transform(texts)[held]

    # student predictions for the SAME windows
    import mlx.core as mx

    from elidedb.ctxtower import load_tower
    model, codec, _ = load_tower(store.dir / "models" / "context")
    seqs = [codec.encode(s) for s in C.window_sequences(store, windows)]
    X = C._pad_stack([s for s, h in zip(seqs, held) if h])
    t = time.perf_counter()
    student = np.array(model(mx.array(X)))
    tower_ms = (time.perf_counter() - t) * 1000
    student /= np.linalg.norm(student, axis=1, keepdims=True) + 1e-8

    ctx = store.table("context").scan()
    key = {(s, a): i for i, (s, a) in enumerate(
        zip(ctx.column("stream").to_pylist(), ctx.column("ts").to_pylist()))}
    AP = np.asarray(ctx.column("appearance").to_pylist(), dtype=np.float32)
    AP = AP[[key[(w[0], w[1])] for w in Wh]]

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-8)

    timings = {}

    def timed(name, fn):
        def wrapped(q):
            t = time.perf_counter()
            r = fn(q)
            timings.setdefault(name, []).append((time.perf_counter() - t) * 1e3)
            return r
        return wrapped

    methods = {
        "appearance": timed("appearance", lambda q: AP @ embed_text(q)),
        "context_exact": timed(
            "context_exact", lambda q: exact @ space.transform([q])[0]),
        "context_student": timed(
            "context_student", lambda q: student @ space.transform([q])[0]),
        "fused_exact": timed("fused_exact", lambda q: (
            0.4 * z(AP @ embed_text(q))
            + 0.6 * z(exact @ space.transform([q])[0]))),
        "fused_student": timed("fused_student", lambda q: (
            0.4 * z(AP @ embed_text(q))
            + 0.6 * z(student @ space.transform([q])[0]))),
    }

    def vlm(q):
        """Retrieve by appearance, then judge the top-12 with the VLM."""
        base = AP @ embed_text(q)
        top = np.argsort(base)[::-1][:12]
        hits = [{"stream": Wh[i][0], "t0": Wh[i][1], "t1": Wh[i][2],
                 "score": float(base[i])} for i in top]
        rr, _ = rerank.rerank_hits(store, hits, q, top_n=12)
        rank = {(h["stream"], h["t0"], h["t1"]): -n
                for n, h in enumerate(rr)}
        return np.array([rank.get(w, -999) for w in Wh], dtype=float)
    methods["vlm_rerank"] = timed("vlm_rerank", vlm)

    res = ctx_eval.evaluate(store, methods, Wh, verbose=True)

    print(f"\n{'method':17s} {'ms/query':>9s} {'judge':>8s} {'overlap':>9s}")
    print("-" * 47)
    out = {}
    for n in methods:
        ms = float(np.median(timings[n]))
        out[n] = {**res[n], "median_ms": round(ms, 2)}
        print(f"{n:17s} {ms:9.1f} {res[n]['judge_mean_top5']:+8.3f} "
              f"{res[n]['overlap_top5']:7.1f}/5")

    out["_tower"] = {"inference_ms_for_all_held_windows": round(tower_ms, 1),
                     "channels": meta["cfg"]["n_hidden"],
                     "params": meta.get("pruned", {}).get("after", {})
                     .get("params")}
    Path("bench") / "bench_context.json".write_text(json.dumps(
        {"summary": out, "held_out_windows": len(Wh),
         "queries": ctx_eval.QUERIES}, indent=2))
    print("\nwrote bench_context.json")


if __name__ == "__main__":
    main()
