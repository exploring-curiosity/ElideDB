"""Does sharp search reach teacher quality at student price?

Three rankers over the same 7,046 windows of lake/bridge4h, graded against
the human task labels (which live OUTSIDE the store):

    student   FDNN-V window embeddings, ranked directly      (~1 ms rank)
    sharp     student shortlist -> teacher re-rank + cache   (~26 ms warm)
    teacher   SigLIP-224 on every window — the ceiling, and
              exactly what sharp converges to as the cache fills

The teacher-everywhere ranking is an EVAL artifact (built once, kept in
eval/, never in the store): it is what you would get by paying 27.7 ms/frame
corpus-wide, which is the cost sharp search exists to avoid.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.cracked import TEACHER, search_sharp            # noqa: E402
from elidedb.embeddings import (_embed_images, _vec_table,   # noqa: E402
                                embed_text)
from elidedb.video import FrameSet                           # noqa: E402

GT = Path("eval/bridge4h_teacher_windows.npz")


def teacher_everywhere(db):
    """Teacher vector for the centre frame of EVERY window (eval-only)."""
    if GT.exists():
        z = np.load(GT, allow_pickle=True)
        return z["keys"].tolist(), z["vecs"]
    from PIL import Image
    tbl, _ = _vec_table(db, "embeddings")
    wins = list(zip(tbl.column("stream").to_pylist(),
                    tbl.column("ts").to_pylist(),
                    tbl.column("t1").to_pylist()))
    frames = db.table("frames").scan()
    keys, vecs = [], []
    t0 = time.time()
    for i in range(0, len(wins), 128):
        block = wins[i:i + 128]
        imgs, ks = [], []
        for (s, a, b) in block:
            mid = (a + b) // 2
            sel = frames.filter(pc.and_(
                pc.equal(frames.column("stream"), s),
                pc.and_(pc.greater_equal(frames.column("ts"),
                                         mid - 2_000_000_000),
                        pc.less_equal(frames.column("ts"),
                                      mid + 2_000_000_000))))
            dec = FrameSet(db, "frames", sel).decode(stream=s, width=448,
                                                     limit=1)
            if dec:
                imgs.append(Image.fromarray(dec[0][1]))
                ks.append((s, int(a), int(b)))
        if imgs:
            for k, v in zip(ks, _embed_images(imgs, TEACHER)):
                keys.append(k)
                vecs.append(v)
        if i % 1024 == 0:
            print(f"  teacher-everywhere {i}/{len(wins)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    vecs = np.stack(vecs)
    np.savez(GT, keys=np.array(keys, dtype=object), vecs=vecs)
    print(f"  built teacher-everywhere: {len(keys)} windows, "
          f"{time.time() - t0:.0f}s (one-time eval cost)")
    return keys, vecs


def to_episodes(hit_wins, eps):
    out = []
    for (s, a, b) in hit_wins:
        best, ov_best = None, 0
        for e in eps:
            if e["stream"] != s:
                continue
            ov = min(b, e["t1"]) - max(a, e["t0"])
            if ov > ov_best:
                best, ov_best = e["ep"], ov
        if best is not None and best not in out:
            out.append(best)
    return out


def metrics(ranked, rel):
    rel = set(rel)
    out = {f"R@{k}": float(len(set(ranked[:k]) & rel) > 0)
           for k in (1, 5, 10)}
    out["MRR"] = next((1.0 / (i + 1) for i, e in enumerate(ranked)
                       if e in rel), 0.0)
    return out


def main():
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    ep_meta = db.table("episodes").scan()
    stream_of = dict(zip(ep_meta.column("episode_index").to_pylist(),
                         ep_meta.column("stream").to_pylist()))
    eps = [{"ep": int(i), "t0": int(a), "t1": int(b), "task": k,
            "stream": stream_of.get(int(i))}
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k and int(i) in stream_of]
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task"], []).append(e["ep"])

    gt_keys, gt_vecs = teacher_everywhere(db)
    gt_pos = {k: i for i, k in enumerate(gt_keys)}

    tbl, svecs = _vec_table(db, "embeddings")
    wins = list(zip(tbl.column("stream").to_pylist(),
                    tbl.column("ts").to_pylist(),
                    tbl.column("t1").to_pylist()))

    rng = np.random.default_rng(0)
    tasks = sorted(by_task)
    queries = [tasks[i] for i in rng.choice(len(tasks), 100, replace=False)]

    res = {n: {"m": [], "ms": []} for n in ("student", "sharp", "teacher")}
    for qi, q in enumerate(queries):
        rel = by_task[q]
        # student
        t0 = time.perf_counter()
        qv = embed_text(q)
        top = np.argsort(-(svecs @ qv))[:10]
        res["student"]["ms"].append((time.perf_counter() - t0) * 1e3)
        res["student"]["m"].append(metrics(
            to_episodes([wins[i] for i in top], eps), rel))
        # sharp (cracking cache warms as the benchmark runs — as in prod)
        t0 = time.perf_counter()
        hits, _ = search_sharp(db, q, k=10)
        res["sharp"]["ms"].append((time.perf_counter() - t0) * 1e3)
        res["sharp"]["m"].append(metrics(
            to_episodes([(h["stream"], h["t0"], h["t1"]) for h in hits],
                        eps), rel))
        # teacher ceiling
        t0 = time.perf_counter()
        qt = embed_text(q, model_id=TEACHER)
        topt = np.argsort(-(gt_vecs @ qt))[:10]
        res["teacher"]["ms"].append((time.perf_counter() - t0) * 1e3)
        res["teacher"]["m"].append(metrics(
            to_episodes([gt_keys[i] for i in topt], eps), rel))
        if (qi + 1) % 20 == 0:
            print(f"  {qi + 1}/100 queries", flush=True)

    print(f"\n{'ranker':9s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} "
          f"{'MRR':>6s} {'ms p50':>8s} {'ms p90':>8s}")
    print("-" * 54)
    summary = {}
    for n, r in res.items():
        agg = {k: float(np.mean([m[k] for m in r["m"]])) for k in r["m"][0]}
        agg["p50_ms"] = float(np.percentile(r["ms"], 50))
        agg["p90_ms"] = float(np.percentile(r["ms"], 90))
        summary[n] = agg
        print(f"{n:9s} {agg['R@1']:6.3f} {agg['R@5']:6.3f} "
              f"{agg['R@10']:6.3f} {agg['MRR']:6.3f} "
              f"{agg['p50_ms']:8.1f} {agg['p90_ms']:8.1f}")

    cache_n = len(db.table("teacher_windows").scan())
    print(f"\ncracked cache after 100 queries: {cache_n}/{len(wins)} windows "
          f"({100 * cache_n / len(wins):.1f}% of corpus teacher-embedded)")
    Path("bench") / "bench_bridge4h_query.json".write_text(json.dumps(
        {"summary": summary, "queries": len(queries),
         "episodes": len(eps), "tasks": len(by_task),
         "cache_windows": cache_n, "total_windows": len(wins)}, indent=2))
    print("wrote bench_bridge4h_query.json")


if __name__ == "__main__":
    main()
