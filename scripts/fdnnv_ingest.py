"""Embed-on-write, demonstrated: EVERY frame of the Bridge store, no stride.

Measures the full write-path pipeline — chunked byte-range decode feeding the
streaming encoder, state carried per stream — and answers three questions:

  1. wall clock: what does every-frame embedding actually cost now?
  2. generalisation: file-000 was NEVER seen in training (different episodes,
     same domain). Fidelity there is the honest number.
  3. retrieval: does TEXT search over student embeddings return the same
     windows as over teacher embeddings? Vector closeness is not the product;
     retrieval agreement is.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402

from elidedb import Store                                    # noqa: E402
from elidedb.embeddings import _embed_images, embed_text     # noqa: E402
from elidedb.fdnnvideo import fdnnv_dir, load_encoder                   # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

WIDTH = 192
CHUNK = 512          # frames per decode call: keeps ffmpeg's buffer ~1 GB


def embed_stream(db, model, stream, rows):
    """Chunked decode -> stateful embed. One continuous pass, state carried
    across chunks — this loop IS the write path."""
    ts_out, vecs = [], []
    h = model.init_state(1)
    dec_s = emb_s = 0.0
    n = len(rows)
    for i in range(0, n, CHUNK):
        t0 = time.perf_counter()
        dec = FrameSet(db, "frames", rows.slice(i, CHUNK)).decode(width=WIDTH)
        dec_s += time.perf_counter() - t0
        if not dec:
            continue
        frames = np.stack([d[1] for d in dec])
        t0 = time.perf_counter()
        x = mx.array(frames.astype(np.float32) / 127.5 - 1.0)[None]
        e, h = model(x, h0=h)
        e = np.array(e[0], dtype=np.float32)
        emb_s += time.perf_counter() - t0
        ts_out.extend(d[0] for d in dec)
        vecs.append(e)
    return np.array(ts_out, np.int64), np.concatenate(vecs), dec_s, emb_s


def main():
    db = Store.open("lake/bridge")
    model, meta = load_encoder(fdnnv_dir())
    frames = db.table("frames").scan()
    streams = sorted(set(frames.column("stream").to_pylist()))

    total_frames = 0
    dec_total = emb_total = 0.0
    per_stream = {}
    t_wall = time.time()
    for s in streams:
        rows = frames.filter(pc.equal(frames.column("stream"), s))
        rows = rows.take(pa.compute.sort_indices(rows.column("ts")))
        ts, vecs, dec_s, emb_s = embed_stream(db, model, s, rows)
        per_stream[s] = (ts, vecs)
        total_frames += len(ts)
        dec_total += dec_s
        emb_total += emb_s
        print(f"  {s[-8:]}: {len(ts):,} frames  decode {dec_s:.1f}s  "
              f"embed {emb_s:.1f}s", flush=True)
    wall = time.time() - t_wall

    print(f"\nEVERY frame: {total_frames:,} frames in {wall:.1f}s wall "
          f"({total_frames / wall:,.0f} frames/s)")
    print(f"  decode {dec_total:.1f}s ({dec_total/total_frames*1000:.2f} "
          f"ms/f) | embed {emb_total:.1f}s "
          f"({emb_total/total_frames*1000:.3f} ms/f)")

    # ---- 2. generalisation: file-000 never seen in training ---------------
    from PIL import Image
    s0 = [s for s in streams if "file-000" in s][0]
    ts0, v0 = per_stream[s0]
    pick = np.linspace(0, len(ts0) - 1, 300).round().astype(int)
    rows0 = frames.filter(pc.equal(frames.column("stream"), s0))
    rows0 = rows0.take(pa.compute.sort_indices(rows0.column("ts")))
    dec = FrameSet(db, "frames", rows0.take(pick)).decode(width=512)
    teacher = _embed_images([Image.fromarray(d[1]) for d in dec],
                            meta["teacher"])
    tmap = {t: v for t, v in zip([d[0] for d in dec], teacher)}
    common = [i for i, t in enumerate(ts0) if int(t) in tmap]
    tv = np.stack([tmap[int(ts0[i])] for i in common])
    cos = (v0[common] * tv).sum(1)
    print(f"\nfile-000 (unseen stream): fidelity mean={cos.mean():.4f} "
          f"p10={np.percentile(cos, 10):.4f} over {len(common)} frames")

    # ---- 3. text-retrieval agreement --------------------------------------
    # Pool per 4s window under BOTH encoders; same text query; compare top-10.
    win = 4_000_000_000
    overlaps = []
    queries = ["a robot arm picking up an object",
               "a pot on the stove",
               "a towel on the counter",
               "opening a drawer"]
    # teacher windows come from the existing teacher frame_vectors
    fv = db.table("frame_vectors").scan()
    for q in queries:
        qv = embed_text(q)
        ranked = {}
        for tag, source in (("student", per_stream), ("teacher", None)):
            wins, keys = [], []
            if tag == "student":
                items = [(s, *per_stream[s]) for s in per_stream
                         if "file-000" not in s]
            else:
                items = []
                for s in set(fv.column("stream").to_pylist()):
                    sub = fv.filter(pc.equal(fv.column("stream"), s))
                    tt = np.array(sub.column("ts").to_pylist())
                    vv = np.asarray(sub.column("vector").to_pylist(),
                                    np.float32)
                    o = np.argsort(tt)
                    items.append((s, tt[o], vv[o]))
            for s, tt, vv in items:
                t = int(tt[0])
                while t < int(tt[-1]):
                    lo, hi = np.searchsorted(tt, [t, t + win])
                    if hi > lo:
                        v = vv[lo:hi].mean(0)
                        v /= np.linalg.norm(v) + 1e-8
                        wins.append(v)
                        keys.append((s, t // win))
                    t += win
                    if len(wins) > 20000:
                        break
            sc = np.stack(wins) @ qv
            ranked[tag] = [keys[i] for i in np.argsort(-sc)[:10]]
        ov = len(set(ranked["student"]) & set(ranked["teacher"]))
        overlaps.append(ov)
        print(f"  text '{q[:40]}': top-10 overlap {ov}/10")
    print(f"text-retrieval agreement: mean {np.mean(overlaps):.1f}/10")

    # ---- extrapolation, honest --------------------------------------------
    per_frame_ms = wall / total_frames * 1000
    full = 8_586_000 * per_frame_ms / 1000 / 3600
    print(f"\nfull 477 h BridgeData2 corpus, EVERY frame, single process: "
          f"{full:.1f} h  (was 38.5 h with SigLIP-fast)")

    Path("bench_fdnnv_ingest.json").write_text(json.dumps({
        "frames": int(total_frames), "wall_s": round(wall, 1),
        "frames_per_s": round(total_frames / wall, 1),
        "decode_ms_per_frame": round(dec_total / total_frames * 1000, 3),
        "embed_ms_per_frame": round(emb_total / total_frames * 1000, 4),
        "file000_fidelity_mean": round(float(cos.mean()), 4),
        "file000_fidelity_p10": round(float(np.percentile(cos, 10)), 4),
        "text_topk_overlap": overlaps,
        "full_corpus_hours_every_frame": round(full, 2)}, indent=2))
    print("wrote bench_fdnnv_ingest.json")


if __name__ == "__main__":
    main()
