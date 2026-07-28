"""Decode-fidelity gate: does the store hand back the ORIGINAL pictures?

Comparing two renditions against each other cannot answer that — it was
measured saying "16 mean abs difference" while one of the two was simply
decoding the wrong scenes. The source file is the only ground truth, so
this decodes each sampled episode through the store's normal byte-range
path and compares it to the exact source frames.

It also reports bytes read per episode: with a keyframe at every episode
start, a window read should fetch that episode and nothing more.

Usage:
  python scripts/verify_media.py <store> --from-store <parent> [--n 16]

`--from-store` supplies the whole file's frame list, whose sorted
timestamps give each frame's presentation index in the source.
"""
from __future__ import annotations

import glob
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.fftools import find  # noqa: E402
from elidedb.store import Store  # noqa: E402
from elidedb.video import FrameSet  # noqa: E402


def originals(store: Store) -> dict[str, str]:
    out = {}
    for h in store.table("frames").history():
        m = h.get("meta", {})
        if "source" in m and "original" in m:
            stem = Path(m["source"]).stem
            out[stem] = m["original"]
            out[stem.rsplit("-", 1)[0]] = m["original"]
        for ref, orig in (m.get("originals") or {}).items():
            stem = Path(ref).stem
            out[stem] = orig
            out[stem.rsplit("-", 1)[0]] = orig
    return out


def decode(store: Store, stream: str, t0: int, t1: int, limit: int):
    out, _ = store.window(t0, t1, tables=["frames"])
    fs = out["frames"]
    if not hasattr(fs, "decode"):
        fs = FrameSet(store, "frames", fs)
    return fs.decode(stream=stream, limit=limit)


def source_frames(src: str, idx: list[int], tmp: str):
    for f in glob.glob(f"{tmp}/ref_*.png"):
        os.unlink(f)
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run([find("ffmpeg"), "-v", "error", "-y", "-i", src,
                    "-vf", f"select='{sel}'", "-vsync", "0",
                    f"{tmp}/ref_%03d.png"], check=True)
    import cv2
    return [cv2.imread(f"{tmp}/ref_{i + 1:03d}.png")[:, :, ::-1]
            for i in range(len(idx))]


def episode_bytes(rows_by_src, src_ref, t0, t1) -> int:
    rows = rows_by_src[src_ref]
    inside = [i for i, r in enumerate(rows) if t0 <= r[0] <= t1]
    if not inside:
        return 0
    k = inside[0]
    while k > 0 and not rows[k][2]:
        k -= 1
    return sum(r[1] for r in rows[k:inside[-1] + 1])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    argv = sys.argv
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 16
    limit = int(argv[argv.index("--frames") + 1]) if "--frames" in argv else 4
    store = Store.open(args[0])
    parent = Store.open(argv[argv.index("--from-store") + 1])
    orig = {**originals(parent), **originals(store)}

    pf = parent.table("frames").scan()
    full: dict[str, list[int]] = {}
    for s, t in zip(pf.column("source").to_pylist(),
                    pf.column("ts").to_pylist()):
        full.setdefault(s, []).append(int(t))
    rank = {s: {t: i for i, t in enumerate(sorted(v))}
            for s, v in full.items()}
    # map this store's media refs onto the parent's by file index
    def parent_ref(sref):
        key = Path(sref).stem.rsplit("-", 1)[0]
        for s in rank:
            if Path(s).stem.rsplit("-", 1)[0] == key:
                return s
        return None

    ft = store.table("frames").scan()
    rows_by_src: dict[str, list] = {}
    for s, t, bo, ps, k in zip(ft.column("source").to_pylist(),
                               ft.column("ts").to_pylist(),
                               ft.column("byte_offset").to_pylist(),
                               ft.column("packet_size").to_pylist(),
                               ft.column("keyframe").to_pylist()):
        rows_by_src.setdefault(s, []).append((int(t), int(ps), bool(k)))
    for v in rows_by_src.values():
        v.sort()
    ts_src = {}
    for s, t in zip(ft.column("source").to_pylist(),
                    ft.column("ts").to_pylist()):
        ts_src[int(t)] = s

    ep = store.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    ep.column("ts").to_pylist(),
                    ep.column("t1").to_pylist()))
    sample = random.Random(0).sample(keys, min(n, len(keys)))
    tmp = tempfile.mkdtemp(prefix="verify_media_")

    maes, total_bytes, worst = [], 0, (0.0, None)
    for stream, t0, t1 in sample:
        t0, t1 = int(t0), int(t1)
        got = decode(store, stream, t0, t1, limit)
        if not got:
            print(f"  NO FRAMES {stream} {t0}")
            continue
        sref = ts_src[int(got[0][0])]
        pref = parent_ref(sref)
        src = orig.get(Path(sref).stem) or \
            orig.get(Path(sref).stem.rsplit("-", 1)[0])
        if not (pref and src and os.path.exists(src)):
            print(f"  SKIP {stream}: no source for {sref}")
            continue
        idx = [rank[pref][int(ts)] for ts, _ in got]
        ref = source_frames(src, idx, tmp)
        for (ts, img), r in zip(got, ref):
            a = np.asarray(img).astype(np.int16)
            b = r.astype(np.int16)
            if a.shape != b.shape:
                print(f"  SHAPE MISMATCH {stream}: {a.shape} vs {b.shape}")
                continue
            m = float(np.abs(a - b).mean())
            maes.append(m)
            if m > worst[0]:
                worst = (m, f"{stream} ts {ts}")
        total_bytes += episode_bytes(rows_by_src, sref, t0, t1)

    if not maes:
        print("nothing compared")
        raise SystemExit(1)
    d = np.array(maes)
    print(f"episodes {len(sample)}   frames vs source {len(maes)}")
    print(f"|pixel diff| vs SOURCE:  mean {d.mean():.2f}  p95 "
          f"{np.percentile(d, 95):.2f}  max {d.max():.2f}   "
          f"(worst: {worst[1]})")
    print(f"bytes read per episode: {total_bytes/len(sample)/1e3:.1f} KB")
    bad = int((d > 8).sum())
    print(f"frames above 8 MAE (wrong picture territory): {bad}/{len(maes)}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
