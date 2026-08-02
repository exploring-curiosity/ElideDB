"""STEP 2 gate: does DINOv3 beat the nano ReID on THIS corpus?

Head-to-head on identical tracks. One decode+track pass over a subset
of segments (the expensive part, paid once), then every candidate
encoder embeds the SAME exemplar crops, and each is scored on the
harness that settled the identity question:

    AUC        proven-same vs proven-diff pairs from interval_pairs -
               the same box-geometry proofs the production writer uses
    recurrence fit_cut sweep: objects recurring across segments at the
               best cut the sweep finds, plus false-merge/recovered

The subset is a STRIDE over segments, not a prefix: a prefix would be
one camera's morning; a stride sees all four streams and the whole
session, so cross-segment recurrence exists to be measured.

Projected from bench_dinov3.json + measured full-write costs:
~7 min at --n 200. Nothing full-corpus is scheduled until this prints
a winner.

    python scripts/ab_identity.py [--n 200] [--store lake/fresh_bench]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
from elidedb.identity import (Stream, features, fit_cut,       # noqa: E402
                              interval_pairs)

N_SEG = 200
VIEWS = 4


def collect_tracks(db, n_seg):
    """Decode + track a stride of segments; closed tracks WITH their
    exemplar crops. Subset-sized, so holding crops in memory is fine -
    the thing that melted the full write was corpus-sized."""
    from elidedb.identity import propose
    ft = db.table("frames").scan()
    ft = ft.take(pc.sort_indices(ft, sort_keys=[("stream", "ascending"),
                                                ("ts", "ascending")]))
    src = ft.column("source").to_pylist()
    stream = ft.column("stream").to_pylist()
    bounds, start = [], 0
    for i in range(1, len(src) + 1):
        if i == len(src) or src[i] != src[start]:
            bounds.append((stream[start], start, i - start))
            start = i
    stride = max(1, len(bounds) // n_seg)
    take = bounds[::stride][:n_seg]

    tracks, cost = [], defaultdict(float)
    for seg_no, (sname, i, n_b) in enumerate(tqdm(take, desc="decode+track")):
        a = time.time()
        chunk = FrameSet(db, "frames", ft.slice(i, n_b)).decode()
        cost["decode"] += time.time() - a
        if not chunk:
            continue
        chunk = sorted(chunk)
        ims = [c[1] for c in chunk]
        a = time.time()
        dets = propose(ims)
        cost["propose"] += time.time() - a
        a = time.time()
        st = Stream(views=VIEWS)
        closed = []
        for (ts, im), b in zip(chunk, dets):
            b = np.asarray(b, np.int32).reshape(-1, 4)
            d = (b, np.ones(len(b), np.float32),
                 np.ones(len(b), np.float32))
            closed += st.update(int(ts), d, im)
        closed += st.flush()
        cost["track"] += time.time() - a
        for _, t in closed:
            if not t["crops"]:
                continue
            tracks.append({
                "stream": sname, "seg": seg_no,
                "ts": t["ts"], "t1": t["t1"], "n": t["n"],
                "box": [int(x) for x in t["box"][0]],
                "crops": [c[1] for c in t["crops"]],
            })
    return tracks, dict(cost)


def descriptors_for(tracks, enc):
    """One pooled unit vector per track from its exemplar crops."""
    if enc == "reid":
        out = []
        for t in tqdm(tracks, desc="embed reid"):
            f = np.stack([features(c, [[0, 0, c.shape[1], c.shape[0]]])[0]
                          for c in t["crops"]])
            v = f.mean(0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out)
    from elidedb import dinov3
    flat, owner = [], []
    for k, t in enumerate(tracks):
        for c in t["crops"]:
            flat.append(c)
            owner.append(k)
    V = cnt = None
    for i in tqdm(range(0, len(flat), 512),
                  desc=f"embed {enc.split('/')[-1][:20]}"):
        E = dinov3.embed(flat[i:i + 512], mid=enc)
        if V is None:
            V = np.zeros((len(tracks), E.shape[1]), np.float32)
            cnt = np.zeros(len(tracks), np.int32)
        for e, k in zip(E, owner[i:i + 512]):
            V[k] += e
            cnt[k] += 1
    V /= np.maximum(cnt[:, None], 1)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)


def score(name, V, neg, pos, epi, t_embed):
    N = np.asarray(neg, np.int64).reshape(-1, 2)
    ns = np.einsum("ij,ij->i", V[N[:, 0]], V[N[:, 1]])
    auc = float("nan")
    if pos:
        P = np.asarray(pos, np.int64).reshape(-1, 2)
        ps = np.einsum("ij,ij->i", V[P[:, 0]], V[P[:, 1]])
        grid = np.linspace(0, 1, 501)
        tpr = (ps[None] >= grid[:, None]).mean(1)
        fpr = (ns[None] >= grid[:, None]).mean(1)
        auc = float(np.trapezoid(tpr[::-1], fpr[::-1]))
    _, rep = fit_cut(V, epi, neg, pos or None)
    best = max(rep, key=lambda r: r["recur"])
    return {"encoder": name, "auc": round(auc, 4), "cut": best["cut"],
            "objects": best["objects"], "recur": best["recur"],
            "singleton_pct": best["singleton_pct"],
            "false_merge_pct": best.get("false_merge_pct"),
            "recovered_pct": best.get("recovered_pct"),
            "embed_s": round(t_embed, 1)}


def main():
    argv = sys.argv
    n_seg = int(argv[argv.index("--n") + 1]) if "--n" in argv else N_SEG
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    t0 = time.time()
    tracks, cost = collect_tracks(db, n_seg)
    print(f"{len(tracks):,} tracks from {n_seg} segments  cost="
          f"{json.dumps({k: round(v, 1) for k, v in cost.items()})}",
          flush=True)

    rows_geo = [(t["stream"], {"ts": t["ts"], "t1": t["t1"],
                               "box": t["box"]}) for t in tracks]
    neg, pos = interval_pairs(rows_geo)
    print(f"proven-diff {len(neg):,}  proven-same {len(pos):,}", flush=True)
    epi = np.array([t["seg"] for t in tracks])

    results = []
    for enc in ("reid",
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"):
        a = time.time()
        try:
            V = descriptors_for(tracks, enc)
            r = score(enc.split("/")[-1], V, neg, pos, epi,
                      time.time() - a)
        except Exception as e:
            r = {"encoder": enc, "error": str(e)[:200]}
        results.append(r)
        print(json.dumps(r), flush=True)

    out = {"segments": n_seg, "tracks": len(tracks),
           "proven_diff": len(neg), "proven_same": len(pos),
           "total_s": round(time.time() - t0, 1), "results": results}
    (ROOT / "bench/ab_identity.json").write_text(json.dumps(out, indent=1))
    print("wrote bench/ab_identity.json")


if __name__ == "__main__":
    main()
