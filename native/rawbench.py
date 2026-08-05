"""SEGMENTATION-FREE evaluation: retrieve TIME RANGES, not demo ids.

The old benchmarks handed the system its retrieval units (kitchen
episode spans came from Bridge's metadata). Here the index knows only
uniform windows over raw video. A query is a set of example TIME
RANGES; the answer is a ranked list of time ranges; a returned range
counts as true if it overlaps a truth span. That is the task a
customer actually has.

Metrics are only the two that matter, k = 1.5*support as a MAX bound
with abstention allowed:
    yield = distinct truth spans hit / support
    prec  = returned ranges that hit a truth span / returned

    python native/rawbench.py --corpus sim
    python native/rawbench.py --corpus bench
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from rawwrite import out_dir, arg                              # noqa: E402


def load_index(corpus):
    """[(file_id, start_s, end_s, scale)], V (N,d)."""
    rows, V = [], []
    for p in sorted(out_dir(corpus).glob("*.npz")):
        z = np.load(p)
        w = z["win"]
        for (a, b, sc) in w:
            rows.append((p.stem, float(a), float(b), float(sc)))
        V.append(z["V"].astype(np.float32))
    if not V:
        raise SystemExit(f"no index for {corpus}")
    return rows, np.concatenate(V)


def truth_sim():
    """{file_id -> [(t0,t1,label)]} from generator truth (EVAL ONLY)."""
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    spans, tmpl = {}, {}
    for e, tm, a, b in zip(t["episode"], t["template"], t["t0"],
                           t["t1"]):
        fid = f"ep{int(e):04d}"
        tmpl[fid] = tm
        lo, hi = spans.get(fid, (1e9, -1e9))
        spans[fid] = (min(lo, float(a)), max(hi, float(b)))
    out = {}
    for fid, (a, b) in spans.items():
        out[fid] = [(a, b, tmpl[fid])]
    return out


def truth_bench():
    """{file_id -> [(t0,t1,demo_index)]} from Bridge meta (EVAL ONLY)
    plus the graded truthset keyed by demo."""
    import pyarrow.parquet as pq
    key = "videos/observation.images.image_0"
    meta = pq.read_table(
        ROOT / "data/bridge/meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", f"{key}/file_index",
                 f"{key}/from_timestamp",
                 f"{key}/to_timestamp"]).to_pydict()
    fi = np.array(meta[f"{key}/file_index"])
    spans = {}
    for j in range(len(fi)):
        f = int(fi[j])
        fid = f"file-{f:03d}"
        spans.setdefault(fid, []).append(
            (float(meta[f"{key}/from_timestamp"][j]),
             float(meta[f"{key}/to_timestamp"][j]),
             int(meta["episode_index"][j])))
    g = pq.read_table(ROOT / "eval/truthsets/graded.parquet") \
        .to_pydict()
    G = {}
    for q, ei, v in zip(g["query_id"], g["episode_index"], g["true"]):
        G[(int(q), int(ei))] = int(v)
    return spans, G


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def query_vec(rows, V, seeds):
    """Mean of window vectors overlapping each seed span, per seed."""
    out = []
    for (fid, s0, s1) in seeds:
        idx = [i for i, (f, a, b, sc) in enumerate(rows)
               if f == fid and overlap(a, b, s0, s1) > 0.5 * (b - a)]
        if idx:
            v = V[idx].mean(0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
    return out


def search(rows, V, qs, exclude, k):
    """Ranked non-overlapping time ranges, seeds' own spans removed."""
    if not qs:
        return []
    sc = np.max(np.stack([V @ q for q in qs]), 0)
    order = np.argsort(-sc)
    picked = []
    for i in order:
        f, a, b, s = rows[i]
        if any(f == ef and overlap(a, b, e0, e1) > 0
               for ef, e0, e1 in exclude):
            continue
        if any(f == pf and overlap(a, b, p0, p1) > 0
               for pf, p0, p1, _ in picked):
            continue
        picked.append((f, a, b, float(sc[i])))
        if len(picked) >= k:
            break
    return picked


def run(corpus):
    rows, V = load_index(corpus)
    print(f"{corpus}: {len(rows):,} windows indexed "
          f"({len({r[0] for r in rows})} files)", flush=True)
    rs = np.random.RandomState(0)
    tasks = []
    if corpus == "sim":
        tr = truth_sim()
        groups = {}
        for fid, spans in tr.items():
            groups.setdefault(spans[0][2], []).append(fid)
        for tm, fids in sorted(groups.items()):
            fids = [f for f in fids if any(r[0] == f for r in rows)]
            if len(fids) < 6:
                continue
            sd = sorted(rs.choice(fids, 5, replace=False))
            seeds = [(f, tr[f][0][0], tr[f][0][1]) for f in sd]
            pos = [(f, tr[f][0][0], tr[f][0][1])
                   for f in fids if f not in sd]
            tasks.append((tm, seeds, pos))
    else:
        spans, G = truth_bench()
        by_demo = {}
        for fid, lst in spans.items():
            for (a, b, ei) in lst:
                by_demo[ei] = (fid, a, b)
        have = {r[0] for r in rows}
        for qi in (3, 4, 5):
            pos_d = [ei for (q, ei), v in G.items()
                     if q == qi and v == 1 and ei in by_demo
                     and by_demo[ei][0] in have]
            if len(pos_d) < 6:
                continue
            sd = list(rs.choice(pos_d, 5, replace=False))
            seeds = [by_demo[ei] for ei in sd]
            pos = [by_demo[ei] for ei in pos_d if ei not in sd]
            tasks.append((f"q{qi}", seeds, pos))

    ys, ps = [], []
    for name, seeds, pos in tasks:
        support = len(pos)
        k = math.ceil(1.5 * support)
        qs = query_vec(rows, V, seeds)
        got = search(rows, V, qs, seeds, k)
        hit_spans = set()
        n_true = 0
        for (f, a, b, s) in got:
            hit = None
            for j, (pf, p0, p1) in enumerate(pos):
                if pf == f and overlap(a, b, p0, p1) > 0:
                    hit = j
                    break
            if hit is not None:
                n_true += 1
                hit_spans.add(hit)
        y = len(hit_spans) / support
        p = n_true / max(len(got), 1)
        ys.append(y)
        ps.append(p)
        print(f"   {name:<20} yield {y:.2f}  prec {p:.2f}  "
              f"(support {support}, returned {len(got)})", flush=True)
    if ys:
        print(f"   MEAN yield {np.mean(ys):.3f}  prec "
              f"{np.mean(ps):.3f}", flush=True)


if __name__ == "__main__":
    run(arg("--corpus", "sim"))
