"""Retrieval over window sequences: temporal alignment variants +
the frozen bench protocol. Iterates in seconds from the seq cache.

Variants measured side by side (all pretrained-only, no fitting):
    pool     mean of window vectors, cosine        (no temporal shape)
    dtw      banded DTW over window cosine         (global alignment)
    dtwd     DTW over DELTA sequence v[t+1]-v[t]   (change profile -
             appearance nuisance cancels inside each difference)
    nov      novelty rhythm 1 - cos(v[t], v[t+1])  (1-D, the old best
             single channel, now in V-JEPA space)

    python native/seqbench.py [--store lake/sim_chains]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from seq import cache_dir, arg                                 # noqa: E402
from chain_channels import bench_S, dtw_sim, DEV, HOLD, norm   # noqa: E402


def load_seqs(store_name):
    out = {}
    for p in sorted(cache_dir(store_name).glob("*.npz")):
        z = np.load(p)
        out[int(p.stem)] = z["V"].astype(np.float32)
    return out


def S_matrix(seqs, eps, kind):
    n = len(eps)
    S = np.zeros((n, n), np.float32)
    reps = {}
    for e in eps:
        V = seqs.get(e)
        if V is None or len(V) < 3:
            continue
        if kind == "pool":
            reps[e] = norm(V.mean(0))
        elif kind == "dtw":
            reps[e] = V
        elif kind == "dtwd":
            D = V[1:] - V[:-1]
            reps[e] = norm(D)
        elif kind == "nov":
            d = 1.0 - (V[1:] * V[:-1]).sum(1)
            k = np.ones(3) / 3
            d = np.convolve(d, k, mode="valid")
            reps[e] = norm(np.stack([d, np.gradient(d)], 1))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = reps.get(eps[i]), reps.get(eps[j])
            if a is None or b is None:
                continue
            if kind == "pool":
                S[i, j] = S[j, i] = float(a @ b)
            else:
                S[i, j] = S[j, i] = dtw_sim(a, b)
    return S


def main():
    import pyarrow.parquet as pq
    store = ROOT / arg("--store", "lake/sim_chains", str)
    seqs = load_seqs(store.name)
    eps = sorted(seqs)
    print(f"{len(eps)} episodes with sequences", flush=True)
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    mats = {}
    for kind in ("pool", "dtw", "dtwd", "nov"):
        import time
        t0 = time.time()
        mats[kind] = S_matrix(seqs, eps, kind)
        print(f"-- {kind} ({time.time()-t0:.0f}s)")
        bench_S(mats[kind], eps, tmpl, DEV, f"{kind} DEV")
        bench_S(mats[kind], eps, tmpl, HOLD, f"{kind} HOLDOUT")
    # simple rank fusion of the two best (frozen choice happens on DEV)
    np.savez(cache_dir(store.name).parent / "seqbench_S.npz",
             eps=np.array(eps), **{f"S_{k}": v for k, v in mats.items()})


if __name__ == "__main__":
    main()
