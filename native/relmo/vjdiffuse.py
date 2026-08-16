"""Read-side rank refinement: similarity diffusion on the corpus graph.

WHY THIS FILE EXISTS. EXPERIMENTS.md §3 records a +50% AP holdout gain
(0.396 -> 0.597) from diffusion on an earlier version of this system, and §7
notes it never made it into the current read path. It is the largest measured
win the pipeline is not using, it needs no training, no labels, and no
re-encoding.

THE IDEA. A query's own similarity to each recording is a noisy, local view.
If A is a strong match for the query and B is a strong match for A, then B is
probably relevant even when the query does not see it directly - the corpus
manifold carries information the pairwise score does not. Diffusion
propagates the initial scores over that manifold:

    y_{t+1} = alpha * S y_t + (1 - alpha) * y_0

with S the symmetrically-normalised, top-k sparsified affinity. Converges to
(1-alpha)(I - alpha S)^-1 y_0; the iteration is cheaper and needs no solve.

COST DISCIPLINE. The corpus-corpus graph is built from MEAN-POOLED cosine
(n^2 dot products - milliseconds), never from DTW (n^2 alignments - hours).
Only the query-to-corpus scores use DTW. So diffusion adds O(n^2 d) once per
corpus plus O(k n) per query, and the per-query read cost stays flat.

Everything here is label-free and fitted on the target corpus, so it applies
to a corpus that did not exist when the model was trained.

    python3 -m relmo.vjdiffuse --ckpt ssl_v1_s0 --dataset rcasa_composite_full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrel, vjrank2  # noqa: E402
from relmo.vjeval import l2  # noqa: E402
from relmo.vjmatch import dtw  # noqa: E402
from relmo.vjrel import parse  # noqa: E402
from relmo.vjreps import build_reps  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402

MIN_SUPPORT = 5


def pooled(Z, ids):
    """One L2-normalised mean vector per recording - the graph's currency."""
    X = np.stack([Z[i].mean(0) for i in ids]).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def affinity(X, k=20, sigma=None):
    """Top-k sparsified, symmetrised, degree-normalised affinity."""
    n = len(X)
    C = X @ X.T
    np.fill_diagonal(C, -np.inf)
    idx = np.argpartition(-C, kth=min(k, n - 2), axis=1)[:, :k]
    W = np.zeros((n, n), np.float32)
    rows = np.repeat(np.arange(n), k)
    vals = C[rows, idx.ravel()]
    if sigma is None:                       # scale from the data, not a prior
        sigma = max(float(np.std(vals)), 1e-3)
    W[rows, idx.ravel()] = np.exp((vals - vals.max()) / sigma)
    W = np.maximum(W, W.T)                  # mutual-friendly symmetrisation
    d = W.sum(1) + 1e-9
    Dm = 1.0 / np.sqrt(d)
    return (W * Dm[:, None]) * Dm[None, :]


def diffuse(y0, S, alpha=0.9, iters=20):
    """y_{t+1} = alpha S y_t + (1-alpha) y_0, row-wise over queries."""
    y = y0.copy()
    for _ in range(iters):
        y = alpha * (y @ S) + (1 - alpha) * y0
    return y


def evaluate(scores, qids, pool, AM, keepmask):
    """prec@support from a (n_q, n_pool) score matrix."""
    pg = np.array([AM[i]["event"] for i in pool])
    prec = []
    for qi, q in enumerate(qids):
        keep = keepmask[qi]
        sv = pg[keep] == AM[q]["event"]
        k = int(sv.sum())
        if k < MIN_SUPPORT:
            continue
        s = scores[qi][keep]
        prec.append(sv[np.argsort(-s)[:k]].sum() / k)
    return float(np.mean(prec)) if prec else float("nan"), prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="rcasa_composite_full")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", default="z + fix + sig WHITENED")
    ap.add_argument("--alphas", default="0.0,0.5,0.7,0.9,0.95")
    ap.add_argument("--ks", default="10,20,50")
    ap.add_argument("--breakdown", default="rcasa")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    AM = vjrel.all_meta(("rcasa", "rcasa_eval", "rcasa_atomic_full",
                         a.dataset))
    seen = {parse(e["id"])["task"]
            for e in R.read_manifest(a.breakdown)["episodes"]} \
        if a.breakdown else None
    D = vjrank2.load_corpus(datasets=(a.dataset,))
    have = sorted(set(D) & set(AM))
    m, ck = vjrank2.load_ckpt(a.ckpt)
    Z = vjrank2.encode_all(m, D, ids=have)
    reps, _ = build_reps(Z, D, have)
    if a.rep not in reps:
        raise SystemExit(f"rep {a.rep!r} not in {list(reps)}")
    Zr = reps[a.rep]
    print(f"{a.dataset}: {len(have)} records | rep {a.rep!r} "
          f"({Zr[have[0]].shape[1]}d) | ckpt {a.ckpt}", flush=True)

    rng = np.random.default_rng(a.seed)
    qids = [have[i] for i in
            sorted(rng.choice(len(have), min(a.queries, len(have)),
                              replace=False))]

    # ---- initial scores: DTW, the expensive part, computed ONCE
    P, ok = _pad([l2(Zr[i]) for i in have])
    L = np.array([len(Zr[i]) for i in have])
    pe = np.array([AM[i]["rollout"] for i in have])
    t0 = time.time()
    S0 = np.zeros((len(qids), len(have)), np.float64)
    keepmask = np.zeros((len(qids), len(have)), bool)
    for qi, q in enumerate(qids):
        C = 1.0 - np.einsum("sd,nkd->nsk", l2(Zr[q]), P)
        C = np.where(ok[:, None, :], C, PAD_COST)
        S0[qi] = -dtw(C, False, L)
        keepmask[qi] = pe != AM[q]["rollout"]
    print(f"  DTW scores {time.time()-t0:.0f}s", flush=True)

    # rank-normalise per query: diffusion must not be driven by scale
    R0 = np.argsort(np.argsort(-S0, 1), 1).astype(np.float32)
    Y0 = np.exp(-R0 / 50.0)
    Y0 = Y0 / (Y0.sum(1, keepdims=True) + 1e-9)

    X = pooled(Zr, have)
    base, _ = evaluate(S0, qids, have, AM, keepmask)
    print(f"\n{'k':>5s} {'alpha':>6s} {'prec':>7s} {'seen':>7s} "
          f"{'UNSEEN':>7s}   (baseline {base:.3f})")
    print("-" * 44)
    out = {"baseline": base}
    for k in [int(x) for x in a.ks.split(",")]:
        S = affinity(X, k=k)
        for al in [float(x) for x in a.alphas.split(",")]:
            Y = Y0 if al == 0 else diffuse(Y0, S, al)
            pr, per = evaluate(Y, qids, have, AM, keepmask)
            cells = []
            for want in (True, False):
                v = [p for q, p in zip(
                    [q for qi, q in enumerate(qids)
                     if (np.array([AM[i]["event"] for i in have])[keepmask[qi]]
                         == AM[q]["event"]).sum() >= MIN_SUPPORT], per)
                    if seen is None or (parse(q)["task"] in seen) is want]
                cells.append(float(np.mean(v)) if v else float("nan"))
            print(f"{k:5d} {al:6.2f} {pr:7.3f} {cells[0]:7.3f} "
                  f"{cells[1]:7.3f}", flush=True)
            out[f"k{k}_a{al}"] = dict(prec=pr, seen=cells[0], unseen=cells[1])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nVERIFIED: wrote {a.out}")
    best = max((v for k, v in out.items() if isinstance(v, dict)),
               key=lambda v: v["prec"], default=None)
    R.log("vjdiffuse", ckpt=a.ckpt, dataset=a.dataset, rep=a.rep,
          baseline=round(base, 4),
          best=round(best["prec"], 4) if best else None)


if __name__ == "__main__":
    main()
