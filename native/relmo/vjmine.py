"""Mine CROSS-VIDEO positive pairs with no labels, for use as training targets.

WHY. ssl_v1 scored 0.211 on the sealed corpus against 0.371 for the frozen
channels, despite PR(z) reaching 46.7. The diagnosis is in the training log:
every positive it ever saw was two crops of the SAME recording, and its
contrastive term hit 0.005 by epoch 2 because "which video is this" is
trivial. It learned within-video consistency and was then asked to do
cross-video retrieval. Nothing taught it what makes two DIFFERENT videos show
the same kind of moment - which is the entire task.

This module manufactures those examples from the corpus itself. The
bootstrap space is the FROZEN fix+sig whitened representation, because that
is the strongest label-free space measured (0.371, and a 0.0 seen-vs-unseen
gap), not the trained one - bootstrapping from a weaker space would just
amplify its errors.

FOUR FILTERS, because a wrong positive poisons and a missing one costs
nothing (there is unlimited unlabeled video):

  1. MUTUAL top-k. A counts B only if B also counts A. One-directional
     attraction is dropped.
  2. CHANNEL CONSENSUS. fix (motion) and sig (appearance) are independent
     witnesses with different failure modes; both must rank the pair above
     their own high percentile. A wall that looks like another wall but moves
     differently is rejected.
  3. TEMPORAL VERIFICATION. The two traces must align under DTW, not merely
     have similar means. A coincidental appearance match does not survive.
  4. ABSTENTION. Keep only the top fraction. Precision of the mined set is
     everything; recall is free.

Cross-view stays barred: pairs sharing a rollout are removed before anything
else. Nothing here reads a label, a task name or a manifest field.

    python3 -m relmo.vjmine --out data/relmo/mined_v1.json
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
from relmo.vjeval import l2  # noqa: E402
from relmo.vjmatch import dtw  # noqa: E402
from relmo.vjreps import apply_w, fit_whiten  # noqa: E402
from relmo.vjssl import POOL, load_pool  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402


def pooled(mats):
    X = np.stack([m.mean(0) for m in mats]).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def mutual_topk(C, k):
    """-> set of (i,j) i<j present in each other's top-k."""
    n = len(C)
    idx = np.argpartition(-C, kth=min(k, n - 2), axis=1)[:, :k]
    top = [set(r) for r in idx]
    out = set()
    for i in range(n):
        for j in top[i]:
            if i != j and i in top[j]:
                out.add((min(i, j), max(i, j)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pct", type=float, default=99.0,
                    help="per-channel percentile both channels must clear")
    ap.add_argument("--keep-frac", type=float, default=0.01,
                    help="fraction of DTW-verified candidates to keep")
    ap.add_argument("--max-dtw", type=int, default=60000)
    ap.add_argument("--out", default=str(R.BASE / "mined_v1.json"))
    a = ap.parse_args()

    print("loading pool...", flush=True)
    D = load_pool(POOL)
    ids = sorted(D)
    n = len(ids)
    print(f"{n} recordings", flush=True)

    fixm = [D[i]["fix"].astype(np.float32) for i in ids]
    sigm = [D[i]["sig"].astype(np.float32) for i in ids]
    wf = fit_whiten(np.concatenate(fixm), 256)
    ws = fit_whiten(np.concatenate(sigm), 256)
    Xf, Xs = pooled([apply_w(m, wf) for m in fixm]), \
        pooled([apply_w(m, ws) for m in sigm])
    roll = np.array([D[i]["rollout"] for i in ids])

    t0 = time.time()
    Cf, Cs = Xf @ Xf.T, Xs @ Xs.T
    np.fill_diagonal(Cf, -np.inf)
    np.fill_diagonal(Cs, -np.inf)
    same = roll[:, None] == roll[None, :]        # cross-view bar
    Cf[same] = -np.inf
    Cs[same] = -np.inf
    C = 0.5 * (Cf + Cs)
    print(f"  affinities {time.time()-t0:.0f}s", flush=True)

    cand = mutual_topk(C, a.k)
    print(f"  mutual top-{a.k}: {len(cand)} pairs", flush=True)
    tf = np.percentile(Cf[np.isfinite(Cf)], a.pct)
    ts = np.percentile(Cs[np.isfinite(Cs)], a.pct)
    cand = [(i, j) for i, j in cand if Cf[i, j] >= tf and Cs[i, j] >= ts]
    print(f"  channel consensus (fix>={tf:.3f} AND sig>={ts:.3f}): "
          f"{len(cand)} pairs", flush=True)
    if not cand:
        raise SystemExit("no candidates survived consensus - lower --pct")

    rng = np.random.default_rng(0)
    if len(cand) > a.max_dtw:
        cand = [cand[k] for k in rng.choice(len(cand), a.max_dtw,
                                            replace=False)]
        print(f"  sampled {len(cand)} for DTW verification", flush=True)

    t0 = time.time()
    Zw = [np.concatenate([apply_w(fixm[i], wf), apply_w(sigm[i], ws)], -1)
          for i in range(n)]
    costs = np.empty(len(cand))
    for c, (i, j) in enumerate(cand):
        P, ok = _pad([l2(Zw[j])])
        Cm = 1.0 - np.einsum("sd,nkd->nsk", l2(Zw[i]), P)
        Cm = np.where(ok[:, None, :], Cm, PAD_COST)
        costs[c] = dtw(Cm, False, np.array([len(Zw[j])]))[0]
    print(f"  DTW verification {time.time()-t0:.0f}s", flush=True)

    keep = int(max(1, len(cand) * a.keep_frac))
    order = np.argsort(costs)[:keep]
    pairs = [[ids[cand[o][0]], ids[cand[o][1]], float(costs[o])]
             for o in order]
    src = {}
    for p in pairs:
        key = tuple(sorted((p[0].split("/")[0], p[1].split("/")[0])))
        src[" x ".join(key)] = src.get(" x ".join(key), 0) + 1
    Path(a.out).write_text(json.dumps(
        dict(k=a.k, pct=a.pct, keep_frac=a.keep_frac, n_pool=n,
             n_pairs=len(pairs), pairs=pairs), indent=1))
    print(f"\nkept {len(pairs)} pairs (top {a.keep_frac*100:g}% by DTW cost)")
    print("cross-corpus composition:")
    for k, v in sorted(src.items(), key=lambda x: -x[1]):
        print(f"   {k:44s} {v:5d}")
    print(f"VERIFIED: wrote {a.out}")
    R.log("vjmine", n_pool=n, n_pairs=len(pairs), k=a.k, pct=a.pct)


if __name__ == "__main__":
    main()
