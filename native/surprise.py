"""STEP 4 - SCORE SURPRISE. Prediction residual over the latent series.

Event Segmentation Theory: a boundary is where prediction fails. The
encoder already produces a temporally smooth latent series, so
"surprise" is how badly the near past predicts the present. No model
is trained here and no boundary is used - this step only produces a
per-time signal.

Several estimators are computed because which one carries the signal
is an empirical question, and the GRADE for this step answers it:
how well does each separate true boundary times from interior times
(AUC)? Truth is used ONLY here, as the grader.

    python native/surprise.py --corpus sim    # grade
    python native/surprise.py --corpus bench
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from rawwrite import out_dir, arg                              # noqa: E402

FINE = 2.0          # scale used as the temporal series


def series(corpus):
    """{file_id -> (times, V)} at the finest scale, time-ordered."""
    out = {}
    for p in sorted(out_dir(corpus).glob("*.npz")):
        z = np.load(p)
        w = z["win"]
        V = z["V"].astype(np.float32)
        m = np.abs(w[:, 2] - FINE) < 1e-6
        if m.sum() < 8:
            continue
        t = (w[m, 0] + w[m, 1]) / 2.0
        o = np.argsort(t)
        out[p.stem] = (t[o], V[m][o])
    return out


def estimators(V, k=3):
    """{name -> surprise series} - all causal or symmetric, none
    fitted to anything."""
    n = len(V)
    out = {}
    # 1. adjacent novelty
    d = np.zeros(n)
    d[1:] = 1.0 - (V[1:] * V[:-1]).sum(1)
    out["adjacent"] = d
    # 2. residual against the mean of the last k
    d = np.zeros(n)
    for i in range(1, n):
        h = V[max(0, i - k):i].mean(0)
        h /= np.linalg.norm(h) + 1e-8
        d[i] = 1.0 - float(V[i] @ h)
    out["hist_mean"] = d
    # 3. linear extrapolation residual (constant-velocity prediction)
    d = np.zeros(n)
    for i in range(2, n):
        pred = 2 * V[i - 1] - V[i - 2]
        pred /= np.linalg.norm(pred) + 1e-8
        d[i] = 1.0 - float(V[i] @ pred)
    out["extrap"] = d
    # 4. two-sided: how different is the near future from the near past
    d = np.zeros(n)
    for i in range(n):
        a0, a1 = max(0, i - k), i
        b0, b1 = i + 1, min(n, i + 1 + k)
        if a1 - a0 < 1 or b1 - b0 < 1:
            continue
        A = V[a0:a1].mean(0)
        B = V[b0:b1].mean(0)
        A /= np.linalg.norm(A) + 1e-8
        B /= np.linalg.norm(B) + 1e-8
        d[i] = 1.0 - float(A @ B)
    out["two_sided"] = d
    # 5. checkerboard-kernel novelty (Foote): correlate a
    # [[+,-],[-,+]] kernel along the diagonal of the self-similarity
    # matrix. Unlike adjacent differences it asks whether the whole
    # block before differs from the whole block after, which is what a
    # boundary IS; multiple widths because event length is unknown.
    S = V @ V.T
    for w in (2, 4, 8):
        d = np.zeros(n)
        for i in range(n):
            a0, a1 = i - w, i
            b0, b1 = i, i + w
            if a0 < 0 or b1 > n:
                continue
            AA = S[a0:a1, a0:a1].mean()
            BB = S[b0:b1, b0:b1].mean()
            AB = S[a0:a1, b0:b1].mean()
            d[i] = 0.5 * (AA + BB) - AB
        out[f"foote{w}"] = d
    # 6. scale-combined novelty: z-sum of the three widths, so a
    # boundary that only shows at one timescale still registers
    z = np.zeros(n)
    for w in (2, 4, 8):
        v = out[f"foote{w}"]
        s = v.std() + 1e-8
        z += (v - v.mean()) / s
    out["foote_multi"] = z
    return out


def truth_boundaries(corpus):
    """{file_id -> [times]} EVAL ONLY."""
    import pyarrow.parquet as pq
    out = {}
    if corpus == "sim":
        t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
            .to_pydict()
        for e, a, b in zip(t["episode"], t["t0"], t["t1"]):
            fid = f"ep{int(e):04d}"
            out.setdefault(fid, set()).update(
                [round(float(a), 2), round(float(b), 2)])
        return {k: sorted(v) for k, v in out.items()}
    key = "videos/observation.images.image_0"
    meta = pq.read_table(
        ROOT / "data/bridge/meta/episodes/chunk-000/file-000.parquet",
        columns=[f"{key}/file_index", f"{key}/from_timestamp",
                 f"{key}/to_timestamp"]).to_pydict()
    fi = np.array(meta[f"{key}/file_index"])
    for j in range(len(fi)):
        fid = f"file-{int(fi[j]):03d}"
        out.setdefault(fid, set()).update(
            [round(float(meta[f"{key}/from_timestamp"][j]), 2),
             round(float(meta[f"{key}/to_timestamp"][j]), 2)])
    return {k: sorted(v) for k, v in out.items()}


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return float((r[y == 1].sum() - len(pos) * (len(pos) - 1) / 2)
                 / max(len(pos) * len(neg), 1))


def main():
    corpus = arg("--corpus", "sim")
    tol = arg("--tol", 1.5, float)
    S = series(corpus)
    B = truth_boundaries(corpus)
    print(f"{corpus}: {len(S)} files with a {FINE}s series; "
          f"boundary tolerance +-{tol}s", flush=True)
    acc = {}
    nb = nn = 0
    for fid, (t, V) in S.items():
        bs = B.get(fid)
        if not bs:
            continue
        est = estimators(V)
        near = np.zeros(len(t), bool)
        for b in bs:
            near |= np.abs(t - b) <= tol
        # interior = far from ANY boundary
        far = np.ones(len(t), bool)
        for b in bs:
            far &= np.abs(t - b) > 2 * tol
        if near.sum() < 2 or far.sum() < 2:
            continue
        nb += int(near.sum())
        nn += int(far.sum())
        for k, d in est.items():
            acc.setdefault(k, [[], []])
            acc[k][0].append(d[near])
            acc[k][1].append(d[far])
    print(f"  {nb:,} boundary-adjacent windows, {nn:,} interior")
    print(f"  {'estimator':<12}{'AUC':<8}")
    best = None
    for k, (P, N) in acc.items():
        a = auc(np.concatenate(P), np.concatenate(N))
        print(f"  {k:<12}{a:.3f}")
        if best is None or a > best[1]:
            best = (k, a)
    print(f"\nSTEP 4 GRADE ({corpus}): best {best[0]} AUC {best[1]:.3f}"
          f"   [>=0.70 usable, >=0.80 good, 0.50 = no signal]")


if __name__ == "__main__":
    main()
