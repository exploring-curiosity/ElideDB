"""DYNAMIC DIRECTIONS. Nothing named, nothing hand-listed.

vtrans scored eleven descriptor families I invented - gist, state bands,
transitions, relational structure, surprise. That was the same error as
naming similarity axes, one level up: a vocabulary of summary
statistics. Computing an invented axis at extraction time does not make
it less invented. This file removes the invention.

WHAT IS STORED: the latent itself. A clip becomes its sequence of patch
grids from the frozen encoder, and nothing is summarised at write time.
There is no descriptor list here to argue about.

WHERE THE AXES COME FROM: the query, at comparison time. A query clip is
a SET - its own frames, its own sub-windows - and that set has internal
structure that is free and label-free. For every direction of the latent
space:

    informative(u) = var_corpus(u) / var_query(u)

A direction the corpus spreads along but this query holds still is a
direction that identifies this query. A direction that wobbles inside
the query itself cannot be what makes it that experience, and a
direction the whole corpus shares carries no information about which
clip this is. The subspace is whatever wins that ratio, recomputed for
every query - so a query about a place and a query about a movement
select different directions out of the same stored latent, with nobody
naming either.

This is the per-query weighting idea moved from CHANNELS (where it was
the repeat offender twice, because it rewarded self-consistency alone)
to DIRECTIONS, and with the corpus term that the old one was missing -
self-consistency divided by corpus spread, not self-consistency.

Guiding lights, kept as guides: encoding specificity says the cue
decides which features count, so the cue computes the subspace;
prediction says the state is what the future depends on, which is the
next experiment (a fitted predictor supplies the directions instead of
a variance ratio).

Scored on the transformation battery, which knows nothing about content.

    SDX_ENC=vits SDX_RES=320 python native/vdyn.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import vaug                                               # noqa: E402
import vcore                                              # noqa: E402
import vsrc                                               # noqa: E402
from vtrans import CLIPS, GRID, encode, spearman          # noqa: E402

CACHE = Path("/private/tmp/claude-501/vdyn")
PCA_DIM = 192      # corpus basis width; a capacity choice, not an axis
CORPUS_SPANS = 96  # spans sampled from the recording to fit that basis


def _l2(V, axis=-1):
    return V / np.maximum(np.linalg.norm(V, axis=axis, keepdims=True), 1e-8)


def latent(corpus, mid, t0, t1, tname, params):
    """THE ONLY THING EXTRACTED: per-frame latent, no summary.

    Kept as per-frame global vectors plus the per-frame coarse grid, so
    both 'what is here' and 'where it is' remain available to whatever
    subspace a query later asks for. Nothing is reduced to a statistic
    that pretends to know which of those matters.
    """
    key = hashlib.sha1(
        f"{mid}|{t0}|{t1}|{tname}|{vcore.ENCODER}|{GRID}|v2".encode()
    ).hexdigest()[:20]
    fp = CACHE / f"{key}.npz"
    if fp.exists():
        z = np.load(fp)
        return z["g"], z["c"]
    src = _srcs(corpus)[mid]
    G = encode(vaug.apply_clip(src.cut(t0, t1), params))
    T = len(G)
    g = _l2(G.reshape(T, GRID * GRID, -1).mean(1))          # (T, d)
    k = GRID // 5
    c = G.reshape(T, 5, k, 5, k, -1).mean((2, 4)).reshape(T, -1)  # (T, 25d)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(fp, g=g.astype(np.float16), c=c.astype(np.float16))
    return g.astype(np.float32), c.astype(np.float32)


_SRCS: dict = {}


def _srcs(corpus):
    if corpus not in _SRCS:
        _SRCS[corpus] = {s.id: s for s in vsrc.sources(corpus)}
    return _SRCS[corpus]


def corpus_basis(which, seed=0):
    """A basis for the recording, fitted on the recording. Width is a
    capacity choice; the DIRECTIONS are the data's, not mine."""
    rs = np.random.RandomState(seed)
    X = []
    for corpus, _, _, _ in CLIPS:
        srcs = _srcs(corpus)
        ids = sorted(srcs)
        for _ in range(CORPUS_SPANS // len(CLIPS)):
            mid = ids[rs.randint(len(ids))]
            dur = srcs[mid].dur
            if dur < 9:
                continue
            t0 = round(float(rs.uniform(0, dur - 8)), 1)
            g, c = latent(corpus, mid, t0, t0 + 8.0, "identity",
                          vaug.identity_params())
            X.append(g if which == "g" else c)
    X = np.concatenate(X).astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:PCA_DIM]
    var = (S[:PCA_DIM] ** 2) / max(len(X) - 1, 1)
    return mu, V, var


def project(x, basis):
    mu, V, _ = basis
    return (np.asarray(x, np.float64) - mu) @ V.T


def dyn_sim(Pq, Pc, cvar, topk=48, eps=1e-6):
    """Compare a query SET against a candidate SET in the subspace the
    query itself selects. No fixed axes: `keep` is recomputed per query.
    """
    qv = Pq.var(0) + eps
    info = cvar / qv                       # corpus spread / query tightness
    keep = np.argsort(-info)[:topk]
    w = np.sqrt(info[keep])
    a = _l2(Pq[:, keep].mean(0) * w)
    b = _l2(Pc[:, keep].mean(0) * w)
    return float(a @ b)


def fixed_sim(Pq, Pc):
    return float(_l2(Pq.mean(0)) @ _l2(Pc.mean(0)))


def main():
    B = vaug.battery()
    names = [n for n, _ in B]
    pmap = dict(B)
    sev = {n: vaug.severity(p) for n, p in B}
    from tqdm import tqdm
    vcore.feature_dim()
    print(f"{len(CLIPS)} clips x {len(B)} transforms; latent only, "
          f"no descriptors\n")

    L = {}
    bar = tqdm(total=len(CLIPS) * len(B), unit="clip", desc="latent")
    for corpus, mid, t0, t1 in CLIPS:
        for n, p in B:
            L[(corpus, n)] = latent(corpus, mid, t0, t1, n, p)
            bar.update(1)
    bar.close()

    keys = [(c, n) for c, _, _, _ in CLIPS for n in names]
    for wi, which in enumerate(("g", "c")):
        t = time.time()
        basis = corpus_basis(which)
        cvar = basis[2]
        P = {k: project(L[k][wi], basis) for k in keys}
        label = "global latent" if which == "g" else "5x5 grid latent"
        print(f"\n=== {label}  (basis {PCA_DIM}d fitted on the recording, "
              f"{time.time()-t:.0f}s) ===")
        print(f"  {'method':<14}{'AUC':>7}{'P@1':>7}{'sev-rho':>9}"
              f"{'struct-rho':>12}")
        methods = {"fixed (mean)": None}
        for tk in (24, 48, 96, 192):
            methods[f"dynamic k={tk}"] = tk
        for mname, tk in methods.items():
            S = np.zeros((len(keys), len(keys)))
            for i, ka in enumerate(keys):
                for j, kb in enumerate(keys):
                    if j < i:
                        continue
                    v = (fixed_sim(P[ka], P[kb]) if tk is None
                         else dyn_sim(P[ka], P[kb], cvar, tk))
                    S[i, j] = S[j, i] = v
            same = np.array([[a[0] == b[0] for b in keys] for a in keys])
            off = ~np.eye(len(keys), dtype=bool)
            pos, neg = S[same & off], S[(~same) & off]
            auc = float((pos[:, None] > neg[None, :]).mean()
                        + 0.5 * (pos[:, None] == neg[None, :]).mean())
            p1 = float(np.mean([
                keys[int(np.argmax(np.where(off[i], S[i], -np.inf)))][0]
                == keys[i][0] for i in range(len(keys))]))
            sr, st = [], []
            for c, _, _, _ in CLIPS:
                ii = [i for i, k in enumerate(keys) if k[0] == c]
                i0 = [i for i in ii if keys[i][1] == "identity"][0]
                sr.append(spearman([S[i0, i] for i in ii if i != i0],
                                   [-sev[keys[i][1]] for i in ii if i != i0]))
                fs, td = [], []
                for a in ii:
                    for b in ii:
                        if a < b:
                            fs.append(S[a, b])
                            td.append(-vaug.tdist(pmap[keys[a][1]],
                                                  pmap[keys[b][1]]))
                st.append(spearman(fs, td))
            print(f"  {mname:<14}{auc:7.3f}{p1:7.3f}{np.mean(sr):9.3f}"
                  f"{np.mean(st):12.3f}")

    print("\nAUC/P@1 = a disguised clip finds its own siblings first")
    print("sev-rho = similarity falls as the disguise gets heavier")
    print("str-rho = similarity mirrors transform-space distance")
    print("\nthe subspace is recomputed for every query; no axis is named "
          "anywhere in this file")


if __name__ == "__main__":
    main()
