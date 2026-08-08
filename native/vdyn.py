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
from vtrans import CLIPS, GRID, encode, spearman, distractors  # noqa: E402

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


def dyn_sim(Pq, Pc, cvar, topk=48, eps=1e-6, Aq=None):
    """Compare in the subspace the query itself selects.

    MEASURED NULL, and why: using the clip's own FRAMES as the query set
    made var_query a temporal variance, so the ratio selected directions
    that are constant through the clip and vary across the corpus -
    which is 'which scene is this', exactly what the plain mean already
    encodes. Every k scored within 0.01 of fixed-mean. It is also
    structurally unable to find an action direction, because a direction
    encoding an action varies WITHIN the clip by definition and the
    criterion suppresses it.

    The invariance that matters is not time, it is NUISANCE. When `Aq`
    is given - the query re-encoded under a few self-generated disguises
    - the denominator becomes variance under disguise, and the surviving
    directions are the ones that identify this clip while ignoring how
    it was filmed. Those disguises come from PROBE primitives, disjoint
    from the ones the battery scores with, so this cannot fit its test.
    """
    qv = (Aq.var(0) if Aq is not None else Pq.var(0)) + eps
    info = cvar / qv                       # corpus spread / query tightness
    keep = np.argsort(-info)[:topk]
    w = np.sqrt(info[keep])
    a = _l2(Pq[:, keep].mean(0) * w)
    b = _l2(Pc[:, keep].mean(0) * w)
    return float(a @ b)


def fixed_sim(Pq, Pc):
    return float(_l2(Pq.mean(0)) @ _l2(Pc.mean(0)))


def sym_dyn(a, b, P, A, cvar, topk):
    """Symmetric: each side proposes its own subspace, scores agree."""
    return 0.5 * (dyn_sim(P[a], P[b], cvar, topk, Aq=A[a])
                  + dyn_sim(P[b], P[a], cvar, topk, Aq=A[b]))


def main():
    B = vaug.battery(prims=vaug.EVAL_PRIMS)      # scored with EVAL only
    probes = vaug.probe_set(3)                   # system disguises: PROBE only
    names = [n for n, _ in B]
    pmap = dict(B)
    sev = {n: vaug.severity(p) for n, p in B}
    from tqdm import tqdm
    vcore.feature_dim()
    D = distractors()
    print(f"{len(CLIPS)} clips x {len(B)} EVAL disguises "
          f"+ {len(D)} undisguised distractors")
    print(f"query-time self-disguises: {len(probes)} (PROBE primitives, "
          f"disjoint from EVAL)\n")

    items, L, A = [], {}, {}
    tot = len(CLIPS) * len(B) * (1 + len(probes)) + len(D)
    bar = tqdm(total=tot, unit="enc", desc="encode")
    for corpus, mid, t0, t1 in CLIPS:
        for n, p in B:
            k = (corpus, n)
            items.append(k)
            L[k] = latent(corpus, mid, t0, t1, n, p); bar.update(1)
            av = []
            for qi, q in enumerate(probes):
                comp = dict(p)
                for pk, pv in q.items():
                    base = vaug.identity_params()[pk]
                    if pv != base:
                        comp[pk] = pv
                av.append(latent(corpus, mid, t0, t1, f"{n}~p{qi}", comp))
                bar.update(1)
            A[k] = av
    for j, (corpus, mid, a, b) in enumerate(D):
        k = (f"dist{j}", "identity")
        items.append(k)
        L[k] = latent(corpus, mid, a, b, "identity",
                      vaug.identity_params()); bar.update(1)
        A[k] = [L[k]]
    bar.close()

    for wi, which in enumerate(("g", "c")):
        basis = corpus_basis(which)
        cvar = basis[2]
        P = {k: project(L[k][wi], basis) for k in items}
        AP = {k: np.concatenate([project(x[wi], basis) for x in A[k]])
              for k in items}
        label = "global latent" if which == "g" else "5x5 grid latent"
        print(f"\n=== {label} ===")
        print(f"  {'method':<20}{'AUC':>7}{'P@1':>7}{'sev-rho':>9}"
              f"{'struct-rho':>12}")
        methods = [("fixed (mean)", None, False)]
        for tk in (24, 48, 96):
            methods.append((f"temporal-var k={tk}", tk, False))
            methods.append((f"nuisance-var k={tk}", tk, True))
        for mname, tk, use_a in methods:
            S = np.zeros((len(items), len(items)))
            for i2, ka in enumerate(items):
                for j2 in range(i2, len(items)):
                    kb = items[j2]
                    if tk is None:
                        v = fixed_sim(P[ka], P[kb])
                    elif use_a:
                        v = sym_dyn(ka, kb, P, AP, cvar, tk)
                    else:
                        v = 0.5 * (dyn_sim(P[ka], P[kb], cvar, tk)
                                   + dyn_sim(P[kb], P[ka], cvar, tk))
                    S[i2, j2] = S[j2, i2] = v
            same = np.array([[a[0] == b[0] for b in items] for a in items])
            off = ~np.eye(len(items), dtype=bool)
            qi = [i2 for i2, k in enumerate(items)
                  if not k[0].startswith("dist")]
            pos = S[np.ix_(qi, range(len(items)))][
                same[np.ix_(qi, range(len(items)))]
                & off[np.ix_(qi, range(len(items)))]]
            neg = S[np.ix_(qi, range(len(items)))][
                (~same[np.ix_(qi, range(len(items)))])
                & off[np.ix_(qi, range(len(items)))]]
            auc = float((pos[:, None] > neg[None, :]).mean()
                        + 0.5 * (pos[:, None] == neg[None, :]).mean())
            p1 = float(np.mean([
                items[int(np.argmax(np.where(off[i2], S[i2], -np.inf)))][0]
                == items[i2][0] for i2 in qi]))
            sr, st = [], []
            for corpus, _, _, _ in CLIPS:
                ii = [i2 for i2, k in enumerate(items) if k[0] == corpus]
                i0 = [i2 for i2 in ii if items[i2][1] == "identity"][0]
                sr.append(spearman([S[i0, i2] for i2 in ii if i2 != i0],
                                   [-sev[items[i2][1]] for i2 in ii
                                    if i2 != i0]))
                fs, td = [], []
                for a2 in ii:
                    for b2 in ii:
                        if a2 < b2:
                            fs.append(S[a2, b2])
                            td.append(-vaug.tdist(pmap[items[a2][1]],
                                                  pmap[items[b2][1]]))
                st.append(spearman(fs, td))
            print(f"  {mname:<20}{auc:7.3f}{p1:7.3f}{np.mean(sr):9.3f}"
                  f"{np.mean(st):12.3f}")

    print("\nAUC/P@1 vs a pool that now includes 60 undisguised real clips")
    print("nuisance-var = subspace from the query's OWN disguises (PROBE),")
    print("scored on disguises it has never seen (EVAL). No axis is named.")


if __name__ == "__main__":
    main()
