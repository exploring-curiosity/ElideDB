"""THE EXPERIMENT LOOP. One command, every variant, scored in seconds.

Objective (user, 2026-08-08): give the system any long recording, hand it
a sample - from inside the recording or from outside - and get back the
closest experiences. Nothing about the data may be named. No fixed axes;
whatever directions matter are discovered from the recording itself.

THERE IS NO TRAINING SET AND NO MANIFEST. Everything fitted here is
fitted on the recording being indexed, unsupervised, at ingest - the
same category as the k-means cells the store already builds from its own
vectors. That is what makes it work on a recording it has never seen:
the structure comes from the data in front of it, not from a corpus
someone curated.

Speed is the point. Patch grids are cached to disk per span, so a new
comparison operator is scored in under a second and only the encoder
ever costs real time.

    SDX_ENC=vits SDX_RES=320 python native/vlab.py            # all
    SDX_ENC=vits SDX_RES=320 python native/vlab.py --only tr_cm
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import vcore                                              # noqa: E402
import vsrc                                               # noqa: E402

GCACHE = Path("/private/tmp/claude-501/vgrids")
# 320px / 16 = 20 patches a side. Pooling to 10x10 was measured to
# DESTROY the signal: the things that move are one to two patches wide,
# so a 2x2 pool buries a placed object under its own neighbourhood and
# every pair came back at 0.27-0.34 - no structure at all.
GRID = int(os.environ.get("SDX_GRID", "20"))
BASIS_SPANS = 64   # spans sampled to fit the recording's own basis


def _l2(V, axis=-1):
    return V / np.maximum(np.linalg.norm(V, axis=axis, keepdims=True), 1e-8)


# ------------------------------------------------------------ extraction

def _encode_grid(F):
    """(T,H,W,3) -> (T,GRID,GRID,d) unit-norm cells. No pooling to one
    vector: arrangement is the whole point."""
    m, proc, dev, dtype, torch = vcore.model()
    out = []
    for i in range(0, len(F), vcore.BATCH):
        chunk = [np.ascontiguousarray(x[..., :3])
                 for x in F[i:i + vcore.BATCH]]
        px = proc(images=chunk, return_tensors="pt",
                  size={"height": 320, "width": 320})["pixel_values"]
        with torch.no_grad():
            r = m(pixel_values=px.to(dev, dtype))
        P = vcore._patches(r, torch).float().cpu().numpy()
        n = P.shape[1]
        s = int(round(n ** 0.5))
        P = P[:, :s * s].reshape(len(chunk), s, s, -1)
        k = s // GRID
        P = P[:, :k * GRID, :k * GRID]
        P = P.reshape(len(chunk), GRID, k, GRID, k, -1).mean((2, 4))
        out.append(P.astype(np.float16))
    return _l2(np.concatenate(out).astype(np.float32))


def grid(corpus, media, t0, t1):
    """Cached patch grid for one span."""
    key = hashlib.sha1(
        f"{corpus}|{media}|{t0}|{t1}|{vcore.ENCODER}|{GRID}".encode()
    ).hexdigest()[:20]
    fp = GCACHE / f"{key}.npy"
    if fp.exists():
        return np.load(fp).astype(np.float32)
    src = _srcs(corpus)[media]
    F = src.cut(t0, t1)
    G = _encode_grid(F)
    GCACHE.mkdir(parents=True, exist_ok=True)
    np.save(fp, G.astype(np.float16))
    return G


_SRC_CACHE: dict = {}


def _srcs(corpus):
    if corpus not in _SRC_CACHE:
        _SRC_CACHE[corpus] = {s.id: s for s in vsrc.sources(corpus)}
    return _SRC_CACHE[corpus]


# -------------------------------------------------- the recording's basis

_BASIS: dict = {}


def basis(corpus, span_s=8.0, seed=0):
    """What this recording does ALL THE TIME - fitted from the recording.

    vext measured the failure: every transition field is dominated by the
    same object, because the same actor is present in every window. That
    object is a COMMON MODE of the recording, exactly as the static scene
    was a common mode of a media. It is not named, not detected, not
    described - it is whatever the leading directions of this recording's
    own transition distribution turn out to be, and it is removed the
    same way the scene basis was.
    """
    if corpus in _BASIS:
        return _BASIS[corpus]
    rs = np.random.RandomState(seed)
    srcs = _srcs(corpus)
    ids = sorted(srcs)
    D, S, M = [], [], []
    for _ in range(BASIS_SPANS):
        mid = ids[rs.randint(len(ids))]
        dur = srcs[mid].dur
        if dur < span_s + 1:
            continue
        t0 = float(rs.uniform(0, dur - span_s))
        G = grid(corpus, mid, round(t0, 1), round(t0 + span_s, 1))
        if len(G) < 8:
            continue
        D.append(_trans_raw(G).ravel())
        S.append(G.reshape(len(G), -1).mean(0))
        M.append(np.linalg.norm(_trans_raw(G), axis=-1).ravel())
    D = np.stack(D)
    S = np.stack(S)
    # leading directions of the transition distribution, and of state
    Dc = D - D.mean(0)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    Sc = S - S.mean(0)
    _, _, Vs = np.linalg.svd(Sc, full_matrices=False)
    # HOW OFTEN DOES THIS CELL CHANGE AT ALL?
    # The change field was measured to be the mover, everywhere, in every
    # span - the top cells sat on it in all four battery spans while the
    # thing that was actually relocated barely registered. A cell that
    # changes in EVERY span carries no information about which span this
    # is; a cell that rarely changes carries a lot. This is inverse
    # document frequency over change, fitted from the recording itself -
    # nothing is detected, named, or given a position prior.
    Mm = np.stack(M).mean(0)
    _BASIS[corpus] = dict(tmean=D.mean(0), tdirs=Vt,
                          smean=S.mean(0), sdirs=Vs,
                          idf=1.0 / (Mm + np.median(Mm)))
    return _BASIS[corpus]


# ------------------------------------------------------- span descriptors

def _trans_raw(G, settle=0.25):
    n = max(int(len(G) * settle), 2)
    return G[-n:].mean(0) - G[:n].mean(0)


def desc(G, corpus, cm=0, mode="trans"):
    """Span -> descriptor. `cm` = how many recording-common modes to
    remove. Everything is a function of pixels plus the recording's own
    statistics."""
    if mode == "gist":
        v = G.reshape(len(G), -1).mean(0)
        if cm:
            b = basis(corpus)
            v = v - b["smean"]
            v = v - (v @ b["sdirs"][:cm].T) @ b["sdirs"][:cm]
        return _l2(v)
    D = _trans_raw(G)
    if cm:
        b = basis(corpus)
        d = D.ravel() - b["tmean"]
        d = d - (d @ b["tdirs"][:cm].T) @ b["tdirs"][:cm]
        D = d.reshape(D.shape)
    return D


def persist(G, corpus, cm=0, lag=0.5):
    """Change that OUTLASTS the mover.

    The actor passes through; what it rearranges stays. Measured as the
    part of the end-state change still present in the final `lag`
    fraction, minus the part explained by the mid-window excursion."""
    n = max(int(len(G) * 0.25), 2)
    m = max(int(len(G) * lag), 2)
    start = G[:n].mean(0)
    end = G[-n:].mean(0)
    mid = G[n:-n].mean(0) if len(G) > 2 * n else G.mean(0)
    D = (end - start) - 0.5 * (mid - start)      # excursion suppressed
    if cm:
        b = basis(corpus)
        d = D.ravel() - b["tmean"]
        d = d - (d @ b["tdirs"][:cm].T) @ b["tdirs"][:cm]
        D = d.reshape(D.shape)
    return D


def effect(G, corpus, cm=0, settle=0.25):
    """Change that is SETTLED AT BOTH ENDS.

    vext's diagnosis was that the mover dominates every transition. The
    mover is not named or detected here - it is simply whatever is still
    in motion when the window ends. A cell is credited with change only
    if it was at rest before AND is at rest after; a thing passing
    through is moving at one end and cancels itself out, while a thing
    that was put somewhere is at rest in both and survives.
    """
    n = max(int(len(G) * settle), 2)
    a, b = G[:n], G[-n:]
    # per-cell rest = how little the cell's own feature varies
    va = np.linalg.norm(a - a.mean(0), axis=-1).mean(0)
    vb = np.linalg.norm(b - b.mean(0), axis=-1).mean(0)
    rest = (1.0 / (1.0 + 8.0 * va)) * (1.0 / (1.0 + 8.0 * vb))
    D = (b.mean(0) - a.mean(0)) * rest[..., None]
    if cm:
        s = basis(corpus)
        d = D.ravel() - s["tmean"]
        d = d - (d @ s["tdirs"][:cm].T) @ s["tdirs"][:cm]
        D = d.reshape(D.shape)
    return D


def relstruct(G, corpus, cm=0, settle=0.25):
    """CHANGE IN THE ARRANGEMENT, not in the contents.

    Cell-to-cell affinity is computed at the settled start and at the
    settled end; the descriptor is how that affinity structure moved.
    Two things becoming adjacent alters this even when both things keep
    their own appearance - which is the case a bag of cell vectors
    cannot express, and the case the product question is made of.
    Absolute appearance cancels: only relations enter.
    """
    n = max(int(len(G) * settle), 2)
    A = _l2(G[:n].mean(0).reshape(GRID * GRID, -1))
    B = _l2(G[-n:].mean(0).reshape(GRID * GRID, -1))
    Sa, Sb = A @ A.T, B @ B.T
    iu = np.triu_indices(GRID * GRID, 1)
    d = (Sb - Sa)[iu]
    if cm:
        # z-score against this recording's own affinity-change spread
        d = d - d.mean()
    return d


def sparse(D, k=12):
    """Keep the k cells that changed most; zero the rest.

    A placed object is one or two patches wide. Averaged over 400 cells
    its contribution is noise, and the mover - which is large - wins the
    descriptor by area alone. Sparsifying asks WHERE something ended up
    different, which is the only part of the field that carries the
    answer. k is a resolution choice, not a fitted constant: it is the
    number of patches an object occupies, times a small factor.
    """
    F = D.reshape(-1, D.shape[-1])
    w = np.linalg.norm(F, axis=1)
    keep = np.argsort(-w)[:k]
    out = np.zeros_like(F)
    out[keep] = F[keep]
    return out.reshape(D.shape)


def idfw(D, corpus, p=1.0):
    """Down-weight cells this recording changes all the time."""
    w = basis(corpus)["idf"].reshape(D.shape[:-1])
    w = (w / w.max()) ** p
    return D * w[..., None]


def quiet(G, corpus, cm=0, settle=0.25, k=1.0):
    """Diff only the cells that were STILL at both ends.

    Measured, not assumed: the change field is the mover. Weighting by
    magnitude picks the mover (it is large); weighting by spatial rarity
    fails because the mover is not in a fixed place. What separates the
    mover from a relocated thing is that the mover is IN MOTION at the
    window's edges while a thing that was picked up was at rest before,
    and a thing that was put down is at rest after.

    So: per cell, temporal spread inside the first quarter and inside
    the last quarter. A cell is admitted only if it is quiet in both,
    against a threshold read off this span's own distribution (median +
    k*MAD - no constant fitted on any benchmark). Everything still
    moving at either edge is dropped outright.

    The descriptor NORM then means something on its own: a span where
    nothing came to rest anywhere new has almost no admitted change.
    """
    n = max(int(len(G) * settle), 2)
    a, b = G[:n], G[-n:]
    va = np.linalg.norm(a - a.mean(0), axis=-1).mean(0)
    vb = np.linalg.norm(b - b.mean(0), axis=-1).mean(0)
    m = np.maximum(va, vb)
    thr = np.median(m) + k * np.median(np.abs(m - np.median(m)))
    keep = (m <= thr)
    D = (b.mean(0) - a.mean(0)) * keep[..., None]
    if cm:
        s = basis(corpus)
        d = D.ravel() - s["tmean"]
        d = d - (d @ s["tdirs"][:cm].T) @ s["tdirs"][:cm]
        D = d.reshape(D.shape) * keep[..., None]
    return D


# ---------------------------------------------------------- comparators

def _assign(A, B):
    """Hungarian assignment over cells: WHERE it happened is free."""
    from scipy.optimize import linear_sum_assignment
    a = A.reshape(-1, A.shape[-1])
    b = B.reshape(-1, B.shape[-1])
    wa, wb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    m = max(wa.max(), wb.max(), 1e-8)
    C = _l2(a) @ _l2(b).T
    W = np.sqrt(np.outer(wa / m, wb / m))
    r, c = linear_sum_assignment(-(C * W))
    return float((C[r, c] * W[r, c]).sum() / max(W[r, c].sum(), 1e-8))


def _flat(A, B):
    return float(_l2(A.ravel()) @ _l2(B.ravel()))


# ------------------------------------------------------------- variants

VARIANTS = {
    "gist_cos":  lambda G, c: ("flat", desc(G, c, 0, "gist")),
    "gist_cm2":  lambda G, c: ("flat", desc(G, c, 2, "gist")),
    "tr_flat":   lambda G, c: ("flat", desc(G, c, 0)),
    "tr_asg":    lambda G, c: ("asg",  desc(G, c, 0)),
    "tr_cm1":    lambda G, c: ("asg",  desc(G, c, 1)),
    "tr_cm2":    lambda G, c: ("asg",  desc(G, c, 2)),
    "tr_cm4":    lambda G, c: ("asg",  desc(G, c, 4)),
    "tr_cm8":    lambda G, c: ("asg",  desc(G, c, 8)),
    "tr_cm16":   lambda G, c: ("asg",  desc(G, c, 16)),
    "ps_cm0":    lambda G, c: ("asg",  persist(G, c, 0)),
    "ps_cm2":    lambda G, c: ("asg",  persist(G, c, 2)),
    "ps_cm4":    lambda G, c: ("asg",  persist(G, c, 4)),
    "ps_cm8":    lambda G, c: ("asg",  persist(G, c, 8)),
    "ps_cm8f":   lambda G, c: ("flat", persist(G, c, 8)),
    "ef_cm0":    lambda G, c: ("asg",  effect(G, c, 0)),
    "ef_cm2":    lambda G, c: ("asg",  effect(G, c, 2)),
    "ef_cm4":    lambda G, c: ("asg",  effect(G, c, 4)),
    "ef_cm8":    lambda G, c: ("asg",  effect(G, c, 8)),
    "ef_cm0f":   lambda G, c: ("flat", effect(G, c, 0)),
    "ef_cm4f":   lambda G, c: ("flat", effect(G, c, 4)),
    "rel":       lambda G, c: ("flat", relstruct(G, c, 0)),
    "rel_z":     lambda G, c: ("flat", relstruct(G, c, 1)),
    "sp_ef8":    lambda G, c: ("asg",  sparse(effect(G, c, 0), 8)),
    "sp_ef16":   lambda G, c: ("asg",  sparse(effect(G, c, 0), 16)),
    "sp_ef32":   lambda G, c: ("asg",  sparse(effect(G, c, 0), 32)),
    "sp_ef16c4": lambda G, c: ("asg",  sparse(effect(G, c, 4), 16)),
    "sp_tr16":   lambda G, c: ("asg",  sparse(desc(G, c, 0), 16)),
    "sp_tr16c4": lambda G, c: ("asg",  sparse(desc(G, c, 4), 16)),
    "sp_ps16":   lambda G, c: ("asg",  sparse(persist(G, c, 0), 16)),
    "sp_ps16c4": lambda G, c: ("asg",  sparse(persist(G, c, 4), 16)),
    "idf_tr":    lambda G, c: ("asg",  idfw(desc(G, c, 0), c)),
    "idf_tr2":   lambda G, c: ("asg",  idfw(desc(G, c, 0), c, 2.0)),
    "idf_ef":    lambda G, c: ("asg",  idfw(effect(G, c, 0), c)),
    "idf_ef2":   lambda G, c: ("asg",  idfw(effect(G, c, 0), c, 2.0)),
    "idf_ef3":   lambda G, c: ("asg",  idfw(effect(G, c, 0), c, 3.0)),
    "idf_ps2":   lambda G, c: ("asg",  idfw(persist(G, c, 0), c, 2.0)),
    "idf_sp":    lambda G, c: ("asg",  sparse(idfw(effect(G, c, 0), c, 2.0), 16)),
    "idf_sp8":   lambda G, c: ("asg",  sparse(idfw(effect(G, c, 0), c, 2.0), 8)),
    "idf_spf":   lambda G, c: ("flat", sparse(idfw(effect(G, c, 0), c, 2.0), 16)),
    "idf_trf":   lambda G, c: ("flat", idfw(desc(G, c, 0), c, 2.0)),
    "qt_k0.5":   lambda G, c: ("asg",  quiet(G, c, 0, k=0.5)),
    "qt_k1":     lambda G, c: ("asg",  quiet(G, c, 0, k=1.0)),
    "qt_k2":     lambda G, c: ("asg",  quiet(G, c, 0, k=2.0)),
    "qt_k1f":    lambda G, c: ("flat", quiet(G, c, 0, k=1.0)),
    "qt_k1c2":   lambda G, c: ("asg",  quiet(G, c, 2, k=1.0)),
    "qt_k1s":    lambda G, c: ("asg",  sparse(quiet(G, c, 0, k=1.0), 12)),
    "qt_k2s":    lambda G, c: ("asg",  sparse(quiet(G, c, 0, k=2.0), 12)),
    "qt_k1i":    lambda G, c: ("asg",  idfw(quiet(G, c, 0, k=1.0), c, 2.0)),
}

OPS = {"flat": _flat, "asg": _assign}


# --------------------------------------------------------------- scoring

def score_variant(name, G, bat, corpus):
    """Margin between the weakest match and the strongest reject.

    A single number that is only positive when EVERY match pair beats
    EVERY reject pair it appears in. That is the product question in one
    scalar - not an average that a good pair can carry."""
    kind_d = {k: VARIANTS[name](G[k], corpus) for k in G}
    op = OPS[kind_d[next(iter(kind_d))][0]]
    S = {}
    for a in G:
        for b in G:
            if a < b:
                S[(a, b)] = op(kind_d[a][1], kind_d[b][1])
    key = lambda a, b: S[(min(a, b), max(a, b))]        # noqa: E731
    mm = [key(a, b) for a, b in bat["match"]]
    rr = [key(a, b) for a, b in bat["reject"]]
    return min(mm) - max(rr), mm, rr, S


def main():
    from flowgebd import arg
    only = arg("--only", "")
    corpus = arg("--corpus", "sim")
    bat = json.loads((ROOT / "eval/battery.json").read_text())[corpus]
    spans = bat["spans"]
    t0 = time.time()
    G = {k: grid(corpus, s["media"], s["t0"], s["t1"])
         for k, s in spans.items()}
    print(f"{len(G)} spans, grids ready in {time.time() - t0:.1f}s "
          f"({vcore.ENCODER.split('/')[-1]}, {GRID}x{GRID})")
    t0 = time.time()
    basis(corpus)
    print(f"recording basis fitted from {BASIS_SPANS} sampled spans "
          f"in {time.time() - t0:.1f}s\n")

    names = [only] if only else list(VARIANTS)
    print(f"  {'variant':<10}{'margin':>9}   {'worst match':>12}"
          f"{'best reject':>13}   verdict")
    best = None
    for n in names:
        t = time.time()
        margin, mm, rr, _ = score_variant(n, G, bat, corpus)
        v = "PASS" if margin > 0 else ""
        print(f"  {n:<10}{margin:+9.3f}   {min(mm):12.3f}{max(rr):13.3f}"
              f"   {v}  ({time.time() - t:.2f}s)")
        if best is None or margin > best[1]:
            best = (n, margin)
    print(f"\nbest: {best[0]} at {best[1]:+.3f}")
    m, mm, rr, S = score_variant(best[0], G, bat, corpus)
    ks = sorted(G)
    print(f"\n{best[0]} full matrix")
    print("      " + "".join(f"{k:>8}" for k in ks))
    for a in ks:
        row = "".join(f"{(1.0 if a == b else S[(min(a,b),max(a,b))]):8.3f}"
                      for b in ks)
        print(f"  {a}   {row}")
    print(f"\nmatch pairs {bat['match']} -> "
          + ", ".join(f"{S[(min(a,b),max(a,b))]:.3f}" for a, b in bat["match"]))


if __name__ == "__main__":
    main()
