"""MOTION-STRUCTURE channel: which distinct thing moved when.

Domain-blind and semantics-free. A track bundle is not an "object" and
carries no role - it is coherently-moving stuff, exactly as valid for
a car, a person or a pallet as for a block. There is no agent, no
rest, no pick/place, no threshold cascade: the representation is a
continuous per-window matrix

    M[w, s] = motion magnitude of bundle-slot s in window w

with slots ordered CANONICALLY by first motion (so the same chain
gives the same slot pattern regardless of appearance or placement),
then flattened to a per-window vector. Retrieval aligns these
sequences like any other channel.

This targets the one thing frozen global encoders provably cannot
express (oracle 0.992 vs every perceptual approach <=0.45): identity
of the moving thing across time, which tracking gives for free.

    python native/motstruct.py --probe     (AUC on cached episodes)
    python native/motstruct.py --build     (all episodes -> cache)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

NSLOT = 6          # canonical slots kept (by first motion)
NWIN = 16          # resampled timeline length


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def bundles_of(tracks, k_means_1d):
    """Rigidity clustering (native/proto.bundle) -> labels, speed."""
    import proto
    labels, spd, mv_cut, coh = proto.bundle(tracks, k_means_1d)
    return labels, spd


def structure(tracks, k_means_1d):
    """-> (NWIN, NSLOT+2) motion-structure matrix, semantics-free."""
    labels, spd = bundles_of(tracks, k_means_1d)
    ids = [b for b in np.unique(labels) if b > 0
           and (labels == b).sum() >= 3]
    if not ids:
        return None
    prof = np.stack([np.median(spd[labels == b], axis=0)
                     for b in ids])                 # B, T-1
    # MERGE fragments of one body: fragments move at the SAME TIMES.
    # (correlation of motion profiles, fitted cut - no appearance,
    # no geometry, no roles.)
    P = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8)
    C = P @ P.T
    keep, groups = list(range(len(ids))), []
    seen = set()
    for i in keep:
        if i in seen:
            continue
        g = [j for j in keep if C[i, j] > 0.8]
        groups.append(g)
        seen.update(g)
    prof = np.stack([prof[g].max(0) for g in groups])
    # canonical slot order: first motion time (scale-free, no roles)
    thr = np.percentile(prof, 75) if prof.size else 0.0
    first = []
    for r in prof:
        idx = np.where(r > thr)[0]
        first.append(idx[0] if len(idx) else len(r) + 1)
    order = np.argsort(first)
    prof = prof[order][:NSLOT]
    # resample the timeline, normalise magnitude per episode
    T = prof.shape[1]
    grid = np.linspace(0, T - 1, NWIN).round().astype(int)
    M = prof[:, grid]                                 # S, NWIN
    M = M / (M.max() + 1e-8)
    out = np.zeros((NWIN, NSLOT + 2), np.float32)
    out[:, :M.shape[0]] = M.T
    out[:, NSLOT] = M.sum(0)                          # total motion
    out[:, NSLOT + 1] = (M > 0.15).sum(0) / max(NSLOT, 1)
    return out


def load_cached(limit=0):
    from track import SCRATCH
    import proto
    from chain_delta import k_means_1d
    tdir = SCRATCH / "proto_tracks_v2"
    out = {}
    for p in sorted(tdir.glob("*_simA.npz")):
        ep = int(p.stem.split("_")[0])
        if limit and ep >= limit:
            continue
        z = np.load(p)
        S = structure(z["tracks"].astype(np.float32), k_means_1d)
        if S is not None:
            out[ep] = S
    return out


def probe():
    """Same-template vs different-template separability (AUC)."""
    import pyarrow.parquet as pq
    reps = load_cached()
    print(f"{len(reps)} episodes with cached tracks", flush=True)
    if len(reps) < 6:
        raise SystemExit("not enough cached episodes")
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    eps = sorted(reps)
    same, diff = [], []
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            a, b = reps[eps[i]], reps[eps[j]]
            A = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            B = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
            s = float((A * B).sum(1).mean())
            (same if tmpl.get(eps[i]) == tmpl.get(eps[j])
             else diff).append(s)
    x = np.array(same + diff)
    y = np.array([1] * len(same) + [0] * len(diff))
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    auc = (r[y == 1].sum() - len(same) * (len(same) - 1) / 2) \
        / max(len(same) * len(diff), 1)
    print(f"same-template pairs {len(same)} mean {np.mean(same):.3f}")
    print(f"diff-template pairs {len(diff)} mean {np.mean(diff):.3f}")
    print(f"AUC {auc:.3f}   (>0.65 = worth building; 0.50 = nothing)")


if __name__ == "__main__":
    probe()
