"""[O]+[A] Objects and agent from tracks alone. PLAN §5.4.

THE structural idea: points on one rigid body share a VELOCITY TIME
SERIES over the whole episode. That single criterion separates a
carried block from the gripper carrying it - they move identically
during the carry, but the gripper also moves during approach and
retreat while the block sits still, so their full-episode signatures
differ. This is why identity rides on tracks: a bundle IS the object,
before/through/after every carry, with no appearance matching, no
re-detection, and no per-frame association to get wrong (the layer
that measured 0.48 recall).

The agent needs no appearance prior either: it is the bundle that
moves during (nearly) every motion burst in the episode, while a block
moves only during its own manipulations. Fitted, not assumed.

All cuts are corpus-fitted from pooled distributions (k-means-1d
midpoints in log space; histograms printed before fitting, per the
Otsu-misfit lesson). No dataset constants anywhere.

    python native/objects.py [--store lake/sim_chains] [--limit N]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import track as _tr_placeholder  # noqa: F401,E402  (path check only)

VER = "v1"
WIN = 2          # velocity window in frames (5 fps -> 0.4 s)
MIN_MOVE_FR = 2  # frames above the speed cut to call a track a mover


def scratch():
    from track import SCRATCH
    return SCRATCH


def views_with_tracks(store_name, limit=0):
    from track import cache_dir
    cdir = cache_dir(store_name)
    out = []
    for p in sorted(cdir.glob("*.npz")):
        ep = int(p.stem.split("_")[0])
        sv = p.stem.split("_", 1)[1]
        if limit and ep >= limit:
            continue
        out.append((ep, sv, p))
    return out


def velocities(tracks):
    """(T,N,2) -> (N, T-WIN, 2) windowed velocity + (N, T-WIN) speed."""
    v = (tracks[WIN:] - tracks[:-WIN]).transpose(1, 0, 2)
    return v, np.linalg.norm(v, axis=2)


def fit_cuts(views, sample=40):
    """Corpus-fitted motion cut (log speed) and coherence cut (log RMS
    velocity difference between tracks)."""
    from chain_delta import k_means_1d
    rng = np.random.default_rng(0)
    sp_pool, coh_pool = [], []
    pick = views[:: max(len(views) // sample, 1)][:sample]
    for ep, sv, p in pick:
        z = np.load(p)
        tr = z["tracks"].astype(np.float32)
        v, sp = velocities(tr)
        sp_pool.append(sp.ravel()[::7])
        idx = rng.choice(len(v), min(300, len(v)), replace=False)
        Vf = v[idx].reshape(len(idx), -1)
        d = np.sqrt(((Vf[:, None, :] - Vf[None, :, :]) ** 2)
                    .sum(-1) / v.shape[1])
        iu = np.triu_indices(len(idx), 1)
        coh_pool.append(d[iu][::11])
    sp = np.concatenate(sp_pool)
    sp = sp[sp > 1e-3]
    ls = np.log10(sp)
    h, e = np.histogram(ls, bins=18)
    print("  speed log-hist " + " ".join(
        f"{10**c:.2f}:{n}" for c, n in zip((e[:-1] + e[1:]) / 2, h)
        if n))
    c = k_means_1d(ls, 3)
    move_cut = 10 ** ((c[1] + c[2]) / 2)
    coh = np.concatenate(coh_pool)
    coh = coh[coh > 1e-3]
    lc = np.log10(coh)
    h, e = np.histogram(lc, bins=18)
    print("  coherence log-hist " + " ".join(
        f"{10**c_:.2f}:{n}" for c_, n in zip((e[:-1] + e[1:]) / 2, h)
        if n))
    cc = k_means_1d(lc, 3)
    coh_cut = 10 ** ((cc[0] + cc[1]) / 2)
    print(f"  FITTED move_cut {move_cut:.2f} px/win   "
          f"coh_cut {coh_cut:.2f} px/win", flush=True)
    return float(move_cut), float(coh_cut)


def bundle_view(tracks, move_cut, coh_cut):
    """-> labels (N,) with -1 for static/background, plus per-bundle
    speed series. Connected components over the velocity-signature
    graph, restricted to tracks that ever move."""
    v, sp = velocities(tracks.astype(np.float32))
    mover = (sp > move_cut).sum(1) >= MIN_MOVE_FR
    idx = np.where(mover)[0]
    labels = np.full(len(tracks[0]), -1, np.int32)
    if len(idx) < 2:
        return labels, v, sp
    Vf = v[idx].reshape(len(idx), -1)
    T = v.shape[1]
    d = np.sqrt(np.maximum(
        (Vf ** 2).sum(1)[:, None] + (Vf ** 2).sum(1)[None, :]
        - 2 * Vf @ Vf.T, 0) / T)
    A = d < coh_cut
    # connected components
    n = len(idx)
    lab = np.full(n, -1, np.int32)
    cur = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = cur
        while stack:
            i = stack.pop()
            for j in np.where(A[i] & (lab < 0))[0]:
                lab[j] = cur
                stack.append(j)
        cur += 1
    labels[idx] = lab
    return labels, v, sp


def bundle_series(labels, sp):
    """{bundle -> median speed series over its member tracks}."""
    out = {}
    for b in np.unique(labels):
        if b < 0:
            continue
        m = labels == b
        if m.sum() < 2:
            continue
        out[int(b)] = np.median(sp[m], axis=0)
    return out


def bursts(series, move_cut, merge_gap):
    """Maximal spans where the bundle moves, short gaps merged."""
    on = series > move_cut
    spans = []
    i0 = None
    for i, o in enumerate(list(on) + [False]):
        if o and i0 is None:
            i0 = i
        elif not o and i0 is not None:
            spans.append([i0, i - 1])
            i0 = None
    out = []
    for s in spans:
        if out and s[0] - out[-1][1] <= merge_gap:
            out[-1][1] = s[1]
        else:
            out.append(s)
    return [tuple(s) for s in out]


def agent_of(bser, move_cut, merge_gap):
    """The agent moves during (nearly) every burst in the episode;
    a block moves only during its own manipulations. Fitted split on
    the per-bundle share of the episode's total motion time."""
    if not bser:
        return set(), {}
    allb = {b: bursts(s, move_cut, merge_gap) for b, s in bser.items()}
    span = {b: sum(e - s + 1 for s, e in v) for b, v in allb.items()}
    tot = max(max(span.values()), 1)
    frac = {b: span[b] / tot for b in span}
    vals = np.array(sorted(frac.values()))
    if len(vals) >= 3:
        from chain_delta import k_means_1d
        c = k_means_1d(vals, 2)
        cut = (c[0] + c[1]) / 2
    else:
        cut = 0.6
    agent = {b for b, f in frac.items() if f >= max(cut, 0.5)}
    return agent, allb


def main():
    from tqdm import tqdm
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else 0
    views = views_with_tracks(store.name, limit)
    print(f"{len(views)} tracked views", flush=True)
    print("fitting corpus cuts...", flush=True)
    move_cut, coh_cut = fit_cuts(views)
    merge_gap = 3          # frames (0.6 s at 5 fps); refit at G2 if needed

    out = {}
    nb, na = [], []
    for ep, sv, p in tqdm(views, desc="bundle", unit="view"):
        z = np.load(p)
        tr = z["tracks"].astype(np.float32)
        ts = z["ts"]
        labels, v, sp = bundle_view(tr, move_cut, coh_cut)
        bser = bundle_series(labels, sp)
        agent, allb = agent_of(bser, move_cut, merge_gap)
        objs = {b: allb[b] for b in bser if b not in agent}
        out[(ep, sv)] = dict(labels=labels, ts=ts, bursts=objs,
                             agent=sorted(agent),
                             agent_bursts={b: allb[b] for b in agent})
        nb.append(len(objs))
        na.append(len(agent))
    print(f"non-agent bundles/view: mean {np.mean(nb):.1f} "
          f"median {np.median(nb):.0f}  agent bundles: "
          f"mean {np.mean(na):.1f}")
    print(f"bursts/view (objects only): "
          f"{np.mean([sum(len(v) for v in d['bursts'].values()) for d in out.values()]):.1f}")
    np.savez_compressed(
        scratch() / f"native_bundles_{VER}_{store.name}.npz",
        data=np.array([(k, val) for k, val in out.items()], object),
        cuts=np.array([move_cut, coh_cut, merge_gap]))
    print(f"saved bundles -> native_bundles_{VER}_{store.name}.npz")


if __name__ == "__main__":
    main()
