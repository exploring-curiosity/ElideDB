"""THE METRIC, not the architecture.

whyfail.py established the fact this file acts on: an MLP readout of
the SHIPPED stored vector names the primitive at 0.859, while cosine
retrieval over that same vector yields 0.382 at k=support (chance
0.283). The primitive is present; the metric cannot see it. Cosine
weights all 320 dimensions alike, and most of that variance is
nuisance - which block, where on the table, which arm pose, which
episode - so ranking collapses toward the majority class (every query
returns 'pick', which is 44% of the corpus).

Every candidate here is LABEL-FREE and per-corpus, so it may ship:

  csls      hubness correction (already in the read path)
  zca       whiten the corpus covariance: dominant variance
            directions ARE the nuisance, so equalizing them stops
            them from dominating the inner product
  diffuse   corpus-graph propagation (in the read path, but only over
            an 800-candidate subset at query time)
  xview     a linear projection trained by InfoNCE on CROSS-VIEW
            pairs: the same moment filmed by two cameras is a
            positive, everything else a negative. No labels, no task
            knowledge - two cameras of one recording are raw data.
            What it must learn is exactly what we want: throw away
            what differs between views of the same event (viewpoint,
            position, scene) and keep what makes an event itself.
  hard      the same, plus temporal-neighbour negatives: windows
            adjacent in the same episode are DIFFERENT events and are
            the hardest negatives available.

Gate: yield at k = support, against the chance floor, episode-disjoint
where a fit is involved.

    python native/metricfix.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
import vwm_qbe  # noqa: E402

FPS = 10.0
PRIMS = ("pick", "place", "stack", "unstack", "push")
CORPORA = ("sim_chains", "sim_eval_bal")


def load():
    idx = None
    for corp in CORPORA:
        part = vwm_qbe.WMIndex(root=f"data/{corp}", corpus=corp)
        if idx is None:
            idx = part
        else:
            idx.traj.update(part.traj)
            idx.kin.update(part.kin)
    V, prims, eps_, tspan = [], [], [], []
    for corp in CORPORA:
        for ep in sorted((ROOT / "data" / corp).glob("ep*")):
            mid = f"{corp}/{ep.name}"
            if mid not in idx.traj:
                continue
            meta = json.loads((ep / "meta.json").read_text())
            views = idx.traj[mid]
            if len(views) < 2:
                continue
            T = min(len(r) for r in views)
            for e in meta["events"]:
                if not e["ok"]:
                    continue
                a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
                if b > T or b - a < 4:
                    continue
                V.append(np.stack(
                    [vwm_qbe._l2(idx.dmulti(r, a, b))
                     for r in views[:2]]))
                prims.append(e["prim"])
                eps_.append(mid)
                tspan.append((a, b))
    return (np.array(V, np.float32), np.array(prims),
            np.array(eps_), np.array(tspan))


def yld(S, prims, eps_, tag, quiet=False):
    n = len(S)
    S = S.copy()
    np.fill_diagonal(S, -9e9)
    per = defaultdict(list)
    for i in range(n):
        mask = eps_ != eps_[i]
        lab = prims[mask] == prims[i]
        sup = int(lab.sum())
        if not sup:
            continue
        order = np.argsort(-S[i][mask])
        per[prims[i]].append((int(lab[order[:sup]].sum()) / sup,
                              sup / int(mask.sum())))
    ally = [r[0] for p in per for r in per[p]]
    allc = [r[1] for p in per for r in per[p]]
    if not quiet:
        line = "  ".join(
            f"{p} {np.mean([r[0] for r in per[p]]):.3f}"
            for p in PRIMS if per[p])
        print(f"{tag:22s} YIELD {np.mean(ally):.3f}  "
              f"(chance {np.mean(allc):.3f}, "
              f"x{np.mean(ally) / max(np.mean(allc), 1e-6):.2f})   {line}",
              flush=True)
    return float(np.mean(ally))


def viewmax(V):
    S = None
    for a in range(2):
        for b in range(2):
            X = V[:, a] @ V[:, b].T
            S = X if S is None else np.maximum(S, X)
    return S


def csls(S, k=50):
    hub = np.sort(S, 1)[:, -k:].mean(1)
    return 2 * S - hub[None, :] - hub[:, None]


def diffuse(S, k=10, alpha=0.9, iters=30):
    n = len(S)
    A = S.copy()
    np.fill_diagonal(A, -9e9)
    thr = np.partition(A, n - k, axis=1)[:, n - k][:, None]
    W = np.where(A >= thr, np.maximum(A, 0.0), 0.0)
    W = np.maximum(W, W.T)
    d = np.sqrt(np.maximum(W.sum(1), 1e-8))
    Wn = W / d[:, None] / d[None, :]
    F = np.eye(n, dtype=np.float32)
    Y = F.copy()
    for _ in range(iters):
        F = alpha * (Wn @ F) + (1 - alpha) * Y
    return F


def pca(V, k=256, whiten=False):
    """Reduce (and optionally whiten) on the corpus's own covariance.
    The stored vector is 122880-d over 2704 samples, so the covariance
    is estimated in the SAMPLE space - a full ZCA there is both
    intractable and meaningless. Fit on the store itself: no labels,
    a corpus property like the hub statistics already shipped.
    Whitening equalizes the dominant directions, which is the point -
    those directions ARE the nuisance that drowns the primitive."""
    from sklearn.decomposition import PCA
    X = V.reshape(-1, V.shape[-1])
    p = PCA(n_components=k, whiten=whiten, svd_solver="randomized",
            random_state=0).fit(X)
    Z = p.transform(X).astype(np.float32)
    Z /= np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-8)
    return Z.reshape(V.shape[0], 2, k)


def xview(V, eps_, tspan, hard=False, dim=128, epochs=400, seed=0):
    """LABEL-FREE metric: a linear projection trained so that the two
    CAMERA VIEWS of one moment agree and different moments do not.
    The only supervision is 'these two clips are the same instant',
    which the recording itself provides."""
    import torch
    torch.manual_seed(seed)
    A = torch.tensor(V[:, 0])
    B = torch.tensor(V[:, 1])
    n, d = A.shape
    W = torch.nn.Linear(d, dim, bias=False)
    torch.nn.init.orthogonal_(W.weight)
    opt = torch.optim.Adam(W.parameters(), lr=3e-3, weight_decay=1e-4)
    tau = 0.07
    # temporal-neighbour mask: same episode & overlapping-ish spans
    same_ep = torch.tensor(eps_[:, None] == eps_[None, :])
    for ep_i in range(epochs):
        idx = torch.randperm(n)[:512]
        a = torch.nn.functional.normalize(W(A[idx]), dim=-1)
        b = torch.nn.functional.normalize(W(B[idx]), dim=-1)
        logits = a @ b.T / tau
        if hard:
            # windows from the SAME episode are different events -
            # push them apart harder by keeping them as negatives
            # with a margin instead of letting the batch dilute them
            m = same_ep[idx][:, idx].clone()
            m.fill_diagonal_(False)
            logits = logits + m.float() * 1.0
        tgt = torch.arange(len(idx))
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt)
                      + torch.nn.functional.cross_entropy(logits.T, tgt))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = torch.nn.functional.normalize(
            W(torch.tensor(V.reshape(-1, d))), dim=-1)
    return P.numpy().reshape(V.shape[0], 2, dim)


def main():
    V, prims, eps_, tspan = load()
    print(f"{len(V)} events, dim {V.shape[-1]}, "
          f"{len(set(eps_))} episodes\n")

    S0 = viewmax(V)
    base = yld(S0, prims, eps_, "raw cosine (shipped)")
    yld(csls(S0), prims, eps_, "+ CSLS")
    yld(diffuse(csls(S0)), prims, eps_, "+ CSLS + diffusion")

    Vp = pca(V, 256, whiten=False)
    yld(viewmax(Vp), prims, eps_, "PCA-256")
    Vw = pca(V, 256, whiten=True)
    Sw = viewmax(Vw)
    yld(Sw, prims, eps_, "PCA-256 whitened")
    yld(csls(Sw), prims, eps_, "whitened + CSLS")
    yld(diffuse(csls(Sw)), prims, eps_, "whitened + CSLS + diff")

    for hard in (False, True):
        Vx = xview(Vp, eps_, tspan, hard=hard)
        Sx = viewmax(Vx)
        tag = "xview+hard" if hard else "xview"
        yld(Sx, prims, eps_, f"{tag} (label-free)")
        yld(csls(Sx), prims, eps_, f"{tag} + CSLS")
        yld(diffuse(csls(Sx)), prims, eps_, f"{tag} + CSLS + diff")

    print()
    selftrain(V, prims, eps_)
    print(f"\nbaseline to beat: {base:.3f}   "
          f"supervised readout of the same vector: 0.859")




def selftrain(V, prims, eps_, rounds=3, topk=6, dim=128,
              epochs=600, seed=0):
    """The measurement above says WHY xview barely helps: the two
    camera views of one moment share the very nuisance that dominates
    (which block, where on the table, which arm pose) - they differ
    only in viewpoint. So cross-view contrast cannot teach the metric
    to ignore what actually drowns the primitive.

    A positive pair that shares the PRIMITIVE while differing in block
    and position is what is needed, and no label-free signal marks one
    directly. But the CORPUS GRAPH is already better than chance
    (diffusion 0.467), so its neighbours can serve as pseudo-positives
    and the metric can be bootstrapped from them - self-training,
    entirely label-free, using only the store's own geometry.

    Reported BOTH ways, because transduction flatters: fitted on the
    whole store (which is legitimate - the store IS the corpus) and
    fitted on 70% of episodes then measured on the held-out 30%."""
    import torch
    from sklearn.decomposition import PCA

    X = V.reshape(-1, V.shape[-1])
    P = PCA(n_components=256, svd_solver="randomized",
            random_state=0).fit(X)
    Z = P.transform(X).astype(np.float32)
    Z /= np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-8)
    Z = Z.reshape(V.shape[0], 2, 256)

    ueps = np.array(sorted(set(eps_)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ueps)
    tr_ep = set(ueps[:int(0.7 * len(ueps))])
    tr = np.array([e in tr_ep for e in eps_])

    cur = Z
    for rd in range(rounds):
        S = diffuse(csls(viewmax(cur)))
        np.fill_diagonal(S, -9e9)
        # pseudo-positives: graph neighbours in OTHER episodes
        pos = []
        for i in range(len(S)):
            cand = np.where(eps_ != eps_[i])[0]
            best = cand[np.argsort(-S[i][cand])[:topk]]
            pos.append(best)
        pos = np.array(pos)

        torch.manual_seed(seed)
        A = torch.tensor(cur[:, 0])
        B = torch.tensor(cur[:, 1])
        W = torch.nn.Linear(256, dim, bias=False)
        torch.nn.init.orthogonal_(W.weight)
        opt = torch.optim.Adam(W.parameters(), lr=3e-3,
                               weight_decay=1e-4)
        tr_idx = np.where(tr)[0]
        for _ in range(epochs):
            sel = rng.choice(tr_idx, min(384, len(tr_idx)),
                             replace=False)
            pk = pos[sel][:, rng.integers(0, topk)]
            a = torch.nn.functional.normalize(W(A[sel]), dim=-1)
            b = torch.nn.functional.normalize(W(B[pk]), dim=-1)
            logits = a @ b.T / 0.07
            tgt = torch.arange(len(sel))
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            nxt = torch.nn.functional.normalize(
                W(torch.tensor(cur.reshape(-1, cur.shape[-1]))),
                dim=-1).numpy().reshape(len(cur), 2, dim)
        Sn = diffuse(csls(viewmax(nxt)))
        yld(Sn, prims, eps_, f"selftrain r{rd + 1} (whole store)")
        ho = ~tr
        yld(Sn[np.ix_(ho, ho)], prims[ho], eps_[ho],
            f"selftrain r{rd + 1} [HELD-OUT eps]")
        cur = nxt
    return cur


if __name__ == "__main__":
    main()
