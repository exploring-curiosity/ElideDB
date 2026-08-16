"""Yield/prec on the ruler for every Stage A variant. Leave-episode-
out; k=support (this session's convention) and the standing
yield@1.5sup/prec operating point. No labels touch any score."""
import sys
sys.path.insert(0, 'native')
import numpy as np, torch
from pathlib import Path
from collections import defaultdict

Z = np.load('data/cache/stagea_prim_v2.npz')
prim, ep, arm = Z['prim'], Z['ep'], Z['arm']
N = len(prim)
PRIMS = ("pick", "place", "stack", "unstack", "push")


def l2(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True),
                          1e-6)


def csls_diffuse(S, topk=12, alpha=0.55, iters=8):
    r = np.sort(S, 1)[:, ::-1][:, 1:topk + 1].mean(1)
    C = 2 * S - r[None, :] - r[:, None]
    W = np.exp((C - C.max()) / 0.35)
    np.fill_diagonal(W, 0)
    thr = np.sort(W, 1)[:, ::-1][:, topk:topk + 1]
    W[W < thr] = 0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    F = S.copy()
    for _ in range(iters):
        F = (1 - alpha) * S + alpha * (W @ F)
    return F


def bench(S, name):
    y1, y15, p15 = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(N):
        mask = ep != ep[i]
        sup = int((prim[mask] == prim[i]).sum())
        if not sup:
            continue
        order = np.argsort(-S[i][mask])
        lab = (prim[mask] == prim[i])
        y1[prim[i]].append(lab[order[:sup]].sum() / sup)
        k15 = min(int(1.5 * sup), mask.sum())
        t = lab[order[:k15]].sum()
        y15[prim[i]].append(t / sup)
        p15[prim[i]].append(t / k15)
    a1 = [v for p in y1 for v in y1[p]]
    a15 = [v for p in y15 for v in y15[p]]
    ap15 = [v for p in p15 for v in p15[p]]
    per = " ".join(f"{p[:2]}{np.mean(y1[p]):.2f}" for p in PRIMS)
    print(f"{name:34s} y@sup {np.mean(a1):.3f}  "
          f"y@1.5 {np.mean(a15):.3f}  p@1.5 {np.mean(ap15):.3f}  "
          f"[{per}]")
    return float(np.mean(a1))


P = l2(Z['pooled'])
Sp = P @ P.T
bench(Sp, 'A pooled cosine')
bench(csls_diffuse(Sp), 'A pooled + L8')

Fl = Z['flow'].astype(np.float32)
Fl = (Fl - Fl.mean(0)) / np.maximum(Fl.std(0), 1e-6)
Sf = -np.abs(Fl[:, None, :] - Fl[None, :, :]).mean(-1)
bench(Sf, 'flow channel alone')

def rankz(S):
    o = np.argsort(np.argsort(-S, 1), 1).astype(np.float32)
    return -o / N
Sfu = rankz(Sp) + rankz(Sf)
bench(Sfu, 'pooled + flow (rank fusion)')
bench(csls_diffuse((Sfu - Sfu.mean()) / Sfu.std()), 'pooled+flow+L8')

# motion-gated late interaction: top-32 moving cells per event,
# PCA-128, symmetric mean-MaxSim
spat, delta = Z['spat'].astype(np.float32), Z['delta'].astype(np.float32)
gidx = np.argsort(-delta, 1)[:, :32]
toks = np.stack([spat[i, gidx[i]] for i in range(N)])
X = toks.reshape(-1, 1024)
X = X - X.mean(0)
U, s, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(
    len(X), 4000, replace=False)], full_matrices=False)
Tk = l2((toks - toks.mean((0, 1))) @ Vt[:128].T)
T = torch.tensor(Tk, device='mps')
Sl = np.zeros((N, N), np.float32)
B = 64
for i0 in range(0, N, B):
    q = T[i0:i0 + B]
    sims = torch.einsum('qtd,ncd->qtnc', q, T)
    m1 = sims.max(-1).values.mean(1)
    m2 = sims.max(1).values.mean(-1)
    Sl[i0:i0 + B] = (0.5 * (m1 + m2)).cpu().numpy()
bench(Sl, 'motion-gated late interaction')
bench(csls_diffuse(Sl), 'late interaction + L8')
S3 = rankz(Sp) + rankz(Sf) + rankz(Sl)
bench(S3, 'ALL THREE (rank fusion)')
bench(csls_diffuse((S3 - S3.mean()) / S3.std()), 'all three + L8')
