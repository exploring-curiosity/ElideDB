"""Relative-part-motion descriptor, and the test that justifies it.

THE QUESTION, in the owner's words: how does "close a LEFT-facing drawer"
differ from "open a RIGHT-facing drawer"? In the world frame it does not.
The handle sweeps the same direction, the point cloud moves the same way,
the predicted trajectory is identical. Any descriptor built from the
displacement field scores them as the same moment. A trajectory predictor
- which is all RelMo-WM v3 is - cannot separate them at any R2.

What separates them is RELATIVE motion between parts: the drawer face
moves AWAY from the cabinet when opening and TOWARD it when closing.
Separation increasing vs decreasing is orientation-free; which way the
cabinet faces cancels out. Measured support from this project: a joint
axis has std 0.0000 in the object's own frame against 0.4332 in camera
and 0.4154 in world - in world coordinates the axis wanders and its sign
means nothing.

NOTHING HERE IS A LABEL. The descriptor is a short vector of geometric
quantities computed from tracks alone: no names, no classes, no text, no
sim state. Retrieval is nearest-neighbour on that vector. Sim `qpos`
appears in ONE place - deciding, after the fact, whether a retrieved
event was an opening or a closing so a number can be printed. Labels
score; they never serve.

Why separation and not the screw axis sign: a screw axis is defined only
up to sign, so "positive travel along the axis" is not a well-defined
quantity across episodes. The distance between two parts is.

    python -m relmo.relmotion --n 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.train_wm2 import lift3d  # noqa: E402

MIN_TRAVEL = 0.08          # rad or m of joint travel to count as an event
WIN = 24                   # frames per event window


# ----------------------------------------------------------------- events
def events(ep_dir: Path, tracks: Path):
    """Windows in which ONE articulated joint does most of the moving.

    Sim state is read here only to LOCATE events and to record their sign
    for scoring. Neither the location nor the sign enters the descriptor."""
    st = np.load(ep_dir / "state.npz", allow_pickle=True)
    if "qpos" not in st.files:
        return []
    qp, jt, ad = st["qpos"], st["jnt_type"], st["jnt_qposadr"]
    T = qp.shape[0]
    out = []
    for j, (t_, a) in enumerate(zip(jt, ad)):
        if int(t_) not in (2, 3) or a >= qp.shape[1]:
            continue
        q = qp[:, a]
        d = np.diff(q)
        if len(d) < WIN:
            continue
        # the WIN-frame window with the largest net travel
        cs = np.concatenate([[0.0], np.cumsum(d)])
        net = cs[WIN:] - cs[:-WIN]
        i = int(np.argmax(np.abs(net)))
        if abs(net[i]) < MIN_TRAVEL:
            continue
        out.append(dict(t0=i, t1=i + WIN, sign=1 if net[i] > 0 else -1,
                        travel=float(net[i]), jtype=int(t_)))
    # keep the single most active joint per episode: overlapping windows
    # from a cabinet whose four doors all jiggle are not distinct events
    return sorted(out, key=lambda e: -abs(e["travel"]))[:1]


# ------------------------------------------------------- label-free parts
def rigid_groups(X, V, k=6):
    """Group points that move as one body. No labels, no trained model.

    Two points on one rigid link keep a CONSTANT distance however the
    link moves. So the affinity is the stability of pairwise distance,
    and the clustering is spectral on that affinity. This is the classic
    common-fate/rigidity cue and it is computable on any video."""
    ok = V.all(0)
    idx = np.where(ok)[0]
    if len(idx) < 12:
        return None, None
    Y = X[:, idx]                                    # (T,N,3)
    D = np.linalg.norm(Y[:, :, None] - Y[:, None, :], axis=-1)   # (T,N,N)
    sd = D.std(0)                                    # distance instability
    sc = max(np.median(D.mean(0)), 1e-6)
    A = np.exp(-(sd / (0.05 * sc)) ** 2)             # 1 = rigidly attached
    d = A.sum(1)
    L = np.eye(len(A)) - A / np.sqrt(np.outer(d, d)).clip(1e-9)
    w, v = np.linalg.eigh(L)
    F = v[:, :k]
    F = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-9)
    # k-means on the spectral embedding, fixed seed, no sklearn
    rng = np.random.default_rng(0)
    C = F[rng.choice(len(F), k, replace=False)]
    for _ in range(25):
        lab = np.argmin(((F[:, None] - C[None]) ** 2).sum(-1), 1)
        for c in range(k):
            if (lab == c).any():
                C[c] = F[lab == c].mean(0)
    return idx, lab


# ------------------------------------------------------------ descriptors
def descriptor_traj(X, V):
    """BASELINE: what a trajectory/flow predictor gives you.

    Pooled world-frame displacement statistics. This is the information
    RelMo-WM v3's per-point `disp` output contains, and the thing that
    provably cannot tell the two drawer cases apart."""
    ok = V.all(0)
    if ok.sum() < 8:
        return None
    Y = X[:, ok]
    dv = Y[-1] - Y[0]
    s = max(np.linalg.norm(Y[0] - Y[0].mean(0), axis=-1).mean(), 1e-6)
    dv = dv / s
    mn = dv.mean(0)
    return np.concatenate([mn, [np.linalg.norm(mn),
                                np.linalg.norm(dv, axis=-1).mean(),
                                np.linalg.norm(dv, axis=-1).std()]])


def _fit(P0, P1):
    """Rigid transform P0 -> P1 (Kabsch)."""
    c0, c1 = P0.mean(0), P1.mean(0)
    H = (P0 - c0).T @ (P1 - c1)
    U, _, Vt = np.linalg.svd(H)
    dt = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, dt]) @ U.T, c1 - c0


def descriptor_rel(X, V, k=6):
    """RELATIVE-PART-MOTION. Orientation-free by construction.

    For the two groups with the most relative motion:
      sep   signed change in inter-group separation - the open/close sign
      rot   |relative rotation| between the groups over the window
      tra   translation of B's centroid in A's frame
      rt    rotation-vs-translation character (hinge-like vs slide-like)
    All normalised by the episode's own spatial scale, so a big cabinet
    and a small drawer land in the same place."""
    idx, lab = rigid_groups(X, V, k)
    if idx is None:
        return None
    Y = X[:, idx]
    s = max(np.linalg.norm(Y[0] - Y[0].mean(0), axis=-1).mean(), 1e-6)
    gs = [np.where(lab == c)[0] for c in range(k)]
    gs = [g for g in gs if len(g) >= 4]
    if len(gs) < 2:
        return None
    # PAIR SELECTION BY AXIS STABILITY, not by "who separates most".
    #
    # Measured why: selecting the largest separation change gave signed
    # AUC 0.3296 - real signal, wrong pair. In a reach-and-pull episode
    # the biggest separation change is the ARM CLOSING ON THE HANDLE, and
    # since opening episodes begin with a reach, it read the reach and
    # inverted the sign. |d_sep| scored 0.7014 on its own, i.e. it was
    # keying on "opens travel further than closes" - a confound.
    #
    # The geometric difference: a hinge or slider moves about ONE FIXED
    # AXIS relative to the static structure it is mounted in, frame after
    # frame. An arm does not - it is a free 6-DoF chain. So the reference
    # is the largest, least-moving group (the scene), and the part is the
    # one whose per-frame relative rotation axis stays put in the
    # reference's own frame. That is WM_PLAN 2b's joint test, and the
    # object frame is where this project measured axis std 0.0000 against
    # 0.4332 in camera.
    mot = np.array([np.linalg.norm(Y[:, g].mean(1) - Y[0, g].mean(0),
                                   axis=-1).mean() for g in gs])
    ref = int(np.argmin(mot / (mot.max() + 1e-9)
                        - 0.5 * np.array([len(g) for g in gs]) / max(
                            sum(len(g) for g in gs), 1)))
    best, bi = -1.0, None
    for j in range(len(gs)):
        if j == ref:
            continue
        A_, B_ = Y[:, gs[ref]], Y[:, gs[j]]
        ax, ok_ = [], True
        for t in range(1, len(Y)):
            Ra_, _ = _fit(A_[t - 1], A_[t])
            Rb_, _ = _fit(B_[t - 1], B_[t])
            Rr = Ra_.T @ Rb_
            w_, v_ = np.linalg.eig(Rr)
            k_ = np.real(v_[:, np.argmin(np.abs(w_ - 1.0))])
            n_ = np.linalg.norm(k_)
            if n_ < 1e-8:
                ok_ = False
                break
            k_ = k_ / n_
            if ax and float(k_ @ ax[-1]) < 0:      # axis sign is arbitrary
                k_ = -k_
            ax.append(k_)
        if not ok_ or len(ax) < 4:
            continue
        ax = np.stack(ax)
        stab = float(np.linalg.norm(ax.mean(0)))    # 1 = perfectly fixed axis
        trav = float(np.abs(np.linalg.norm(B_.mean(1) - A_.mean(1), axis=-1)[-1]
                            - np.linalg.norm(B_.mean(1) - A_.mean(1), axis=-1)[0]))
        sc_ = stab * trav
        if sc_ > best:
            best, bi = sc_, (ref, j)
    if bi is None:
        return None
    A, B = Y[:, gs[bi[0]]], Y[:, gs[bi[1]]]
    ca, cb = A.mean(1), B.mean(1)
    sep = np.linalg.norm(cb - ca, axis=-1)
    d_sep = float((sep[-1] - sep[0]) / s)            # SIGNED. the whole point.
    # relative rotation of B in A's frame, first frame -> last
    Ra, _ = _fit(A[0], A[-1])
    Rb, _ = _fit(B[0], B[-1])
    Rrel = Ra.T @ Rb
    ang = float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
    tra = float(np.linalg.norm(Ra.T @ (cb[-1] - ca[-1]) - (cb[0] - ca[0])) / s)
    rt = float(ang / (ang + tra + 1e-6))
    return np.array([d_sep, abs(d_sep), ang, tra, rt,
                     float(np.sign(d_sep))], np.float32)


# ------------------------------------------------------------------ test
def build(dataset="rcasa_v1", n=150):
    files = sorted((R.TRACKS / dataset).glob("*.npz"))[:n]
    rows = []
    for f in files:
        z = np.load(f)
        if "gdist" not in z.files:
            continue
        ep = f.stem
        cand = list((R.dataset_dir(dataset)).glob(f"shard_*/{ep}"))
        if not cand:
            continue
        evs = events(cand[0], f)
        if not evs:
            continue
        X, V = lift3d(z), z["gvis"]
        for e in evs:
            t0, t1 = e["t0"], min(e["t1"], len(X))
            if t1 - t0 < WIN:
                continue
            xs, vs = X[t0:t1], V[t0:t1]
            dt_, dr_ = descriptor_traj(xs, vs), descriptor_rel(xs, vs)
            if dt_ is None or dr_ is None:
                continue
            rows.append(dict(ep=ep, sign=e["sign"], travel=e["travel"],
                             traj=dt_, rel=dr_))
    return rows


def retrieval(rows, key, k=10):
    """Leave-one-out query-by-example. Every clip is the query once; we
    ask what fraction of its k nearest neighbours share its sign.
    Chance = the majority-class rate, which is reported beside it."""
    M = np.stack([r[key] for r in rows])
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    S = np.asarray([r["sign"] for r in rows])
    D = ((M[:, None] - M[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    hits = []
    for i in range(len(rows)):
        nn = np.argsort(D[i])[:k]
        hits.append(float((S[nn] == S[i]).mean()))
    return float(np.mean(hits))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa_v1")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    rows = build(a.dataset, a.n)
    # BALANCE THE CLASSES. Unbalanced, this metric measures the class
    # prior: the first run returned 130 opening / 19 closing, majority
    # chance 0.8725, and BOTH descriptors scored under it (0.789 / 0.808)
    # while telling us nothing. Balanced, chance is exactly 0.5.
    rng_ = np.random.default_rng(0)
    pos = [r for r in rows if r["sign"] > 0]
    neg = [r for r in rows if r["sign"] < 0]
    m_ = min(len(pos), len(neg))
    if m_ >= 10:
        pos = [pos[i] for i in rng_.choice(len(pos), m_, replace=False)]
        neg = [neg[i] for i in rng_.choice(len(neg), m_, replace=False)]
        rows = pos + neg
    if len(rows) < 20:
        print(json.dumps(dict(error="too few events", n=len(rows))))
        raise SystemExit(1)
    S = np.array([r["sign"] for r in rows])
    maj = float(max((S > 0).mean(), (S < 0).mean()))
    rep = dict(dataset=a.dataset, events=len(rows), k=a.k,
               opening=int((S > 0).sum()), closing=int((S < 0).sum()),
               chance_majority=round(maj, 4),
               precision_trajectory=round(retrieval(rows, "traj", a.k), 4),
               precision_relative=round(retrieval(rows, "rel", a.k), 4))
    R.log("artic_qbe", **rep)
    print(json.dumps(rep, indent=1))
