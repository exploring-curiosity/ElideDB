"""Experience as a RECURRENCE over expectation and outcome.

Spec: docs/superpowers/specs/2026-08-14-experience-recurrence-design.md

Everything before this was history-free: compute a feature per window, stack,
let DTW align. A history-free descriptor cannot condition on what preceded, so
"door halfway, moving" reads identically whether the door started closed or
open - which is exactly the open/close confusion that owns 61-72% of drawer
errors.

Per step, both relative to act(t) so the scene cancels:

    a_t = pred(t+1) - act(t)     expected change   IDENTITY
    b_t = act(t+1) - act(t)      realised change   CONFIRMATION

pred(t) is deliberately absent - including it would let f form the barred
error term pred(t) - act(t).

    g_t = max(0, <a_t, b_t>)                    confirmation gate
    n_t = a_t - <a_t, m_{t-1}> m_{t-1}          novelty vs history
    m_t = normalize(L*m_{t-1} + (1-L)*g_t*a_t)  motif
    z_t = [m_t ; n_t ; g_t]

Roles follow the audit: gating is worth +0.12 over no gate while the gate is at
ceiling (a perfect object mask buys only +0.021), so reality GATES and
expectation CARRIES.

LAMBDA BY PRINCIPLE, not by sweep. L = 1 - 1/K with K the predictor's horizon
(WIN=4), so L = 0.75. The state's memory matches the timescale the model can
actually forecast over. The layer-6 choice was made by peeking at the graded
score and that is on record; this one is not.

    python -m relmo.vjchain --dataset rcasa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjeval import group_key, l2, parse  # noqa: E402
from relmo.vjeval5 import dtw_from_cost  # noqa: E402
from relmo.vjrec4 import OUT4, WIN  # noqa: E402

LAM = 1.0 - 1.0 / WIN            # 0.75 - see module docstring


def chain(a, b, lam=LAM):
    """(T,D) expected and realised change -> motif, novelty, gate."""
    A, B = l2(a), l2(b)
    T, D = A.shape
    m = np.zeros(D)
    M = np.zeros((T, D), np.float32)
    N = np.zeros((T, D), np.float32)
    G = np.zeros(T, np.float32)
    for t in range(T):
        g = max(0.0, float(A[t] @ B[t]))
        # novelty is measured against the state BEFORE this step is folded in
        n = A[t] - (A[t] @ m) * m
        m = lam * m + (1.0 - lam) * g * A[t]
        nn = np.linalg.norm(m)
        if nn > 1e-9:
            m = m / nn
        M[t], N[t], G[t] = m, n, g
    return M, N, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--lam", type=float, default=LAM)
    a_ = ap.parse_args()

    d4 = OUT4 / f"{a_.dataset}_L{a_.layer}"
    files = sorted(d4.glob("*.npz"))
    meta = [parse(p.stem) for p in files]
    G_ = np.array([group_key(m) for m in meta])
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    print(f"{len(files)} episodes | lambda {a_.lam:.3f} "
          f"(= 1 - 1/{WIN}, from the predictor horizon)")

    base, MM, NN, GG = [], [], [], []
    for p in files:
        z = np.load(p)
        a, b = z["pred_change"], z["obs_change"]
        M, N, g = chain(a, b, a_.lam)
        base.append(a)
        MM.append(M)
        NN.append(N)
        GG.append(g)
    BASE, M_, N_ = l2(np.stack(base)), l2(np.stack(MM)), l2(np.stack(NN))
    Gv = np.stack(GG)
    print(f"gate g: mean {Gv.mean():.3f}  sd {Gv.std():.3f}  "
          f"frac zero {(Gv <= 0).mean():.3f}")

    arms = {
        "pred_change (baseline)": [(BASE, 1.0)],
        "m  motif only": [(M_, 1.0)],
        "n  novelty only": [(N_, 1.0)],
        "m + n": [(M_, 1.0), (N_, 1.0)],
        "m + n + baseline": [(M_, 1.0), (N_, 1.0), (BASE, 1.0)],
    }
    tot = {k: [0, 0] for k in list(arms) + ["random"]}
    fam = {k: {} for k in arms}
    for i in range(len(files)):
        full = np.where(epid != epid[i])[0]
        sv = G_[full] == G_[i]
        sup = int(sv.sum())
        if sup < 5:
            continue
        for k, blocks in arms.items():
            C = None
            for X, w in blocks:
                c = 1.0 - np.einsum("sd,nkd->nsk", X[i], X[full])
                C = w * c if C is None else C + w * c
            s = -dtw_from_cost(C / len(blocks))
            c_ = int(sv[np.argsort(-s)[:sup]].sum())
            tot[k][0] += c_
            tot[k][1] += sup
            f = fam[k].setdefault(meta[i]["task"], [0, 0])
            f[0] += c_
            f[1] += sup
        tot["random"][0] += sup * sup / len(full)
        tot["random"][1] += sup
    r = tot["random"][0] / tot["random"][1]
    print(f"\nk=support | group-aware | chance {r:.3f}")
    print(f"{'arm':24s} {'true/returned':>15s} {'prec':>7s} {'lift':>7s}")
    print("-" * 58)
    for k in ["random"] + list(arms):
        c_, n_ = tot[k]
        print(f"{k:24s} {f'{c_:.0f}/{n_}':>15s} {c_/n_:7.3f} {(c_/n_)/r:6.2f}x")

    b_ = "pred_change (baseline)"
    best = max(arms, key=lambda k: tot[k][0] / tot[k][1])
    if best != b_:
        print(f"\nper-family: {best} vs baseline")
        print(f"{'task family':26s} {'best':>8s} {'base':>8s} {'delta':>8s}")
        for t_ in sorted(fam[best],
                         key=lambda x: -fam[best][x][0] / fam[best][x][1]):
            p1 = fam[best][t_][0] / fam[best][t_][1]
            p0 = fam[b_][t_][0] / fam[b_][t_][1]
            print(f"{t_:26s} {p1:8.3f} {p0:8.3f} {p1-p0:+8.3f}")
    R.log("vjchain", dataset=a_.dataset, lam=a_.lam, episodes=len(files),
          chance=round(r, 4),
          **{k.split()[0].replace("+", "_"): round(tot[k][0]/tot[k][1], 4)
             for k in arms})


if __name__ == "__main__":
    main()
