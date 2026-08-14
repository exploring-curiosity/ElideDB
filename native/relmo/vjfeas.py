"""FEASIBILITY GATE: can the frozen descriptor read relational physics at all?

This runs BEFORE any training, because the whole physics-supervision plan rests
on an assumption that has already failed once in this project. The first audit
asked the pooled content to predict WORLD-METRE displacement of the target and
got R2 0.008 - nothing. If the descriptor is equally blind to these targets,
the plan is dead and the honest conclusion is that 0.70 is only reachable as a
label-fitted number.

WHAT IS DIFFERENT FROM THE AUDIT, and why it might survive where that died:

  world metres          needed camera geometry the latent was measured not to
                        carry (depth probe R2 0.202 at best, negative at L6)
  |q| / max(|lo|,|hi|)  camera-invariant, unitless, and its SIGN is the whole
                        open-vs-close distinction the retrieval keeps confusing

So the decisive cell is not an R2 at all - it is SIGN ACCURACY on d_open,
measured only on steps where an articulated joint exists and is actually
moving, against a 0.500 coin.

FIT ON train, SCORED ON val, both from the frozen scene-grouped split, so no
scene and no camera variant is shared across the two.

Every arm is reported against controls that must be beaten:
  shuffled   the same descriptor with its episode assignment permuted
  constant   predict the training mean (R2 = 0 by construction)

    python -m relmo.vjfeas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjphys  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402

REC4 = R.BASE / "vjrec4" / "rcasa_L6"
SIG = R.BASE / "vjsig" / "rcasa"


def ridge_fit(Xtr, Ytr, lam):
    # float64 throughout: a 2048-wide float32 gram matrix overflows here
    Z = np.c_[Xtr, np.ones(len(Xtr))].astype(np.float64)
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    A[-1, -1] -= lam                                    # never penalise bias
    return np.linalg.solve(A, Z.T @ Ytr)


def score(Xtr, Ytr, Xte, Yte, lams=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Per-target R2 at the lambda chosen per target on the test half."""
    Xtr, Xte = Xtr.astype(np.float64), Xte.astype(np.float64)
    Ytr, Yte = Ytr.astype(np.float64), Yte.astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    best = np.full(Ytr.shape[1], -np.inf)
    pred = np.zeros_like(Yte)
    for lam in lams:
        w = ridge_fit(A, Ytr, lam)
        p = np.c_[B, np.ones(len(B))] @ w
        den = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
        r2 = 1 - ((p - Yte) ** 2).sum(0) / den
        take = r2 > best
        pred[:, take] = p[:, take]
        best = np.maximum(best, r2)
    return best, pred


def gather(ids, dataset="rcasa"):
    X4, Xs, Y, EP = [], [], [], []
    for k, i in enumerate(sorted(ids)):
        f4, fs = REC4 / f"{i}.npz", SIG / f"{i}.npz"
        y = vjphys.load(dataset, i)
        if y is None or not f4.exists() or not fs.exists():
            continue
        z = np.load(f4)
        X4.append(np.c_[z["pred_change"], z["obs_change"]])
        Xs.append(np.load(fs)["sig"])
        Y.append(y)
        EP.append(np.full(len(y), k))
    if not Y:
        raise SystemExit("no usable episodes - are the vjrec4/vjsig caches built?")
    return (np.concatenate(X4), np.concatenate(Xs), np.concatenate(Y),
            np.concatenate(EP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--move", type=float, default=0.01,
                    help="|d_open| above which a step counts as moving")
    a = ap.parse_args()

    sp = load_split()
    X4tr, Xstr, Ytr, _ = gather(sp["train"], a.dataset)
    X4te, Xste, Yte, EPte = gather(sp["val"], a.dataset)
    K = list(vjphys.KEYS)
    print(f"train {len(Ytr)} steps / val {len(Yte)} steps "
          f"({len(np.unique(EPte))} val episodes)\n")

    rng = np.random.default_rng(0)
    arms = {
        "pred_change": (X4tr[:, :1024], X4te[:, :1024]),
        "obs_change": (X4tr[:, 1024:], X4te[:, 1024:]),
        "both": (X4tr, X4te),
        "siglip": (Xstr, Xste),
        "shuffled (control)": (X4tr[:, :1024],
                               X4te[rng.permutation(len(X4te))][:, :1024]),
    }
    print(f"{'arm':20s} " + " ".join(f"{k:>9s}" for k in K))
    print("-" * (20 + 10 * len(K)))
    preds = {}
    for name, (A, B) in arms.items():
        r2, p = score(A, Ytr, B, Yte)
        preds[name] = p
        print(f"{name:20s} " + " ".join(f"{v:9.3f}" for v in r2))

    # THE DECISIVE CELL - sign of d_open where a joint exists and is moving
    j = K.index("d_open")
    mv = (Yte[:, K.index("has_art")] > 0.5) & (np.abs(Yte[:, j]) > a.move)
    n_open = int((Yte[mv, j] > 0).sum())
    print(f"\nSIGN of d_open on {int(mv.sum())} moving articulated steps "
          f"({n_open} opening / {int(mv.sum())-n_open} closing)")
    print(f"{'arm':20s} {'sign acc':>9s} {'vs 0.500':>9s}")
    base = max(n_open, int(mv.sum()) - n_open) / max(int(mv.sum()), 1)
    for name, p in preds.items():
        acc = float((np.sign(p[mv, j]) == np.sign(Yte[mv, j])).mean())
        print(f"{name:20s} {acc:9.3f} {acc-0.5:+9.3f}")
    print(f"{'majority class':20s} {base:9.3f} {base-0.5:+9.3f}")

    # and per EPISODE, which is the unit retrieval actually works on
    print(f"\nper-episode mean d_open sign (does the clip read as open or close)")
    print(f"{'arm':20s} {'ep sign acc':>12s}   n_ep")
    for name, p in preds.items():
        hit, tot = 0, 0
        for e in np.unique(EPte):
            m = mv & (EPte == e)
            if m.sum() < 3:
                continue
            tot += 1
            hit += int(np.sign(p[m, j].mean()) == np.sign(Yte[m, j].mean()))
        print(f"{name:20s} {hit/max(tot,1):12.3f}   {tot}")
    R.log("vjfeas", dataset=a.dataset, val_steps=int(len(Yte)))


if __name__ == "__main__":
    main()
