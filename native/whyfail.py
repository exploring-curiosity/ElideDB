"""WHY IS THE YIELD LOW. Three measurements, no architecture.

The owner's question: L0-L8 was designed and built without ever
checking whether it moves yield on a PRIMITIVE. So this asks the
prior question that should have been asked first - is the primitive
even present in what the system stores?

  1. YIELD AT k = support, against the CHANCE floor. A yield number
     without its chance floor is meaningless: 'pick' is 44% of this
     corpus, so a coin gets 0.44 on it.
  2. IS THE PRIMITIVE IN THE REPRESENTATION? A supervised probe on
     the SHIPPED vector (dmulti over r20), episode-disjoint. This
     does NOT claim to bound the problem - it answers one narrow
     question: can ANY readout of this stored vector name the
     primitive? If not, no scorer, ranker or graph built on top can,
     and every L4-L8 result in this project was predetermined.
  3. WHAT COMES BACK INSTEAD. Confusion over the returned set.

    python native/whyfail.py
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
    """Every true event as the SHIPPED vector, per view."""
    idx = None
    for corp in CORPORA:
        part = vwm_qbe.WMIndex(root=f"data/{corp}", corpus=corp)
        if idx is None:
            idx = part
        else:
            idx.traj.update(part.traj)
            idx.kin.update(part.kin)
    V, prims, eps_ = [], [], []
    for corp in CORPORA:
        for ep in sorted((ROOT / "data" / corp).glob("ep*")):
            mid = f"{corp}/{ep.name}"
            if mid not in idx.traj:
                continue
            meta = json.loads((ep / "meta.json").read_text())
            views = idx.traj[mid]
            T = min(len(r) for r in views)
            for e in meta["events"]:
                if not e["ok"]:
                    continue
                a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
                if b > T or b - a < 4:
                    continue
                vv = [vwm_qbe._l2(idx.dmulti(r, a, b)) for r in views]
                V.append(np.stack(vv[:2] if len(vv) >= 2
                                  else [vv[0], vv[0]]))
                prims.append(e["prim"])
                eps_.append(mid)
    return np.array(V, np.float32), np.array(prims), np.array(eps_)


def main():
    V, prims, eps_ = load()
    n = len(V)
    cnt = Counter(prims)
    print(f"{n} events | " + "  ".join(f"{p} {cnt[p]}" for p in PRIMS))

    # ---------- 1. yield at k = support, vs the chance floor -------
    S = None
    for a in range(2):
        for b in range(2):
            X = V[:, a] @ V[:, b].T
            S = X if S is None else np.maximum(S, X)
    np.fill_diagonal(S, -9e9)

    def yield_at_support(S, tag):
        per = defaultdict(list)
        for i in range(n):
            mask = eps_ != eps_[i]
            lab = prims[mask] == prims[i]
            sup = int(lab.sum())
            if not sup:
                continue
            order = np.argsort(-S[i][mask])
            hit = int(lab[order[:sup]].sum())      # k = support
            per[prims[i]].append((hit / sup, sup, int(mask.sum())))
        print(f"\n--- {tag}: yield at k = support ---")
        print(f"{'prim':9s} {'n':>4s} {'support':>8s} {'CHANCE':>7s} "
              f"{'yield':>7s} {'x chance':>9s}")
        ally, allc = [], []
        for p in PRIMS:
            if not per[p]:
                continue
            y = np.mean([r[0] for r in per[p]])
            ch = np.mean([r[1] / r[2] for r in per[p]])
            ally += [r[0] for r in per[p]]
            allc += [r[1] / r[2] for r in per[p]]
            print(f"{p:9s} {len(per[p]):4d} "
                  f"{np.mean([r[1] for r in per[p]]):8.0f} {ch:7.3f} "
                  f"{y:7.3f} {y / max(ch, 1e-6):9.2f}")
        print(f"{'ALL':9s} {len(ally):4d} {'':8s} {np.mean(allc):7.3f} "
              f"{np.mean(ally):7.3f} "
              f"{np.mean(ally) / max(np.mean(allc), 1e-6):9.2f}")
        return np.mean(ally)

    yield_at_support(S, "SHIPPED vector, cosine")

    # ---------- 2. is the primitive IN the stored vector? ----------
    # episode-disjoint split; a probe that fails here means no
    # downstream scorer could ever have succeeded
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    ueps = np.array(sorted(set(eps_)))
    rng = np.random.default_rng(0)
    rng.shuffle(ueps)
    tr_ep = set(ueps[:int(0.7 * len(ueps))])
    tr = np.array([e in tr_ep for e in eps_])
    X = V[:, 0]                     # one view, the stored vector
    y = np.array([PRIMS.index(p) for p in prims])
    print(f"\n--- is the primitive present in the stored vector? "
          f"(train {tr.sum()} / test {(~tr).sum()}, episode-disjoint) ---")
    for name, clf in (("linear probe",
                       LogisticRegression(max_iter=2000, C=1.0)),
                      ("MLP probe (256)",
                       MLPClassifier((256,), max_iter=600,
                                     random_state=0))):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[~tr])
        acc = float((pred == y[~tr]).mean())
        maj = float(Counter(y[~tr]).most_common(1)[0][1] / (~tr).sum())
        print(f"{name:16s} accuracy {acc:.3f}   "
              f"(majority-class floor {maj:.3f})")
        per = {}
        for pi, p in enumerate(PRIMS):
            m = y[~tr] == pi
            if m.sum():
                per[p] = float((pred[m] == pi).mean())
        print("                 per-prim recall: "
              + "  ".join(f"{p} {per.get(p, 0):.2f}" for p in PRIMS))

    # ---------- 3. what comes back instead ----------
    print("\n--- what the cosine ranking returns at k = support ---")
    conf = defaultdict(Counter)
    for i in range(n):
        mask = eps_ != eps_[i]
        lab = prims[mask] == prims[i]
        sup = int(lab.sum())
        if not sup:
            continue
        idx2 = np.where(mask)[0][np.argsort(-S[i][mask])[:sup]]
        for j in idx2:
            conf[prims[i]][prims[j]] += 1
    print(f"{'query':9s} " + " ".join(f"{p:>8s}" for p in PRIMS))
    for p in PRIMS:
        tot = sum(conf[p].values()) or 1
        print(f"{p:9s} " + " ".join(f"{conf[p][q] / tot:8.3f}"
                                    for q in PRIMS))
    print("\n(rows = query primitive, cols = what was returned, "
          "fraction of the returned set)")


if __name__ == "__main__":
    main()
