"""All four datasets, ONE protocol. Car and drone had never been evaluated.

Audit finding: sim and bridge had real QbE numbers; KITTI and AGZ had
only a representation-coherence ratio, which is not comparable to
anything. This closes that gap.

The problem with car/drone is that they have no repeated labelled
events to retrieve - bridge has 484 episodes of "open the drawer", AGZ
is one continuous flight. A retrieval task has to be CONSTRUCTED, and
any construction is a place to smuggle in domain knowledge. So the
construction is fixed and identical for both:

  1. Cut the media into uniform overlapping windows (no segmentation
     opinion - a fixed grid already beat learned segmentation).
  2. Describe each window by its own SENSOR truth: speed, |turn rate|,
     climb rate, z-scored per domain.
  3. Define classes by k-means on that state with k fixed in advance.
     I do not choose which classes exist or which are interesting.
  4. Run the identical QbE protocol used on sim and bridge: 5 seeds,
     k = ceil(1.5 x support) as a MAX bound, seed-calibrated abstention.

The classes come from motion sensors, the encoder sees only pixels, and
nothing about either domain is hand-written. The same siglip2 + rank
pooling encoder is used everywhere so the four numbers are comparable.

    python native/alldom.py --domains agz,kitti --k 8 --cap 300
"""
from __future__ import annotations

import collections
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402
from crossdom import units_for                                 # noqa: E402
from style import kinematic_state, uniform                     # noqa: E402
from unitenc import _norm                                      # noqa: E402

SEPARATION = 60.0        # s; a neighbour this close is trivially similar


def qbe(vecs, lab, media, times, seed_n=5, sep=SEPARATION):
    """The protocol used on sim and bridge, unchanged.

    One addition, required here and only here: candidates within `sep`
    seconds of a seed in the same media are EXCLUDED. AGZ is a single
    continuous flight, so without this a query would retrieve its own
    temporal neighbours and score near 1.0 while learning nothing.
    """
    groups = collections.defaultdict(list)
    for i, l in enumerate(lab):
        groups[l].append(i)
    rs = np.random.RandomState(0)
    ys, ps, sups, rets, per = [], [], [], [], {}
    for cl, pool in sorted(groups.items()):
        if len(pool) < seed_n + 3:
            continue
        sd = sorted(rs.choice(pool, seed_n, replace=False).tolist())

        def far(i):
            return all(media[i] != media[s] or
                       abs(times[i] - times[s]) > sep for s in sd)
        cand = [i for i in range(len(vecs)) if i not in sd and far(i)]
        if len(cand) < 4:
            continue
        support = sum(1 for i in cand if lab[i] == cl)
        if support < 2:
            continue
        k = math.ceil(1.5 * support)
        S = {i: max(float(vecs[i] @ vecs[s]) for s in sd) for i in cand}
        loo = [max(float(vecs[a] @ vecs[b]) for b in sd if b != a)
               for a in sd]
        cut = min(loo) if loo else -np.inf
        got = [i for i in sorted(S, key=lambda x: -S[x])[:k]
               if S[i] >= cut]
        tr = sum(1 for i in got if lab[i] == cl)
        ys.append(tr / support)
        ps.append(tr / len(got) if got else 0.0)
        sups.append(support)
        rets.append(len(got))
        per[cl] = (tr / support, tr / len(got) if got else 0.0, support)
    return ys, ps, sups, rets, per


def build(name, cap, enc):
    """-> vectors, class labels, media ids, start times."""
    import domains as D
    from tqdm import tqdm
    if name == "agz":
        d = D.agz()
        sp = uniform(d["t"][-1] - d["t"][0])[:cap]
        V, kept = units_for(sp, lambda i, d=d: d["frame_path"](
            max(int(i) + 1, 1)), d["fps"], enc)
        from motionrep import agz_state
        K = agz_state(kept)
        med = ["agz"] * len(kept)
    else:
        Vs, ks, med, Ks = [], [], [], []
        per_drive = max(cap // max(len(D.kitti_drives()), 1), 4)
        for dr in tqdm(D.kitti_drives(), desc="kitti", unit="drive",
                       leave=False):
            dd = D.kitti(dr)
            t = dd["t"]
            if len(t) < 20:
                continue
            sp = uniform(t[-1] - t[0], 2.0, 0.5)[:per_drive]
            v, kp = units_for(sp, dd["frame_path"], dd["fps"], enc)
            if not len(v):
                continue
            ox = sorted((dr / "oxts/data").glob("*.txt"))
            A = np.array([list(map(float, f.read_text().split()))
                          for f in ox])[:len(t)]
            dt = np.gradient(t) + 1e-9
            turn = np.gradient(np.unwrap(A[:, 5])) / dt
            speed = np.hypot(A[:, 6], A[:, 7])
            climb = np.gradient(A[:, 2]) / dt
            Vs.append(v)
            ks += kp
            med += [dr.name] * len(kp)
            Ks.append(kinematic_state(t, speed, turn, climb, kp))
        if not Vs:
            return None
        V, kept, K = _norm(np.concatenate(Vs)), ks, np.concatenate(Ks)
    return V, K, med, [s[0] for s in kept]


def main():
    from sklearn.cluster import KMeans
    want = arg("--domains", "agz,kitti").split(",")
    cap = arg("--cap", 300, int)
    K_CLS = arg("--k", 8, int)
    enc = arg("--enc", "siglip2_rank")

    print(f"ALL-DOMAIN QbE — encoder {enc}, {K_CLS} sensor-derived "
          f"classes, same protocol as sim/bridge\n")
    print(f"{'domain':<10}{'units':<8}{'classes':<9}{'yield':<9}"
          f"{'prec':<9}{'support':<9}{'returned'}")
    for name in want:
        got = build(name, cap, enc)
        if got is None:
            print(f"{name:<10}no units")
            continue
        V, K, med, times = got
        if len(V) < 40:
            print(f"{name:<10}too few units ({len(V)})")
            continue
        lab = KMeans(K_CLS, n_init=10,
                     random_state=0).fit_predict(K)
        ys, ps, sups, rets, per = qbe(V, lab, med, times)
        if not ys:
            print(f"{name:<10}{len(V):<8}no class had enough support")
            continue
        print(f"{name:<10}{len(V):<8}{len(per):<9}{np.mean(ys):<9.3f}"
              f"{np.mean(ps):<9.3f}{np.mean(sups):<9.1f}"
              f"{np.mean(rets):.1f}", flush=True)
    print("\nreference, same protocol and encoder:")
    print("  bridge (real robot, top-10 tasks)  0.734 / 0.584")
    print("  sim    (synthetic arm)             0.492 / 0.328")


if __name__ == "__main__":
    main()
