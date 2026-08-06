"""Unsupervised STYLE structure - no taxonomy, no names, no text.

Every evaluation before this one asked "does the vector recover a class
I defined?" - prim labels on the arm, and on drone/car a kinematic
vocabulary (still/cruise/turn/climb) that I INVENTED. Scoring an
encoder against invented categories measures the invention. It is the
same mistake as tuning boundary F1 while span F1 collapsed.

QbE needs no names. It needs one property: similar things land near
each other. So this measures the property directly.

TRUTH AS A RULER, NOT A TAXONOMY. Each unit has a vision-independent
kinematic state from GPS/IMU/pose - a continuous vector (speed, turn
rate, climb rate), not a label. The question becomes:

    are a unit's NEAREST NEIGHBOURS in embedding space more
    kinematically similar to it than random units are?

That is exactly what QbE requires, it needs no vocabulary, and the
ruler cannot be gamed by relabelling. Reported as a RATIO against the
random baseline, so 1.0 means "no better than chance" in every domain
regardless of how that domain's motion is distributed.

Two further label-free checks:

  stability   cluster two disjoint temporal halves separately; do they
              agree about which units belong together? A structure that
              changes with the sample is not structure.
  usable k    how many clusters emerge, and are they populated? One
              giant cluster is as useless as one cluster per unit.

Units are UNIFORM OVERLAPPING WINDOWS, following the measured result
that a fixed grid beats learned segmentation (0.408 vs 0.317) - no
segmentation opinion imposed here either.

    python native/style.py --domain agz --enc r50_rank
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402
from crossdom import units_for                                 # noqa: E402
from unitenc import _norm                                      # noqa: E402

SEPARATION = 60.0


def kinematic_state(t, speed, turn, climb, spans):
    """Continuous state per span, z-scored so the three axes are
    comparable and no single unit (m/s vs rad/s) dominates."""
    S = []
    for a, b, _ in spans:
        m = (t >= a) & (t <= b)
        if m.sum() < 2:
            m = np.zeros_like(t, bool)
            m[np.argmin(np.abs(t - a))] = True
        S.append([speed[m].mean(), np.abs(turn[m]).mean(),
                  climb[m].mean()])
    S = np.asarray(S, float)
    mu, sd = S.mean(0), S.std(0) + 1e-8
    return (S - mu) / sd


def neighbour_coherence(V, K, spans, media, k=5):
    """Ratio: kinematic distance to embedding-neighbours vs to random.

    < 1.0 means neighbours in embedding space really are more alike in
    the physical sense. 1.0 means the embedding carries nothing a QbE
    user could exploit. This is the whole test.
    """
    n = len(V)
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    # respect the same separation rule used elsewhere: a neighbour that
    # is 3 s away is trivially similar and would flatter the result
    for i in range(n):
        for j in range(n):
            if media[i] == media[j] and \
                    abs(spans[i][0] - spans[j][0]) <= SEPARATION:
                S[i, j] = -np.inf
    rng = np.random.default_rng(0)
    near, rand = [], []
    for i in range(n):
        cand = np.where(np.isfinite(S[i]))[0]
        if len(cand) < k + 1:
            continue
        nn = cand[np.argsort(-S[i, cand])[:k]]
        near += [np.linalg.norm(K[i] - K[j]) for j in nn]
        rr = rng.choice(cand, size=k, replace=False)
        rand += [np.linalg.norm(K[i] - K[j]) for j in rr]
    if not near:
        return float("nan"), float("nan"), 0
    return (float(np.mean(near)), float(np.mean(rand)), len(near))


def stability(V, spans, k=8):
    """Cluster disjoint temporal halves; do they agree about pairs?"""
    from sklearn.cluster import KMeans
    t = np.array([s[0] for s in spans])
    half = np.median(t)
    a, b = t <= half, t > half
    if a.sum() < k * 2 or b.sum() < k * 2:
        return float("nan"), 0
    def cent(X):
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(X)
        C = np.stack([X[lab == c].mean(0) for c in range(k)
                      if (lab == c).any()])
        return _norm(C)
    Ca, Cb = cent(V[a]), cent(V[b])
    asg_a = np.argmax(V @ Ca.T, 1)
    asg_b = np.argmax(V @ Cb.T, 1)
    n = len(V)
    iu = np.triu_indices(n, 1)
    A = (asg_a[:, None] == asg_a[None, :])[iu]
    B = (asg_b[:, None] == asg_b[None, :])[iu]
    return float((A == B).mean()), k


def uniform(dur, w=3.0, st=1.0):
    out, x = [], 0.0
    while x + w <= dur + 1e-6:
        out.append((x, x + w, "?"))
        x += st
    return out


def main():
    import domains as D
    from tqdm import tqdm
    enc = arg("--enc", "r50_rank")
    want = arg("--domains", "agz,kitti").split(",")
    cap = arg("--cap", 500, int)

    print(f"STYLE structure (no taxonomy) — encoder {enc}\n", flush=True)
    print(f"{'domain':<12}{'units':<8}{'nn dist':<10}{'rand dist':<11}"
          f"{'ratio':<9}{'stability':<11}{'k'}")

    for name in want:
        if name == "agz":
            d = D.agz()
            dur = d["t"][-1] - d["t"][0]
            sp = uniform(dur)[:cap]
            V, kept = units_for(sp, lambda i, d=d: d["frame_path"](
                max(int(i) + 1, 1)), d["fps"], enc)
            med = ["agz"] * len(kept)
            t, tr = d["t"], None
            # rebuild continuous state on the same series domains.py used
            import csv
            rows = list(csv.reader(open(
                D.AGZ / "Log Files/OnboardGPS.csv", newline="")))[1:]
            lat = np.array([float(r[2]) for r in rows])
            lon = np.array([float(r[3]) for r in rows])
            alt = np.array([float(r[4]) for r in rows])
            x = (lon - lon.mean()) * 111320.0 * np.cos(np.radians(lat.mean()))
            y = (lat - lat.mean()) * 111320.0
            dt = np.gradient(t) + 1e-9
            vx, vy = np.gradient(x) / dt, np.gradient(y) / dt
            speed = np.hypot(vx, vy)
            turn = np.gradient(np.unwrap(np.arctan2(vy, vx))) / dt
            climb = np.gradient(alt) / dt
            K = kinematic_state(t, speed, turn, climb, kept)
        elif name == "kitti":
            Vs, ks, med, Ks = [], [], [], []
            for dr in tqdm(D.kitti_drives(), desc="kitti", unit="drive",
                           leave=False):
                dd = D.kitti(dr)
                t = dd["t"]
                if len(t) < 20:
                    continue
                sp = uniform(t[-1] - t[0], 2.0, 0.5)
                v, kp = units_for(sp, dd["frame_path"], dd["fps"], enc)
                if not len(v):
                    continue
                import datetime as _d  # noqa: F401
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
                print(f"{name:<12}no units")
                continue
            V, kept, K = _norm(np.concatenate(Vs)), ks, np.concatenate(Ks)
        else:
            continue

        if len(V) < 20:
            print(f"{name:<12}too few units ({len(V)})")
            continue
        nn, rd, npairs = neighbour_coherence(V, K, kept, med)
        st, kk = stability(V, kept)
        ratio = nn / rd if rd else float("nan")
        print(f"{name:<12}{len(V):<8}{nn:<10.3f}{rd:<11.3f}"
              f"{ratio:<9.3f}{st:<11.3f}{kk}", flush=True)

    print("\nratio < 1.0 = embedding neighbours ARE physically more "
          "alike (what QbE needs).  1.0 = nothing usable.")


if __name__ == "__main__":
    main()
