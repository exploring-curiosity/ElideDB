"""Re-run the audit across every gate variant. Fix each claim, or show it cannot be.

For each of the five claims, the question is not "does it work" but "does ANY
available construction make it work" - because the first audit tested exactly
one construction and reported its failure as a property of the approach.

TWO CONTROLS DO THE REAL WORK

  ORACLE GATE (gorc)  content pooled over the TRUE object mask. Content is
      normally pooled weighted by the gate, so if the gate is wrong the content
      is pooled from the wrong patches and claim C fails for claim A's reason.
      Oracle pooling separates them. Scoring only; never available at serve.

  IMAGE-SPACE TARGET  the first audit asked the content to predict WORLD
      displacement in metres. The model only ever sees pixels, and recovering
      world coordinates additionally requires camera geometry the latent was
      measured not to carry (linear depth probe R2 0.202 at best, negative at
      layer 6). Asking for world metres conflates "does not encode motion" with
      "does not encode 3D". Image-space displacement - the motion of the true
      mask's centroid - is what a pixel model could in principle know, so it is
      the fair target. World displacement is kept as a second row.

Claim D is also re-specified. Correlating gate mass with contact COUNT was the
wrong test: making contact typically STOPS motion, so a real relationship would
show as anticorrelation or as a transient. Contact ONSET against gate CHANGE is
the test that could actually detect it.

    python -m relmo.vjaudit2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import GRID, TUBELET  # noqa: E402
from relmo.vjfix import GATES, OUTF  # noqa: E402
from relmo.vjrec4 import CTX  # noqa: E402

PRED = [g for g in GATES if g != "gorc"]        # gorc is the oracle, not an arm


def ridge_r2(X, Y, ep, lams=(1e2, 1e3, 1e4, 1e5)):
    """Held-out R2 at the best lambda. Split by EPISODE, never by row."""
    cut = np.median(ep)
    tr, te = ep <= cut, ep > cut
    if tr.sum() < 50 or te.sum() < 50:
        return float("nan")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    best = -np.inf
    for lam in lams:
        A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z[tr].T @ Y[tr])
        p = Z[te] @ w
        r2 = 1 - ((p - Y[te]) ** 2).sum() / (((Y[te] - Y[te].mean(0)) ** 2).sum())
        best = max(best, r2)
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    a = ap.parse_args()

    d = OUTF / a.dataset
    files = sorted(d.glob("*.npz"))
    if len(files) < 30:
        raise SystemExit(f"only {len(files)} vjfix records - let it finish")
    man = R.read_manifest(a.dataset)
    by = {e["id"]: e for e in man["episodes"]}
    steps = list(range(CTX, 32))
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    POS = np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float64)

    A = {g: [0, 0] for g in PRED}
    chance = []
    Bc = {g: [] for g in PRED}
    E = {g: [] for g in PRED}
    Dc = {g: [] for g in PRED}
    CX = {g: [] for g in GATES}
    Yimg, Ywld, Ycon, EP = [], [], [], []
    k = 0
    for p in files:
        e = by.get(p.stem)
        if e is None:
            continue
        st = np.load(R.dataset_dir(a.dataset) / e["shard"] / p.stem / "state.npz")
        z = np.load(p)
        tb = st["target_bodies"]
        xpos, xvel = st["xpos"], st["xvel"]
        cp, cn = st["contact_pairs"], st["contact_n"]
        T = len(xpos)
        idx = np.linspace(0, T - 1, 64).round().astype(int)
        orc = z["gorc"]
        vis = orc.sum(1) > 0.05
        if vis.sum() < 8:
            continue
        # image-space displacement of the TRUE mask centroid
        cen = np.zeros((len(steps), 2))
        for si in range(len(steps)):
            w = orc[si]
            cen[si] = (POS * w[:, None]).sum(0) / (w.sum() + 1e-9) if w.sum() > 0 \
                else np.nan
        dimg = np.vstack([np.zeros((1, 2)), np.diff(cen, axis=0)])
        spd, dwld, con = [], [], []
        for si, t in enumerate(steps):
            f = min(idx[t * TUBELET], T - 1)
            f0 = min(idx[max(t - 1, CTX) * TUBELET], T - 1)
            spd.append(np.abs(xvel[f][tb][:, :3]).sum())
            dwld.append((xpos[f][tb][:, :3] - xpos[f0][tb][:, :3]).mean(0))
            kk = int(cn[f])
            con.append(float(np.isin(cp[f][:kk, 1:3], tb).any(1).sum()) if kk else 0.0)
        spd = np.array(spd)
        con = np.array(con)
        ok = vis & np.isfinite(dimg).all(1)
        for g in PRED:
            gg = z[g]
            m = gg.sum(1)
            for si in range(len(steps)):
                if not vis[si] or orc[si].max() <= 0:
                    continue
                mk = orc[si] > 0.10 * orc[si].max()
                if mk.sum() == 0:
                    continue
                A[g][0] += int(mk[gg[si].argmax()])
                A[g][1] += 1
                if g == PRED[0]:
                    chance.append(mk.mean())
            if m.std() > 0 and spd.std() > 0:
                Bc[g].append(np.corrcoef(m, spd)[0, 1])
                ga, ta = m > np.median(m), spd > np.median(spd)
                E[g].append((ga & ta).sum() / max((ga | ta).sum(), 1))
            dm, dc = np.abs(np.diff(m)), np.abs(np.diff(con))
            if dm.std() > 0 and dc.std() > 0:
                Dc[g].append(np.corrcoef(dm, dc)[0, 1])
        for g in GATES:
            CX[g].append(z["c_" + g][ok])
        Yimg.append(dimg[ok])
        Ywld.append(np.array(dwld)[ok])
        Ycon.append(con[ok, None])
        EP.append(np.full(int(ok.sum()), k))
        k += 1

    ch = float(np.mean(chance))
    EPa = np.concatenate(EP)
    Yi, Yw, Yc = (np.concatenate(v) for v in (Yimg, Ywld, Ycon))
    print(f"AUDIT v2 - {k} episodes, {len(EPa)} steps. sim state SCORING only\n")
    print("A  WHERE - gate peak lands on the true object mask")
    print(f"   {'gate':8s} {'hit':>7s} {'chance':>8s} {'lift':>7s}")
    for g in PRED:
        h = A[g][0] / max(A[g][1], 1)
        print(f"   {g:8s} {h:7.3f} {ch:8.3f} {h/ch:6.2f}x")
    print("\nB  WHEN - corr(gate mass, true target speed)")
    for g in PRED:
        print(f"   {g:8s} {np.mean(Bc[g]):+7.3f}")
    print("\nC  WHAT - held-out ridge R2 from pooled content (split by episode)")
    print(f"   {'pooling':10s} {'image-space':>12s} {'world (m)':>11s} "
          f"{'contact':>9s}")
    for g in GATES:
        X = np.concatenate(CX[g])
        tag = g + (" *ORACLE" if g == "gorc" else "")
        print(f"   {tag:10s} {ridge_r2(X, Yi, EPa):12.3f} "
              f"{ridge_r2(X, Yw, EPa):11.3f} {ridge_r2(X, Yc, EPa):9.3f}")
    print("\nD  CONTACT - corr(|d gate mass|, |d contact count|)")
    for g in PRED:
        print(f"   {g:8s} {np.mean(Dc[g]):+7.3f}")
    print("\nE  SPAN - IoU of gate-active vs actually-moving (chance ~0.333)")
    for g in PRED:
        print(f"   {g:8s} {np.mean(E[g]):7.3f}")


if __name__ == "__main__":
    main()
