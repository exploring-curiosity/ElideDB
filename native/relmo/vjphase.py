"""Remove the WINDOW-PHASE signature from stream-time descriptors.

THE MEASUREMENT THAT FORCED THIS. On v6 records, with a cubic trend in episode
time already removed and phase labels permuted WITHIN each recording as the
null, window phase p = (step - CTX) mod 8 explains

    a_t   67.7% of the direction variance   (null 0.1%, p < 0.005)
    b_t   27.4%                             (null 0.1%, p < 0.005)

of the L2-normalised, per-recording-centred descriptors. That is not a detail
riding on top of the content; at 68% it IS the descriptor. Subsequence DTW over
cosine would spend most of its budget aligning window phase, which is precisely
the failure the v4 header warns about for context ramps: "DTW aligning two
traces that both ramp will match the ramps, not the events".

WHY IT APPEARS NOW AND NOT IN v4. Two sources, both structural:

  * the ENCODER. V-JEPA carries positional embeddings over the 16 temporal
    positions of its window, so a tubelet's representation depends on where it
    sits in the clip. This is why b_t - pure observation, no predictor
    involved, no anchoring choice available - shows the effect at all.
  * the PREDICTOR. Each window runs two blocks from different contexts, at
    horizons 1..5. Block 1 and block 2 are different prediction regimes.

Under v4 every clip was sampled to an identical grid, so step t of one clip and
step t of another carried the SAME phase signature and it cancelled as common
mode. Under stream time two recordings of the same event can sit at different
offsets relative to their window grids, and DTW is free to align across that
offset - so the signature stops cancelling and starts competing.

THE CORRECTION. Estimate the mean unit descriptor per phase over a corpus and
subtract it, then re-normalise; likewise remove the per-phase mean of the log
norm-ratio. Nothing here looks at a label, a task name, a verb, an object or a
scene - it is a property of the ENCODER and the WINDOW GEOMETRY, the same
category as the affine predictor calibration the pipeline already carries, and
it travels to out-of-domain corpora unchanged for the same reason: refitting it
per target domain would be exactly the per-domain tuning an OOD claim may not
use.

It is fitted on the TRAIN split only, so the reported drop on val/test/ood is a
real generalisation and not the tautology of subtracting a mean from the data
it was computed on.

    python -m relmo.vjphase --fit          # fit on train, report held-out R2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjrec4 import CTX  # noqa: E402

PERIOD = 8                # descriptor steps per window; see vjrec6.geometry
OUT = R.BASE / "vjphase"


def phase_of(step, period=PERIOD):
    return (np.asarray(step) - CTX) % period


def unit(x, eps=1e-9):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def fit(recs, period=PERIOD):
    """{id: npz-like} -> phase model. Uses no labels of any kind."""
    ua = [[] for _ in range(period)]
    ub = [[] for _ in range(period)]
    ur = [[] for _ in range(period)]
    for z in recs:
        p = phase_of(z["step"], period)
        a = z["pred_change"].astype(np.float64)
        b = z["obs_change"].astype(np.float64)
        r = np.log((np.linalg.norm(b, axis=-1) + 1e-9)
                   / (np.linalg.norm(a, axis=-1) + 1e-9))
        A, B = unit(a), unit(b)
        for k in range(period):
            m = p == k
            if m.any():
                ua[k].append(A[m])
                ub[k].append(B[m])
                ur[k].append(r[m])
    mu_a = np.stack([np.concatenate(x).mean(0) if x else np.zeros(1024)
                     for x in ua]).astype(np.float32)
    mu_b = np.stack([np.concatenate(x).mean(0) if x else np.zeros(1024)
                     for x in ub]).astype(np.float32)
    mu_r = np.array([float(np.concatenate(x).mean()) if x else 0.0
                     for x in ur], np.float32)
    n = np.array([sum(len(y) for y in x) for x in ua], np.int64)
    return dict(mu_a=mu_a, mu_b=mu_b, mu_r=mu_r, n=n,
                period=np.int32(period))


def apply(a, b, step, mdl):
    """-> (a', g'). Direction with the phase mean removed, then renormalised."""
    p = phase_of(step, int(mdl["period"]))
    A = unit(unit(a.astype(np.float32)) - mdl["mu_a"][p])
    B = unit(unit(b.astype(np.float32)) - mdl["mu_b"][p])
    r = np.log((np.linalg.norm(b, axis=-1) + 1e-9)
               / (np.linalg.norm(a, axis=-1) + 1e-9)) - mdl["mu_r"][p]
    g = np.stack([(A * B).sum(-1), r], -1).astype(np.float32)
    return A.astype(np.float32), g


def phase_r2(U, p, period=PERIOD):
    """Share of the centred direction variance explained by per-phase means."""
    U = U - U.mean(0)
    tot = float((U ** 2).sum())
    if tot <= 0:
        return float("nan")
    return sum(int((p == k).sum()) * float((U[p == k].mean(0) ** 2).sum())
               for k in range(period)) / tot


def load(name="rcasa_train"):
    f = OUT / f"{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"no phase model at {f} - run relmo.vjphase "
                                f"--fit")
    return dict(np.load(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--name", default="rcasa_train")
    ap.add_argument("--min-steps", type=int, default=32,
                    help="recordings shorter than 4 windows cannot separate "
                         "window phase from position-in-episode; they are "
                         "excluded from the DIAGNOSTIC, never from the fit")
    a = ap.parse_args()

    from relmo.vjsplit import load as load_split
    sp = load_split()
    rec = R.BASE / "vjrec6" / f"{a.dataset}_L{a.layer}"
    files = {p.stem: p for p in rec.glob("*.npz")
             if not p.name.startswith(".")}
    if not files:
        raise SystemExit(f"no v6 records in {rec}")
    tr = sorted(set(sp["train"]) & set(files))
    print(f"{len(files)} records | fitting on {len(tr)} TRAIN recordings")

    mdl = fit([np.load(files[i]) for i in tr])
    if a.fit:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"{a.name}.npz", **mdl)
        print(f"wrote {OUT / (a.name + '.npz')}  "
              f"steps per phase {mdl['n'].tolist()}")

    print(f"\nphase R2 on the L2-normalised, per-recording-centred descriptor")
    print(f"(recordings with >= {a.min_steps} steps only, so window phase is "
          f"not confounded with position in the episode)\n")
    print(f"{'split':10s} {'n_rec':>6s} {'a raw':>8s} {'a fixed':>8s} "
          f"{'b raw':>8s} {'b fixed':>8s}")
    for split in ("train", "val", "test"):
        ids = [i for i in sorted(set(sp[split]) & set(files))
               if len(np.load(files[i])["step"]) >= a.min_steps]
        if not ids:
            continue
        raw_a, raw_b, fix_a, fix_b, P = [], [], [], [], []
        for i in ids:
            z = np.load(files[i])
            st = z["step"]
            ra, rb = unit(z["pred_change"]), unit(z["obs_change"])
            fa, _ = apply(z["pred_change"], z["obs_change"], st, mdl)
            fb = unit(unit(z["obs_change"]) - mdl["mu_b"][phase_of(st)])
            for L, v in ((raw_a, ra), (raw_b, rb), (fix_a, fa), (fix_b, fb)):
                L.append(v - v.mean(0))          # centre per recording
            P.append(phase_of(st))
        P = np.concatenate(P)
        vals = [phase_r2(np.concatenate(L), P) for L in
                (raw_a, fix_a, raw_b, fix_b)]
        print(f"{split:10s} {len(ids):6d} " +
              " ".join(f"{v:8.4f}" for v in vals))
    R.log("vjphase", dataset=a.dataset, name=a.name, fitted=bool(a.fit),
          n_train=len(tr))


if __name__ == "__main__":
    main()
