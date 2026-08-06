"""Is the motion even IN the pixels? Appearance vs global vs residual.

style.py found that a frozen appearance encoder clusters drone and car
footage by SCENE, not by motion: neighbours in embedding space are only
2-6% more kinematically alike than random units (ratio 0.977 agz,
0.941 kitti) despite the structure being stable (0.80-0.82).

Before reaching for a trained latent-action model, the cheap question
is whether the motion is recoverable from pixels at all. Global image
motion is a direct projection of ego-motion, so a descriptor built from
it should score far better than appearance IF the information is there.

Three descriptors, one ruler (the continuous kinematic state):

  app     r50 rank-pooled appearance  - today's encoder, the baseline
  global  the 6 affine parameters of the DOMINANT image motion per
          frame pair, rank-pooled. This is ego-motion as the camera
          sees it: pan, zoom, rotation.
  resid   flow AFTER the global component is removed - what moved
          INDEPENDENTLY of the camera. This is the "actor A acted on
          actor B" signal, with the camera's own motion subtracted.

Reading the outcome:
  global scores well  -> motion is in the pixels; the appearance
                         encoder simply does not expose it, and the fix
                         is a motion channel, not a bigger model.
  global scores badly -> not recoverable without a trained dynamics
                         model, and the latent-action route is the
                         honest next step rather than a preference.
  resid  is the one that matters for scene-actions rather than
         ego-actions, and it is EXPECTED to score worse on this ruler -
         the ruler measures the ego, which resid deliberately removes.

    python native/motionrep.py --domains agz,kitti
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
from style import (SEPARATION, kinematic_state,                # noqa: E402
                   neighbour_coherence, stability, uniform)
from unitenc import NF, _norm                                  # noqa: E402

FW = 160          # flow resolution; motion needs far less than semantics


def _rank(P):
    """mean + parameter-free rank pooling, as used for appearance."""
    P = np.asarray(P, np.float32)
    if len(P) < 2:
        return np.zeros(P.shape[1] * 2, np.float32)
    m = P.mean(0)
    T = len(P)
    al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
    r = (al[:, None] * P).sum(0)
    return np.concatenate([m, r / max(np.linalg.norm(r), 1e-8)])


def motion_units(spans, frame_at, fps, mode="global", grid=3):
    """Per-unit motion descriptor straight from pixels. No model."""
    import cv2
    from PIL import Image
    vecs, keep = [], []
    for (a, b, lab) in spans:
        # CONSECUTIVE frames, not NF spread across the span. Optical
        # flow needs a small baseline: at 30 fps an 8-sample spread over
        # 3 s puts 0.43 s between samples, far beyond what Farneback's
        # pyramid can track, and the descriptor becomes noise (measured:
        # ratio 1.003, i.e. exactly chance). Sampling a short burst from
        # the middle of the span keeps displacements trackable while
        # still describing that span.
        c = int((a + b) * 0.5 * fps)
        ids = np.arange(c - NF // 2, c + NF - NF // 2)
        ids = ids[ids >= 0]
        gs = []
        for j in ids:
            try:
                im = Image.open(frame_at(int(j))).convert("L")
            except Exception:                            # noqa: BLE001
                continue
            h = max(int(im.height * FW / im.width), 8)
            gs.append(np.asarray(im.resize((FW, h))))
        if len(gs) < 3:
            continue
        prof = []
        for i in range(len(gs) - 1):
            fl = cv2.calcOpticalFlowFarneback(
                gs[i], gs[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = fl.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            # dominant motion as an affine field: [u v] = A [x y 1]
            X = np.stack([xx.ravel(), yy.ravel(),
                          np.ones(h * w, np.float32)], 1)
            U = fl[..., 0].ravel()
            V = fl[..., 1].ravel()
            au, *_ = np.linalg.lstsq(X, U, rcond=None)
            av, *_ = np.linalg.lstsq(X, V, rcond=None)
            if mode == "global":
                # normalise translation by frame size so the descriptor
                # means the same thing at any resolution
                prof.append([au[0], au[1], au[2] / w,
                             av[0], av[1], av[2] / h])
            else:
                ru = U - X @ au
                rv = V - X @ av
                ru = ru.reshape(h, w)
                rv = rv.reshape(h, w)
                ch, cw = h // grid, w // grid
                cells = []
                for r0 in range(grid):
                    for c0 in range(grid):
                        su = ru[r0 * ch:(r0 + 1) * ch,
                                c0 * cw:(c0 + 1) * cw]
                        sv = rv[r0 * ch:(r0 + 1) * ch,
                                c0 * cw:(c0 + 1) * cw]
                        cells += [su.mean(), sv.mean(),
                                  float(np.hypot(su, sv).mean())]
                prof.append(cells)
        if len(prof) < 2:
            continue
        vecs.append(_rank(prof))
        keep.append((a, b, lab))
    return (_norm(np.stack(vecs)) if vecs else np.zeros((0, 2))), keep


def agz_state(kept):
    import csv
    import domains as D
    rows = list(csv.reader(open(
        D.AGZ / "Log Files/OnboardGPS.csv", newline="")))[1:]
    t = np.array([float(r[0]) for r in rows]) * 1e-6
    t -= t[0]
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
    return kinematic_state(t, speed, turn, climb, kept)


def main():
    import domains as D
    from tqdm import tqdm
    want = arg("--domains", "agz,kitti").split(",")
    cap = arg("--cap", 400, int)
    modes = arg("--modes", "global,resid").split(",")

    print("MOTION vs APPEARANCE — same ruler, same units\n", flush=True)
    print(f"{'domain':<10}{'desc':<9}{'units':<8}{'nn':<9}{'rand':<9}"
          f"{'ratio':<9}{'stability'}")

    for name in want:
        if name == "agz":
            d = D.agz()
            sp = uniform(d["t"][-1] - d["t"][0])[:cap]
            fa = lambda i, d=d: d["frame_path"](max(int(i) + 1, 1))
            for mode in modes:
                V, kept = motion_units(sp, fa, d["fps"], mode)
                if len(V) < 20:
                    print(f"{name:<10}{mode:<9}too few")
                    continue
                K = agz_state(kept)
                med = ["agz"] * len(kept)
                nn, rd, _ = neighbour_coherence(V, K, kept, med)
                st, _ = stability(V, kept)
                print(f"{name:<10}{mode:<9}{len(V):<8}{nn:<9.3f}"
                      f"{rd:<9.3f}{nn / rd:<9.3f}{st:.3f}", flush=True)
        elif name == "kitti":
            for mode in modes:
                Vs, ks, med, Ks = [], [], [], []
                for dr in tqdm(D.kitti_drives(), desc=f"kitti/{mode}",
                               unit="drive", leave=False):
                    dd = D.kitti(dr)
                    t = dd["t"]
                    if len(t) < 20:
                        continue
                    sp = uniform(t[-1] - t[0], 2.0, 0.5)
                    v, kp = motion_units(sp, dd["frame_path"],
                                         dd["fps"], mode)
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
                    print(f"{name:<10}{mode:<9}no units")
                    continue
                V = _norm(np.concatenate(Vs))
                K = np.concatenate(Ks)
                nn, rd, _ = neighbour_coherence(V, K, ks, med)
                st, _ = stability(V, ks)
                print(f"{name:<10}{mode:<9}{len(V):<8}{nn:<9.3f}"
                      f"{rd:<9.3f}{nn / rd:<9.3f}{st:.3f}", flush=True)

    print("\nbaseline (appearance, r50_rank): agz 0.977, kitti 0.941")
    print("ratio << 1.0 means the motion IS in the pixels.")


if __name__ == "__main__":
    main()
