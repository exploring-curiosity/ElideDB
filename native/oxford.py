"""OXFORD as a third domain for layer grading (driving footage).

Only 19 s of video, so it is not a retrieval corpus - it is a
GENERALISATION CHECK: does a layer built and graded on manipulation
video behave sensibly on driving video, where "events" are stops,
starts and turns rather than picks and places?

Its truth is SENSOR-DERIVED and vision-independent: INS velocity and
yaw give boundary times without anyone labelling pixels, which makes
it an unusually clean grader for step 4.

    python native/oxford.py --prepare    # PNG sequence -> mp4
    python native/oxford.py --grade      # step-4 grade on driving
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

SRC = ROOT / "data/oxfordDataset"
OUT = ROOT / "data/oxford_clip"
MP4 = OUT / "drive.mp4"


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def prepare():
    """Bayer PNG sequence -> a normal mp4 so the SHARED encode path
    (native/encode.py) handles it exactly like any other upload."""
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    ts = np.loadtxt(SRC / "stereo.timestamps", usecols=0,
                    dtype=np.int64)
    files = sorted((SRC / "stereo/centre").glob("*.png"))
    fps = len(ts) / ((ts[-1] - ts[0]) / 1e6)
    tmp = OUT / "frames"
    tmp.mkdir(exist_ok=True)
    for i, p in enumerate(files):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im.ndim == 2:                       # Bayer -> RGB
            im = cv2.cvtColor(im, cv2.COLOR_BayerGB2BGR)
        cv2.imwrite(str(tmp / f"{i:05d}.png"), im)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate",
                    f"{fps:.4f}", "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(MP4)], check=True)
    np.save(OUT / "ts.npy", ts)
    print(f"wrote {MP4}  ({len(files)} frames, {fps:.1f} fps, "
          f"{(ts[-1]-ts[0])/1e6:.1f}s)")


def truth_boundaries(tol_speed=0.5, tol_yawrate=0.05):
    """Event boundaries from INS: stop/start and turn onset/offset.
    Vision-independent by construction."""
    ts = np.load(OUT / "ts.npy")
    t0, t1 = int(ts[0]), int(ts[-1])
    rows = list(csv.DictReader(open(SRC / "gps/ins.csv")))
    t = np.array([int(r["timestamp"]) for r in rows])
    m = (t >= t0) & (t <= t1)
    t = t[m]
    sp = np.hypot(
        np.array([float(r["velocity_north"]) for r in rows])[m],
        np.array([float(r["velocity_east"]) for r in rows])[m])
    yaw = np.unwrap(np.array([float(r["yaw"]) for r in rows])[m])
    rel = (t - t0) / 1e6
    dt = np.gradient(rel) + 1e-9
    yawrate = np.abs(np.gradient(yaw) / dt)
    moving = sp > tol_speed
    turning = yawrate > tol_yawrate
    bnds = []
    for series in (moving, turning):
        ch = np.where(series[1:] != series[:-1])[0]
        bnds += [float(rel[i]) for i in ch]
    # merge boundaries within 0.5 s
    bnds.sort()
    out = []
    for b in bnds:
        if not out or b - out[-1] > 0.5:
            out.append(b)
    return out, rel, sp, yawrate


def grade():
    import encode as E
    import surprise as S
    WIN = arg("--win", 0.5, float)
    STRIDE = arg("--stride", 0.125, float)
    FPS = arg("--fps", 14.0, float)
    TOL = arg("--tol", 1.0, float)
    bs, rel, sp, yr = truth_boundaries()
    dur = float(rel[-1])
    print(f"oxford: {dur:.1f}s driving, {len(bs)} INS boundaries at "
          f"{[round(b,1) for b in bs]}", flush=True)
    E.DEC_FPS = FPS
    E.frames_for.__defaults__ = (E.NF, FPS)
    F = E.decode(MP4, fps=FPS)
    spans, times = [], []
    x = 0.0
    while x + WIN <= dur:
        spans.append((x, x + WIN))
        times.append(x + WIN / 2)
        x += STRIDE
    V = E.encode_spans(F, spans)
    times = np.array(times)
    est = S.estimators(V, k=3)
    near = np.zeros(len(times), bool)
    far = np.ones(len(times), bool)
    for b in bs:
        near |= np.abs(times - b) <= TOL
        far &= np.abs(times - b) > 2 * TOL
    print(f"  {int(near.sum())} boundary-adjacent, {int(far.sum())} "
          f"interior windows", flush=True)
    if near.sum() < 2 or far.sum() < 2:
        print("  too few windows to grade")
        return
    best = None
    for k, v in est.items():
        a = S.auc(v[near], v[far])
        print(f"  {k:<18}{a:.3f}")
        if best is None or a > best[1]:
            best = (k, a)
    print(f"STEP 4 GRADE (oxford/driving): best {best[0]} "
          f"AUC {best[1]:.3f}")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    if "--grade" in sys.argv:
        grade()
