"""Multi-domain event truth, derived WITHOUT the camera.

The action-AUC ceiling (0.721 -> yield ~0.49) was measured on one
embodiment: a MuJoCo robot arm. Whether that is a physical-AI ceiling
or an artifact of that corpus cannot be settled on more arm data, so
this exposes other embodiments through one interface.

The discipline, inherited from native/oxford.py, is that truth must be
VISION-INDEPENDENT. Every domain below derives its events from motion
sensing - GPS, IMU, OXTS, laser-tracker pose - never from pixels. If
events were derived from appearance and then scored with an appearance
encoder, the result would be circular and worthless.

Domains:
  agz      45 min drone flight over Zurich; per-image GPS lat/lon/alt,
           joined to frames by imgid (an index join, no interpolation)
  uzhfpv   aggressive quadrotor flight; laser-tracker 6-DoF pose
  kitti    car; per-frame OXTS GPS/IMU (lat lon alt roll pitch yaw vf)
  sim      robot arm; the existing truth.parquet prim labels

Event vocabulary is deliberately the SAME shape in every domain -
kinematic states any moving platform has - because the question is
whether one encoder discriminates events across embodiments, and that
question is meaningless if each domain gets a different taxonomy.

    python native/domains.py --domain agz --limit 400
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

AGZ = ROOT / "data/drone/agz_full/AGZ"
KITTI = ROOT / "data/kitti"


def _label(speed, turn, climb, sp_hi, turn_hi, climb_hi):
    """One kinematic vocabulary for every embodiment.

    Deliberately coarse and platform-neutral: a car, a drone and an arm
    all either hold still, proceed, turn, or change height. Thresholds
    are per-domain QUANTILES of that domain's own distribution, so
    nothing is a hand-picked constant and no domain is handed an easier
    problem than another.
    """
    if speed < sp_hi[0]:
        return "still"
    if abs(turn) > turn_hi:
        return "turn"
    if abs(climb) > climb_hi:
        return "climb" if climb > 0 else "descend"
    return "cruise"


def _smooth(x, t, win_s=2.0):
    """Moving average over a TIME window.

    Necessary, not cosmetic: AGZ logs GPS slower than it captures
    frames, so consecutive rows repeat a position and a raw gradient is
    mostly quantisation noise. Unsmoothed, every kinematic label
    flickers frame to frame, no run survives the minimum length, and
    the whole corpus collapses to one class (measured: 90/90 "cruise").
    The window is expressed in SECONDS so it means the same thing at
    30 Hz on a drone and 10 Hz on a car.
    """
    hz = len(t) / max(t[-1] - t[0], 1e-6)
    w = max(int(win_s * hz), 3)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")


def _spans_from_series(t, speed, turn, climb, min_len=1.5):
    """Contiguous runs of one kinematic label -> labelled spans."""
    speed = _smooth(speed, t)
    turn = _smooth(turn, t)
    climb = _smooth(climb, t)
    sp_hi = (np.quantile(speed, 0.25),)
    # Heading rate is meaningless while stationary - atan2 of a zero
    # velocity is noise - so the turn threshold is fitted ONLY over
    # moving samples. Fitted over everything, the 114 "still" spans in
    # AGZ swamped the 80th percentile and "turn" fired once in 45 min.
    mv = speed >= sp_hi[0]
    turn_hi = (np.quantile(np.abs(turn[mv]), 0.80) if mv.sum() > 10
               else np.quantile(np.abs(turn), 0.80))
    climb_hi = np.quantile(np.abs(climb), 0.85)
    lab = [_label(s, r, c, sp_hi, turn_hi, climb_hi)
           for s, r, c in zip(speed, turn, climb)]
    out, i = [], 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        if t[j] - t[i] >= min_len:
            out.append((float(t[i]), float(t[j]), lab[i]))
        i = j + 1
    return out


def agz():
    """45-minute urban drone flight. Truth = onboard GPS."""
    import csv
    rows = list(csv.reader(
        open(AGZ / "Log Files/OnboardGPS.csv", newline="")))[1:]
    t = np.array([float(r[0]) for r in rows]) * 1e-6      # us -> s
    t -= t[0]
    imgid = np.array([int(r[1]) for r in rows])
    lat = np.array([float(r[2]) for r in rows])
    lon = np.array([float(r[3]) for r in rows])
    alt = np.array([float(r[4]) for r in rows])
    # local metric frame (small area, so a flat approximation is fine)
    m_lat = 111320.0
    x = (lon - lon.mean()) * m_lat * np.cos(np.radians(lat.mean()))
    y = (lat - lat.mean()) * m_lat
    dt = np.gradient(t) + 1e-9
    vx, vy = np.gradient(x) / dt, np.gradient(y) / dt
    speed = np.hypot(vx, vy)
    head = np.unwrap(np.arctan2(vy, vx))
    turn = np.gradient(head) / dt
    climb = np.gradient(alt) / dt
    spans = _spans_from_series(t, speed, turn, climb)

    def frame_path(i):
        return AGZ / "MAV Images" / f"{i:05d}.jpg"

    return dict(name="agz", t=t, imgid=imgid, spans=spans,
                frame_path=frame_path, fps=float(len(t) / (t[-1] - t[0])))


def uzhfpv(seq="outdoor_forward_1"):
    """Quadrotor with laser-tracker ground-truth pose."""
    import zipfile
    zp = ROOT / f"data/drone/uzhfpv_{seq}.zip"
    z = zipfile.ZipFile(zp)
    gt = [l.split() for l in
          z.read("groundtruth.txt").decode().splitlines()
          if l and not l.startswith("#")]
    t = np.array([float(r[0]) for r in gt])
    t -= t[0]
    p = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in gt])
    dt = np.gradient(t) + 1e-9
    v = np.gradient(p, axis=0) / dt[:, None]
    speed = np.linalg.norm(v[:, :2], axis=1)
    head = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn = np.gradient(head) / dt
    climb = v[:, 2]
    return dict(name=f"uzhfpv_{seq}", t=t, zip=zp,
                spans=_spans_from_series(t, speed, turn, climb),
                fps=float(len(t) / (t[-1] - t[0])))


def kitti_drives():
    """Every extracted KITTI drive with OXTS."""
    out = []
    for d in sorted(KITTI.glob("2011_09_26/2011_09_26_drive_*_sync")):
        ox = d / "oxts/data"
        im = d / "image_02/data"
        if ox.is_dir() and im.is_dir():
            out.append(d)
    return out


def kitti(drive):
    """Car. Truth = per-frame OXTS (lat lon alt roll pitch yaw ... vf)."""
    ox = sorted((drive / "oxts/data").glob("*.txt"))
    vals = [list(map(float, f.read_text().split())) for f in ox]
    A = np.array(vals)
    ts = (drive / "oxts/timestamps.txt").read_text().splitlines()
    import datetime as dtm
    tt = [dtm.datetime.fromisoformat(s.strip()[:26]) for s in ts if s.strip()]
    t = np.array([(x - tt[0]).total_seconds() for x in tt])[:len(A)]
    yaw = np.unwrap(A[:, 5])
    dt = np.gradient(t) + 1e-9
    turn = np.gradient(yaw) / dt
    speed = np.hypot(A[:, 6], A[:, 7]) if A.shape[1] > 7 else A[:, 8]
    climb = np.gradient(A[:, 2]) / dt

    def frame_path(i):
        return drive / "image_02/data" / f"{i:010d}.png"

    return dict(name=drive.name, t=t, spans=_spans_from_series(
        t, speed, turn, climb), frame_path=frame_path,
        fps=float(len(t) / max(t[-1] - t[0], 1e-6)))


def main():
    which = arg("--domain", "agz")
    if which == "agz":
        d = agz()
    elif which == "kitti":
        ds = kitti_drives()
        print(f"kitti drives extracted: {len(ds)}")
        if not ds:
            return
        d = kitti(ds[0])
    else:
        d = uzhfpv()
    from collections import Counter
    c = Counter(l for _, _, l in d["spans"])
    dur = d["t"][-1] - d["t"][0]
    print(f"{d['name']}: {dur:.0f}s @ {d['fps']:.1f} Hz, "
          f"{len(d['spans'])} labelled spans")
    print("  events:", dict(c))
    for s in d["spans"][:6]:
        print(f"   {s[0]:8.1f} {s[1]:8.1f}  {s[2]}")


if __name__ == "__main__":
    main()
