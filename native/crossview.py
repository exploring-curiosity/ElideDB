"""VISION ONLY. Videos and nothing else - no text, no sensors, no labels.

Every evaluation before this one leaked a non-vision signal into the
GROUND TRUTH, even when the encoder saw only pixels:

    bridge  classes were task STRINGS ("open the drawer") - text
    car     classes were k-means on OXTS GPS/IMU        - sensors
    drone   classes were k-means on GPS                 - sensors
    sim     classes were template names from a generator - labels

The user's constraint is stricter than I had been reading it: the only
admissible input is the video. So the ground truth has to be derivable
from video alone, and there is exactly one thing that qualifies -

    THE SAME PHYSICAL MOMENT SEEN BY A SECOND CAMERA.

Query with a window from camera A; the correct answers are the windows
of camera B that overlap it in time. Same event, different viewpoint.
No words, no sensors, no annotation - only the fact that two files were
recorded simultaneously, which is a property of the recording, not a
semantic judgement about its contents.

This is also the right question for a QbE database: given this clip,
find this event again. A system that cannot recognise the same event
from a second angle cannot retrieve a similar event from a new episode.

Support comes from temporal overlap: with W-second windows at stride S,
several B-windows overlap a query, so support > 1 and yield/prec are
meaningful at k = ceil(1.5 x support) with the usual abstention.

CAVEAT recorded up front: the viewpoint change differs per domain. sim's
cam1/cam2 are genuinely different angles; KITTI image_02/03 and UZH-FPV
image_0/1 are stereo pairs with a short baseline, so those are an
EASIER test and their numbers are not comparable to sim's.

    python native/crossview.py --domains sim,kitti,fpv
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
from unitenc import _norm                                      # noqa: E402

# stride 1.0 with a 3 s window makes adjacent windows overlap at
# IoU 0.5, so a query has ~3 correct answers instead of 1. With
# support 1 and k = ceil(1.5) = 2, precision is capped at 0.5 by
# construction - the same arithmetic trap found earlier in the sim
# harness. Support 3 lifts the cap to 0.6 and makes yield less binary.
W, STRIDE, NF = 3.0, 1.0, 8
IOU_TRUE = 0.5


def windows(dur, w=W, st=STRIDE):
    out, t = [], 0.0
    while t + w <= dur + 1e-6:
        out.append((t, t + w))
        t += st
    return out


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def enc_frames(frames, mode="rank"):
    from gebdback import BACKBONES, KIND
    fn, mid = KIND[BACKBONES["siglip2"][0]], BACKBONES["siglip2"][1]
    V = _norm(fn(mid, frames))
    m = V.mean(0)
    if mode == "mean":
        return m / max(np.linalg.norm(m), 1e-8)
    T = len(V)
    al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
    r = (al[:, None] * V).sum(0)
    r = r / max(np.linalg.norm(r), 1e-8)
    v = np.concatenate([m, r])
    return v / max(np.linalg.norm(v), 1e-8)


def encode_stream(get_frame, fps, wins, mode):
    out = []
    for (a, b) in wins:
        idx = np.linspace(a * fps, b * fps, NF).round().astype(int)
        fr = []
        for i in idx:
            f = get_frame(int(i))
            if f is not None:
                fr.append(f)
        if len(fr) < 3:
            out.append(None)
            continue
        out.append(enc_frames(fr, mode))
    return out


def qbe_cross(VA, WA, VB, WB):
    """Query each A-window; correct answers are overlapping B-windows."""
    ia = [i for i, v in enumerate(VA) if v is not None]
    ib = [j for j, v in enumerate(VB) if v is not None]
    if len(ia) < 4 or len(ib) < 6:
        return None
    A = np.stack([VA[i] for i in ia])
    B = np.stack([VB[j] for j in ib])
    S = A @ B.T
    ys, ps, sups, rets = [], [], [], []
    for qi in range(len(ia)):
        truth = [n for n, j in enumerate(ib)
                 if tiou(WA[ia[qi]], WB[j]) >= IOU_TRUE]
        sup = len(truth)
        if sup < 1:
            continue
        k = math.ceil(1.5 * sup)
        order = np.argsort(-S[qi])
        # abstention: keep only candidates above the query's own
        # self-consistency floor, estimated from the score distribution
        # (no labels, no seeds available in a single-query setting)
        sc = S[qi][order]
        cut = sc.mean() + sc.std()
        got = [int(o) for o in order[:k] if S[qi][o] >= cut]
        tr = sum(1 for o in got if o in truth)
        ys.append(tr / sup)
        ps.append(tr / len(got) if got else 0.0)
        sups.append(sup)
        rets.append(len(got))
    if not ys:
        return None
    return (float(np.mean(ys)), float(np.mean(ps)),
            float(np.mean(sups)), float(np.mean(rets)), len(ys),
            len(ib))


def run_sim(mode, limit):
    import encode as E
    from tqdm import tqdm
    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))[:limit]
    agg = []
    for d in tqdm(dirs, desc="sim", unit="ep", leave=False):
        # Sim assigns a RANDOM camera pair per episode (cam0/cam2,
        # cam1/cam3, cam2/cam3, ...). Hard-coding cam1/cam2 silently
        # dropped 120 of 150 episodes and left this measurement at n=2.
        # Take whichever two the episode actually has - which also makes
        # the viewpoint change vary across episodes rather than being
        # one fixed baseline.
        cams = sorted(d.glob("cam*.mp4"))
        if len(cams) < 2:
            continue
        c1, c2 = cams[0], cams[1]
        dur = E.probe_duration(c1)
        F1, F2 = E.decode(c1, fps=4.0, w=256), E.decode(c2, fps=4.0,
                                                        w=256)
        wins = windows(dur)
        if len(wins) < 4:
            continue
        VA = encode_stream(lambda i: F1[min(i, len(F1) - 1)], 4.0,
                           wins, mode)
        VB = encode_stream(lambda i: F2[min(i, len(F2) - 1)], 4.0,
                           wins, mode)
        r = qbe_cross(VA, wins, VB, wins)
        if r:
            agg.append(r)
    return agg


def run_kitti(mode, limit):
    from PIL import Image
    from tqdm import tqdm
    import domains as D
    agg = []
    for dr in tqdm(D.kitti_drives()[:limit], desc="kitti", unit="drive",
                   leave=False):
        d2, d3 = dr / "image_02/data", dr / "image_03/data"
        if not (d2.is_dir() and d3.is_dir()):
            continue
        n = len(sorted(d2.glob("*.png")))
        if n < 30:
            continue
        fps, dur = 10.0, n / 10.0

        def mk(base):
            def g(i):
                p = base / f"{i:010d}.png"
                if not p.exists():
                    return None
                im = Image.open(p).convert("RGB")
                h = max(int(im.height * 256 / im.width), 8)
                return np.asarray(im.resize((256, h)))
            return g
        wins = windows(dur)
        if len(wins) < 4:
            continue
        VA = encode_stream(mk(d2), fps, wins, mode)
        VB = encode_stream(mk(d3), fps, wins, mode)
        r = qbe_cross(VA, wins, VB, wins)
        if r:
            agg.append(r)
    return agg


def run_fpv(mode, limit):
    import io
    import zipfile
    from PIL import Image
    from tqdm import tqdm
    agg = []
    zips = sorted((ROOT / "data/drone").glob("uzhfpv_*.zip"))[:limit]
    for zp in tqdm(zips, desc="fpv", unit="seq", leave=False):
        z = zipfile.ZipFile(zp)
        names = z.namelist()
        c0 = {int(n.split("_")[-1].split(".")[0]): n for n in names
              if n.startswith("img/image_0_")}
        c1 = {int(n.split("_")[-1].split(".")[0]): n for n in names
              if n.startswith("img/image_1_")}
        if len(c0) < 60 or len(c1) < 60:
            continue
        fps = 30.0
        dur = min(max(c0), max(c1)) / fps

        def mk(dd):
            def g(i):
                nm = dd.get(i)
                if nm is None:
                    return None
                im = Image.open(io.BytesIO(z.read(nm))).convert("RGB")
                h = max(int(im.height * 256 / im.width), 8)
                return np.asarray(im.resize((256, h)))
            return g
        wins = windows(dur)
        if len(wins) < 4:
            continue
        VA = encode_stream(mk(c0), fps, wins, mode)
        VB = encode_stream(mk(c1), fps, wins, mode)
        r = qbe_cross(VA, wins, VB, wins)
        if r:
            agg.append(r)
    return agg


def main():
    want = arg("--domains", "sim,kitti,fpv").split(",")
    mode = arg("--mode", "rank")
    limit = arg("--limit", 12, int)
    print("VISION-ONLY cross-view QbE — query camera A, retrieve the "
          "same moment from camera B")
    print(f"encoder siglip2_{mode}, {W:.0f}s windows / {STRIDE:.1f}s "
          f"stride, truth = temporal IoU >= {IOU_TRUE}\n")
    print(f"{'domain':<9}{'media':<8}{'queries':<9}{'pool':<7}"
          f"{'yield':<9}{'prec':<9}{'support'}")
    runner = {"sim": run_sim, "kitti": run_kitti, "fpv": run_fpv}
    for name in want:
        if name not in runner:
            continue
        agg = runner[name](mode, limit)
        if not agg:
            print(f"{name:<9}no usable media")
            continue
        y = np.mean([a[0] for a in agg])
        p = np.mean([a[1] for a in agg])
        s = np.mean([a[2] for a in agg])
        q = sum(a[4] for a in agg)
        pool = np.mean([a[5] for a in agg])
        print(f"{name:<9}{len(agg):<8}{q:<9}{pool:<7.0f}{y:<9.3f}"
              f"{p:<9.3f}{s:.1f}", flush=True)
    print("\nchance yield ~ returned/pool; a system that cannot match "
          "the same event across two angles cannot do QbE at all.")


if __name__ == "__main__":
    main()
