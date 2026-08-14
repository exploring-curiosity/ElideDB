"""Validate that an episode's GROUND TRUTH is actually correct.

This exists because the previous corpus passed a geometry gate and was
still unusable. pixelgt asserted lift 1.000 / roundtrip 0.0000px / seg
1.0000 on rcasa_v1 - and every frame of it was collision geometry drawn
over the visual meshes, its segmentation merged every body above id 255
into one, and the manipulation target moved in 4.5% of episodes. Each
number was true; together they proved nothing.

So this checks the things that were silently wrong, not only the things
that were easy to measure:

  A SEG RANGE      no body id is saturated at a dtype ceiling. uint8
                   clipped 318-507-body kitchens at 255, which merged
                   bodies AND made the visibility gate reject whole
                   tasks (CloseDrawer's targets are 323/324/325/329).
  B PROJECTION     lift a pixel to 3D through depth + the camera model,
                   attach it to the body segmentation says owns it, then
                   re-project it. Error in pixels. Validates camera
                   intrinsics, extrinsics, depth scale and seg jointly.
  C PERSISTENCE    carry that point forward on its body's pose and check
                   it still lands on the SAME body id. Validates xpos /
                   xquat against seg over time - the thing a world model
                   actually consumes.
  D CONTENT        the manipulation target moves, is visible, and the
                   gripper is visible. A geometrically perfect recording
                   of nothing happening is still worthless.
  E CONTACT        contacts are recorded, and some involve the target.
  F RENDER         the visual layer is what was captured: flat untextured
                   collision geometry shows up as large regions of
                   near-identical saturated colour, so that is measured
                   and reported rather than assumed away.

    python -m relmo.gtcheck --dataset rcasa_v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.pixelgt import open_gt  # noqa: E402

MAX_PROJ_PX = 1.0     # roundtrip must be sub-pixel
MIN_PERSIST = 0.80    # fraction of carried points still on their body
MIN_TARGET_MOVE = 0.05
# clean decoded video measured 0.010 median / 0.042 max on this corpus;
# a collision-geom render measured 0.187-0.211. Sits between, with room.
MAX_FLICKER = 0.08


def _frames(mp4, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def check(ep_dir: Path, n_pts=200, seed=0):
    z = np.load(ep_dir / "state.npz", allow_pickle=True)
    W, H = int(z["width"]), int(z["height"])
    seg, dep = z["seg"], z["depth"].astype(np.float32)
    ds = int(z["seg_ds"])
    nb = len(z["body_names"])
    T = len(seg)
    rep = dict(id=ep_dir.name, T=int(T), n_bodies=int(nb),
               camera=str(z["camera"]), instruction=str(z["instruction"]))

    # -- A: segmentation range
    mx = int(seg.max())
    ceil_hit = float((seg == np.iinfo(seg.dtype).max).mean())
    rep["seg_dtype"] = str(seg.dtype)
    rep["seg_max_id"] = mx
    rep["seg_saturated_frac"] = round(ceil_hit, 6)
    rep["A_seg_range_ok"] = bool(mx < np.iinfo(seg.dtype).max and mx <= nb)

    # -- B/C: projection roundtrip and persistence
    gt = open_gt(ep_dir)
    rng = np.random.default_rng(seed)
    errs, persist, lifted = [], [], 0
    for _ in range(n_pts):
        u = float(rng.integers(4, W - 4))
        v = float(rng.integers(4, H - 4))
        lt = gt.lift_track(0, u, v)
        if lt is None or not lt["vis"][0]:
            continue
        lifted += 1
        errs.append(float(np.hypot(lt["xy"][0, 0] - u, lt["xy"][0, 1] - v)))
        # does the point stay on its own body as the body moves?
        b = int(lt["body"])
        hit = tot = 0
        for t in range(0, T, max(T // 12, 1)):
            if not lt["vis"][t]:
                continue
            x, y = lt["xy"][t]
            ix, iy = int(x / ds), int(y / ds)
            if not (0 <= ix < seg.shape[2] and 0 <= iy < seg.shape[1]):
                continue
            tot += 1
            hit += int(seg[t, iy, ix] == b)
        if tot:
            persist.append(hit / tot)
    rep["points_lifted"] = lifted
    rep["B_proj_err_px_mean"] = round(float(np.mean(errs)), 5) if errs else None
    rep["B_proj_err_px_max"] = round(float(np.max(errs)), 5) if errs else None
    rep["B_projection_ok"] = bool(errs and np.max(errs) <= MAX_PROJ_PX)
    rep["C_persistence"] = round(float(np.mean(persist)), 4) if persist else None
    rep["C_persistence_ok"] = bool(persist and np.mean(persist) >= MIN_PERSIST)

    # -- D: content
    tg = z["target_bodies"]
    xp = z["xpos"]
    mv = float(np.linalg.norm(xp[-1][tg] - xp[0][tg], axis=-1).max())
    tv, gv = z["target_visible"], z["gripper_visible"]
    rep["D_target_move_m"] = round(mv, 4)
    rep["D_target_vis_mean"] = round(float(tv.mean()), 4)
    rep["D_grip_vis_mean"] = round(float(gv.mean()), 4)
    rep["D_content_ok"] = bool(mv >= MIN_TARGET_MOVE and tv.mean() > 0
                               and gv.mean() > 0)

    # -- E: contacts, and do any involve the target?
    ncon = z["contact_n"]
    cp = z["contact_pairs"]
    tgt_con = 0
    for t in range(T):
        n = int(ncon[t])
        if n:
            pr = cp[t, :n, :2].astype(int)
            tgt_con += int(np.isin(pr, tg).any(1).sum())
    rep["E_contact_frames"] = int((ncon > 0).sum())
    rep["E_target_contacts"] = int(tgt_con)
    rep["E_contact_ok"] = bool((ncon > 0).any() and tgt_con > 0)

    # -- F: is the render the VISUAL layer?
    #
    # This gate used to ask whether a frame held large flat SATURATED
    # regions, because that is what untextured collision geometry looked
    # like. relmo.rendercontrol re-rendered two episodes from identical
    # physics state, once visual-only and once with every geom group on,
    # and measured that detector against the bug it was built for:
    #
    #                     flat_sat clean   flat_sat BUGGY
    #   painted kitchen        0.2762           0.2898
    #   plain kitchen          0.0003           0.0255
    #
    # It fails both ways. Red-painted cabinets read 0.2762 with nothing
    # wrong (33/448 of this corpus failed on paint), and on the plain
    # kitchen the genuinely buggy render read 0.0255 - UNDER the 0.05
    # threshold, so the gate would have passed the bug it exists to
    # catch. Flat and saturated is what paint looks like.
    #
    # What the bug actually is: collision geoms are coincident with the
    # visual meshes, so which one wins is decided per pixel and flips as
    # the view changes. It is a TEMPORAL defect. Measure it directly -
    # on pixels whose body id is unchanged between consecutive frames,
    # how often does the colour jump? Same control, same episodes:
    #
    #                     flicker clean    flicker BUGGY
    #   painted kitchen       0.00089          0.21051
    #   plain kitchen         0.00241          0.18657
    #
    # ~100x separation, and it does not care what colour the cabinets
    # are. On h.264-decoded video rather than raw renders, compression
    # residual lifts the clean floor to a measured 0.010 median / 0.042
    # max over this corpus, still far below the bug's 0.19.
    F = _frames(ep_dir / "frames.mp4", W, H)
    f = F[len(F) // 2].astype(np.int16)
    sat = (f.max(-1) - f.min(-1)) > 90
    lap = np.abs(np.diff(f.mean(-1), axis=0)[:, :-1]) + \
        np.abs(np.diff(f.mean(-1), axis=1)[:-1, :])
    flat = lap < 2.0
    rep["F_flat_saturated_frac"] = round(float((sat[:-1, :-1] & flat).mean()), 4)
    Tf = min(len(F), T)
    Fd = F[:Tf, ::ds, ::ds].astype(np.int16)[:, :seg.shape[1], :seg.shape[2]]
    fl = []
    for t in np.linspace(0, Tf - 2, min(8, max(Tf - 1, 1))).astype(int):
        same = seg[t] == seg[t + 1]
        if same.sum() < 50:
            continue
        dd = np.abs(Fd[t + 1] - Fd[t]).max(-1)
        fl.append(float((dd[same] > 30.0).mean()))
    rep["F_flicker"] = round(float(np.mean(fl)), 5) if fl else 0.0
    rep["F_render_ok"] = bool(rep["F_flicker"] < MAX_FLICKER)

    # -- G: is xvel the velocity of xpos? MuJoCo's cvel is about the
    # subtree CoM, so storing it unshifted next to xpos gives a pair that
    # silently disagrees (measured 1.43 relative error before the fix).
    xv = z["xvel"][:, :, :3].astype(np.float64)
    fd = np.diff(xp.astype(np.float64), axis=0) * float(z["fps"])
    movm = np.linalg.norm(fd, axis=-1) > 0.02
    if movm.any():
        e = (np.linalg.norm(fd[movm] - xv[:-1][movm], axis=-1)
             / np.maximum(np.linalg.norm(fd[movm], axis=-1), 1e-6))
        rep["G_vel_rel_err"] = round(float(np.median(e)), 4)
    else:
        rep["G_vel_rel_err"] = 0.0
    rep["G_velocity_ok"] = bool(rep["G_vel_rel_err"] < 0.20)

    # -- H: was the contact buffer big enough? A truncated contact
    # channel is worse than none, and 64 dropped contacts in 35.8% of
    # frames before this was measured.
    ov = int(z["contact_overflow_frames"]) if "contact_overflow_frames" \
        in z.files else -1
    rep["H_contact_overflow_frames"] = ov
    rep["H_contact_ok"] = bool(ov == 0)

    rep["PASS"] = bool(rep["A_seg_range_ok"] and rep["B_projection_ok"]
                       and rep["C_persistence_ok"] and rep["D_content_ok"]
                       and rep["E_contact_ok"] and rep["F_render_ok"]
                       and rep["G_velocity_ok"] and rep["H_contact_ok"])
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa_v2")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    root = R.dataset_dir(a.dataset)
    eps = sorted(p.parent for p in root.glob("shard_*/*/state.npz"))
    rows = [check(e, a.n) for e in eps]
    hdr = (f"{'episode':50s} {'A':>3s} {'B px':>8s} {'C':>6s} "
           f"{'D move':>7s} {'E con':>6s} {'F flick':>8s} {'G vel':>6s} "
           f"{'H ovf':>6s}  PASS")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['id'][:50]:50s} {'ok' if r['A_seg_range_ok'] else 'BAD':>3s} "
              f"{(r['B_proj_err_px_max'] if r['B_proj_err_px_max'] is not None else -1):8.4f} "
              f"{(r['C_persistence'] or 0):6.3f} {r['D_target_move_m']:7.3f} "
              f"{r['E_target_contacts']:6d} {r['F_flicker']:8.5f} "
              f"{r['G_vel_rel_err']:6.3f} {r['H_contact_overflow_frames']:6d}  "
              f"{'PASS' if r['PASS'] else 'FAIL'}")
    n_ok = sum(r["PASS"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} passed")
    for r in rows:
        if not r["PASS"]:
            bad = [k for k in ("A_seg_range_ok", "B_projection_ok",
                               "C_persistence_ok", "D_content_ok",
                               "E_contact_ok", "F_render_ok",
                               "G_velocity_ok", "H_contact_ok") if not r[k]]
            print(f"  {r['id'][:50]:50s} failed: {', '.join(bad)}")
    R.log("gtcheck", dataset=a.dataset, n=len(rows), passed=n_ok,
          rows=[{k: v for k, v in r.items() if k != "instruction"}
                for r in rows])
