"""Positive control: re-render one episode WITH collision geometry on.

The F gate fails 33/448 episodes of rcasa, and every one inspected is a
kitchen with red or orange painted cabinets. Flicker and hue-diversity
say those 33 are indistinguishable from the passing corpus. That is
evidence F is a false positive - but it is not evidence the gate would
still catch the real bug, and a gate that cannot catch its own bug is
worse than no gate.

The buggy corpus was deleted, so this reproduces the bug on purpose:
replay a few frames of a real episode twice from the same physics state,
once with the visual-only scene option the replayer now uses, and once
with all geom groups visible the way MuJoCo's Renderer defaults. Then
run the same measurements on both.

If flat-saturation and hue-diversity separate the two renders, the gate
can be redefined on the signature that actually distinguishes them. If
they do not separate, the whole detector is worthless and should go.

    python -m relmo.rendercontrol --id CloseCabinet_episode_000016
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.flicker import FLICKER_TOL, MIN_SAT  # noqa: E402


def flicker_on_static(F, S):
    """The statistic flicker.py computes: colour jumps on pixels whose
    body id did not change between consecutive frames. Frames must be
    CONSECUTIVE simulation steps, else real motion swamps it."""
    out = []
    for i in range(len(F) - 1):
        same = S[i] == S[i + 1]
        if same.sum() < 50:
            continue
        d = np.abs(F[i + 1].astype(np.int16) - F[i].astype(np.int16)).max(-1)
        out.append(float((d[same] > FLICKER_TOL).mean()))
    return round(float(np.mean(out)), 5) if out else None


def measure(F):
    """The three numbers, on a stack of rendered frames."""
    f = F[len(F) // 2].astype(np.int16)
    mx, mn = f.max(-1), f.min(-1)
    sat = (mx - mn) > MIN_SAT
    lap = np.abs(np.diff(f.mean(-1), axis=0)[:, :-1]) + \
        np.abs(np.diff(f.mean(-1), axis=1)[:-1, :])
    flat = lap < 2.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    c = np.maximum(mx - mn, 1)
    h = np.where(mx == r, ((g - b) / c) % 6,
                 np.where(mx == g, (b - r) / c + 2, (r - g) / c + 4)) * 60
    hues = 0
    if sat.sum() > 100:
        hist, _ = np.histogram(h[sat] % 360, bins=12, range=(0, 360))
        hues = int((hist > 0.05 * hist.sum()).sum())
    d = np.abs(F[1:].astype(np.int16) - F[:-1].astype(np.int16)).max(-1)
    return dict(flat_sat=round(float((sat[:-1, :-1] & flat).mean()), 4),
                sat=round(float(sat.mean()), 4), hues=hues,
                px_change=round(float((d > FLICKER_TOL).mean()), 4))


def run(ep_name, n=12, out=None):
    import mujoco
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from relmo.rcreplay import _visual_only, episodes

    task = ep_name.split("_episode_")[0]
    demo = "episode_" + re.search(r"_episode_(\d+)", ep_name).group(1)
    src = None
    for sp in ("pretrain", "target"):
        for e in episodes(task, sp):
            if e.name == demo:
                src = e
                break
        if src:
            break
    if src is None:
        raise SystemExit(f"source demo not found: {task}/{demo}")

    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron",
                         controller_configs=cfg, has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False,
                         control_freq=20, ignore_done=True)
    env.reset()
    env.set_ep_meta(json.loads((src / "ep_meta.json").read_text()))
    env.reset()
    env.reset_from_xml_string(gzip.open(src / "model.xml.gz", "rt").read())
    env.sim.reset()
    m, sim = env.sim.model._model, env.sim
    st = np.load(src / "states.npz")["states"]

    # how many geoms are collision-only? this is the population that the
    # default renderer draws over the visual meshes.
    ng0 = int((m.geom_group == 0).sum())
    ng1 = int((m.geom_group == 1).sum())

    cam = "robot0_agentview_center"
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    if cid < 0:
        cid = 0
    ren = mujoco.Renderer(m, 240, 320)
    vis = _visual_only(mujoco)
    allg = mujoco.MjvOption()          # MuJoCo default: every group on
    # CONSECUTIVE steps from the middle of the demo, so the flicker
    # statistic sees the same near-static scene the corpus measurement
    # saw. Spreading frames over the whole episode measures motion.
    t0 = max(len(st) // 2 - n // 2, 0)
    ts = np.arange(t0, min(t0 + n, len(st)))

    outs = {}
    for label, opt in (("visual_only", vis), ("all_groups", allg)):
        F, S = [], []
        for t in ts:
            sim.set_state_from_flattened(st[t])
            sim.forward()
            d_ = sim.data._data
            ren.disable_segmentation_rendering()
            ren.update_scene(d_, camera=cid, scene_option=opt)
            F.append(ren.render().copy())
            ren.enable_segmentation_rendering()
            ren.update_scene(d_, camera=cid, scene_option=opt)
            s = ren.render()[..., 0]
            sg = np.zeros_like(s, np.uint16)
            ok = (s >= 0) & (s < m.ngeom)
            sg[ok] = m.geom_bodyid[s[ok]].astype(np.uint16)
            S.append(sg)
            ren.disable_segmentation_rendering()
        F, S = np.stack(F), np.stack(S)
        outs[label] = measure(F)
        outs[label]["flicker"] = flicker_on_static(F, S)
        if out:
            import imageio.v3 as iio
            iio.imwrite(Path(out) / f"{label}.png", F[len(F) // 2])
    ren.close()
    outs["geoms_group0_collision"] = ng0
    outs["geoms_group1_visual"] = ng1
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="CloseCabinet_episode_000016")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    r = run(a.id, a.n, a.out or None)
    print(json.dumps(r, indent=1))
    v, g = r["visual_only"], r["all_groups"]
    print(f"\nflat_sat  visual {v['flat_sat']:.4f}  vs  collision-on "
          f"{g['flat_sat']:.4f}")
    print(f"hues      visual {v['hues']}       vs  collision-on {g['hues']}")
