"""Autonomous pipeline driver.

Owner is unavailable (2026-08-12): "start autonomous work to proceed
without my directions one after other... validate once before training
and begin everything autonomously."

So this is a state machine over ARTIFACTS, not a script of steps. It
looks at what exists on disk, decides the single next thing to do, and
does it. Safe to run repeatedly - every stage is idempotent and skips
if its output already exists. If a stage's GATE fails, it STOPS and
records why rather than training on data it has not validated.

    python -m relmo.pipeline --step     # advance one stage
    python -m relmo.pipeline --status   # what would happen next
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import daemon as D  # noqa: E402
from relmo import registry as R  # noqa: E402

NATIVE = Path(__file__).resolve().parents[1]
TARGET_RCASA = 500          # enough to start; generation continues
GATE = R.ROOT / "data" / "relmo" / "gates.json"


def _n_eps(ds):
    try:
        return R.read_manifest(ds).get("n_episodes", 0)
    except Exception:
        return 0


def _n_tracks(ds):
    d = R.TRACKS / ds
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def gates():
    return json.loads(GATE.read_text()) if GATE.exists() else {}


def set_gate(k, v):
    g = gates()
    g[k] = v
    GATE.write_text(json.dumps(g, indent=1))
    R.log("gate", name=k, **(v if isinstance(v, dict) else {"value": v}))


def state():
    """What exists right now."""
    return dict(
        rcasa_eps=_n_eps("rcasa_v1"), rcasa_tracks=_n_tracks("rcasa_v1"),
        movi_tracks=_n_tracks("movi_e"), phys_tracks=_n_tracks("physgen_v3"),
        running={j: D.alive(j) for j in ("rcgen", "tracks2", "train_wm2")
                 if D.alive(j)},
        gates=gates())


def next_step(s):
    """One decision, in dependency order."""
    if s["running"]:
        return ("wait", f"running: {list(s['running'])}")
    if s["rcasa_eps"] < TARGET_RCASA:
        return ("generate", f"rcasa {s['rcasa_eps']}/{TARGET_RCASA}")
    if s["rcasa_tracks"] < s["rcasa_eps"] * 0.9:
        return ("track", f"rcasa tracks {s['rcasa_tracks']}/{s['rcasa_eps']}")
    if "G0" not in s["gates"]:
        return ("gate_g0", "learnability of the 3D targets")
    if not s["gates"]["G0"].get("passed"):
        return ("halt", f"G0 FAILED: {s['gates']['G0']}")
    return ("train", "all gates clear")


def do_generate():
    D.spawn("rcgen", ["--name", "rcasa_v1", "--episodes",
                      str(TARGET_RCASA + 100), "--T", "120"])
    return "rcgen launched"


def do_track():
    """Launch tracking, but refuse to relaunch a stage that already ran
    and produced nothing.

    Measured the hard way: tracks2 swallowed 609 per-episode exceptions
    (a missing MPS kernel), reported 600/600, and wrote zero caches - so
    the driver cheerfully relaunched it forever. A stage that has run
    and produced no artifact is a HALT, not a retry."""
    g = gates()
    tried = g.get("track_attempts", 0)
    if tried >= 2 and _n_tracks("rcasa_v1") == 0:
        set_gate("track_attempts", tried)
        return ("HALT: tracks2 ran %d times and produced 0 caches - "
                "check ledger for track2_error" % tried)
    set_gate("track_attempts", tried + 1)
    D.spawn("tracks2", ["--name", "rcasa_v1"])
    return "tracks2 launched on rcasa_v1 (attempt %d)" % (tried + 1)


def do_gate_g0(n=200, dataset="rcasa_v1"):
    """G0: is the 3D target learnable AT ALL?

    Const-velocity extrapolation on the GT tracks, scored exactly the
    way the trainer scores itself: motion-R2 over DISPLACEMENT from the
    last context frame, on moving points only. If this is not positive
    at h=1 the target is noise and no model can win - this project lost
    12k steps to exactly that, so the gate runs BEFORE training.

    Tests the 3D target, not 2D: the model predicts 3D (train_wm2
    lifts via gdist + the camera model), and a gate that validates a
    different quantity than the model consumes proves nothing.
    """
    from relmo import splits as SP
    from relmo.train_wm2 import lift3d, TC, HZ
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    if len(files) < 20:
        return "not enough tracks for G0"
    tr = SP.partition(files, dataset)[SP.TRAIN][:n]
    sse = np.zeros(HZ)
    sst = np.zeros(HZ)
    sse_still = np.zeros(HZ)
    rng = np.random.default_rng(0)
    used = 0
    for f in tr:
        z = np.load(f)
        if "gdist" not in z.files:
            continue
        X = lift3d(z)
        V = z["gvis"]
        T = X.shape[0]
        if T < TC + HZ:
            continue
        t0 = int(rng.integers(0, T - TC - HZ + 1))
        X, V = X[t0:t0 + TC + HZ], V[t0:t0 + TC + HZ]
        s = max(np.linalg.norm(
            X[:TC] - X[:TC].mean((0, 1)), axis=-1).mean(), 1e-6)
        Xn = (X - X[:TC].mean((0, 1))) / s
        base = Xn[TC - 1]
        tgt = Xn[TC:]                                  # (H,P,3)
        m = V[TC:] & V[TC - 1][None]
        disp = tgt - base[None]
        mag = np.linalg.norm(disp, axis=-1)
        if not m.any():
            continue
        thr = max(mag[m].mean(), 1e-6) * 0.5
        mv = m & (mag > thr)
        if not mv.any():
            continue
        used += 1
        v = Xn[TC - 1] - Xn[TC - 2]
        for h in range(HZ):
            sel = mv[h]
            if not sel.any():
                continue
            d_true = disp[h][sel]
            d_cv = (v * (h + 1))[sel]
            sse[h] += ((d_cv - d_true) ** 2).sum()
            sse_still[h] += (d_true ** 2).sum()
            sst[h] += (d_true ** 2).sum()
    r2 = [float(1 - sse[h] / max(sst[h], 1e-9)) for h in range(HZ)]
    r2_still = [float(1 - sse_still[h] / max(sst[h], 1e-9))
                for h in range(HZ)]
    rep = dict(dataset=dataset, episodes_used=used,
               const_vel_r2=[round(x, 4) for x in r2],
               stillness_r2=[round(x, 4) for x in r2_still],
               h1=round(r2[0], 4), passed=bool(r2[0] > 0))
    set_gate("G0", rep)
    return ("G0 h1_const_vel_r2=%.4f stillness=%.4f passed=%s (n=%d)"
            % (rep["h1"], r2_still[0], rep["passed"], used))


# The driver only consults G0, so on its own it would relaunch whatever
# config it launched last. Two configs have now been MEASURED to fail G4
# 0/8 (run A readout=mean pooled 0.082; run B readout=last+mean pooled
# 0.124 peak, both horizon-flat), so the launch config is pinned here and
# bumped deliberately when a hypothesis is refuted - never re-run a
# refuted arm just because the state machine sees an idle slot.
RUN = dict(run="relmowm3_fdnn", trainer="train_wm3", head="fdnn",
           datasets="rcasa_v1,arctic_v1", steps="40000", enabled=True,
           why="v3: per-point displacement (PointWorld-style), annealed-WTA "
               "modes, GNS noise, slots demoted to auxiliary. v2's slot "
               "latents measured pairwise-cos 0.9796 - one global rigid "
               "motion for the whole scene - and its own oracle ceiling "
               "(0.826 @ h=1) sat below const-velocity (0.957).")


def do_train():
    """Launch the pinned arm - or refuse, if none is pinned.

    The driver only consults G0, so left alone it relaunches whatever it
    launched last. It has already tried to resume run C twice after that
    run was deliberately stopped. Three arms are now MEASURED failures of
    G4 (A 0.082, B 0.124 peak, C 0.040 falling to 0.016, all 0/8), so an
    idle training slot is not a reason to spend four hours re-running
    one. enabled=False makes the state machine report and wait."""
    if not RUN.get("enabled"):
        return ("NO ARM PINNED - %s. Set RUN['enabled']=True with a new "
                "run id once the next hypothesis is testable." % RUN["why"])
    D.spawn(RUN["trainer"], ["--datasets", RUN["datasets"],
                             "--steps", RUN["steps"],
                             "--head", RUN["head"], "--run", RUN["run"]])
    return "train_wm2 launched: %s (%s)" % (RUN["run"], RUN["why"])


ACTIONS = dict(generate=do_generate, track=do_track,
               gate_g0=do_gate_g0, train=do_train)


def step():
    s = state()
    what, why = next_step(s)
    R.log("pipeline", step=what, why=why, **{k: v for k, v in s.items()
                                             if k != "gates"})
    if what in ("wait", "halt"):
        return f"{what}: {why}"
    return f"{what}: {why} -> {ACTIONS[what]()}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    s = state()
    if a.status or not a.step:
        print(json.dumps(dict(state={k: v for k, v in s.items()},
                              next=next_step(s)), indent=1, default=str))
    else:
        print(step())
