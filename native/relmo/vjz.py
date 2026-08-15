"""Latent loading. Nothing else.

REVERTED 2026-08-14 on the owner's rule. This module previously trained a
recurrence against hand-written physical targets - openness, articulation type,
rotation, speed, and gripper-relative quantities read from sim state. Every one
of those is ME DESCRIBING THE EVENT and handing the description to the model.
Dropping the gripper-relative half was not a fix; it removed the robot-specific
part of the same violation and left the rest in place.

The rule is: no description of what is happening in the scene may be supplied.
Anything the model knows about an event has to be extracted BY the model, from
latents. So the only inputs that exist here are

    a_t = pred(t+1) - act(t)     V-JEPA expectation, 1024-d
    g_t = [<a,b>, log|b|/|a|]    V-JEPA realised change, as 2 scalars
    sig_t                        SigLIP appearance embedding, 768-d

and `b_t = act(t+1) - act(t)` is consumed here so the barred error term
pred(t+1) - act(t+1) never leaves as a vector - see relmo/vjrank.py for why
that bar has to be structural rather than a policy.

relmo/vjphys.py stays on disk but is no longer imported by anything that
trains. It is diagnostics and oracle measurement only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

REC4 = R.BASE / "vjrec4" / "rcasa_L6"
SIG = R.BASE / "vjsig" / "rcasa"
# stream-time records: `t` is absolute time and T varies with duration, so
# everything downstream of gather() must treat the result as RAGGED
REC6 = R.BASE / "vjrec6" / "rcasa_L6"
SIG6 = R.BASE / "vjsig6" / "rcasa"
CKPT = R.BASE / "models" / "vjz"


def dirs(root="vjrec6", dataset="rcasa", layer=6, suffix=""):
    """-> (rec_dir, sig_dir) for a records generation. v6 = stream time."""
    sig = "vjsig6" if root == "vjrec6" else "vjsig"
    return (R.BASE / root / f"{dataset}_L{layer}{suffix}",
            R.BASE / sig / f"{dataset}{suffix}")


def channels(z, phase=None, primary="a"):
    """npz -> (a_t, g_t). b_t is consumed here and never leaves as a vector.

    `phase` is an optional relmo.vjphase model. Stream-time records carry a
    strong window-position signature (68% of a_t's direction variance,
    measured); subtracting its per-phase mean is a property of the encoder and
    the window geometry, not of any label. Note the a-channel is returned as a
    unit vector when corrected, which loses nothing: every consumer either
    L2-normalises it (the DTW matcher) or LayerNorms it (the ranker), and both
    are invariant to its scale.
    """
    a, b = z["pred_change"].astype(np.float32), z["obs_change"].astype(np.float32)
    if phase is not None and "step" in z:
        from relmo.vjphase import apply as phase_apply
        a, g = phase_apply(a, b, z["step"], phase)
        # phase_apply returns the corrected A as the vector; when b is primary
        # we need its corrected form instead, so recompute it the same way
        if primary == "b":
            from relmo.vjphase import apply as _ap, phase_of, unit
            B = unit(unit(b) - phase["mu_b"][phase_of(z["step"])])
            return B.astype(np.float32), g
        return a, g
    na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
    cos = ((a / na) * (b / nb)).sum(-1, keepdims=True)
    # THE BAR IS SYMMETRIC. pred(t+1)-act(t+1) = a - b is formable by any linear
    # layer that receives BOTH as vectors, so at most one of them may be one.
    # It does not say WHICH: v6 chose `a` by default, and the frozen sweep says
    # that was backwards - b alone outscores a alone (0.442 vs 0.420 prec).
    # `a` is what the model GUESSED and is therefore partly a function of the
    # prior, which is the owner's own objection to the error channel; `b` is
    # purely a function of what happened. Either way g stays two scalars and
    # cannot reconstruct a 1024-d residual.
    g = np.concatenate([cos, np.log(nb / na)], -1).astype(np.float32)
    return (b if primary == "b" else a), g


def gather(ids, dataset="rcasa", want_y=False, rec_dir=None, sig_dir=None,
           phase=None, arc=0.0, primary="a"):
    """{id: dict(a, g, sig)}. `want_y` is accepted and ignored - there are no
    targets any more; the parameter stays so existing callers keep working."""
    rec_dir = rec_dir or REC4
    sig_dir = sig_dir or SIG
    out = {}
    for i in sorted(ids):
        f = rec_dir / f"{i}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        a, g = channels(z, phase, primary)
        s = sig_dir / f"{i}.npz"
        v = np.load(s)["sig"].astype(np.float32) if s.exists() else None
        if arc > 0 and "where_map" in z:
            # re-index by CUMULATIVE CHANGE so a fast and a slow execution of
            # one event emit the same number of steps. All three channels ride
            # the same grid or they stop describing the same instants.
            from relmo.vjmatch import arc_resample
            gate = z["where_map"].reshape(len(a), -1).sum(1)
            aux = [g] + ([v] if v is not None and len(v) == len(a) else [])
            a, rest = arc_resample(a, gate, arc, aux=aux, max_len=256)
            g = rest[0]
            if len(rest) > 1:
                v = rest[1]
            elif v is not None:
                v = None
        out[i] = dict(a=a, g=g, sig=v)
    return out
