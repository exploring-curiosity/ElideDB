"""2D track + depth + camera model -> 3D. One implementation, three sources.

WHY THIS MODULE EXISTS (L1.3, the gate). train_wm2.lift3d built its 3D
from gxy (GT projections of body-fixed points) and gdist (GT planar-z).
Those two jointly ARE the 3D answer, so the world model never had to
recover geometry from anything; and neither field exists at serve time.
Every 3D number this project produced was therefore privileged, and the
cost of being video-only was unmeasured.

The lift is now parameterised by WHERE the two inputs come from:

    gt        gxy   + gdist     fully privileged. The old behaviour, kept
                               as the upper bound to measure against.
    gtdepth   gxy   + ddist_g   GT pixels, PREDICTED depth. Isolates the
                               depth model's error from the tracker's.
    video     xy    + ddist     CoTracker pixels, predicted depth.
                               SERVE-LEGAL: every input is derived from
                               raw video (relmo/preddepth.py).

Only `video` obeys the serve-time rule. The other two exist so a drop
can be attributed instead of guessed.

GEOMETRY. Planar-z unprojection, x = (u - cx) * z / f. That convention
is not assumed: relmo/depthcheck.py casts mujoco.mj_ray through sampled
pixels and settled planar-z vs ray-length against the collision model
(and confirmed the buffer's absolute scale to < 1 cm). y is negated
because MuJoCo's camera looks down -z with +y up while image rows grow
downward.

depth_noise multiplies depth by (1 + eps * N(0,1)) per point per frame.
It exists to answer the question the failure of the video path forces:
HOW ACCURATE must depth be before the relation is recoverable? Injecting
graded error into the privileged path puts a number on the requirement
instead of leaving it to opinion.
"""
from __future__ import annotations

import numpy as np

# (xy field, depth field, visibility field) per mode.
SOURCES = {
    "gt":      ("gxy", "gdist",   "gvis"),
    "gtdepth": ("gxy", "ddist_g", "gvis"),
    "video":   ("xy",  "ddist",   "vis"),
    # video_s: the same, after relmo/depthfix.py removes the per-frame
    # global scale jitter using world-static tracked points as anchors.
    # Serve-legal - it reads only xy, vis and the predicted depth.
    "video_s": ("xy",  "ddist_s", "vis"),
    "gtdepth_s": ("gxy", "ddist_g_s", "gvis"),
    # --- controls, never for training ---
    # leak: tracker pixels with TRUE depth. Isolates the tracker's
    #   contribution, and doubles as the positive control for the gate -
    #   if a privileged field ever leaks back into the video path, the
    #   video arm's score moves toward this one.
    "leak":    ("xy",  "gdist",   "vis"),
    # flat: tracker pixels with depth held at a constant. The lift then
    #   carries NO depth information at all, so it is the floor any
    #   "3D helps" claim must clear.
    "flat":    ("xy",  "ddist",   "vis"),
}
MODES = tuple(SOURCES)
NOISE_KINDS = ("iid", "static", "global")


NOMINAL_FOVY = 60.0    # the middle of the corpus's 45/60/75 spread


def focal(z, fixed_fov=None):
    """Pixels per unit at unit depth.

    SERVE-TIME AUDIT. cam_fovy is a PER-EPISODE sim value on this corpus
    (measured: 45, 60 and 75 degrees appear across rcasa and rcasa_eval),
    so reading it is reading the simulator, not the video - the same
    class of violation as reading gdist. A real deployment calibrates a
    camera once; it does not receive a different true focal length per
    clip. fixed_fov substitutes one nominal calibration for every
    episode, which is what a serve-time system would actually have, and
    lets the cost of that substitution be measured rather than assumed.
    Nothing else in the lift touches the camera: the output is in the
    CAMERA frame, so cam_mat and cam_pos (extrinsics, which are not
    recoverable from a single stream) are never read."""
    W = float(z["width"]) if "width" in z.files else 640.0
    H = float(z["height"]) if "height" in z.files else 480.0
    if fixed_fov:
        return W, H, (H / 2.0) / np.tan(np.deg2rad(fixed_fov) / 2.0)
    if "cam_fovy" in z.files:
        fs = (H / 2.0) / np.tan(np.deg2rad(float(z["cam_fovy"])) / 2.0)
    elif "focal_length" in z.files:
        fs = float(z["focal_length"]) / float(z["sensor_width"]) * W
    else:
        fs = H
    return W, H, fs


def have(z, mode):
    return all(k in z.files for k in SOURCES[mode])


def vis_of(z, mode="gt"):
    """Visibility from the SAME source as the coordinates.

    Mixing them is a silent leak: scoring video-lifted points with gvis
    would hand the model the sim's occlusion answer."""
    return z[SOURCES[mode][2]].astype(bool)


def add_noise(d, eps, kind="iid", seed=0, salt=0):
    """Relative depth error of a given SIZE and a given STRUCTURE.

    Structure is not a detail, it is the finding. Measured on rcasa
    (relmo/relg3.py): iid noise at AbsRel 0.04 costs the relational
    probe MORE than the real depth network costs it at AbsRel 0.22.
    Independent per-frame jitter destroys the temporal consistency that
    every rigidity/velocity feature is built on, while a network's error
    is smooth in space and time - a whole surface is wrong together and
    STAYS wrong the same way, which those features are largely blind to.
    So a single noise model cannot stand in for "depth is X% wrong":

      iid     per point per FRAME   - worst case, breaks temporal consistency
      static  per point, fixed over the window - breaks the shape of the
              scene but keeps every point's error constant in time
      global  per frame, one scale for the whole scene - pure scale error,
              the failure monocular depth is known to have
    """
    # SALT IS NOT OPTIONAL. Without it every episode drew from
    # default_rng(seed) and got the IDENTICAL noise array - measured,
    # 100% identical values between two different episodes for iid,
    # static and global alike. Shared noise is not noise: it is a fixed
    # per-point pattern the probe can learn, and it silently corrupted
    # the first version of this sweep. Caught by an impossible result -
    # injected noise RAISING the AUC above the clean privileged arm.
    rng = np.random.default_rng((int(seed), int(salt)))
    if eps <= 0:
        return d
    if kind == "iid":
        n = rng.standard_normal(d.shape)
    elif kind == "static":
        n = rng.standard_normal((1,) + d.shape[1:])
    elif kind == "global":
        n = rng.standard_normal((d.shape[0],) + (1,) * (d.ndim - 1))
    elif kind == "drift":
        # THE CELL THE FIRST SWEEP MISSED, and the one that matches the
        # real predictor. static/iid/global are the corners: constant in
        # time, white in time, or shared by the whole frame. depthnet's
        # actual error is none of these - measured on held-out episodes
        # its per-point time-varying part has std 0.119, lag-1
        # autocorrelation 0.686 and 56% of its variance in a quadratic
        # trend. That is SLOW PER-POINT DRIFT: each surface's depth
        # wanders on its own, smoothly. Synthesised here as a low
        # frequency Fourier sum per point, normalised to unit temporal
        # variance so eps means the same thing as in the other arms.
        T = d.shape[0]
        t = np.arange(T, dtype=np.float64) / max(T - 1, 1)
        sh = (1,) + d.shape[1:]
        n = np.zeros_like(d, dtype=np.float64)
        for k in (1, 2):
            n += (rng.standard_normal(sh)
                  * np.sin(2 * np.pi * k * t.reshape((-1,) + (1,) * (d.ndim - 1))
                           + rng.uniform(0, 2 * np.pi, sh)))
        s = n.std(0, keepdims=True)
        n = n / np.maximum(s, 1e-9)
    else:
        raise ValueError(f"unknown noise kind {kind!r}")
    return np.clip(d * (1.0 + eps * n).astype(np.float32), 0.02, None)


def smooth_depth(d, k):
    """Centred moving MEDIAN of width k along time, per point.

    Serve-legal: it touches only the predicted depth track, no GT. It is
    here because of what the noise sweep measured — a per-point depth
    error that is constant over a window costs the relational probe
    almost nothing even at 27% relative error, while frame-to-frame
    jitter costs it 0.13-0.16 AUC at a quarter of that size. depthnet's
    error is 0.54 static bias and 0.24 temporal jitter, i.e. mostly the
    harmless kind plus enough of the harmful kind to matter. Removing
    the jitter is therefore the cheap intervention, and it needs no
    better depth model.

    Median rather than mean: the failure is spikes as a point crosses a
    depth discontinuity, and a mean spreads a spike over the window
    instead of rejecting it."""
    if k < 3:
        return d
    T = d.shape[0]
    r = k // 2
    idx = np.clip(np.arange(T)[:, None] + np.arange(-r, r + 1)[None, :],
                  0, T - 1)
    return np.median(d[idx], axis=1).astype(d.dtype)


def lift3d(z, mode="gt", depth_noise=0.0, seed=0, key=None,
           noise_kind="iid", depth_smooth=0, fixed_fov=None, salt=0):
    """(T,P,3) camera-frame XYZ. Camera frame, not world: it is the only
    frame available on real video, and it stops the model memorising a
    world origin that will not exist in production."""
    if key is not None:          # legacy call sites passed key="g"
        mode = "gt" if key == "g" else mode
    kxy, kd, _ = SOURCES[mode]
    if kd not in z.files:
        raise KeyError(f"lift mode {mode!r} needs {kd!r}; run "
                       f"`python -m relmo.preddepth` for this dataset")
    xy = z[kxy].astype(np.float32)
    d = z[kd].astype(np.float32)
    if mode == "flat":
        # control: constant depth. Every point sits on one fronto-
        # parallel plane, so the "3D" is the 2D track scaled - it holds
        # no depth information whatsoever.
        d = np.ones_like(d)
    d = add_noise(d, depth_noise, noise_kind, seed, salt)
    d = smooth_depth(d, depth_smooth)
    W, H, fs = focal(z, fixed_fov)
    x = (xy[..., 0] - W / 2.0) * d / fs
    y = -(xy[..., 1] - H / 2.0) * d / fs
    return np.stack([x, y, d], -1)
