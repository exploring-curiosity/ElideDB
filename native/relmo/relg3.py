"""L1.3 — what does the relational target cost when the 3D comes from VIDEO?

relg0 established the target is learnable from pixels: a linear probe on
2D CoTracker features scores AUC 0.816 held-out against an oracle
ceiling of 0.997. Every 3D number beside it, though, was lifted through
gxy + gdist — GT projections and GT depth, which jointly ARE the answer
and do not exist at serve time.

This measures the delta. Same windows, same features, same probe; only
the source of the 3D changes:

    gt        gxy + gdist      privileged upper bound
    gtdepth   gxy + ddist_g    predicted depth, perfect pixels
                               -> isolates the DEPTH model's error
    video     xy  + ddist      predicted depth, tracker pixels
                               -> serve-legal; depth + tracker error

and the 2D probe runs on the identical window set as the reference, so
"does 3D help at all" and "what does video-only cost" are separate,
answerable questions rather than one confounded number.

THE NOISE SWEEP. The depth model is weak — measured on held-out rcasa,
AbsRel 0.333 and only 56% of points within 25% of true depth, against
~0.05 / 98% for published monocular depth. So a bad video result would
be ambiguous: is video-only geometry hopeless, or is THIS predictor bad?
Injecting graded relative noise into the PRIVILEGED depth answers it —
it draws the curve of probe AUC against depth error, and the point where
that curve leaves the privileged value is the accuracy the downstream
task actually requires.

Read the sweep as a LOWER BOUND on the requirement, not an equivalence.
Injected noise is iid per point per frame, so a 100-point moving set
averages most of it away. A depth network's error is spatially and
temporally correlated — a whole surface is wrong together, and no
averaging removes that. If the video arm sits well below the iid arm at
the same AbsRel, the structure of the error is the problem, not its size.

    python -m relmo.relg3 --dataset rcasa
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import lift as LF  # noqa: E402
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.lift import SOURCES  # noqa: E402
from relmo.relg0 import fit_probe, pixel_feats  # noqa: E402
from relmo.relsep import auc, rel_feats  # noqa: E402

MOVE_Q = 0.75          # same "moving point" rule as relg0, so the moving
                       # sets are comparable between the 2D and 3D probes
NPAIR = 300

F2 = ("motion_contrast", "rigidity", "rot_ratio", "decel", "coherence",
      "path_curv", "spread_change")
F3 = ("m3_contrast", "m3_rigidity", "m3_rot", "m3_decel", "m3_coherence",
      "m3_curv", "m3_spread", "m3_arc", "m3_flat", "m3_depthfrac",
      "m3_net_m", "m3_size_m")

# CONTACT GEOMETRY - the gap between the moving set and everything else.
# Separate from F3 because it is the feature family the contact question
# lives or dies on, and because this project has a standing measurement
# that the IMAGE-PLANE version of it is worthless: the box gap between
# two bodies measured 0.00 median for touching AND separated pairs alike,
# because a projection puts a near hand and a far object at the same
# apparent distance. If depth is ever load-bearing anywhere, it is here.
FC = ("c3_gap_min", "c3_gap_rel", "c3_gap_start", "c3_gap_end",
      "c3_gap_ratio", "c3_near_frac", "c3_gap_dz")
NGAP_M, NGAP_S, NGAP_T = 60, 150, 6      # subsamples for the gap search


# 2D CO-MOTION. Contact is a millimetre-scale geometric fact - measured,
# the per-frame 3D displacement that carries it is 4.55 mm at 731 mm
# range, while even SOTA monocular depth errs by 21.9 mm there. So it
# cannot be MEASURED from monocular depth. But contact has a kinematic
# signature that needs no depth at all: bodies in contact move TOGETHER,
# and contact change is the moment that stops being true. Co-motion is
# an image-plane property, so it is available exactly where depth is not.
F2C = ("cm_rigid_a", "cm_rigid_b", "cm_rigid_chg", "cm_link_var",
       "cm_link_trend", "cm_grp_chg", "cm_vel_corr")


def comotion(z, t0, W):
    """2D-only kinematic coupling between the mover and the rest.

    Deliberately NOT a distance: an image-plane gap was measured at 0.00
    median for touching AND separated pairs alike. This asks whether two
    point sets move as one, which a projection preserves."""
    xy, vs = z["xy"].astype(np.float64), z["vis"].astype(bool)
    T = len(xy)
    a, b = t0, min(t0 + W, T)
    if b - a < 8:
        return None
    X, V = xy[a:b], vs[a:b]
    ok = V.all(0)
    if ok.sum() < 20:
        return None
    X = X[:, ok]
    d = np.linalg.norm(X[-1] - X[0], axis=-1)
    thr = np.quantile(d, MOVE_Q)
    mv = d >= max(thr, 1e-9)
    if mv.sum() < 4 or (~mv).sum() < 4:
        return None
    M, S = X[:, mv], X[:, ~mv]
    h = len(X) // 2

    def rigid_frac(P):
        """1 - residual/total of a single AFFINE fit to the motion field.

        One body moving rigidly is explained by one affine map; two
        bodies moving differently are not. This is the quantity that
        should change when contact forms or breaks."""
        if len(P) < 3:
            return 0.0
        dv = P[-1] - P[0]
        A = np.concatenate([P[0], np.ones((len(P[0]), 1))], 1)
        coef, *_ = np.linalg.lstsq(A, dv, rcond=None)
        res = dv - A @ coef
        tot = ((dv - dv.mean(0)) ** 2).sum()
        return float(1.0 - (res ** 2).sum() / (tot + 1e-12))

    # mean 2D distance between the two sets, over time: constant while
    # they are locked together, moving when they are not
    mi = np.linalg.norm(M[:, :, None, :] - S[:, None, :, :], axis=-1)
    link = mi.mean((1, 2))
    sc = np.linalg.norm(M[0] - M[0].mean(0), axis=-1).mean() + 1e-9
    # does the moving/static PARTITION itself change between halves?
    da = np.linalg.norm(X[h] - X[0], axis=-1)
    db = np.linalg.norm(X[-1] - X[h], axis=-1)
    ga = da >= max(np.quantile(da, MOVE_Q), 1e-9)
    gb = db >= max(np.quantile(db, MOVE_Q), 1e-9)
    vm = np.diff(M.mean(1), axis=0)
    vsx = np.diff(S.mean(1), axis=0)
    nm = np.linalg.norm(vm, axis=-1) + 1e-12
    ns = np.linalg.norm(vsx, axis=-1) + 1e-12
    return dict(
        cm_rigid_a=rigid_frac(X[:h + 1]), cm_rigid_b=rigid_frac(X[h:]),
        cm_rigid_chg=abs(rigid_frac(X[:h + 1]) - rigid_frac(X[h:])),
        cm_link_var=float(link.std() / sc),
        cm_link_trend=float((link[-1] + 1e-9) / (link[0] + 1e-9)),
        cm_grp_chg=float((ga != gb).mean()),
        cm_vel_corr=float(((vm * vsx).sum(-1) / (nm * ns)).mean()),
    )


def feat3(X, V, t0, W, rng):
    """3D analogues of relg0's features, plus what only 3D can express.

    The first seven mirror the 2D set exactly, so any difference is the
    lift and not the feature design. The last five have no 2D form:

      m3_arc        second/first eigenvalue of the centroid path — a
                    hinge sweeps an arc, a carry runs straight
      m3_flat       third/first — is that path planar (a hinge is)
      m3_depthfrac  share of the net motion along the optical axis, the
                    component a projection destroys
      m3_net_m      net displacement in METRES
      m3_size_m     the moving set's extent in METRES

    The last two are the only absolutely-scaled features here. That is
    deliberate: monocular depth loses global scale before it loses shape,
    so keeping scale-dependent features in the set lets the sweep show
    WHICH kind of feature dies first."""
    T = len(X)
    a, b = t0, min(t0 + W, T)
    if b - a < 4:
        return None
    Xw, Vw = X[a:b].astype(np.float64), V[a:b]
    ok = (Vw.all(0) & np.isfinite(Xw).all((0, 2))
          & (Xw[..., 2] > 0.05).all(0) & (Xw[..., 2] < 20.0).all(0))
    if ok.sum() < 12:
        return None
    M0 = Xw[:, ok]
    d = np.linalg.norm(M0[-1] - M0[0], axis=-1)
    thr = np.quantile(d, MOVE_Q)
    mv = d >= max(thr, 1e-9)
    if mv.sum() < 4:
        return None
    M = M0[:, mv]                                       # (W,P,3) metres
    cen = M.mean(1)
    step = np.linalg.norm(np.diff(cen, axis=0), axis=-1)
    net = float(np.linalg.norm(cen[-1] - cen[0]))
    path = float(step.sum()) + 1e-12
    # rigidity: a rigid body holds its internal 3D distances EXACTLY,
    # whatever it does. Its 2D projection does not - which is the single
    # clearest reason to want the lift at all.
    i0, i1 = np.triu_indices(M.shape[1], 1)
    if len(i0) > NPAIR:
        sel = rng.choice(len(i0), NPAIR, replace=False)
        i0, i1 = i0[sel], i1[sel]
    pd = np.linalg.norm(M[:, i0] - M[:, i1], axis=-1)
    rig = float(np.median(pd.std(0) / (pd.mean(0) + 1e-9)))
    # principal axis rotation, sign-free (an eigenvector's sign is
    # arbitrary, so a flipped axis must not read as a 180 deg turn)
    ax = []
    for t in (0, len(M) - 1):
        c = M[t] - M[t].mean(0)
        _, vec = np.linalg.eigh(c.T @ c)
        ax.append(vec[:, -1])
    dang = float(np.arccos(np.clip(abs(float(ax[0] @ ax[1])), 0, 1)))
    spread = float(np.linalg.norm(M[0] - M[0].mean(0), axis=-1).mean())
    h = max(len(step) // 3, 1)
    dv = M[-1] - M[0]
    dvn = dv / (np.linalg.norm(dv, axis=-1, keepdims=True) + 1e-12)
    s1 = float(np.linalg.norm(M[-1] - M[-1].mean(0), axis=-1).mean())
    rest = d[~mv]
    cc = cen - cen.mean(0)
    lam = np.sort(np.linalg.eigvalsh(cc.T @ cc))[::-1]
    lam = np.maximum(lam, 0.0)
    # ---- contact geometry: how close does the mover get to the rest? ----
    # The moving set is the thing being manipulated (or the manipulator);
    # the static points are the scene. Contact is a statement about the
    # distance between them, and it is the one quantity a projection is
    # known to destroy. Computed on a subsample of points and frames -
    # the full cross product is 30k distances per frame and adds nothing.
    S = M0[:, ~mv]
    gap = dict.fromkeys(FC, 0.0)
    if S.shape[1] >= 4:
        mi = rng.choice(M.shape[1], min(NGAP_M, M.shape[1]), replace=False)
        si = rng.choice(S.shape[1], min(NGAP_S, S.shape[1]), replace=False)
        ts = np.unique(np.linspace(0, len(M) - 1, NGAP_T).astype(int))
        A, B = M[np.ix_(ts, mi)], S[np.ix_(ts, si)]
        D = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1)
        per_t = D.min((1, 2))
        # the same gap computed on the DEPTH axis alone, which is exactly
        # the component the image plane cannot see
        Dz = np.abs(A[..., None, 2] - B[:, None, :, 2])
        gap = dict(
            c3_gap_min=float(per_t.min()),
            c3_gap_rel=float(per_t.min() / (spread + 1e-9)),
            c3_gap_start=float(per_t[0]),
            c3_gap_end=float(per_t[-1]),
            c3_gap_ratio=float((per_t[-1] + 1e-6) / (per_t[0] + 1e-6)),
            c3_near_frac=float((D < max(spread, 1e-6)).mean()),
            c3_gap_dz=float(Dz.min((1, 2)).min()),
        )
    return dict(**gap, **dict(
        m3_contrast=float(d[mv].mean() / (rest.mean() + 1e-9))
        if rest.size else 1.0,
        m3_rigidity=rig,
        m3_rot=float(dang * spread / (net + 1e-9)),
        m3_decel=float((step[:h].mean() + 1e-12) / (step[-h:].mean() + 1e-12)),
        m3_coherence=float(np.linalg.norm(dvn.mean(0))),
        m3_curv=float(net / path),
        m3_spread=float(s1 / (spread + 1e-9)),
        m3_arc=float(lam[1] / (lam[0] + 1e-15)),
        m3_flat=float(lam[2] / (lam[0] + 1e-15)),
        m3_depthfrac=float(abs(cen[-1, 2] - cen[0, 2]) / (net + 1e-9)),
        m3_net_m=net,
        m3_size_m=spread,
    ))


def depth_stats(pred, true, ok):
    """The standard monocular-depth report, at the tracked points.

    AbsRel and d1 rather than R2 alone: R2 is dominated by the scene's
    depth spread, so a model can post a respectable R2 while every point
    is 30% wrong. d1 (the share within a 1.25x factor) is the number the
    depth literature is judged on."""
    p, y = pred[ok].astype(np.float64), true[ok].astype(np.float64)
    if len(p) < 100:
        return None
    rel = np.abs(p - y) / np.maximum(y, 1e-6)
    rat = np.maximum(p / np.maximum(y, 1e-6), np.maximum(y, 1e-6) / p)
    return dict(absrel=float(rel.mean()), d1=float((rat < 1.25).mean()),
                med_err_m=float(np.median(np.abs(p - y))),
                r2=float(1 - ((p - y) ** 2).sum()
                         / (((y - y.mean()) ** 2).sum() + 1e-12)))


def build_arms(sweep_eps=(), seeds=(), kinds=(), smooth=()):
    """name -> lift3d kwargs.

    leak and flat are POSITIVE CONTROLS for this measurement, not
    results. leak feeds the video path TRUE depth: if the video arm ever
    reads near it, a privileged field has leaked back in. flat feeds it
    constant depth, so nothing the 3D features report can come from
    depth - the floor any "3D helps" claim must clear.

    video_fov is the SERVE-TIME AUDIT arm. The other video arms read
    cam_fovy, which varies per episode on this corpus (45/60/75 deg) and
    is therefore a simulator value, not a video one. This arm gives every
    episode the same nominal calibration, which is what a deployed system
    has, and measures what that substitution costs."""
    arms = {m: dict(mode=m) for m in ("gt", "gtdepth", "video", "video_s",
                                      "gtdepth_s", "leak", "flat")}
    arms["video_fov"] = dict(mode="video", fixed_fov=LF.NOMINAL_FOVY)
    arms["gt_fov"] = dict(mode="gt", fixed_fov=LF.NOMINAL_FOVY)
    for k in smooth:
        arms[f"video_sm{k}"] = dict(mode="video", depth_smooth=k)
    for k in kinds:
        for e in sweep_eps:
            for sd in seeds:
                arms[f"{k}{e:g}_s{sd}"] = dict(mode="gt", depth_noise=e,
                                               seed=sd, noise_kind=k)
    return arms


def boot_delta(sa, sb, y, ids, n=2000, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b), RESAMPLING EPISODES.

    Windows from one episode are not independent - six windows of the
    same drawer share a scene, a camera and a label. Resampling windows
    would report an interval several times too narrow. Resampling whole
    episodes respects the clustering, and pairing (both arms scored on
    the identical resample) removes the between-episode variance that
    both arms share, which is most of it."""
    rng = np.random.default_rng(seed)
    uid = np.unique(ids)
    byep = [np.where(ids == u)[0] for u in uid]        # once, not per draw
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(uid), len(uid))
        m = np.concatenate([byep[i] for i in pick])
        yy = y[m]
        if yy.sum() < 2 or (1 - yy).sum() < 2:
            continue
        d.append(auc(sa[m][yy > .5], sa[m][yy < .5])
                 - auc(sb[m][yy > .5], sb[m][yy < .5]))
    d = np.array(d)
    return float(d.mean()), float(np.quantile(d, 0.025)), \
        float(np.quantile(d, 0.975))


def collect(dataset, W, arms):
    from tqdm import tqdm
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    which = {f.stem: nm for nm, key in (("train", SP.TRAIN), ("val", SP.VAL),
                                        ("test", SP.TEST))
             for f in part[key]}
    rows, dstat = [], {k: [] for k in arms}
    rng = np.random.default_rng(0)
    for f in tqdm(sorted(files), unit="ep", desc=f"relg3/{dataset}"):
        if f.stem not in by:
            continue
        z = np.load(f)
        need = ("contact_pairs", "window_starts", "ddist", "ddist_g",
                "ddist_s", "ddist_g_s")
        if any(k not in z.files for k in need):
            continue
        Xs = {}
        # per-episode salt so injected noise is INDEPENDENT across
        # episodes; without it every episode gets the same array
        salt = int(hashlib.sha256(f.stem.encode()).hexdigest()[:8], 16)
        for nm, kw in arms.items():
            Xs[nm] = (LF.lift3d(z, salt=salt, **kw),
                      LF.vis_of(z, kw["mode"]))
        gd, gv = z["gdist"], z["gvis"].astype(bool)
        okd = gv & (gd > 0.05) & (gd < 20)
        for nm, kw in arms.items():
            mode = kw["mode"]
            # Every arm's depth error is measured against gdist — the
            # true depth of the point the track is SUPPOSED to be on —
            # so noise levels and the real predictor land on one
            # comparable AbsRel axis. For the video arm this deliberately
            # charges tracker drift to the depth error: if the tracker
            # slides onto another surface, the lifted 3D really is wrong
            # by that amount, and pretending otherwise would flatter it.
            ok = (okd & z["vis"].astype(bool)
                  if SOURCES[mode][0] == "xy" else okd)
            s = depth_stats(Xs[nm][0][..., 2], gd, ok)
            if s:
                dstat[nm].append(s)
        for t0 in z["window_starts"]:
            t0 = int(t0)
            px = pixel_feats(z, t0, W)
            if px is None:
                continue
            cm = comotion(z, t0, W)
            if cm is None:
                continue
            px = dict(px, **cm)
            f3 = {}
            bad = False
            for nm, (X, V) in Xs.items():
                # a FRESH seeded rng per arm: the pair subsample used by
                # the rigidity feature must not differ between arms for
                # the incidental reason that one arm was evaluated second
                v = feat3(X, V, t0, W, np.random.default_rng(t0 + 1))
                if v is None:
                    bad = True
                    break
                f3[nm] = v
            if bad:
                # a window is kept only if EVERY arm produces features.
                # Different arms scored on different windows would make
                # the deltas meaningless.
                continue
            lab = rel_feats(z, t0, W)                     # ORACLE, label only
            # THREE targets, one window set. Articulation was the L1.3
            # target; contact is the one the prior image-plane finding
            # says depth should be required for, and the two must be
            # scored on identical windows or the comparison is not a
            # comparison.
            rows.append(dict(
                id=f.stem, split=which.get(f.stem, "train"), px=px, f3=f3,
                y_articulation=float(lab["articulation"] > 0.05),
                # grip_frac > 0.5 was tried first and is DEGENERATE on a
                # manipulation corpus: the robot touches the target in
                # 81% of windows, so the label is nearly constant and the
                # probe scores it at ~0.5 whatever the features. third
                # contact - the target resting on / touching a NON-robot
                # body - is 41% positive and is the state question that
                # actually distinguishes held from placed.
                y_contact_state=float(lab["third_frac"] > 0.5),
                y_grip_state=float(lab["grip_frac"] > 0.5),
                y_contact_change=float(lab["contact_changes"] > 0.0)))
    return rows, dstat


def depthnet_unseen(dataset, train_ep=60):
    """Episodes depthnet never trained on.

    depthnet.load_split takes partition(sorted(files))[TRAIN][:train_ep],
    so the set is reconstructible exactly. It matters: predicted depth on
    an episode depthnet memorised is not the predicted depth the system
    would have at serve, and pooling those windows into an evaluation
    would flatter the video arm by an unknown amount. Anything not in
    this set is excluded from CV scoring."""
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    seen = {f.stem for f in SP.partition(files, dataset)[SP.TRAIN][:train_ep]}
    return {f.stem for f in files} - seen


def score_cv(rows, feats, target, arm=None, folds=5, eval_ids=None, seed=0):
    """Grouped K-fold by EPISODE -> pooled out-of-fold scores.

    The fixed 80/10/10 split leaves 26-28 held-out episodes, and at that
    size the paired interval on a 0.07 AUC gap straddles zero — the same
    run scored -0.082 [-0.155,-0.020] on one window set and -0.066
    [-0.145,+0.014] on another. Grouped CV scores EVERY eligible episode
    out of fold instead of 30% of them, which is the only honest way to
    buy power here: no episode is ever in both the fit and the score, so
    it is not a relaxation of the held-out rule."""
    elig = np.array(sorted(eval_ids if eval_ids is not None
                           else {r["id"] for r in rows}))
    rng = np.random.default_rng(seed)
    fold = {e: i for i, e in enumerate(rng.permutation(elig))}

    def flat(r):
        d = dict(y=r[target])
        d.update(r["px"])
        if arm:
            d.update(r["f3"][arm])
        return d
    S, Y, I = [], [], []
    for k in range(folds):
        hold = {e for e, i in fold.items() if i % folds == k}
        te = [r for r in rows if r["id"] in hold]
        tr = [r for r in rows if r["id"] not in hold]
        if not te or len({r[target] for r in tr}) < 2:
            continue
        _, sc, y = fit_probe([flat(r) for r in tr], [flat(r) for r in te],
                             feats, return_scores=True)
        S.append(sc)
        Y.append(y)
        I.append(np.array([r["id"] for r in te]))
    S, Y, I = np.concatenate(S), np.concatenate(Y), np.concatenate(I)
    return auc(S[Y > .5], S[Y < .5]), S, Y, I


TARGETS = ("y_articulation", "y_contact_state", "y_contact_change")
CORE = ("gt", "leak", "flat", "video", "video_s", "gtdepth", "gtdepth_s",
        "video_fov", "gt_fov")


def report_target(rows, target, ev, feats3, tag):
    """One arm table + the paired contrasts, for a single target."""
    a2, s2, y2, ids = score_cv(rows, F2, target, None, 5, ev)
    npos = int(sum(r[target] for r in rows if r["id"] in ev))
    print(f"\n=== TARGET {target}  ({npos} positive of "
          f"{sum(1 for r in rows if r['id'] in ev)} scorable windows, "
          f"{len(ev)} episodes) ===")
    print(f"  2D only  AUC {a2:.3f}          [features: {tag}]")
    print("  {:11s} {:>8s} {:>10s}".format("arm", "AUC 3D", "AUC 2D+3D"))
    S3, S23 = {}, {}
    for nm in CORE:
        u3, s3, y3, _ = score_cv(rows, list(feats3), target, nm, 5, ev)
        u23, s23, _, _ = score_cv(rows, list(F2) + list(feats3), target,
                                  nm, 5, ev)
        S3[nm], S23[nm] = s3, s23
        print("  {:11s} {:8.3f} {:10.3f}".format(nm, u3, u23))
    out = {"auc_2d": round(a2, 4),
           "auc_3d": {k: round(float(auc(S3[k][y2 > .5], S3[k][y2 < .5])), 4)
                      for k in S3},
           "auc_2d3d": {k: round(float(auc(S23[k][y2 > .5], S23[k][y2 < .5])),
                                 4) for k in S23},
           "contrasts": {}}
    print("  contrasts (paired episode bootstrap, 95% CI):")
    for lab, sa, sb in (
            ("privileged 3D vs 2D-only", S3["gt"], s2),
            ("privileged 3D vs flat (NO depth)", S3["gt"], S3["flat"]),
            ("video 3D vs privileged 3D", S3["video"], S3["gt"]),
            ("video 3D vs flat (NO depth)", S3["video"], S3["flat"]),
            ("leak (true depth, tracker px) vs privileged",
             S3["leak"], S3["gt"]),
            ("video_fov (nominal fovy) vs video (sim fovy)",
             S3["video_fov"], S3["video"]),
            ("video_s (scale-stabilised) vs video", S3["video_s"],
             S3["video"]),
            ("video_s (scale-stabilised) vs privileged 3D", S3["video_s"],
             S3["gt"]),
            ("gtdepth_s (stabilised) vs gtdepth", S3["gtdepth_s"],
             S3["gtdepth"]),
            ("gt_fov (nominal fovy) vs gt (sim fovy)",
             S3["gt_fov"], S3["gt"]),
            ("2D+3D gt vs 2D-only", S23["gt"], s2),
            ("2D+3D video vs 2D-only", S23["video"], s2)):
        m, lo, hi = boot_delta(sa, sb, y2, ids)
        out["contrasts"][lab] = [round(m, 4), round(lo, 4), round(hi, 4)]
        star = "" if lo <= 0 <= hi else "  *"
        print(f"    {lab:44s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--win", type=int, default=24)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--targets",
                    default="y_articulation,y_contact_state,y_contact_change")
    a = ap.parse_args()
    arms = build_arms()
    rows, dstat = collect(a.dataset, a.win, arms)
    ev = {r["id"] for r in rows} & depthnet_unseen(a.dataset)
    print(f"\n{a.dataset}: {len(rows)} windows kept by ALL arms; "
          f"{len(ev)} episodes are depthnet-unseen and therefore scorable")
    print("depth at tracked points, held-out:")
    for nm in CORE:
        ds = dstat[nm]
        if ds:
            d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
            print(f"  {nm:11s} AbsRel {d['absrel']:.3f}  d1 {d['d1']:.3f}"
                  f"  medErr {d['med_err_m']:.3f} m")
    rep = dict(dataset=a.dataset, win=a.win, windows=len(rows),
               episodes=len(ev), folds=a.folds, targets={})
    # MOTION features for articulation (what L1.3 measured), and the
    # CONTACT-GAP family for the contact targets. Scoring contact with
    # motion features only would answer a different question - whether
    # motion predicts contact - not whether depth expresses it.
    if "y_articulation" in a.targets:
        rep["targets"]["y_articulation"] = report_target(
            rows, "y_articulation", ev, F3, "motion 3D")
    for tgt in [t for t in ("y_contact_state", "y_contact_change")
                if t in a.targets]:
        rep["targets"][tgt] = report_target(
            rows, tgt, ev, tuple(F3) + FC, "motion 3D + contact gap")
        rep["targets"][tgt + "__gaponly"] = report_target(
            rows, tgt, ev, FC, "contact gap ONLY")
    R.log("rel_g3_targets", **rep)
    (R.BASE / "relg3_targets.json").write_text(json.dumps(rep, indent=1))
    print("\nreference: relg0 2D probe 0.816 (articulation, fixed split) | "
          "oracle relation 0.997")
