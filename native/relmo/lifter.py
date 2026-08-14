"""G1 LIFT — the state estimator. 2D tracks in, per-point 3D trajectory out.

WHY THIS IS THE THING TO BUILD, and why it is not another forecaster.
gates.json settles it numerically: on rcasa val the oracle that is handed
the TRUE parts and rolls them forward rigidly scores 0.971, while
const-velocity scores 0.458. The gap between a model and const-velocity
is the dynamics problem; the gap between const-velocity and 0.971 is the
STATE problem, and it is twice as large. wmP/wmR/wmS spent themselves on
the small half and reached parity (+0.880 vs +0.881). So: estimate state.

WHAT IT PREDICTS. Depth, one scalar per point per frame, and nothing
else. The 3D then follows by exact unprojection with a FIXED nominal
camera:

    X = (u - W/2) * z / f ,  Y = -(v - H/2) * z / f ,  Z = z

Predicting z rather than free XYZ is deliberate. The camera model is
knowledge we have; making the network rediscover it would spend capacity
to arrive at a worse version of an identity we can write down, and it
would let the network move a point sideways to reduce a depth error,
which is exactly the kind of cheat that produces a good loss and a
useless state. Here every degree of freedom the model has is depth.

f is computed from a FIXED fovy of 60 deg for every episode, never from
z["cam_fovy"]. cam_fovy is a per-episode sim value on this corpus (45,
60 and 75 appear), so reading it is reading the simulator. A real
deployment calibrates once. Target and prediction use the same fixed f,
so the comparison is internally consistent.

THE MEASUREMENT THIS EXISTS TO MAKE. The previous sweep injected noise
into GT depth and concluded depth was unreachable. It tested per-frame
error models (iid / static / global / drift) against a per-frame depth
network. But a MOVING OBJECT under a static camera generates its own
parallax: its 3D structure is constrained by its own motion over the
window. That is structure-from-motion on the object, it lives in the
TEMPORAL axis, and a per-frame network cannot express it and the noise
sweep could not have simulated it. So the model here is temporal by
construction, and the decisive control is B1 — the identical network
with the temporal axis removed. If the temporal model beats B1 on MOVING
points, motion carries depth the per-frame path cannot see. If it does
not, the sweep's conclusion stands and this is the experiment that
earned it.

SERVE-LEGAL INPUTS, exhaustively: xy, vis, width, height. That is all
this module reads at inference. gxy/gdist/gvis are read ONLY to build
targets and to score; pdepth, xpos, contact_pairs, qpos, jnt_*, bid,
ident, cam_fovy, cam_mat, cam_pos are never read at all.

    python -m relmo.lifter train --steps 4000
    python -m relmo.lifter gate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R      # noqa: E402
from relmo import splits as SP       # noqa: E402
from relmo.device import DEVICE      # noqa: E402

W = 24                  # window length, the project-wide convention
STRIDE = 12
FOVY = 60.0             # the fixed nominal calibration; never per-episode
RUN = "lifter_v1"

# A point counts as MOVING if its GT 3D displacement across the window
# exceeds this. Used for EVALUATION STRATIFICATION ONLY - it reads GT,
# which evaluation may do and inference may not. 1 cm is well above the
# tracker's own noise floor (trackval: background EPE median 0.445 px,
# which at ~1 m and f=208 px is ~2 mm).
MOVE_M = 0.01


# ----------------------------------------------------------------- data
def focal(width, height, fovy=FOVY):
    return (float(height) / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)


def rays(xy, width, height, f):
    """Pixels -> normalised ray directions (x/z, y/z).

    This is the natural input for a depth predictor: it removes the
    camera from the learning problem entirely, so the network sees
    geometry rather than an image convention, and the same weights
    transfer to a camera with a different focal length."""
    u = (xy[..., 0] - float(width) / 2.0) / f
    v = -(xy[..., 1] - float(height) / 2.0) / f
    return np.stack([u, v], -1).astype(np.float32)


def load_episode(path):
    """One episode -> serve-legal inputs + GT targets, or None.

    Returns dict with:
      ray  (T,P,2) f32   from TRACKED xy   - serve-legal
      vis  (T,P)   bool  from tracker      - serve-legal
      gray (T,P,2) f32   from GT gxy       - target construction only
      gz   (T,P)   f32   GT depth          - target construction only
      gvis (T,P)   bool  GT visibility     - target construction only
    """
    z = np.load(path)
    need = ("xy", "vis", "gxy", "gdist", "gvis", "width", "height")
    if not all(k in z.files for k in need):
        return None
    wd, ht = float(z["width"]), float(z["height"])
    f = focal(wd, ht)
    gz = z["gdist"].astype(np.float32)
    gv = z["gvis"].astype(bool)
    # Reject non-physical depths rather than letting them poison a log.
    gv &= np.isfinite(gz) & (gz > 0.05) & (gz < 20.0)
    if gv.mean() < 0.05:
        return None
    return dict(
        ray=rays(z["xy"].astype(np.float32), wd, ht, f),
        vis=z["vis"].astype(bool),
        gray=rays(z["gxy"].astype(np.float32), wd, ht, f),
        gz=gz, gvis=gv, f=f, wd=wd, ht=ht,
        ddist=(z["ddist"].astype(np.float32) if "ddist" in z.files else None),
        stem=Path(path).stem, cam=camera_of(Path(path).stem))


def camera_of(stem: str) -> str:
    """Which camera rendered this episode, from its id.

    Not cosmetic. CoTracker's error measured against sim GT on rcasa:
    fixed agentview median EPE 0.27 px, eye_in_hand 1.24 px - 4.6x worse,
    and the corpus's single worst episode (37 px) is a wrist camera. Those
    episodes were pooled UNSTRATIFIED into every number this project has
    reported, so a moving-camera effect has been free to masquerade as
    model variance. Every eval stratifies on this from here on."""
    tail = stem.rsplit("__", 1)[-1]
    if "eye_in_hand" in tail:
        return "eye_in_hand"
    if "agentview" in tail or "frontview" in tail:
        return "fixed"
    return "other"


def windows_of(T, w=W, stride=STRIDE):
    if T < w:
        return []
    return list(range(0, T - w + 1, stride))


def load_split(dataset, split, limit=None, quiet=False):
    from tqdm import tqdm
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    sel = part[split] if split in part else files
    if limit:
        sel = sel[:limit]
    out = []
    it = sel if quiet else tqdm(sel, desc=f"load {dataset}/{split}", unit="ep")
    for p in it:
        e = load_episode(p)
        if e is not None:
            out.append(e)
    return out


# ---------------------------------------------------------------- model
class Lifter(nn.Module):
    """Temporal encode per point -> attention ACROSS points -> depth.

    The cross-point attention is not decoration. Monocular depth from a
    single object's motion is scale-ambiguous in isolation; what fixes
    the scale is the RELATION between what moves and what does not, and
    between points at different depths moving at different image speeds.
    A per-point model cannot represent that, so the points must talk.
    Attention is permutation-equivariant, so the model does not learn a
    point ORDER, and it accepts any P.

    temporal=False strips every temporal path and makes each frame
    independent. That is B1, the control - same modules, same width,
    same depth, same optimiser, only the time axis removed.
    """

    def __init__(self, d=128, n_spatial=3, n_head=4, temporal=True):
        super().__init__()
        self.temporal = temporal
        # per (point, frame): ray direction, and - only when temporal -
        # its velocity and its offset from the window start.
        fin = 2 + 1 + (4 if temporal else 0)
        self.inp = nn.Sequential(nn.Linear(fin, d), nn.GELU(),
                                 nn.Linear(d, d))
        if temporal:
            # depthwise-separable temporal conv stack. Dilations 1,2,4
            # give a receptive field of 15 frames inside a 24-frame
            # window, so a point's whole trajectory shape is visible.
            self.tconv = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d, d, 5, padding=2 * dl, dilation=dl, groups=d),
                    nn.Conv1d(d, d, 1), nn.GELU())
                for dl in (1, 2, 4)])
            self.tnorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        enc = nn.TransformerEncoderLayer(
            d, n_head, dim_feedforward=2 * d, batch_first=True,
            norm_first=True, dropout=0.0, activation="gelu")
        self.spatial = nn.TransformerEncoder(enc, n_spatial)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d),
                                  nn.GELU(), nn.Linear(d, 1))
        # Predict log-depth as an offset from a learned global constant,
        # initialised at log(1.0 m) - roughly this corpus's median. The
        # model therefore starts at the flat baseline B0 and can only
        # improve on it, instead of spending its first thousand steps
        # discovering the scene scale.
        self.logz0 = nn.Parameter(torch.zeros(1))

    def forward(self, ray, vis):
        """ray (B,T,P,2), vis (B,T,P) bool -> logz (B,T,P)."""
        B, T, P, _ = ray.shape
        v = vis.float().unsqueeze(-1)
        if self.temporal:
            d1 = torch.zeros_like(ray)
            d1[:, 1:] = ray[:, 1:] - ray[:, :-1]
            off = ray - ray[:, :1]
            x = torch.cat([ray, v, d1, off], -1)
        else:
            x = torch.cat([ray, v], -1)
        h = self.inp(x)                                   # B,T,P,d
        if self.temporal:
            # (B*P, d, T) for temporal convolution over the time axis
            g = h.permute(0, 2, 3, 1).reshape(B * P, -1, T)
            for cv, nm in zip(self.tconv, self.tnorm):
                g = g + cv(g)[..., :T]
                g = nm(g.transpose(1, 2)).transpose(1, 2)
            h = g.reshape(B, P, -1, T).permute(0, 3, 1, 2)   # B,T,P,d
            ctx = h.mean(1)                                  # B,P,d
        else:
            ctx = h.reshape(B * T, P, -1)
        # attention across POINTS
        if self.temporal:
            s = self.spatial(ctx)                            # B,P,d
            s = s.unsqueeze(1).expand(-1, T, -1, -1)         # B,T,P,d
        else:
            s = self.spatial(ctx).reshape(B, T, P, -1)
        out = self.head(torch.cat([h, s], -1)).squeeze(-1)   # B,T,P
        return out + self.logz0


# ------------------------------------------------------------- batching
def sample_batch(eps, bs, rng, device, w=W):
    ray, vis, gray, gz, gv = [], [], [], [], []
    for _ in range(bs):
        e = eps[rng.integers(len(eps))]
        T = e["ray"].shape[0]
        ws = windows_of(T, w)
        if not ws:
            continue
        t0 = ws[rng.integers(len(ws))]
        sl = slice(t0, t0 + w)
        ray.append(e["ray"][sl]); vis.append(e["vis"][sl])
        gray.append(e["gray"][sl]); gz.append(e["gz"][sl])
        gv.append(e["gvis"][sl])
    t = lambda a, dt: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    return (t(ray, torch.float32), t(vis, torch.bool), t(gray, torch.float32),
            t(gz, torch.float32), t(gv, torch.bool))


def unproject(ray, z):
    """(B,T,P,2) rays + (B,T,P) depth -> (B,T,P,3) camera-frame XYZ."""
    return torch.cat([ray * z.unsqueeze(-1), z.unsqueeze(-1)], -1)


def loss_fn(logz, ray, gray, gz, gv):
    """L1 on log-depth + L1 on 3D DISPLACEMENT.

    Two terms because the two things being asked for are different. The
    log-depth term buys per-frame accuracy (what AbsRel measures). The
    displacement term buys TRAJECTORY accuracy: it is invariant to a
    constant per-point depth offset, which is exactly the error mode the
    earlier sweep measured as nearly free downstream, so without it the
    optimiser would happily trade away the quantity the product needs to
    buy the quantity the metric happens to show."""
    m = gv.float()
    n = m.sum().clamp(min=1.0)
    l_z = (torch.abs(logz - torch.log(gz.clamp(min=0.05))) * m).sum() / n
    # displacement: predicted 3D uses the TRACKED ray (serve condition),
    # target 3D uses the GT ray. Tracker error is part of the problem.
    P3 = unproject(ray, torch.exp(logz))
    G3 = unproject(gray, gz)
    dP = P3 - P3[:, :1]
    dG = G3 - G3[:, :1]
    mm = (m * gv[:, :1].float()).unsqueeze(-1)
    l_d = (torch.abs(dP - dG) * mm).sum() / mm.sum().clamp(min=1.0) / 3.0
    return l_z + l_d, l_z.item(), l_d.item()


# ------------------------------------------------------------ baselines
def flat_logz(eps):
    """B0: one constant depth for the whole corpus, the train median."""
    v = np.concatenate([e["gz"][e["gvis"]] for e in eps])
    return float(np.log(np.median(v)))


# ----------------------------------------------------------- evaluation
def _r2(pred, targ, mask):
    """R2 against the mean of the TARGET, computed on masked entries.

    DEGENERATE-VARIANCE GUARD, added after the smoke run returned
    dispR2 = -197385 on STATIC points. That number was not a model
    failure, it was a metric failure: a static point's GT displacement is
    ~0 at every frame, so ss_tot ~ 0 and R2 explodes for any nonzero
    residual. R2 is only meaningful where the target actually varies.
    Static points are therefore scored with RMSE (see eval_episode) and
    R2 is withheld rather than reported as a spectacular negative."""
    p, t = pred[mask], targ[mask]
    if t.size < 8:
        return np.nan
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean(0, keepdims=True)) ** 2).sum())
    # 1 mm^2 per element: below this the stratum carries no signal to
    # explain and the ratio is numerical noise.
    if ss_tot < 1e-6 * max(t.size, 1):
        return np.nan
    return 1.0 - ss_res / ss_tot


def _rmse(pred, targ, mask):
    """Displacement error in METRES. Always well posed, including on
    static points, and directly interpretable against the 1.5 mm figure
    the contact sweep established as the physically meaningful scale."""
    if mask.sum() < 8:
        return np.nan
    d = pred[mask] - targ[mask]
    return float(np.sqrt((d ** 2).sum(-1).mean()))


def eval_episode(e, predict, w=W, in_ray="tracked", lift_ray="tracked"):
    """Per-episode metrics.

    predict(ray, vis, e, t0) -> logz (T,P). The episode and window start
    are passed so a baseline that reads a CACHED per-frame prediction
    (B2 depthnet) can index it, without the learned model ever touching
    either - the learned predictor ignores both arguments.

    in_ray / lift_ray select TRACKED (serve-legal) or GT (privileged)
    pixels for, respectively, what the predictor SEES and what the depth
    is unprojected THROUGH. Both default to tracked, which is the only
    serve-legal setting and the only one a gate may use. The GT settings
    exist to ATTRIBUTE error, because the headline `lifter` number
    confounds two sources and cannot be acted on:

        in_ray  lift_ray   what the residual contains
        tracked tracked    depth error + tracker error   (= the lifter)
        tracked GT         depth error + tracker-as-input error
        GT      GT         depth error ALONE
        (with an oracle depth head, GT/GT is a POSITIVE CONTROL that
         must return ~0 m and R2 ~ 1.0; if it does not, this harness is
         broken and no other row on the table means anything.)"""
    T = e["ray"].shape[0]
    ws = windows_of(T, w)
    if not ws:
        return None
    # abs_* is ABSOLUTE 3D position error, added because dispR2 and
    # rmse_* are computed on X_t - X_t0 and are therefore BLIND BY
    # CONSTRUCTION to a constant per-point depth offset. Contact,
    # support and scene-relative state need absolute position, so a
    # displacement-only table can call a model adequate for a job it
    # cannot do. The static-offset arm of the depth sweep exists to
    # make that blindness visible rather than argued about.
    acc = dict(mov=[], sta=[], all=[], absrel=[], d1=[],
               rmse_mov=[], rmse_sta=[], rmse_all=[],
               abs_mov=[], abs_sta=[], abs_all=[])
    for t0 in ws:
        sl = slice(t0, t0 + w)
        ray, vis = e["ray"][sl], e["vis"][sl]
        gray, gz, gv = e["gray"][sl], e["gz"][sl], e["gvis"][sl]
        x_in = ray if in_ray == "tracked" else gray
        x_lift = ray if lift_ray == "tracked" else gray
        logz = predict(x_in, vis, e, t0)
        z = np.exp(logz)
        # per-frame depth accuracy
        m = gv
        if m.sum() >= 8:
            ar = np.abs(z[m] - gz[m]) / gz[m]
            rt = np.maximum(z[m] / gz[m], gz[m] / z[m])
            acc["absrel"].append(float(ar.mean()))
            acc["d1"].append(float((rt < 1.25).mean()))
        # 3D displacement
        P3 = np.concatenate([x_lift * z[..., None], z[..., None]], -1)
        G3 = np.concatenate([gray * gz[..., None], gz[..., None]], -1)
        dP, dG = P3 - P3[:1], G3 - G3[:1]
        valid = gv & gv[:1]                       # (T,P)
        disp = np.linalg.norm(dG[-1] - dG[0], axis=-1)   # (P,)
        moving = disp > MOVE_M
        for key, sel in (("all", np.ones_like(moving)),
                         ("mov", moving), ("sta", ~moving)):
            mk = valid & sel[None, :]
            r = _r2(dP, dG, mk)
            if np.isfinite(r):
                acc[key].append(r)
            q = _rmse(dP, dG, mk)
            if np.isfinite(q):
                acc["rmse_" + key].append(q)
            a = _rmse(P3, G3, mk)          # ABSOLUTE, not bias-cancelled
            if np.isfinite(a):
                acc["abs_" + key].append(a)
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}


def boot_paired(a, b, n=4000, seed=0):
    """Paired bootstrap over EPISODES on the contrast a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=int(a.size))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n, a.size))
    d = (a[idx] - b[idx]).mean(1)
    return dict(mean=float((a - b).mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), n=int(a.size))


# ------------------------------------------------------------- training
def train(eps_tr, eps_va, steps=4000, d=128, temporal=True, bs=8,
          lr=3e-4, seed=0, tag="wm", quiet=False):
    from tqdm import tqdm
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Lifter(d=d, temporal=temporal).to(DEVICE)
    net.logz0.data.fill_(flat_logz(eps_tr))
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine decay is REQUIRED on this project: a constant LR collapsed a
    # previous run from peak 0.92 to -0.09. Not a preference, a scar.
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    print(f"[{tag}] {npar/1e6:.2f}M params, temporal={temporal}, "
          f"device={DEVICE}, {len(eps_tr)} train eps", flush=True)
    best, best_state, hist = np.inf, None, []
    bar = range(steps) if quiet else tqdm(range(steps), desc=f"train/{tag}",
                                          unit="step")
    for st in bar:
        net.train()
        ray, vis, gray, gz, gv = sample_batch(eps_tr, bs, rng, DEVICE)
        logz = net(ray, vis)
        loss, lz, ld = loss_fn(logz, ray, gray, gz, gv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        if not quiet and st % 50 == 0:
            bar.set_postfix(loss=f"{loss.item():.4f}", lz=f"{lz:.4f}",
                            ld=f"{ld:.4f}", gn=f"{float(gn):.2f}")
        if (st + 1) % 500 == 0 or st + 1 == steps:
            vl = validate(net, eps_va, rng)
            hist.append(dict(step=st + 1, val=vl, loss=float(loss.item())))
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, dict(params=npar, best_val=best, hist=hist)


@torch.no_grad()
def validate(net, eps, rng, n=24, bs=8):
    net.eval()
    tot = 0.0
    for _ in range(n):
        ray, vis, gray, gz, gv = sample_batch(eps, bs, rng, DEVICE)
        logz = net(ray, vis)
        loss, _, _ = loss_fn(logz, ray, gray, gz, gv)
        tot += float(loss.item())
    return tot / n


def predictor(net):
    """The MODEL. Reads ray and vis only - both derived from xy/vis."""
    @torch.no_grad()
    def f(ray, vis, e=None, t0=None):
        net.eval()
        r = torch.as_tensor(ray, dtype=torch.float32, device=DEVICE)[None]
        v = torch.as_tensor(vis, dtype=torch.bool, device=DEVICE)[None]
        return net(r, v)[0].float().cpu().numpy()
    return f


def const_predictor(logz):
    """B0 flat: one constant for everything."""
    return lambda ray, vis, e=None, t0=None: np.full(
        ray.shape[:2], logz, np.float32)


def epmedian_predictor():
    """B3: the episode's own median GT depth. An ORACLE trivial bound -
    it reads GT, so it is an upper bound on what a depth-blind predictor
    with perfect scene-scale knowledge could do, not a competitor."""
    def f(ray, vis, e=None, t0=None):
        m = float(np.log(np.median(e["gz"][e["gvis"]])))
        return np.full(ray.shape[:2], m, np.float32)
    return f


def depthnet_predictor():
    """B2: the existing per-frame CNN's depth (ddist), read from cache."""
    def f(ray, vis, e=None, t0=None):
        dd = e["ddist"][t0:t0 + ray.shape[0]]
        return np.log(np.clip(dd, 0.05, None)).astype(np.float32)
    return f


def oracle_predictor():
    """CEILING: the TRUE depth, applied to the TRACKED ray.

    Privileged, never a baseline to beat - it is the score a PERFECT
    depth model would get given CoTracker's pixel error. Its gap below
    1.0 is the tracker's contribution, so a shortfall can be attributed
    instead of argued about."""
    def f(ray, vis, e=None, t0=None):
        gz = e["gz"][t0:t0 + ray.shape[0]]
        return np.log(np.clip(gz, 0.05, None)).astype(np.float32)
    return f


# ------------------------------------------------------------ the gate
def score_all(eps, preds, desc="eval"):
    """{name: predictor | (predictor, in_ray, lift_ray)}
       -> {name: {metric: [per-episode values]}}.

    `cam` and `stem` ride along per episode so every table can be
    stratified by camera after the fact without re-running anything."""
    from tqdm import tqdm
    METRICS = ("mov", "sta", "all", "absrel", "d1",
               "rmse_mov", "rmse_sta", "rmse_all",
               "abs_mov", "abs_sta", "abs_all")
    out = {k: {m: [] for m in METRICS} for k in preds}
    for k in preds:
        out[k]["cam"], out[k]["stem"] = [], []
    for e in tqdm(eps, desc=desc, unit="ep"):
        for name, spec in preds.items():
            if name == "depthnet" and e["ddist"] is None:
                continue
            fn, i_r, l_r = spec if isinstance(spec, tuple) else (
                spec, "tracked", "tracked")
            r = eval_episode(e, fn, in_ray=i_r, lift_ray=l_r)
            if r is None:
                continue
            for m, v in r.items():
                out[name][m].append(v)
            out[name]["cam"].append(e["cam"])
            out[name]["stem"].append(e["stem"])
    return out


def run_gate(args):
    t_start = time.time()
    tr = load_split("rcasa", SP.TRAIN)
    va = load_split("rcasa", SP.VAL)
    te = load_split("rcasa", SP.TEST)
    print(f"episodes  train {len(tr)}  val {len(va)}  test {len(te)}")
    if min(len(tr), len(va), len(te)) < 3:
        print("ABORT: a split is too small to bootstrap over.")
        return

    # --- the model, and B1 the capacity-matched no-time control -------
    net, meta = train(tr, va, steps=args.steps, d=args.d, temporal=True,
                      bs=args.bs, seed=args.seed, tag="lift")
    b1, meta1 = train(tr, va, steps=args.steps, d=args.d, temporal=False,
                      bs=args.bs, seed=args.seed, tag="B1-2Donly")

    preds = {
        "lifter":    predictor(net),
        "B1_2donly": predictor(b1),
        "B0_flat":   const_predictor(flat_logz(tr)),
        "B3_epmed":  epmedian_predictor(),
        "depthnet":  depthnet_predictor(),
        # CEILING, not a competitor: TRUE depth applied to the TRACKED
        # ray. Everything above it is unreachable while the tracker has
        # error, so it says how much of any shortfall is the depth model
        # and how much is CoTracker.
        "C_gtdepth": oracle_predictor(),
    }
    res = score_all(te, preds, desc="gate/test")

    print("\n" + "=" * 88)
    print("G1 LIFT — held-out TEST episodes (rcasa), bootstrap over EPISODES")
    print("  dispR2 = R2 of 3D displacement X_t - X_t0 over a 24-frame window")
    print("  STATIC R2 is withheld by design: GT displacement ~ 0 there, so")
    print("  its variance denominator is degenerate. Read rmse_sta instead.")
    print("=" * 88)
    print(f"{'arm':<12}{'dispR2 MOV':>11}{'dispR2 all':>11}"
          f"{'rmseMOV_m':>11}{'rmseSTA_m':>11}{'AbsRel':>9}{'d1':>7}{'n':>4}")
    for k, v in res.items():
        f = lambda m: (np.nanmean(v[m]) if v[m] else np.nan)
        print(f"{k:<12}{f('mov'):>11.4f}{f('all'):>11.4f}"
              f"{f('rmse_mov'):>11.4f}{f('rmse_sta'):>11.4f}"
              f"{f('absrel'):>9.4f}{f('d1'):>7.4f}{len(v['mov']):>4}")

    # --- pre-registered contrasts ------------------------------------
    print("\nPRE-REGISTERED CONTRASTS (paired per-episode bootstrap, 95% CI)")
    contrasts = {}
    # B3_epmed is IN this list now. It was computed, printed and left a
    # spectator while c1/c2 gated against the two WEAKEST arms - so the
    # run advertised "+9.07 over 2D-only" when the honest margin over the
    # strongest trivial predictor was +0.93. Gate against your best
    # baseline or the gate is decoration.
    for name in ("B1_2donly", "B0_flat", "B3_epmed"):
        c = boot_paired(res["lifter"]["mov"], res[name]["mov"], seed=args.seed)
        contrasts[f"mov_lifter_minus_{name}"] = c
        sig = "SIG" if (c["lo"] > 0) else ("n.s." if c["hi"] > 0 else "WORSE")
        print(f"  dispR2 MOVING  lifter - {name:<10} "
              f"{c['mean']:+.4f}  [{c['lo']:+.4f},{c['hi']:+.4f}]  "
              f"n={c['n']}  {sig}")

    ar_l = np.nanmean(res["lifter"]["absrel"])
    ar_d = (np.nanmean(res["depthnet"]["absrel"])
            if res["depthnet"]["absrel"] else np.nan)
    c_ar = boot_paired(res["depthnet"]["absrel"], res["lifter"]["absrel"],
                       seed=args.seed) if res["depthnet"]["absrel"] else None
    if c_ar:
        print(f"  AbsRel         depthnet - lifter      "
              f"{c_ar['mean']:+.4f}  [{c_ar['lo']:+.4f},{c_ar['hi']:+.4f}]"
              f"  n={c_ar['n']}  (positive = lifter better)")

    g = contrasts["mov_lifter_minus_B1_2donly"]
    h = contrasts["mov_lifter_minus_B0_flat"]
    j = contrasts["mov_lifter_minus_B3_epmed"]
    c1, c2 = g["lo"] > 0, h["lo"] > 0
    # c3 WAS `ar_l < 0.248`, a hardcoded literal carried over from
    # depthnet's OWN protocol (different split, dense per-frame). The
    # like-for-like depthnet AbsRel measured inside this very run is
    # 0.3246 - so the threshold and its own comparator disagreed by 31%,
    # and the gate could not be trusted in either direction. The paired
    # contrast c_ar was already computed and printed one screen above,
    # and was neither gated on nor saved. It is the criterion now.
    # The CRITERION was right - an absolute-scale gate must exist or the
    # model drifts into scale-free depth and every physical-unit relation
    # downstream dies. Only the threshold was wrong.
    c3 = bool(c_ar is not None and c_ar["lo"] > 0)
    # c4: beat the STRONGEST trivial predictor, not the weakest.
    c4 = j["lo"] > 0
    # c5: dispR2 MOV > 0. R2=0 is "emit the window's mean displacement
    # vector" - and that constant is fitted on the target itself, so it
    # is an ORACLE, not a serve-time rival. Clearing it is therefore a
    # stiff bar and a necessary one: below zero the model has explained
    # none of the displacement structure it exists to explain.
    r2_mov = float(np.nanmean(res["lifter"]["mov"]))
    c5 = bool(np.isfinite(r2_mov) and r2_mov > 0)
    passed = bool(c1 and c2 and c3 and c4 and c5)
    # SKILL SCORE: where the model sits between the best trivial
    # predictor and the achievable ceiling. The raw dispR2 is negative
    # for every arm, so a bare "+9.07 over 2D-only" is unreadable and
    # flatters; this is the number to quote.
    b3, ceil = (float(np.nanmean(res["B3_epmed"]["mov"])),
                float(np.nanmean(res["C_gtdepth"]["mov"])))
    skill = (r2_mov - b3) / (ceil - b3) if ceil > b3 else float("nan")
    print("\nGATE (revised 2026-08-13 after the c3 threshold was found "
          "to disagree with its own in-run comparator):")
    print(f"  1. dispR2 MOV > B1 2D-only, CI excludes 0 ... "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  2. dispR2 MOV > B0 flat,    CI excludes 0 ... "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"  3. AbsRel depthnet-lifter, CI lo > 0      ... "
          f"{'PASS' if c3 else 'FAIL'}"
          + (f"  (lo={c_ar['lo']:+.4f})" if c_ar else "  (no depthnet)"))
    print(f"  4. dispR2 MOV > B3 epmed,   CI excludes 0 ... "
          f"{'PASS' if c4 else 'FAIL'}  ({j['mean']:+.4f} "
          f"[{j['lo']:+.4f},{j['hi']:+.4f}])")
    print(f"  5. dispR2 MOV > 0 (oracle window mean)    ... "
          f"{'PASS' if c5 else 'FAIL'}  ({r2_mov:+.4f})")
    print(f"  G1 {'PASSED' if passed else 'FAILED'}")
    print(f"\n  SKILL vs B3_epmed -> C_gtdepth ceiling: {skill:.3f}  "
          f"(B3 {b3:+.3f} | lifter {r2_mov:+.3f} | ceiling {ceil:+.3f})")
    print(f"  rmse_mov: lifter {1000*np.nanmean(res['lifter']['rmse_mov']):.1f} mm"
          f" | ceiling {1000*np.nanmean(res['C_gtdepth']['rmse_mov']):.1f} mm"
          f"  vs the 4.55 mm/frame contact signal")

    rec = dict(gate="G1 LIFT", passed=passed, dataset="rcasa",
               n_test=len(te), n_train=len(tr), steps=args.steps,
               params=meta["params"], params_b1=meta1["params"],
               best_val=meta["best_val"], best_val_b1=meta1["best_val"],
               absrel_lifter=float(ar_l), absrel_depthnet=float(ar_d),
               r2_mov=r2_mov, skill_vs_b3=float(skill),
               criteria=dict(c1_vs_2donly=c1, c2_vs_flat=c2,
                             c3_absrel_paired=c3, c4_vs_b3epmed=c4,
                             c5_r2_positive=c5),
               contrasts=contrasts,
               # c_ar was the run's most decisive number and it was
               # printed and dropped from the artifact. That is exactly
               # how the 0.248 error survived a whole run. It is saved.
               contrast_absrel_depthnet_minus_lifter=c_ar,
               cam_counts={c: res["lifter"]["cam"].count(c)
                           for c in set(res["lifter"]["cam"])},
               summary={k: {m: float(np.nanmean(v[m])) if v[m] else None
                            for m in v if m not in ("cam", "stem")}
                        for k, v in res.items()},
               secs=round(time.time() - t_start, 1))
    outp = R.BASE / "lifter_g1.json"
    outp.write_text(json.dumps(rec, indent=1))
    R.log("gate_g1_lift", **{k: rec[k] for k in
                             ("passed", "n_test", "absrel_lifter",
                              "absrel_depthnet", "params", "secs")})
    torch.save(net.state_dict(), R.BASE / "lifter_v1.pt")
    print(f"\nwrote {outp}")
    return rec



# ------------------------------------------------------------------ OOD
def run_ood(args):
    """Transfer, REPORTED not gating (G5 material, not G1).

    rcasa_eval  same renderer, unseen episodes.
    arctic_v1   a DIFFERENT EMBODIMENT - human bimanual manipulation,
                2800x2000 frames, objects at ~0.55 m instead of ~1.0 m.

    ARCTIC CAVEAT, stated because it changes what the numbers mean:
    arctic ships no cam_fovy, so `rays` divides by a focal derived from
    the same nominal 60 deg used everywhere here. If ARCTIC's true field
    of view differs, the ray directions are scaled by an unknown
    constant and the reconstructed 3D lives in a distorted frame. The
    comparison between ARMS stays valid (every arm uses the same rays;
    only depth differs), but ARCTIC's absolute metric depth numbers are
    NOT claims about real-world accuracy. Read the scale-invariant
    column there, not AbsRel.
    """
    net = Lifter(d=args.d, temporal=True).to(DEVICE)
    ck = R.BASE / "lifter_v1.pt"
    if not ck.exists():
        print(f"no checkpoint at {ck}; run `gate` first.")
        return
    net.load_state_dict(torch.load(ck, map_location=DEVICE))
    tr = load_split("rcasa", SP.TRAIN, quiet=True)
    out = {}
    for ds in ("rcasa_eval", "arctic_v1"):
        d = R.TRACKS / ds
        if not d.exists() or not any(d.glob("*.npz")):
            print(f"skip {ds}: no tracks on disk")
            continue
        eps = load_split(ds, SP.TEST if ds == "rcasa_eval" else "all",
                         limit=args.limit)
        if len(eps) < 3:
            eps = load_split(ds, "all", limit=args.limit)
        if len(eps) < 3:
            print(f"skip {ds}: only {len(eps)} usable episodes")
            continue
        preds = {"lifter": predictor(net),
                 "B0_flat": const_predictor(flat_logz(tr)),
                 "B3_epmed": epmedian_predictor(),
                 "C_gtdepth": oracle_predictor()}
        res = score_all(eps, preds, desc=f"ood/{ds}")
        print(f"\n--- OOD {ds}  ({len(eps)} episodes) ---")
        print(f"{'arm':<12}{'dispR2 MOV':>12}{'dispR2 all':>12}"
              f"{'rmseMOV_m':>12}{'AbsRel':>10}")
        for k, v in res.items():
            f = lambda m: (np.nanmean(v[m]) if v[m] else np.nan)
            print(f"{k:<12}{f('mov'):>12.4f}{f('all'):>12.4f}"
                  f"{f('rmse_mov'):>12.4f}{f('absrel'):>10.4f}")
        c = boot_paired(res["lifter"]["mov"], res["B0_flat"]["mov"])
        print(f"  lifter - B0_flat (dispR2 MOV): {c['mean']:+.4f} "
              f"[{c['lo']:+.4f},{c['hi']:+.4f}] n={c['n']}")
        out[ds] = {k: {m: (float(np.nanmean(v[m])) if v[m] else None)
                       for m in v if m not in ("cam", "stem")}
                   for k, v in res.items()}
        out[ds]["_contrast_vs_flat"] = c
    (R.BASE / "lifter_ood.json").write_text(json.dumps(out, indent=1))
    R.log("lifter_ood", datasets=list(out))
    print(f"\nwrote {R.BASE / 'lifter_ood.json'}")


def run_attrib(args):
    """ATTRIBUTION 2x2 — is the remaining error DEPTH or the TRACKER?

    The `lifter` arm confounds both and therefore cannot direct effort.
    Holding the trained checkpoint fixed (no retraining), vary the two
    inputs independently:

        arm                 ray in    ray lifted   residual contains
        lifter              tracked   tracked      depth + tracker
        C_gtdepth           tracked   tracked      tracker alone (oracle z)
        C_mixray_predz      tracked   GT           depth + tracker-as-input
        C_gtray_predz       GT        GT           DEPTH ALONE
        C_gtray_gtdepth     GT        GT           POSITIVE CONTROL ~ 0

    THE DECIDING CONTRAST is C_gtray_predz vs C_gtdepth. If depth-alone
    error is at or below tracker-alone error, the depth head has stopped
    being the binding constraint and every further hour belongs to the
    tracker.

    The positive control is not optional. C_gtray_gtdepth feeds GT pixels
    and GT depth through the identical path; it MUST return ~0 m and
    dispR2 ~ 1.0. If it does not, the harness is measuring something
    other than what these column headings claim and the whole table is
    void - which is the failure this project has already paid for once."""
    net = Lifter(d=args.d, temporal=True).to(DEVICE)
    ck = R.BASE / "lifter_v1.pt"
    if not ck.exists():
        print(f"no checkpoint at {ck}; run `gate` first.")
        return
    net.load_state_dict(torch.load(ck, map_location=DEVICE))
    te = load_split("rcasa", SP.TEST)
    tr = load_split("rcasa", SP.TRAIN, quiet=True)
    print(f"test episodes: {len(te)}")

    P, O = predictor(net), oracle_predictor()
    preds = {
        "lifter":          (P, "tracked", "tracked"),
        "C_gtdepth":       (O, "tracked", "tracked"),
        "C_mixray_predz":  (P, "tracked", "gt"),
        "C_gtray_predz":   (P, "gt", "gt"),
        "C_gtray_gtdepth": (O, "gt", "gt"),
        "B3_epmed":        (epmedian_predictor(), "tracked", "tracked"),
        "B0_flat":         (const_predictor(flat_logz(tr)), "tracked",
                            "tracked"),
    }
    res = score_all(te, preds, desc="attrib")

    print("\n" + "=" * 84)
    print("ATTRIBUTION — which input is the binding constraint?")
    print("=" * 84)
    print(f"{'arm':<18}{'rayIN':>8}{'rayLIFT':>9}{'dispR2MOV':>11}"
          f"{'rmseMOV_mm':>12}{'AbsRel':>9}{'n':>4}")
    for k, v in res.items():
        i_r, l_r = (preds[k][1], preds[k][2])
        f = lambda m: (np.nanmean(v[m]) if v[m] else np.nan)
        print(f"{k:<18}{i_r:>8}{l_r:>9}{f('mov'):>11.4f}"
              f"{1000*f('rmse_mov'):>12.2f}{f('absrel'):>9.4f}"
              f"{len(v['mov']):>4}")

    # --- POSITIVE CONTROL, checked not eyeballed ---------------------
    pc = 1000 * np.nanmean(res["C_gtray_gtdepth"]["rmse_mov"])
    pc_r2 = float(np.nanmean(res["C_gtray_gtdepth"]["mov"]))
    ok = bool(pc < 1.0 and pc_r2 > 0.99)
    print(f"\nPOSITIVE CONTROL  C_gtray_gtdepth: rmse_mov {pc:.4f} mm, "
          f"dispR2 {pc_r2:.5f}  -> {'OK' if ok else 'HARNESS BROKEN'}")
    if not ok:
        print("  Every row above is void until this returns ~0 mm / ~1.0.")

    print("\nDECIDING CONTRASTS (paired bootstrap over EPISODES, 95% CI)")
    out_c = {}
    for a, b in (("C_gtray_predz", "C_gtdepth"),
                 ("lifter", "C_gtray_predz"),
                 ("lifter", "C_gtdepth")):
        c = boot_paired(res[a]["rmse_mov"], res[b]["rmse_mov"],
                        seed=args.seed)
        out_c[f"rmsemov_{a}_minus_{b}"] = c
        print(f"  rmse_mov  {a:<15} - {b:<14} "
              f"{1000*c['mean']:+8.2f} mm  "
              f"[{1000*c['lo']:+.2f},{1000*c['hi']:+.2f}]  n={c['n']}")

    # --- CAMERA STRATIFICATION (Signal 3) ----------------------------
    print("\nSTRATIFIED BY CAMERA (fixed agentview vs moving wrist)")
    cams = res["lifter"]["cam"]
    print(f"  test-set composition: "
          + ", ".join(f"{c}={cams.count(c)}" for c in sorted(set(cams))))
    strat = {}
    print(f"\n{'arm':<18}" + "".join(f"{c:>22}" for c in sorted(set(cams))))
    for k, v in res.items():
        row, cells = [], {}
        for c in sorted(set(cams)):
            idx = [i for i, cc in enumerate(v["cam"]) if cc == c]
            r2 = np.nanmean([v["mov"][i] for i in idx]) if idx else np.nan
            rm = (1000 * np.nanmean([v["rmse_mov"][i] for i in idx])
                  if idx else np.nan)
            cells[c] = dict(r2=float(r2), rmse_mm=float(rm), n=len(idx))
            row.append(f"{r2:>10.3f}/{rm:>8.1f}mm")
        strat[k] = cells
        print(f"{k:<18}" + "".join(f"{s:>22}" for s in row))
    print("  (cells are dispR2_MOV / rmse_mov_mm)")

    rec = dict(kind="lifter_attrib", n_test=len(te),
               cam_counts={c: cams.count(c) for c in set(cams)},
               positive_control=dict(rmse_mm=pc, r2=pc_r2, ok=ok),
               contrasts=out_c, stratified=strat,
               summary={k: {m: float(np.nanmean(v[m])) if v[m] else None
                            for m in v if m not in ("cam", "stem")}
                        for k, v in res.items()})
    outp = R.BASE / "lifter_attrib.json"
    outp.write_text(json.dumps(rec, indent=1))
    R.log("lifter_attrib", n_test=len(te), pc_ok=ok,
          pc_rmse_mm=pc, cam_counts=rec["cam_counts"])
    print(f"\nwrote {outp}")
    return rec


def run_regate(args):
    """Re-adjudicate G1 against the CORRECTED criteria without retraining.

    WHY THIS EXISTS. The full `gate` retrains two nets (4000 steps each).
    Re-run concurrently with tracking it advanced 50 steps in 10 minutes
    of wall clock - the two jobs thrash the same MPS device - and would
    have needed ~9 h. The corrected criteria c2/c3/c4/c5 are all
    functions of the SAVED checkpoint's test-set scores, so they cost
    ~10 s to recompute exactly.

    c1 (vs B1_2donly) is the ONE criterion that needs a second trained
    net. It is CARRIED FORWARD from the previous run, not recomputed,
    and is labelled as such in the artifact. That is honest because c1
    is unchanged by this revision (its comparator was never in dispute)
    and it passed with the widest margin on the board. It is NOT
    evidence produced today and must not be quoted as if it were."""
    net = Lifter(d=args.d, temporal=True).to(DEVICE)
    ck = R.BASE / "lifter_v1.pt"
    if not ck.exists():
        print(f"no checkpoint at {ck}; run `gate` first.")
        return
    net.load_state_dict(torch.load(ck, map_location=DEVICE))
    tr = load_split("rcasa", SP.TRAIN, quiet=True)
    te = load_split("rcasa", SP.TEST)
    preds = {"lifter": predictor(net),
             "B0_flat": const_predictor(flat_logz(tr)),
             "B3_epmed": epmedian_predictor(),
             "depthnet": depthnet_predictor(),
             "C_gtdepth": oracle_predictor()}
    res = score_all(te, preds, desc="regate/test")

    print("\n" + "=" * 84)
    print(f"G1 RE-ADJUDICATION — saved checkpoint, {len(te)} test episodes")
    print("=" * 84)
    print(f"{'arm':<12}{'dispR2MOV':>11}{'rmseMOV_mm':>12}{'AbsRel':>9}{'n':>4}")
    for k, v in res.items():
        f = lambda m: (np.nanmean(v[m]) if v[m] else np.nan)
        print(f"{k:<12}{f('mov'):>11.4f}{1000*f('rmse_mov'):>12.2f}"
              f"{f('absrel'):>9.4f}{len(v['mov']):>4}")

    contrasts = {}
    for name in ("B0_flat", "B3_epmed"):
        contrasts[f"mov_lifter_minus_{name}"] = boot_paired(
            res["lifter"]["mov"], res[name]["mov"], seed=args.seed)
    c_ar = boot_paired(res["depthnet"]["absrel"], res["lifter"]["absrel"],
                       seed=args.seed)
    r2_mov = float(np.nanmean(res["lifter"]["mov"]))
    b3 = float(np.nanmean(res["B3_epmed"]["mov"]))
    ceil = float(np.nanmean(res["C_gtdepth"]["mov"]))
    skill = (r2_mov - b3) / (ceil - b3) if ceil > b3 else float("nan")

    prev = {}
    p = R.BASE / "lifter_g1.json"
    if p.exists():
        prev = json.loads(p.read_text())
    c1 = bool(prev.get("criteria", {}).get("c1_vs_2donly", False))
    c2 = contrasts["mov_lifter_minus_B0_flat"]["lo"] > 0
    c3 = bool(c_ar["lo"] > 0)
    c4 = contrasts["mov_lifter_minus_B3_epmed"]["lo"] > 0
    c5 = bool(np.isfinite(r2_mov) and r2_mov > 0)
    passed = bool(c1 and c2 and c3 and c4 and c5)

    print("\nGATE (corrected criteria):")
    print(f"  1. dispR2 MOV > B1 2D-only     ... "
          f"{'PASS' if c1 else 'FAIL'}   [CARRIED FORWARD, not recomputed]")
    for lbl, ok, c in (("2. dispR2 MOV > B0 flat     ", c2,
                        contrasts["mov_lifter_minus_B0_flat"]),
                       ("4. dispR2 MOV > B3 epmed    ", c4,
                        contrasts["mov_lifter_minus_B3_epmed"])):
        print(f"  {lbl}... {'PASS' if ok else 'FAIL'}   "
              f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] n={c['n']}")
    print(f"  3. AbsRel depthnet-lifter lo>0 ... {'PASS' if c3 else 'FAIL'}   "
          f"{c_ar['mean']:+.4f} [{c_ar['lo']:+.4f},{c_ar['hi']:+.4f}] "
          f"n={c_ar['n']}")
    print(f"  5. dispR2 MOV > 0              ... {'PASS' if c5 else 'FAIL'}   "
          f"{r2_mov:+.4f}")
    print(f"  G1 {'PASSED' if passed else 'FAILED'}")
    print(f"\n  SKILL vs B3 -> ceiling: {skill:.3f}   "
          f"(B3 {b3:+.3f} | lifter {r2_mov:+.3f} | ceiling {ceil:+.3f})")
    print(f"  rmse_mov lifter {1000*np.nanmean(res['lifter']['rmse_mov']):.1f} mm"
          f" | ceiling {1000*np.nanmean(res['C_gtdepth']['rmse_mov']):.1f} mm"
          f" | contact signal 4.55 mm")

    cams = res["lifter"]["cam"]
    strat = {}
    for k, v in res.items():
        strat[k] = {c: dict(
            r2=float(np.nanmean([v["mov"][i] for i, cc in
                                 enumerate(v["cam"]) if cc == c])),
            rmse_mm=float(1000 * np.nanmean([v["rmse_mov"][i] for i, cc in
                                             enumerate(v["cam"]) if cc == c])),
            n=v["cam"].count(c)) for c in sorted(set(cams))}
    print("\nSTRATIFIED dispR2 MOV: " + " | ".join(
        f"{c} n={cams.count(c)} lifter {strat['lifter'][c]['r2']:+.3f}"
        for c in sorted(set(cams))))

    rec = dict(gate="G1 LIFT (re-adjudicated)", passed=passed,
               dataset="rcasa", n_test=len(te),
               criteria=dict(c1_vs_2donly=c1, c2_vs_flat=c2,
                             c3_absrel_paired=c3, c4_vs_b3epmed=c4,
                             c5_r2_positive=c5),
               c1_carried_forward_not_recomputed=True,
               contrasts=contrasts,
               contrast_absrel_depthnet_minus_lifter=c_ar,
               r2_mov=r2_mov, skill_vs_b3=float(skill),
               cam_counts={c: cams.count(c) for c in set(cams)},
               stratified=strat,
               summary={k: {m: float(np.nanmean(v[m])) if v[m] else None
                            for m in v if m not in ("cam", "stem")}
                        for k, v in res.items()})
    outp = R.BASE / "lifter_g1_revised.json"
    outp.write_text(json.dumps(rec, indent=1))
    R.log("gate_g1_readjudicated", passed=passed, criteria=rec["criteria"],
          skill_vs_b3=float(skill), r2_mov=r2_mov)
    print(f"\nwrote {outp}")
    return rec


def run_depthsweep(args):
    """THE DEPTH STOPPING RULE — what depth accuracy does THE LIFT need?

    c5 failed: dispR2 MOV -0.409, so the lifter is still beaten by
    "emit the window's mean displacement". Attribution says depth binds
    (removing ALL tracker ray error buys 3.2%). The question that
    decides whether to spend 9 h retraining is therefore NOT "can we
    train a better depth head" but "how good would a depth head have to
    be, and is that reachable at all". L1.1b asked this for CONTACT. It
    has never been asked for the LIFT.

    Method: take GT depth, corrupt it at controlled magnitude, push it
    through the SAME harness that scored every arm, and read off the
    AbsRel at which dispR2 crosses each threshold. No training, no
    tracking, no GPU.

    THREE NOISE STRUCTURES, because they are not interchangeable:
      iid     per point per frame          - pure sensor noise
      static  per point, constant in time  - a wrong but STABLE depth
      global  per frame, whole-scene scale - the scene breathes

    `static` is the control that answers the metric-blindness charge:
    a constant per-point offset cancels exactly in X_t - X_t0, so it
    should leave dispR2 almost untouched while wrecking absolute
    position. If that is what happens, dispR2 is confirmed insensitive
    to the error mode that matters for contact, and abs_mov is the
    honest column.

    PRE-REGISTERED INTERPRETATION (fixed before the run, so it cannot be
    rationalised after):
      required AbsRel ~0.002  -> lift route dead for the same physical
                                 reason contact was. STOP, say so.
      required 0.10 - 0.20    -> lifter at 0.310, gap ~2x, ordinary
                                 training problem, 9 h retrain justified.
      required >= 0.30        -> depth head already good enough; the
                                 shortfall is ARCHITECTURAL and
                                 retraining the same model is wrong.
    """
    te = load_split("rcasa", SP.TEST)
    print(f"test episodes: {len(te)}   seeds: {args.seeds}")
    # sigma is multiplicative on depth: z' = z * exp(sigma * eps).
    # AbsRel is then ~ sigma*sqrt(2/pi) for small sigma, so this sweep
    # spans AbsRel from ~0.0008 to ~0.6 and brackets every threshold.
    SIG = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10,
           0.15, 0.20, 0.30, 0.50, 0.80]

    def noisy_predictor(sigma, kind, seed):
        def f(ray, vis, e=None, t0=None):
            gz = e["gz"][t0:t0 + ray.shape[0]]
            T, P = gz.shape
            # deterministic per (episode, window, level, seed)
            h = abs(hash((e["stem"], t0, kind, float(sigma), seed))) % (2**31)
            rng = np.random.default_rng(h)
            if kind == "iid":
                n = rng.standard_normal((T, P))
            elif kind == "static":
                n = np.repeat(rng.standard_normal((1, P)), T, 0)
            else:                                  # global per-frame scale
                n = np.repeat(rng.standard_normal((T, 1)), P, 1)
            z = np.clip(gz, 0.05, None) * np.exp(sigma * n)
            return np.log(np.clip(z, 0.05, None)).astype(np.float32)
        return f

    rows = []
    for kind in ("iid", "static", "global"):
        for sigma in SIG:
            per_seed = []
            for sd in range(args.seeds):
                preds = {"n": noisy_predictor(sigma, kind, sd)}
                r = score_all(te, preds, desc=f"{kind} s={sigma} seed{sd}")["n"]
                per_seed.append(dict(
                    absrel=float(np.nanmean(r["absrel"])),
                    mov=float(np.nanmean(r["mov"])),
                    abs_mov=float(np.nanmean(r["abs_mov"])),
                    rmse_mov=float(np.nanmean(r["rmse_mov"]))))
            m = {k: float(np.mean([p[k] for p in per_seed]))
                 for k in per_seed[0]}
            m.update(kind=kind, sigma=sigma,
                     mov_sd=float(np.std([p["mov"] for p in per_seed])))
            rows.append(m)
            print(f"  {kind:<7} sig={sigma:<6} AbsRel={m['absrel']:.4f}  "
                  f"dispR2={m['mov']:+.4f} (sd {m['mov_sd']:.4f})  "
                  f"abs={1000*m['abs_mov']:7.2f}mm  "
                  f"disp={1000*m['rmse_mov']:6.2f}mm", flush=True)

    print("\n" + "=" * 78)
    print("REQUIRED DEPTH ACCURACY FOR THE LIFT (AbsRel at each threshold)")
    print("=" * 78)
    THRESH = [("dispR2 > 0        (clears c5)", 0.0),
              ("dispR2 >= -0.409  (= today's lifter)", -0.409),
              ("dispR2 >= +0.683  (= C_gtdepth ceiling)", 0.683)]
    req = {}
    for kind in ("iid", "static", "global"):
        rs = sorted([r for r in rows if r["kind"] == kind],
                    key=lambda r: r["absrel"])
        for label, thr in THRESH:
            # coarsest (largest AbsRel) level still meeting the threshold
            ok = [r for r in rs if r["mov"] >= thr]
            best = max(ok, key=lambda r: r["absrel"]) if ok else None
            req[f"{kind}|{label}"] = best["absrel"] if best else None
            got = f"{best['absrel']:.4f}" if best else "UNREACHABLE"
            print(f"  {kind:<7} {label:<40} AbsRel <= {got}")

    lift = 0.3101
    print(f"\n  lifter's measured AbsRel today: {lift:.4f}")
    key = req.get("iid|dispR2 > 0        (clears c5)")
    if key is None:
        verdict = ("UNREACHABLE at any tested noise level — investigate "
                   "the harness before concluding anything")
    elif key < 0.005:
        verdict = ("STOP: the lift route is dead for the same physical "
                   "reason contact was")
    elif key < 0.10:
        verdict = (f"HARD: needs AbsRel <= {key:.4f} vs {lift:.4f} today, "
                   f"a {lift/key:.1f}x improvement")
    elif key < 0.30:
        verdict = (f"ORDINARY TRAINING PROBLEM: needs <= {key:.4f} vs "
                   f"{lift:.4f}, gap {lift/key:.1f}x — retrain justified")
    else:
        verdict = (f"ARCHITECTURAL: depth head at {lift:.4f} already "
                   f"meets the required {key:.4f}; retraining the same "
                   f"model is the wrong move")
    print(f"  PRE-REGISTERED VERDICT: {verdict}")

    st = [r for r in rows if r["kind"] == "static"]
    ii = [r for r in rows if r["kind"] == "iid"]
    print("\n  METRIC-BLINDNESS CHECK (static offset cancels in displacement):")
    for s in (0.05, 0.20):
        a = next((r for r in st if r["sigma"] == s), None)
        b = next((r for r in ii if r["sigma"] == s), None)
        if a and b:
            print(f"    sigma={s}: static dispR2 {a['mov']:+.4f} / abs "
                  f"{1000*a['abs_mov']:.1f}mm   vs   iid dispR2 "
                  f"{b['mov']:+.4f} / abs {1000*b['abs_mov']:.1f}mm")

    rec = dict(kind="lifter_depthsweep", n_test=len(te), seeds=args.seeds,
               lifter_absrel=lift, rows=rows, required=req, verdict=verdict,
               preregistered=dict(dead="<0.005", hard="0.005-0.10",
                                  training="0.10-0.30", architectural=">=0.30"))
    outp = R.BASE / "lifter_depthsweep.json"
    outp.write_text(json.dumps(rec, indent=1))
    R.log("lifter_depthsweep", n_test=len(te), seeds=args.seeds,
          required_iid_c5=key, verdict=verdict)
    print(f"\nwrote {outp}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "smoke", "ood", "attrib",
                                    "regate", "depthsweep"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "smoke":
        tr = load_split("rcasa", SP.TRAIN, limit=4)
        va = load_split("rcasa", SP.VAL, limit=2)
        net, m = train(tr, va, steps=60, d=64, bs=4, tag="smoke")
        print("smoke ok", m["params"], "best_val", round(m["best_val"], 4))
        r = eval_episode(va[0], predictor(net))
        print("eval keys", {k: round(v, 4) for k, v in r.items()})
    elif a.cmd == "ood":
        run_ood(a)
    elif a.cmd == "attrib":
        run_attrib(a)
    elif a.cmd == "regate":
        run_regate(a)
    elif a.cmd == "depthsweep":
        run_depthsweep(a)
    else:
        run_gate(a)


if __name__ == "__main__":
    main()
