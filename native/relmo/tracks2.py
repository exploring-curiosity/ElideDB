"""Track extraction v2 — motion-seeded queries + depth supervision.

Two changes over v1, both aimed at signal density:

1. QUERY POINTS ARE MOTION-SEEDED, not a uniform grid. A uniform grid
   put only 4-16% of its points on the objects that matter (measured);
   the rest tracked floor. Here frame-differencing finds where the
   recording changes and puts most points there, with a thin uniform
   background sample kept so the model still sees static context.
   Crucially this uses MOTION, not categories - it is computable at
   deployment on any video, so training and inference see the same
   kind of point distribution. Sampling that needed labels would have
   introduced a train/serve gap.

2. DEPTH is carried through as per-point supervision, so RelMo can be
   trained to lift 2D tracks into latent 3D. Contact and support are
   3D facts; the image plane provably cannot express them (box gap
   measured 0.00 median for BOTH touching and separated pairs).

Deployment note: seeding needs only the video. Depth and body ids are
privileged and are used at training time only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# CoTracker3 uses aten::grid_sampler_3d, which MPS does not implement.
# Measured: 609/609 rcasa episodes failed with that op missing while the
# job still reported 600/600 "done". The CPU fallback for that one
# kernel produces GT coverage 1.000 but is SLOW, and an earlier version
# of this comment understated it as "~14s/episode".
#
# MEASURED, from this module's own tqdm bar (tracks2.log):
#     rcasa       100 eps / 33,110 frames @ 320x240 -> 1:57:19
#                 = 70.40 s/ep = 212.6 ms/frame = 4.7 fps = 0.16x realtime
#     physgen_v3  2000 eps (shorter, smaller) -> ~7.4 s/ep
# The per-EPISODE figure is not portable (episode length varies 163-1082
# frames); 212.6 ms/FRAME is the number to reason with. One hour of
# 30 fps video = 6.4 hours of tracking.
#
# This is ~70x over the owner's stated serve budget (< 1 s/episode) and
# is the single dominant serve-time cost — everything downstream is
# sub-10 ms. Treat any latency claim elsewhere in the repo as wrong
# unless it cites this bar.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

NQ = 384                 # query points per clip
BG_FRAC = 0.25           # fraction spent on static context
SEG_DS = 2
VER = 3
# The FROZEN tracker. The same checkpoint that runs at eval runs
# here, its id is stamped into every cache, and trackval.py scores it
# against pixel-exact sim GT before a dataset is accepted - owner
# directive: "if cotracker itself fails during eval then what?"
TRACKER = os.environ.get("RELMO_TRACKER", "cotracker3_online")
DEV = os.environ.get("RELMO_DEV", "mps")
_M = None


def model():
    global _M
    if _M is None:
        import torch
        _M = torch.hub.load("facebookresearch/co-tracker",
                            TRACKER).to(DEV).eval()
    return _M


def frames(mp4, wh=None):
    """Decode to (T,H,W,3). Resolution is PROBED, not assumed - the
    corpus is no longer one generator: physgen renders 640x480 and
    imported Kubric MOVi is 256x256."""
    if wh is None:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(mp4)], stdout=subprocess.PIPE, check=True)
        w, h = (int(x) for x in pr.stdout.decode().strip().split(","))
    else:
        w, h = wh
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def seed_queries(F, n=NQ, rng=None):
    """Where does this recording CHANGE? Put points there. Video-only,
    no labels, identical at train and deploy."""
    rng = rng or np.random.default_rng(0)
    T, H, W = F.shape[:3]
    idx = np.linspace(0, T - 1, min(T, 12)).astype(int)
    S = F[idx].astype(np.int16)
    E = np.abs(np.diff(S, axis=0)).sum(-1).max(0).astype(np.float32)
    E = E / max(E.max(), 1e-6)
    ys, xs = np.where(E > 0.12)
    n_mov = int(n * (1 - BG_FRAC))
    q = []
    if len(ys) > 0:
        take = rng.choice(len(ys), min(n_mov, len(ys)),
                          replace=len(ys) < n_mov)
        q += [(0.0, float(xs[i]), float(ys[i])) for i in take]
    n_bg = n - len(q)
    g = int(np.ceil(np.sqrt(max(n_bg, 1))))
    for yy in np.linspace(8, H - 8, g):
        for xx in np.linspace(8, W - 8, g):
            if len(q) >= n:
                break
            q.append((0.0, float(xx), float(yy)))
    return np.asarray(q[:n], np.float32)


def _run_tracker(F, q):
    """Tracks for the whole clip, with memory that does not grow with T.

    The offline predictor takes the entire video as one float32 tensor on
    device and holds features for every frame. MEASURED on rcasa: a T=444
    episode drove RSS to ~7 GB, swap to 94% full, and the job fell from
    283% CPU to 15% thrashing - it stopped writing artifacts entirely.
    The remaining corpus has median T=225 but p90 527 and max 1125, so
    offline could not have finished this dataset at all.

    The online predictor is the supported constant-memory path: it slides
    a window of step*2 frames, keeps its own state across calls, and
    returns tracks accumulated over every frame seen so far, so point
    identity still spans the whole episode. The video stays on the CPU as
    uint8 and only the current window is converted and moved to device.

    It is a DIFFERENT checkpoint (scaled_online.pth), so it is a
    different frozen tracker and trackval re-scores it against pixel GT
    before the corpus is accepted - the same rule the offline one passed.
    """
    import torch
    M = model()
    Fc = torch.from_numpy(np.ascontiguousarray(F))       # uint8, CPU
    Q = torch.tensor(q)[None].to(DEV)

    def win(a, b):
        return Fc[a:b].permute(0, 3, 1, 2)[None].float().to(DEV)

    # Only the online predictor has .step; the offline one takes the whole
    # video in one call. Dispatch on the capability rather than on the
    # tracker name so either checkpoint runs through this function - which
    # is what makes an online-vs-offline A/B on identical episodes
    # possible at all.
    step = getattr(M, "step", None)
    if step is None:
        with torch.no_grad():
            tr, vs = M(win(0, len(Fc)), queries=Q)
        return tr[0].cpu().numpy(), vs[0].cpu().numpy()

    with torch.no_grad():
        M(video_chunk=win(0, step * 2), is_first_step=True, queries=Q)
        tr = vs = None
        for i in range(0, max(len(Fc) - step, 1), step):
            tr, vs = M(video_chunk=win(i, i + step * 2))
    if tr is None:
        raise RuntimeError(f"tracker returned nothing for T={len(Fc)}")
    return tr[0].cpu().numpy(), vs[0].cpu().numpy()


def extract(ep_dir: Path, out: Path):
    import gc

    import torch
    F = frames(ep_dir / "frames.mp4")
    st = np.load(ep_dir / "state.npz")
    seg = st["seg"]
    dep = st["depth"].astype(np.float32) if "depth" in st else None
    q = seed_queries(F)
    xy, vs = _run_tracker(F, q)
    gc.collect()
    torch.mps.empty_cache()
    T = min(len(xy), len(seg))
    xy, vs, seg = xy[:T], vs[:T], seg[:T]
    H, W = seg.shape[1:]
    # seg/depth are stored at their own scale (physgen half-res,
    # imported MOVi full-res) - derive it instead of assuming
    ds = max(F.shape[2] // W, 1)
    ix = np.clip((xy[..., 0] / ds).astype(int), 0, W - 1)
    iy = np.clip((xy[..., 1] / ds).astype(int), 0, H - 1)
    flat = (iy * W + ix)
    # uint16: scenes carry 300-500 bodies, uint8 merged everything
    # above id 255 into one and corrupted the part labels.
    bid = np.take_along_axis(seg.reshape(T, -1), flat, 1).astype(np.uint16)
    pdep = (np.take_along_axis(dep[:T].reshape(T, -1), flat, 1)
            .astype(np.float16) if dep is not None else None)
    P = xy.shape[1]
    ident = np.zeros(P, np.uint16)
    for p in range(P):
        b = bid[:, p][vs[:, p] > 0.5]
        if len(b):
            v, c = np.unique(b, return_counts=True)
            ident[p] = v[np.argmax(c)]
    # float32 ON PURPOSE: fp16 put a 0.25-0.5px staircase under every
    # slow motion (the most common per-frame steps in the v2 cache
    # were literally the fp16 grid values) - measured, not guessed
    payload = dict(ver=VER, tracker=TRACKER, xy=xy.astype(np.float32),
                   vis=vs.astype(bool), bid=bid, ident=ident)
    # privileged extras are per-source; MOVi ships object velocities
    # instead of MuJoCo contact forces, so carry whatever exists
    for k_, dst in (("contact", "contact"), ("xpos", "xpos"),
                    ("xvel", "xvel"), ("obj_positions", "xpos"),
                    ("obj_velocities", "xvel")):
        if k_ in st.files and dst not in payload:
            payload[dst] = st[k_]
    # RELATIONAL CHANNEL. The copy list above asks for a key named
    # "contact"; the replayer writes "contact_pairs"/"contact_n", so the
    # contact channel never reached a single track file - the same
    # key-mismatch shape as window_starts. It matters more than the
    # trajectory does: motion-R2 over displacement is INVARIANT to the
    # distinction this project exists to make, because pushing a block
    # left-to-right and closing a drawer left-to-right give the same point
    # displacements. What separates them is relational - which bodies are
    # in contact, and whether the moving body is articulated against a
    # parent (bounded by jnt_range) or free on a support. All of it is
    # recorded per frame and none of it was reaching the model.
    # Privileged: training/scoring signal only, never read at serve time.
    for k_ in ("contact_pairs", "contact_n", "qpos", "xquat",
               "body_parentid", "jnt_type", "jnt_axis", "jnt_range",
               "jnt_bodyid", "jnt_qposadr", "target_bodies", "body_mass"):
        if k_ in st.files:
            payload[k_] = st[k_]
    # window_starts rides along because train_wm2.load() samples t0 from
    # it and opens the TRACK file, not the episode - without it that
    # branch is dead and every sample silently takes the uniform-t0
    # fallback it exists to avoid.
    #
    # Measured on THIS corpus, the two are the same: uniform windows have
    # median max-displacement 0.0750 m with 9% under 1 cm, window_starts
    # 0.0722 m with 10% under 1 cm. It matters on a corpus full of dead
    # time; here the replayer's per-window visibility and motion gates
    # already rejected those cameras at generation time, so 91% of
    # uniform windows contain real motion. Carried anyway - it costs
    # three small arrays, and the trainer should not depend on a
    # generation-time gate staying this strict.
    for k_ in ("window_starts", "window_len", "n_windows"):
        if k_ in st.files:
            payload[k_] = st[k_]
    if pdep is not None:
        payload["pdepth"] = pdep
    # GT TARGETS from sim state: lift each query pixel through
    # depth+seg into a body-fixed point and project it through the
    # stored poses. Training predicts gxy (exact physics); the noisy
    # tracker xy is the INPUT - so the model learns to see through
    # the same tracker it will meet at eval, but is never graded on
    # reproducing that tracker's noise.
    from relmo.pixelgt import open_gt
    gt = open_gt(ep_dir)
    gxy = np.zeros_like(xy, np.float32)
    gvis = np.zeros(xy.shape[:2], bool)
    # gdist: GT depth per point per frame -> with the camera model this
    #   makes an EXACT 3D track, which is what the world model predicts
    # gbody: which rigid body the point sits on -> the part-segmentation
    #   answer key for G2. Both are training-only scaffolding.
    gdist = np.zeros(xy.shape[:2], np.float32)
    gbody = np.full(P, -1, np.int16)
    for p_ in range(P):
        lt = gt.lift_track(0, float(xy[0, p_, 0]), float(xy[0, p_, 1]))
        if lt is None:
            continue
        n_ = min(T, len(lt["xy"]))
        gxy[:n_, p_] = lt["xy"][:n_]
        gvis[:n_, p_] = lt["vis"][:n_]
        gdist[:n_, p_] = lt["dist"][:n_]
        gbody[p_] = lt["body"]
    payload["gxy"] = gxy
    payload["gvis"] = gvis
    payload["gdist"] = gdist
    payload["gbody"] = gbody
    # camera model, so 2D+depth can be lifted to 3D without reopening
    # the episode (per-frame: RoboCasa/Kubric cameras MOVE)
    for k_ in ("cam_pos", "cam_mat", "cam_fovy", "cam_positions",
               "cam_quaternions", "focal_length", "sensor_width",
               "width", "height"):
        if k_ in st.files:
            payload[k_] = st[k_]
    np.savez_compressed(out, **payload)
    return dict(P=int(P), T=int(T), on_body=int((ident > 0).sum()),
                bodies=int(len(np.unique(ident[ident > 0]))))


def build(name="physgen_v2", limit=None, subset=None):
    from tqdm import tqdm
    man = R.read_manifest(name)
    out_dir = R.TRACKS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model()
    eps = man["episodes"] if limit is None else man["episodes"][:limit]
    if subset:
        keep = {ln.strip() for ln in Path(subset).read_text().split()
                if ln.strip()}
        eps = [e for e in eps if e["id"] in keep]
        missing = keep - {e["id"] for e in eps}
        if missing:
            print(f"WARNING: {len(missing)} subset ids not in manifest")
        print(f"subset: {len(eps)} of {len(man['episodes'])} episodes")
    todo = [e for e in eps if not (out_dir / f"{e['id']}.npz").exists()]
    made = 0
    for e in tqdm(todo, unit="ep", desc=f"tracks2/{name}"):
        ep_dir = R.dataset_dir(name) / e["shard"] / e["id"]
        tmp = out_dir / f".tmp_{e['id']}.npz"
        try:
            extract(ep_dir, tmp)
            tmp.rename(out_dir / f"{e['id']}.npz")
            made += 1
        except Exception as exc:
            R.log("track2_error", dataset=name, id=e["id"],
                  error=str(exc)[:200])
            tmp.unlink(missing_ok=True)
    # ARTIFACT COUNT, and it must be the real one. This was glob("ep*.npz"),
    # which matches physgen/movi ids but NOT rcasa's (CloseCabinet_episode_*).
    # Measured: 100 track files on disk for rcasa, ledger recorded total=0;
    # movi_e logged made=1616 total=0. The one field you would use to verify
    # "did this job actually write anything" reported zero for every dataset
    # whose ids are not ep-prefixed. Never pattern-match an artifact count.
    R.log("tracks2_done", dataset=name, made=made,
          total=len(list(out_dir.glob("*.npz"))))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="physgen_v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", default=None,
                    help="file of episode ids, one per line (see relmo.subset)")
    a = ap.parse_args()
    print("new:", build(a.name, a.limit, a.subset))
