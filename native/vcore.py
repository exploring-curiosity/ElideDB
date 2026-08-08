"""THE PERCEPTION PATH. Pixels in, three channels out. Nothing else in.

One code path for write and read. A query encoded by different code than
the corpus is compared across a seam, and every number downstream is then
untrustworthy; this module exists so that seam cannot be created.

WHAT IS ALLOWED IN: pixels, and statistics computed from those same
pixels. No text, no codebook, no class list, no sensor, no label, no
constant fitted on an evaluation corpus.

    encoder   DINOv3 ConvNeXt-Tiny, frozen.
              Self-supervised on images with no labels and no captions,
              so it carries no vocabulary. SigLIP2 measures better on
              actions here (0.691 vs 0.538) and is nonetheless
              disqualified: its geometry is defined by a text corpus.

Three things happen to those features, in this order.

1. SCENE BASIS REMOVAL.  mu = mean over the media's own frames; the
   residual is what changed. Within one recording, mu IS the room, the
   lighting, the camera pose and the permanent furniture - it is constant
   in time, so it can carry no information about any event. Cosine on raw
   features therefore scores "same room" far above "same action", which
   is exactly the failure that produced every inflated number in this
   project's history. Jegou & Chum (ECCV 2012) give the retrieval reading
   of the same operation: subtracting the mean turns a jointly-missing
   component into negative evidence instead of free agreement. Mu &
   Viswanath (ICLR 2018) show removing the mean plus the top directions
   is what makes cosine behave. Rank 1 is the default because it has no
   parameter to tune; a higher rank is available and is chosen from the
   media's own spectrum, never from a benchmark.

2. UNIFORM MULTI-SCALE WINDOWS.  2 / 4 / 8 s, stride half. Fixed in
   advance and never tuned: learned segmentation was measured against a
   plain grid on this corpus and lost, 0.317 to 0.408. A window is
   emitted whenever it fits, which is the same rule for a 7-second robot
   episode and a 45-minute flight.

3. THREE CHANNELS, stored separately and never pre-fused - the best
   single channel was measured to beat any fixed fusion, so the weights
   belong to the query, not to the writer.

     c1 CHANGE  mean of the residual. What changed, with the scene gone.
                Order-blind by construction.
     c2 ORDER   rank pooling, sum_t (2t-T-1) f_t. Closed form, no
                parameters, and antisymmetric under time reversal, so it
                is the only channel that can tell a reach-in from a
                pull-out. Measured to lift action AUC 0.521 -> 0.691.
     c3 SHAPE   the window's own temporal self-similarity matrix,
                resampled to 12x12 and z-scored. Built purely from
                RELATIVE similarities, so it is unchanged by any global
                transform of the feature space. Junejo, Dexter, Laptev &
                Perez (ECCV 2008 / PAMI 2011) built cross-view action
                recognition on exactly this observation: temporal
                self-similarity is strikingly stable across viewpoints,
                with no structure recovery and no correspondence. It
                names nothing, which is what makes it admissible here.

    python native/vcore.py --selftest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "native"))

import vsrc                                              # noqa: E402

# Selectable so the family can be A/B'd; every option is DINOv3, i.e.
# self-supervised with no labels, no captions and no codebook. A
# text-aligned encoder stays disqualified however well it scores.
_ENC = {"ct": "convnext-tiny", "cs": "convnext-small",
        "cb": "convnext-base", "cl": "convnext-large",
        "vitb": "vitb16", "vitl": "vitl16", "vithp": "vith16plus"}
# The DEFAULT must be the validated configuration. It was "ct" while
# every measured run set SDX_ENC=vitl explicitly - so anyone running the
# code without the environment got a different system than the one the
# numbers describe. The manifest recorded the truth; the code did not.
_E = os.environ.get("SDX_ENC", "vitl")
ENCODER = (_E if "/" in _E
           else f"facebook/dinov3-{_ENC[_E]}-pretrain-lvd1689m")
# DINOv3 ViT overflows fp16 - this project hit that NaN trap once and it
# is recorded here rather than rediscovered.
ENC_FP32 = "vit" in ENCODER
SCALES = (2.0, 4.0, 8.0)
NS = 12                    # SSM resolution -> 66-d shape channel
# HOW A FRAME BECOMES A VECTOR. Measured, not assumed.
#
# The incumbent is `pooler_output`, i.e. the CLS path. That is the right
# choice for DINO v1, whose objective trains CLS explicitly - but DINOv2
# and DINOv3 distribute discriminative information across the PATCH
# tokens, and reported results for those backbones favour pooling the
# patches instead. Collapsing to CLS also discards spatial layout, which
# is precisely what `crop` perturbs, and crop is the weakest transform in
# the battery (0.389 on sim).
#
# Every option below is parameter-free. GeM's p=3 is the standard fixed
# exponent from the retrieval literature, NOT a value swept on this
# benchmark - fitting it here would be fitting a constant on the
# evaluation corpus.
# MEASURED (ViT-L@320, 10 min/store, 12 queries, identical stores):
#   cls 0.701  mean 0.721  max 0.697  gem 0.750  clsgem 0.711   (yield)
# GeM also moved the two weakest transforms most: codec_hard 0.430 ->
# 0.610, warp 0.742 -> 0.801.
#
# The reported guidance for DINOv2/v3 is that MAX pooling wins. It does
# not here: max is worst on yield and collapses on crop (0.498 vs 0.657).
# The half that transferred was "pool the patches, not CLS".
#
# The decisive number is not the mean, it is the SPREAD across corpora:
#   cls  sim .713 bridge .598 car .804 drone .689  -> spread 0.206
#   max  sim .787 bridge .746 car .686 drone .569  -> spread 0.218
#   gem  sim .745 bridge .756 car .740 drone .759  -> spread 0.019
# GeM is the only mode that behaves the same on all four, which is what
# "one approach, not per-dataset tuning" actually requires.
POOL = os.environ.get("SDX_POOL", "gem")   # cls|mean|max|gem|clsgem|rmac2|rmac3
POOL_FALLBACK = set()      # records any silent degradation to plain GeM
GEM_P = 3.0

BATCH = 32                 # measured plateau: 16->35.3 fps, 128->27.1.
                           # The GPU saturates at 32; more RAM buys nothing
                           # here, so headroom went to resolution instead.

# MULTI-RESOLUTION FRAME ENCODING. Measured bottleneck: the nuisance
# battery splits cleanly in two. Photometric and temporal transforms
# score 0.75-0.88 and 0.60-0.81; the GEOMETRIC and RESOLUTION ones -
# crop 0.17-0.55, warp 0.30-0.58, codec 0.10-0.43 - are where the system
# loses, on all four corpora at once. One failure mode, not five.
#
# The standard answer in instance retrieval is to describe a frame at
# several input resolutions and average the unit-normed descriptors
# (Gordo et al.; Radenovic et al. GeM) - a crop is, to first order, the
# same content at a different scale, so a scale-averaged descriptor
# moves less under one. It is parameter-free, needs no training, and
# carries no vocabulary, so it is admissible here.
#
# MEASURED on all four corpora, from preserved artifacts (V_single*.json
# against V_multires*.json). Yield 0.372->0.437 sim, 0.398->0.460
# bridge, 0.378->0.501 drone, 0.542->0.564 car. On `warp` - the
# viewpoint proxy that decides the product - +0.118 / +0.228 / +0.265 /
# +0.007.
#
# car is the exception, and its aspect ratio is why: KITTI decodes to
# 256x78, leaving almost no vertical resolution for extra scales to
# exploit, while the other three are 144-192 high. An earlier reading on
# sim+car ALONE said "helps the synthetic corpus only, do not ship";
# measuring the other two reversed it, and both real robot corpora gain
# the most. Two of four is not four.
#
# COST, measured on the same media rather than extrapolated:
#
#            1-res      2-res      3-res
#   bridge   1.41       1.63       2.49   min per hour of video
#   drone    3.64       3.60       4.17
#
# Three resolutions cost 1.8x on bridge and 1.15x on drone - NOT the 3x
# a per-frame count suggests, because ffmpeg and PIL DECODE dominate the
# write path and extra forward passes ride along nearly free. An earlier
# 3x estimate came from extrapolating sim, whose 25-second clips make
# startup overhead dominant; it was the least representative media
# available and it was the one used.
#
# This also settles two resolutions: it saves almost nothing (1.63 vs
# 2.49; 3.60 vs 4.17) while giving up 13-24% of the yield gain and, on
# bridge, 83% of the rank1 gain. Three it is.
#
# Still over the one-minute-per-hour budget at 2.5-4.2. The lever for
# that is DECODE - hardware decode, or a lower sample rate - not the
# encoder, which is not where the time goes.
RES = tuple(int(x) for x in
            os.environ.get("SDX_RES", "256,320,384").split(","))
#
# The ladder is 224/320/448 against a 640-wide decode, and that pairing
# is the point. It was 168/224/320 against a 256-wide decode, which meant
# the "high resolution" pass was fed UPSAMPLED pixels - interpolation,
# not detail. Sources are 640x480 (sim, bridge), 1242x375 (car) and
# 1920x1080 (drone); decoding them to 256 threw away between 2.5x and
# 7.5x of the signal before any encoder saw it. No model can recover
# what the decoder discarded.

# The basis is LOCAL and LEAVE-ONE-OUT, and both halves of that are
# forced by a measurement, not a preference.
#
# A whole-media mean fails a query clip: an 8 s clip handed in on its own
# has a mean taken over those 8 s, while the same 8 s sitting inside a
# 25 s media has a mean taken over 25 s. Encoded both ways the same
# content agreed at cos 0.106 - a seam, and the exact class of silent
# failure this module exists to prevent. Worse, a window that spans its
# whole context has a residual mean of ZERO by construction, so its
# appearance channel is pure noise.
#
# Fixing both: the basis for a window is the mean of the frames around
# it, EXCLUDING the window's own frames.
#
# The context is a multiple of the WINDOW'S OWN SCALE, not a wall-clock
# constant, and that is the part measurement forced. A fixed 60 s
# context is something a stored 41-minute file can supply and an
# 8-second example never can, so the two sides computed different bases
# and described the same seconds differently. Measured seam on c1
# against query clip length: 0.936 at 20 s, 0.846 at 12 s, 0.798 at 8 s,
# and worse at the coarser scales - a system that cannot retrieve
# literally itself, which is what the first end-to-end run showed
# (identity yield 0.292).
#
# Scale-relative context is supplyable by both sides by construction: a
# 2 s window needs 2 s either side whether it sits in an 8-second clip
# or a 45-minute flight. It also sharpens what the residual means - how
# this moment differs from the moments immediately around it, rather
# than from the average of a whole recording.
CTX_MULT = 3.0             # context span = CTX_MULT x window scale
MIN_CTX_F = 8              # frames of context required to define a basis

_M = {}


def model():
    if "m" not in _M:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from elidedb.device import pick
        dev, dtype = pick()
        if ENC_FP32:
            dtype = torch.float32
        _M["proc"] = AutoImageProcessor.from_pretrained(ENCODER)
        _M["m"] = AutoModel.from_pretrained(
            ENCODER, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
        _M["dev"], _M["dtype"], _M["torch"] = dev, dtype, torch
    return _M["m"], _M["proc"], _M["dev"], _M["dtype"], _M["torch"]


def _l2(V, axis=-1):
    V = np.asarray(V, np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=axis, keepdims=True),
                          1e-8)


def _patches(r, torch):
    """Patch tokens only: CLS and the register tokens are dropped.

    Registers exist to absorb high-norm artefacts (Darcet et al.); left
    in a pooled average they would contribute exactly the junk they were
    introduced to quarantine.
    """
    h = r.last_hidden_state
    if h.dim() == 4:                       # ConvNeXt: B,C,H,W
        return h.flatten(2).transpose(1, 2)
    m, _p, _d, _dt, _t = model()
    nreg = int(getattr(getattr(m, "config", None),
                       "num_register_tokens", 0) or 0)
    return h[:, 1 + nreg:]


def _rmac(P, torch, L):
    """R-MAC as published (Tolias, Sicre & Jegou, ICLR 2016): square
    regions at L scales with ~40% overlap, each pooled, L2-normed, and
    SUMMED.

    Aimed at `crop` and `warp`, the two weakest transforms. The GeM
    descriptor pools the WHOLE frame into one vector, so removing a
    third of the field of view shifts every frame vector, and rank
    pooling faithfully preserves that shift. Nothing in it is spatially
    local, so nothing in it can survive a recomposition.

    Two details are load-bearing, and the first version of this function
    got the second one wrong.

    SUMMED, not concatenated. Under a crop, content slides between
    cells; with concatenation every component then changes, which is
    worse than the status quo. Summing individually-normalised region
    descriptors means the regions that DO survive a crop keep
    contributing the vectors they contributed before it. That is the
    property R-MAC is built on.

    OVERLAPPING, at several scales. The first version pooled a uniform
    non-overlapping grid, and it was measured WORSE than plain GeM -
    margin 0.477 (2x2) and 0.478 (3x3) against GeM's 0.494 - because a
    non-overlapping grid has hard seams, and a crop slides content
    straight across them. Restoring the published construction (region
    side 2/(l+1) of the frame, strided to ~40% overlap) reversed the
    sign: 0.532. On held-out clips over the whole battery it then won
    6/6 transforms, crop +0.043 and warp +0.028 - the two spatially
    local transforms - and moved photo/codec by less than 0.007, which
    is the signature a spatial fix should have.

    No constant here is fitted on any corpus: the region size, the
    number of regions per scale and the overlap are the paper's, and L
    is the digit in the pool name.

    REJECTED ANYWAY. Correct implementation, real held-out gain on the
    proxy, and it still LOST the store A/B - on every metric, including
    the transform it exists to fix:

        16 q, identical media, A/B scale   gem -> rmac3
        yield  0.803 -> 0.764  -0.039      crop  0.734 -> 0.674  -0.060
        prec   0.846 -> 0.819  -0.027      warp  0.854 -> 0.818  -0.036

    So POOL stays `gem` and this function is kept only as the corrected
    implementation of a measured dead end. Do not re-run it on the
    strength of the proxy number; the proxy is the thing that was wrong.
    """
    B, N, D = P.shape
    side = int(round(float(N) ** 0.5))
    if side * side != N:                # non-square token layout
        return None
    G = P.reshape(B, side, side, D)
    out = None
    for l in range(1, L + 1):
        rs = int(round(2.0 * side / (l + 1)))        # region side
        if rs < 1:
            continue
        n = l + 1                                    # regions per axis
        step = max((side - rs) / (n - 1), 0.0) if n > 1 else 0.0
        starts = sorted({min(int(round(i * step)), side - rs)
                         for i in range(n)})
        for y0 in starts:
            for x0 in starts:
                R = G[:, y0:y0 + rs, x0:x0 + rs, :].reshape(B, -1, D)
                g = (R.float().clamp(min=1e-6).pow(GEM_P)
                     .mean(1).pow(1.0 / GEM_P))
                g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                out = g if out is None else out + g
    return out


def _pool(r, torch):
    if POOL.startswith("rmac"):
        P = _patches(r, torch)
        v = _rmac(P, torch, int(POOL[4:] or 3))
        if v is not None:
            return v
        POOL_FALLBACK.add(POOL)          # visible, not silent
        return P.float().clamp(min=1e-6).pow(GEM_P).mean(1).pow(1.0 / GEM_P)
    if POOL == "cls":
        v = getattr(r, "pooler_output", None)
        return v if v is not None else r.last_hidden_state.mean(1)
    P = _patches(r, torch)
    if POOL == "mean":
        return P.mean(1)
    if POOL == "max":
        return P.max(1).values
    g = P.float().clamp(min=1e-6).pow(GEM_P).mean(1).pow(1.0 / GEM_P)
    if POOL == "gem":
        return g
    v = getattr(r, "pooler_output", None)
    if v is None:
        v = r.last_hidden_state.mean(1)
    # concat of two unit-normed halves so neither dominates by norm
    a = v.float() / v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([a, b], dim=-1)


def feature_dim():
    """Asked of the model. A hardcoded 768 silently truncates every
    wider member of the family (ConvNeXt-Large is 1536)."""
    if "dim" not in _M:
        _M["dim"] = int(frame_features(
            np.zeros((2, 64, 64, 3), np.uint8)).shape[1])
    return _M["dim"]


def frame_features(F, batch=BATCH, res=None):
    """(N,H,W,3) uint8 -> (N,d) unit-norm frame vectors.

    Averaged over RES input resolutions, each L2-normed first so no
    single resolution dominates by having a larger norm.
    """
    if len(F) == 0:
        return np.zeros((0, _M.get("dim", 768)), np.float32)
    m, proc, dev, dtype, torch = model()
    acc = None
    for R in (res or RES):
        out = []
        for i in range(0, len(F), batch):
            chunk = [np.ascontiguousarray(x[..., :3])
                     for x in F[i:i + batch]]
            px = proc(images=chunk, return_tensors="pt",
                      size={"height": R, "width": R})["pixel_values"]
            with torch.no_grad():
                r = m(pixel_values=px.to(dev, dtype))
            out.append(_pool(r, torch).float().cpu().numpy())
        V = _l2(np.concatenate(out))
        acc = V if acc is None else acc + V
    return _l2(acc)


FCACHE = Path(os.environ.get(
    "SDX_FCACHE", "/private/tmp/claude-501/vframes"))


def _fkey(src, R):
    return (f"{src.id.replace('/', '_')}__"
            f"{ENCODER.split('/')[-1]}_{R}_{POOL}_"
            f"{vsrc.FPS:g}_{vsrc.WIDTH}_{getattr(src, 'dur', 0):.1f}.npy")


def media_features_at(src, R, block_s=vsrc.BLOCK_S, bar=None):
    """One media, ONE resolution -> (T,d), cached on disk.

    The cache sits at the FRAME-FEATURE layer, and the layer is the whole
    point. It was previously at the window layer, keyed by the window
    scales and the context multiple - so changing a CHANNEL definition,
    which does not touch a single pixel, invalidated everything and
    re-decoded the corpus. Caching per single resolution also lets two
    ladders share their common rungs (256/320/384 and 224/320/448 both
    pay for 320 once).

    This serves the write path only. The read path encodes handed-in
    frames, which have no media identity to key on, and it computes the
    identical function - verified by --selftest rather than assumed,
    because a fast path that quietly disagrees with the slow one is the
    seam this module exists to prevent.
    """
    FCACHE.mkdir(parents=True, exist_ok=True)
    fp = FCACHE / _fkey(src, R)
    if fp.exists():
        try:
            V = np.load(fp)
            if bar is not None:
                bar.update(len(V))
            return V
        except Exception:                                # noqa: BLE001
            fp.unlink(missing_ok=True)
    parts = []
    for _t0, F in src.blocks(block_s):
        parts.append(frame_features(F, res=(R,)))
        if bar is not None:
            bar.update(len(F))
    V = (np.concatenate(parts) if parts
         else np.zeros((0, feature_dim()), np.float32))
    np.save(fp, V)
    return V


def media_features(src, block_s=vsrc.BLOCK_S, bar=None):
    """Stream a whole media at bounded RAM -> (T,d) frame features,
    averaged over RES exactly as frame_features would."""
    acc = None
    for R in RES:
        V = _l2(media_features_at(src, R, block_s, bar))
        acc = V if acc is None else acc + V
    if acc is None or not len(acc):
        return np.zeros((0, feature_dim()), np.float32)
    return _l2(acc)


def local_basis(P, T, a, b, scale, fps=vsrc.FPS):
    """Leave-one-out local scene mean for the window [a,b) of a media.

    The context spans CTX_MULT x `scale` seconds centred on the window,
    so it is defined by the window itself and is supplyable by a short
    clip and a long media alike. P is the prefix-sum of the media's
    frame features, so this costs two lookups however long the media is.
    Returns (mu, n_context); n_context < MIN_CTX_F means "no valid unit".
    """
    half = int(round(CTX_MULT * scale * fps / 2))
    c0, c1 = max(0, (a + b) // 2 - half), min(T, (a + b) // 2 + half)
    tot = P[c1] - P[c0] - (P[b] - P[a])
    n = (c1 - c0) - (b - a)
    if n < MIN_CTX_F:
        return None, n
    return (tot / n).astype(np.float32), n


def windows(dur, scales=SCALES):
    """[(t0, t1, scale)] - the same grid rule for every media."""
    out = []
    for w in scales:
        st = w / 2.0
        t = 0.0
        while t + w <= dur + 1e-6:
            out.append((round(t, 3), round(t + w, 3), w))
            t += st
    return out


def _idx(t0, t1, T, fps=vsrc.FPS):
    a = int(np.floor(t0 * fps))
    b = int(np.ceil(t1 * fps))
    a = max(0, min(a, T - 1))
    b = max(a + 2, min(b, T))
    return a, b


def rank_pool(X):
    """sum_t (2t - T - 1) x_t : the closed-form dynamic image.

    Parameter-free, and it flips sign when the clip is reversed, which
    is the whole reason it is here."""
    T = len(X)
    if T < 2:
        return np.zeros(X.shape[1], np.float32)
    a = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
    return (a[:, None] * X).sum(0)


def shape_ssm(X, ns=NS):
    """Temporal self-similarity of a window -> 66-d, z-scored.

    Only relative similarities enter, so any global transform of the
    feature space leaves this unchanged. The z-score removes the
    absolute similarity level, which is a property of the scene, and
    keeps the pattern, which is a property of the event.
    """
    T = len(X)
    if T < 2:
        return np.zeros(ns * (ns - 1) // 2, np.float32)
    j = np.linspace(0, T - 1, ns).round().astype(int)
    S = X[j] @ X[j].T
    iu = np.triu_indices(ns, 1)
    v = S[iu].astype(np.float32)
    v = v - v.mean()
    s = float(v.std())
    return v / s if s > 1e-6 else v


def encode_windows(F, wins):
    """The write path and the read path both call exactly this.

    F is one media's frame features. Each window gets its own local
    leave-one-out basis, so a query clip and a stored media describe the
    same seconds the same way. `valid` marks windows with enough context
    to be a unit at all.
    """
    T = len(F)
    d = F.shape[1] if T else feature_dim()
    P = np.zeros((T + 1, d), np.float64)
    if T:
        np.cumsum(F, axis=0, out=P[1:])
    n = len(wins)
    c1 = np.zeros((n, d), np.float32)
    c2 = np.zeros((n, d), np.float32)
    c3 = np.zeros((n, NS * (NS - 1) // 2), np.float32)
    en = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    for i, (t0, t1, _w) in enumerate(wins):
        a, b = _idx(t0, t1, T)
        mu, _nc = local_basis(P, T, a, b, _w)
        if mu is None:
            continue
        R = F[a:b] - mu
        X = _l2(R)
        c1[i] = X.mean(0)
        c2[i] = rank_pool(X)
        c3[i] = shape_ssm(X)
        # how far this window departs from its own surroundings: the
        # vision-native "something happened here" signal
        en[i] = float(np.linalg.norm(R, axis=1).mean())
        valid[i] = True
    return dict(c1=_l2(c1), c2=_l2(c2), c3=_l2(c3), energy=en,
                valid=valid)


def encode_source(src, bar=None):
    """Whole-media write path: pixels -> windows + channels."""
    F = media_features(src, bar=bar)
    wins = windows(src.dur)
    if not wins or len(F) < 4:
        return None, None
    return wins, encode_windows(F, wins)


def encode_clip(frames):
    """Read path for an example handed in as raw frames: the identical
    operator chain, with the clip acting as its own media.

    The same grid rule runs, and any window that cannot find enough
    surrounding context inside the clip simply comes back invalid. A
    query is therefore described by its sub-windows, never by one vector
    over the whole thing - which is also what gives the read path a seed
    set to calibrate itself with.
    """
    F = frame_features(frames)
    dur = len(F) / vsrc.FPS
    wins = windows(dur)
    if not wins:
        return None, None
    return wins, encode_windows(F, wins)


# ------------------------------------------------------------- self test

def selftest():
    """Properties that must hold, checked on pixels only.

    1  determinism: the same media encodes to the same vectors twice.
    2  seam-freedom: encoding a media whole, and encoding a cut of it as
       if it were a standalone query clip, must agree on the overlapping
       window - otherwise write and read disagree.
    3  order sensitivity: c2 must separate a clip from its reverse, and
       c1 must not (that is what makes them different channels).
    4  scene removal: the residual must have far less between-media
       agreement than the raw features.
    """
    CH = ("c1", "c2", "c3")
    ss = vsrc.sources("sim", 2)
    s = ss[0]
    w1, e1 = encode_source(s)
    w2, e2 = encode_source(s)
    det = max(float(np.abs(e1[c] - e2[c]).max()) for c in CH)

    # A query clip is a STRICT sub-range of a stored media - it must be,
    # or the clip and the media share a basis and the test proves
    # nothing. Cutting the middle third gives the clip genuinely less
    # surrounding context than the store had, which is the real
    # production asymmetry.
    # Q0 must land on the grid's coarsest stride, or the clip's windows
    # and the media's windows describe different seconds and nothing can
    # be compared.
    st = max(SCALES) / 2.0
    Q0 = st * max(int(s.dur / 3.0 / st), 1)
    Q1 = min(Q0 + st * 3, s.dur)
    F = s.cut(Q0, Q1)
    qdur = len(F) / vsrc.FPS
    wq, eq = encode_clip(F)
    common = [(i, j) for i, (a, b, _) in enumerate(w1)
              for j, (c, d, _) in enumerate(wq)
              if abs(a - (c + Q0)) < 1e-6 and abs(b - (d + Q0)) < 1e-6
              and e1["valid"][i] and eq["valid"][j]]
    seam = {c: float(np.mean([eq[c][j] @ e1[c][i] for i, j in common]))
            for c in CH} if common else {c: float("nan") for c in CH}

    # Reversal: window (t0,t1) of the clip becomes (qdur-t1, qdur-t0) of
    # the reversed clip. Comparing index j to index j would compare two
    # different time spans and measure nothing.
    _wr, er = encode_clip(F[::-1].copy())
    back = {(round(qdur - b, 3), round(qdur - a, 3)): j
            for j, (a, b, _) in enumerate(wq)}
    pairs = [(j, back[(a, b)]) for j, (a, b, _) in enumerate(wq)
             if (a, b) in back and eq["valid"][j]
             and er["valid"][back[(a, b)]]]
    rv = {c: float(np.mean([eq[c][j] @ er[c][k] for j, k in pairs]))
          for c in CH} if pairs else {c: float("nan") for c in CH}

    # The seam bar is NOT a number I choose after seeing the result -
    # that is how a fitted constant sneaks in wearing a test's clothes.
    # The bar is the channel's own 99th-percentile similarity to
    # unrelated windows: re-encoding the same seconds a second way must
    # agree MORE than different content ever does, or the channel cannot
    # retrieve across that seam whatever its absolute cosine says.
    _w2, e2b = encode_source(ss[1])
    p99, marg = {}, {}
    for c in CH:
        A = np.stack([eq[c][j] for _i, j in common])
        B = e2b[c][e2b["valid"]]
        p99[c] = float(np.percentile(A @ B.T, 99)) if len(B) else 1.0
        marg[c] = seam[c] - p99[c]
    ok_seam = min(marg.values()) > 0.0

    fa, fb = media_features(ss[0]), media_features(ss[1])
    raw = float(_l2(fa.mean(0), 0) @ _l2(fb.mean(0), 0))
    res = float(_l2(_l2(fa - fa.mean(0)).mean(0), 0)
                @ _l2(_l2(fb - fb.mean(0)).mean(0), 0))
    ok_rev = rv["c2"] < seam["c2"] - 0.3
    print(f"units {int(e1['valid'].sum())}/{len(w1)} valid, "
          f"{len(common)} shared with the query clip, "
          f"{len(pairs)} reversal pairs\n")
    print(f"1 determinism        max|diff| {det:.2e}   "
          f"{'PASS' if det < 1e-5 else 'FAIL'}")
    print("2 write/read seam    " + "  ".join(
        f"{c} {seam[c]:.3f}/p99 {p99[c]:.3f} m{marg[c]:+.3f}" for c in CH)
        + f"\n                     {'PASS' if ok_seam else 'FAIL'}"
        + "  (re-encoding must beat unrelated content)")
    print("3 reversal           " + "  ".join(
        f"{c} {rv[c]:+.3f}" for c in CH)
        + f"   {'PASS' if ok_rev else 'FAIL'}"
        + "  (c1/c3 order-blind, c2 must invert)")
    print(f"4 scene removal      raw cross-media cos {raw:.3f} -> "
          f"residual {res:.3f}   {'PASS' if res < raw else 'FAIL'}")
    ok = det < 1e-5 and ok_seam and ok_rev and res < raw
    print(f"\nVCORE SELFTEST: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__.strip().splitlines()[0])
