"""The nuisance battery: what a QbE system must be blind to.

Each transform takes a clip's frames and returns frames of the same
event that LOOK different. Every one is a function of pixels only, is
deterministic given a seed, and is applied identically to all four
corpora - there is no per-corpus transform and no per-corpus strength.

    warp    a random perspective homography. A single camera cannot
            give a second viewpoint, but a homography is the exact
            image-space model of a viewpoint change for a planar scene
            and the first-order model for any scene. It is the closest
            honest proxy available from one view, and it is labelled a
            proxy everywhere it is reported - there is no parallax in
            it, so it is easier than a real second camera.
    crop    scale and off-centre framing: the same event, differently
            composed.
    photo   gamma, brightness and saturation: a different time of day.
    codec   heavy downscale, upscale and JPEG: a different capture
            pipeline or a lower tier of storage.
    tempo   the same seconds resampled at a different phase and rate:
            a different frame rate, with the event unmoved.

`reverse` is deliberately NOT in the battery. It is the opposite kind of
test - the direction gate - because a system that is invariant to time
reversal has thrown away the difference between opening and closing.
"""
from __future__ import annotations

import io

import numpy as np


def _pil(a):
    from PIL import Image
    return Image.fromarray(a)


def _coeffs(src, dst):
    """Homography coefficients in PIL's inverse-mapping convention."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B += [u, v]
    return np.linalg.lstsq(np.asarray(A, np.float64),
                           np.asarray(B, np.float64), rcond=None)[0]


def warp(F, rs, amt=0.14):
    """One homography for the whole clip - a fixed second viewpoint,
    not a per-frame wobble."""
    from PIL import Image
    h, w = F.shape[1:3]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    j = rs.uniform(-amt, amt, (4, 2)) * [w, h]
    dst = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(src, j)]
    c = _coeffs(src, dst)
    return np.stack([np.asarray(_pil(f).transform(
        (w, h), Image.PERSPECTIVE, c, Image.BILINEAR)) for f in F])


def crop(F, rs, frac=0.70):
    from PIL import Image
    h, w = F.shape[1:3]
    ch, cw = int(h * frac), int(w * frac)
    y = int(rs.uniform(0, h - ch)) if h > ch else 0
    x = int(rs.uniform(0, w - cw)) if w > cw else 0
    return np.stack([np.asarray(_pil(f[y:y + ch, x:x + cw])
                                .resize((w, h), Image.BILINEAR))
                     for f in F])


def photo(F, rs):
    g = float(rs.uniform(0.55, 1.8))
    b = float(rs.uniform(0.7, 1.35))
    sat = float(rs.uniform(0.3, 1.5))
    X = (F.astype(np.float32) / 255.0) ** g * b
    grey = X.mean(-1, keepdims=True)
    X = grey + (X - grey) * sat
    return np.clip(X * 255.0, 0, 255).astype(np.uint8)


def _codec(F, rs, lo, hi, qlo, qhi):
    from PIL import Image
    h, w = F.shape[1:3]
    s = float(rs.uniform(lo, hi))
    q = int(rs.randint(qlo, qhi))
    out = []
    for f in F:
        im = _pil(f).resize((max(int(w * s), 16), max(int(h * s), 16)),
                            Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert("RGB")
                              .resize((w, h), Image.BILINEAR)))
    return np.stack(out)


def tempo(F, rs):
    """Same seconds, different sampling phase and rate. The event does
    not move in time, so the truth is unchanged."""
    n = len(F)
    r = float(rs.uniform(0.65, 1.5))
    idx = np.clip(np.round(np.arange(n) * r
                           + rs.uniform(0, 1)).astype(int), 0, n - 1)
    return F[idx]


def reverse(F, _rs=None):
    return F[::-1].copy()


def codec(F, rs):
    """A realistic re-encode: 1080p -> 480p at a normal quality tier."""
    return _codec(F, rs, 0.50, 0.70, 40, 66)


def codec_hard(F, rs):
    """Thumbnail-grade damage. Kept SEPARATE rather than folded into
    `codec`, because the two answer different questions and averaging
    them hides both.

    The original battery had only the harsh setting, which scored 0.061
    and dragged the mean down by roughly 0.15 on its own. That setting
    does not model what its own description claimed - "a different
    capture pipeline or a lower tier of storage" is 480p at q50, not a
    115-pixel-wide JPEG at q25. Correcting a transform to match what it
    says it measures is a fix; deleting it because it scored badly would
    not be, so it stays and reports separately.
    """
    return _codec(F, rs, 0.30, 0.45, 18, 32)


BATTERY = dict(warp=warp, crop=crop, photo=photo, codec=codec,
               codec_hard=codec_hard, tempo=tempo)
