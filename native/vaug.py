"""The transformation battery: one clip, many disguises, KNOWN distance.

The point of this file is that it supplies a ground truth that contains
no information about the data. A transformation is something I apply, so
its parameters are exactly known and are a property of the operation,
not of the pixels - which makes "same episode, made to look different"
measurable without a label, a verb, or a human.

Every transform is a point in a normalised parameter space where the
origin is identity. Two quantities fall straight out:

    severity(T)     distance from identity  - how disguised is it
    dist(Ti, Tj)    distance between transforms - how differently
                    disguised are these two

and both are things a representation should track: a mild disguise
should stay closer than a heavy one, and two clips wearing similar
disguises should be more alike than two wearing different ones. That is
a RANKING requirement, not a threshold, so it cannot be passed by a
system that simply calls everything similar.

Primitives compose. Sampling is over 1..4 simultaneous primitives at
random strengths, so the battery is dominated by combinations rather
than by the textbook single-axis cases.

    python native/vaug.py --preview       # look at what it produces
"""
from __future__ import annotations

import io

import numpy as np

# Each primitive: (name, lo, hi) in NATURAL units, plus a normaliser so
# that a "full strength" application contributes ~1.0 of severity.
PRIMS = {
    # geometric
    "rot":     dict(lo=-40.0, hi=40.0,  norm=40.0),   # degrees
    "mirror":  dict(lo=0, hi=1,         norm=1.0),    # binary
    "vflip":   dict(lo=0, hi=1,         norm=1.0),
    "zoom":    dict(lo=0.55, hi=1.45,   norm=0.45, mid=1.0),
    "crop":    dict(lo=0.45, hi=1.0,    norm=0.55, mid=1.0),
    "tx":      dict(lo=-0.30, hi=0.30,  norm=0.30),   # frac of width
    "ty":      dict(lo=-0.30, hi=0.30,  norm=0.30),
    "shear":   dict(lo=-0.30, hi=0.30,  norm=0.30),
    "persp":   dict(lo=0.0, hi=0.22,    norm=0.22),
    "aspect":  dict(lo=0.65, hi=1.55,   norm=0.45, mid=1.0),
    "rot90":   dict(lo=0, hi=3,         norm=1.5),    # quarter turns
    # photometric
    "bright":  dict(lo=0.55, hi=1.65,   norm=0.55, mid=1.0),
    "contrast": dict(lo=0.45, hi=1.75,  norm=0.60, mid=1.0),
    "gamma":   dict(lo=0.50, hi=2.00,   norm=0.75, mid=1.0),
    "sat":     dict(lo=0.0, hi=1.9,     norm=1.0, mid=1.0),
    "hue":     dict(lo=-0.5, hi=0.5,    norm=0.5),
    "gray":    dict(lo=0, hi=1,         norm=1.0),
    "invert":  dict(lo=0, hi=1,         norm=1.0),
    # capture / degradation
    "blur":    dict(lo=0.0, hi=3.2,     norm=3.2),
    "noise":   dict(lo=0.0, hi=0.13,    norm=0.13),
    "jpeg":    dict(lo=8, hi=95,        norm=87.0, mid=95, invert_sev=True),
    "downup":  dict(lo=0.14, hi=1.0,    norm=0.86, mid=1.0),
    "cutout":  dict(lo=0.0, hi=0.28,    norm=0.28),   # frac of area
    "vignette": dict(lo=0.0, hi=0.9,    norm=0.9),
    # temporal
    "speed":   dict(lo=0.5, hi=1.9,     norm=0.7, mid=1.0),
    "tphase":  dict(lo=0.0, hi=1.0,     norm=1.0),
    "tdrop":   dict(lo=0.0, hi=0.45,    norm=0.45),   # frac frames dropped
}

GEOM = ("rot", "mirror", "vflip", "zoom", "crop", "tx", "ty", "shear",
        "persp", "aspect", "rot90")
PHOTO = ("bright", "contrast", "gamma", "sat", "hue", "gray", "invert")
CAPTURE = ("blur", "noise", "jpeg", "downup", "cutout", "vignette")
TEMPORAL = ("speed", "tphase", "tdrop")
FAMILY = {}
for _f, _ns in (("geom", GEOM), ("photo", PHOTO),
                ("capture", CAPTURE), ("temporal", TEMPORAL)):
    for _n in _ns:
        FAMILY[_n] = _f


def identity_params():
    p = {}
    for k, s in PRIMS.items():
        p[k] = s.get("mid", 0.0 if s["lo"] < 0 else s["lo"])
    return p


def severity(p):
    """Distance from identity in normalised parameter space."""
    base = identity_params()
    v = []
    for k, s in PRIMS.items():
        d = (p[k] - base[k]) / s["norm"]
        v.append(d)
    return float(np.linalg.norm(v))


def pvec(p):
    base = identity_params()
    return np.array([(p[k] - base[k]) / PRIMS[k]["norm"] for k in PRIMS],
                    np.float64)


def tdist(pa, pb):
    return float(np.linalg.norm(pvec(pa) - pvec(pb)))


# ------------------------------------------------------------- appliers

def _pil(a):
    from PIL import Image
    return Image.fromarray(a)


def apply_clip(F, p):
    """Apply a full parameter set to a clip (T,H,W,3) uint8."""
    from PIL import Image, ImageEnhance, ImageFilter
    T, H, W, _ = F.shape

    # ---- temporal first: it decides which frames exist at all
    idx = np.arange(T, dtype=np.float64)
    if p["speed"] != 1.0:
        n = max(int(round(T / p["speed"])), 4)
        idx = np.linspace(0, T - 1, n)
    if p["tphase"] > 0:
        idx = idx + p["tphase"]
    idx = np.clip(np.round(idx).astype(int), 0, T - 1)
    if p["tdrop"] > 0 and len(idx) > 8:
        rs = np.random.RandomState(int(p["tdrop"] * 1e6) % 2**31)
        keep = rs.rand(len(idx)) >= p["tdrop"]
        if keep.sum() >= 8:
            idx = idx[keep]
    F = F[idx]

    out = []
    for fr in F:
        im = _pil(fr)
        # ---- geometric
        if p["rot90"]:
            im = im.rotate(90 * int(p["rot90"]), expand=True)
        if p["mirror"]:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if p["vflip"]:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        w, h = im.size
        if p["crop"] < 1.0:
            cw, ch = int(w * p["crop"]), int(h * p["crop"])
            x0 = int((w - cw) * (0.5 + p["tx"]))
            y0 = int((h - ch) * (0.5 + p["ty"]))
            x0 = max(0, min(x0, w - cw)); y0 = max(0, min(y0, h - ch))
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h))
        if p["zoom"] != 1.0:
            zw, zh = max(int(w / p["zoom"]), 8), max(int(h / p["zoom"]), 8)
            x0, y0 = (w - zw) // 2, (h - zh) // 2
            if p["zoom"] > 1.0:
                im = im.crop((x0, y0, x0 + zw, y0 + zh)).resize((w, h))
            else:
                bg = Image.new("RGB", (zw, zh))
                bg.paste(im.resize((w, h)), (-x0, -y0))
                im = bg.resize((w, h))
        if p["aspect"] != 1.0:
            im = im.resize((max(int(w * p["aspect"]), 8), h)).resize((w, h))
        if p["rot"] != 0.0:
            im = im.rotate(p["rot"], Image.BILINEAR, expand=False)
        if p["shear"] != 0.0:
            im = im.transform((w, h), Image.AFFINE,
                              (1, p["shear"], -p["shear"] * h / 2, 0, 1, 0),
                              Image.BILINEAR)
        if p["persp"] > 0:
            a = p["persp"]
            src = [(0, 0), (w, 0), (w, h), (0, h)]
            dst = [(a * w, a * h), (w - a * w, 0),
                   (w, h - a * h), (0, h)]
            im = im.transform((w, h), Image.PERSPECTIVE,
                              _coeffs(src, dst), Image.BILINEAR)
        # ---- photometric
        if p["gray"]:
            im = im.convert("L").convert("RGB")
        if p["sat"] != 1.0:
            im = ImageEnhance.Color(im).enhance(p["sat"])
        if p["bright"] != 1.0:
            im = ImageEnhance.Brightness(im).enhance(p["bright"])
        if p["contrast"] != 1.0:
            im = ImageEnhance.Contrast(im).enhance(p["contrast"])
        a = np.asarray(im).astype(np.float32) / 255.0
        if p["gamma"] != 1.0:
            a = a ** p["gamma"]
        if p["hue"] != 0.0:
            a = _hue(a, p["hue"])
        if p["invert"]:
            a = 1.0 - a
        if p["vignette"] > 0:
            yy, xx = np.mgrid[0:h, 0:w]
            r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2
                        + ((xx - w / 2) / (w / 2)) ** 2)
            a = a * (1.0 - p["vignette"] * np.clip(r - 0.4, 0, None))[..., None]
        if p["cutout"] > 0:
            rs = np.random.RandomState(int(p["cutout"] * 1e6) % 2**31)
            side = int(np.sqrt(p["cutout"]) * min(h, w))
            y0 = rs.randint(0, max(h - side, 1)); x0 = rs.randint(0, max(w - side, 1))
            a[y0:y0 + side, x0:x0 + side] = 0.5
        im = _pil(np.clip(a * 255, 0, 255).astype(np.uint8))
        # ---- capture
        if p["blur"] > 0:
            im = im.filter(ImageFilter.GaussianBlur(p["blur"]))
        if p["downup"] < 1.0:
            im = im.resize((max(int(w * p["downup"]), 8),
                            max(int(h * p["downup"]), 8))).resize((w, h))
        if p["jpeg"] < 95:
            b = io.BytesIO(); im.save(b, "JPEG", quality=int(p["jpeg"]))
            b.seek(0); im = Image.open(b).convert("RGB")
        a = np.asarray(im).astype(np.float32)
        if p["noise"] > 0:
            rs = np.random.RandomState(int(p["noise"] * 1e6) % 2**31)
            a = a + rs.randn(*a.shape) * p["noise"] * 255.0
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return np.stack(out)


def _coeffs(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B += [u, v]
    return np.linalg.lstsq(np.asarray(A, np.float64),
                           np.asarray(B, np.float64), rcond=None)[0]


def _hue(a, shift):
    import colorsys
    mx = a.max(-1); mn = a.min(-1); v = mx
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-8), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.zeros_like(mx)
    m = (d > 1e-8)
    idx = m & (mx == r); h[idx] = ((g - b)[idx] / d[idx]) % 6
    idx = m & (mx == g); h[idx] = ((b - r)[idx] / d[idx]) + 2
    idx = m & (mx == b); h[idx] = ((r - g)[idx] / d[idx]) + 4
    h = (h / 6.0 + shift) % 1.0
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    pp, qq, tt = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.zeros_like(a)
    for k, (rr, gg, bb) in enumerate([(v, tt, pp), (qq, v, pp), (pp, v, tt),
                                      (pp, qq, v), (tt, pp, v), (v, pp, qq)]):
        sel = (i == k)
        out[sel] = np.stack([rr, gg, bb], -1)[sel]
    return out


# ------------------------------------------------------------- sampling

def battery(n_combo=44, seed=0):
    """A named set of parameter dicts. Graded singles + compositions.

    Singles at three strengths give the severity axis a clean read; the
    bulk is 1..4 simultaneous primitives so the battery measures what
    actually happens to footage rather than one textbook axis at a time.
    """
    rs = np.random.RandomState(seed)
    out = [("identity", identity_params())]
    # graded singles - every primitive, three strengths
    for k, s in PRIMS.items():
        base = s.get("mid", 0.0 if s["lo"] < 0 else s["lo"])
        for lvl, frac in (("lo", 0.34), ("md", 0.67), ("hi", 1.0)):
            p = identity_params()
            if k in ("mirror", "vflip", "gray", "invert"):
                if lvl != "hi":
                    continue
                p[k] = 1
            elif k == "rot90":
                p[k] = {"lo": 1, "md": 2, "hi": 3}[lvl]
            else:
                far = s["hi"] if abs(s["hi"] - base) > abs(base - s["lo"]) \
                    else s["lo"]
                p[k] = base + (far - base) * frac
            out.append((f"{k}.{lvl}", p))
    # compositions
    keys = list(PRIMS)
    for i in range(n_combo):
        p = identity_params()
        m = rs.randint(2, 5)
        picks = rs.choice(keys, m, replace=False)
        for k in picks:
            s = PRIMS[k]
            base = s.get("mid", 0.0 if s["lo"] < 0 else s["lo"])
            if k in ("mirror", "vflip", "gray", "invert"):
                p[k] = 1
            elif k == "rot90":
                p[k] = rs.randint(1, 4)
            else:
                p[k] = float(rs.uniform(s["lo"], s["hi"]))
            _ = base
        out.append((f"combo{i:02d}x{m}", p))
    return out


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native"))
    import vsrc
    from PIL import Image, ImageDraw
    B = battery()
    print(f"{len(B)} transforms")
    fam = {}
    for n, p in B:
        fam.setdefault(n.split(".")[0] if "." in n else "combo", []).append(n)
    print(f"  singles over {len(PRIMS)} primitives, "
          f"{sum(1 for n, _ in B if n.startswith('combo'))} compositions")
    sv = sorted((severity(p), n) for n, p in B)
    print(f"  severity range {sv[0][0]:.2f} ({sv[0][1]}) .. "
          f"{sv[-1][0]:.2f} ({sv[-1][1]})")
    if "--preview" in sys.argv:
        s = vsrc.sources("sim", 1)[0]
        F = s.cut(10.5, 14.5)
        sel = B[:1] + B[1:60:6] + [b for b in B if b[0].startswith("combo")][:18]
        W, H, C = 150, 112, 8
        sheet = Image.new("RGB", (W * C, (H + 13) * ((len(sel) + C - 1) // C)),
                          (18, 20, 24))
        d = ImageDraw.Draw(sheet)
        for i, (n, p) in enumerate(sel):
            G = apply_clip(F, p)
            im = Image.fromarray(G[len(G) // 2]).resize((W, H))
            sheet.paste(im, ((i % C) * W, (i // C) * (H + 13) + 13))
            d.text(((i % C) * W + 2, (i // C) * (H + 13) + 2),
                   f"{n} s={severity(p):.1f}", fill=(255, 255, 0))
        sheet.save("/private/tmp/claude-501/aug_preview.png")
        print("wrote /private/tmp/claude-501/aug_preview.png", sheet.size)


if __name__ == "__main__":
    main()
