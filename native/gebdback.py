"""STEP 4 backbone study: which features should the Ncut run on?

graphgebd.py reached sim 0.711 / oxford 0.500 on DINOv3 ConvNeXt-Tiny,
which is the in-repo backbone - chosen for IDENTITY (crop-vs-crop
re-identification, AUC 0.9994), a different job. The published
GraphGEBD number (0.732 Kinetics) is on ResNet50 / DINOv2. So the
substitution was never tested; this tests it.

The algorithm is held EXACTLY fixed (same recursive contiguous Ncut,
same depth, same MIN_SEG, same emitter). Only frame_features changes.

Two affinity scalings are reported for every backbone, because sigma
lives on cosine DISTANCE and different embedding spaces have different
distance scales - a fixed sigma silently favours whichever backbone
happens to match it:

  fixed  sigma = 0.25 (what graphgebd.py shipped)
  self   sigma = median off-diagonal distance of THIS media

`self` is the honest comparison and is also one fewer constant of
mine: it is the self-tuning-affinity idea (Zelnik-Manor & Perona),
fitted per media from the media's own distribution.

Features are cached to disk per (backbone, media) so an interrupt
costs one backbone, not the run.

    python native/gebdback.py --corpus sim
    python native/gebdback.py --corpus oxford
    python native/gebdback.py --corpus sim --backbones dinov3_ct,dinov2
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import f1_at, media_jobs, arg                     # noqa: E402
from graphgebd import recursive_ncut                            # noqa: E402

FPS = 4.0
CACHE = Path("/private/tmp/claude-501/gebdback")

# name -> (kind, model id).  Every one of these is already in the local
# HF cache except resnet50 (torchvision, ~100 MB), so nothing here
# depends on a download that could stall the run.
BACKBONES = {
    # what graphgebd.py currently uses
    "dinov3_ct":  ("hf", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"),
    # what the PAPER uses
    "dinov2_b":   ("hf", "facebook/dinov2-base"),
    "resnet50":   ("tv", "resnet50"),
    # bigger / different-family DINOv3
    "dinov3_vits": ("hf", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    "dinov3_vitb": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    # semantic (text-aligned) - a genuinely different inductive bias
    "siglip2":    ("siglip", "google/siglip2-so400m-patch14-384"),
    # low-level control: reconstruction, not discrimination
    "mae":        ("hf", "facebook/vit-mae-base"),
    # video-native: features that see motion, not stills
    "vjepa2":     ("vjepa", "facebook/vjepa2-vitl-fpc64-256"),
}
ORDER = ["dinov3_ct", "dinov2_b", "resnet50", "dinov3_vits",
         "dinov3_vitb", "siglip2", "mae", "vjepa2"]


def _norm(V):
    V = np.asarray(V, np.float32)
    if np.isnan(V).any():
        raise FloatingPointError("NaN features")
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)


def feats_hf(mid, F, batch=32):
    """Pooled per-frame features from any HF image encoder."""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from elidedb.device import pick
    dev, dtype = pick()
    if "vit" in mid.rsplit("/", 1)[-1]:
        dtype = torch.float32            # DINOv3 ViT overflows fp16
    proc = AutoImageProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid, dtype=dtype,
                                      low_cpu_mem_usage=True) \
        .to(dev).eval()
    out = []
    for i in range(0, len(F), batch):
        chunk = [np.ascontiguousarray(x[..., :3]) for x in F[i:i + batch]]
        px = proc(images=chunk, return_tensors="pt",
                  size={"height": 224, "width": 224})["pixel_values"]
        with torch.no_grad():
            r = model(pixel_values=px.to(dev, dtype))
        v = getattr(r, "pooler_output", None)
        if v is None:                    # MAE has no pooler
            v = r.last_hidden_state.mean(1)
        out.append(v.float().cpu().numpy())
    del model
    return np.concatenate(out)


def feats_tv(_mid, F, batch=32):
    """ImageNet ResNet50 global-pool features - the paper's baseline."""
    import torch
    import torchvision as tv
    from elidedb.device import pick
    dev, _ = pick()
    m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m = m.to(dev).eval()
    tf = tv.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    out = []
    for i in range(0, len(F), batch):
        x = torch.stack([tf(torch.from_numpy(
            np.ascontiguousarray(f[..., :3])).permute(2, 0, 1))
            for f in F[i:i + batch]]).to(dev)
        with torch.no_grad():
            out.append(m(x).float().cpu().numpy())
    del m
    return np.concatenate(out)


def feats_siglip(mid, F, batch=16):
    import torch
    from transformers import AutoModel, AutoProcessor
    from elidedb.device import pick
    dev, dtype = pick()
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid, dtype=dtype,
                                      low_cpu_mem_usage=True) \
        .to(dev).eval()
    out = []
    for i in range(0, len(F), batch):
        chunk = [np.ascontiguousarray(x[..., :3]) for x in F[i:i + batch]]
        px = proc(images=chunk, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            v = model.get_image_features(pixel_values=px.to(dev, dtype))
        out.append(v.float().cpu().numpy())
    del model
    return np.concatenate(out)


def feats_vjepa(mid, F, nframe=16, stride=4):
    """Video-native per-frame features.

    A clip is encoded around each stride-th frame and its tokens are
    mean-pooled; frames in between inherit the nearest clip vector.
    Temporal resolution therefore drops to FPS/stride = 1 Hz, which is
    still far finer than the F1@0.05 tolerance (>=0.7 s here).
    """
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid, dtype=torch.float16) \
        .to("mps").eval()
    n = len(F)
    centres = list(range(0, n, stride))
    vecs = []
    for c in centres:
        s = int(np.clip(c - nframe // 2, 0, max(n - nframe, 0)))
        idx = np.linspace(s, min(s + nframe - 1, n - 1),
                          nframe).round().astype(int)
        clip = [np.ascontiguousarray(F[i][..., :3]) for i in idx]
        px = proc([clip], return_tensors="pt")["pixel_values_videos"]
        with torch.no_grad():
            r = model(pixel_values_videos=px.to("mps", torch.float16),
                      skip_predictor=True)
        vecs.append(r.last_hidden_state[0].float().mean(0).cpu().numpy())
    del model
    V = np.stack(vecs)
    near = np.clip(np.round(np.arange(n) / stride).astype(int),
                   0, len(centres) - 1)
    return V[near]


KIND = {"hf": feats_hf, "tv": feats_tv, "siglip": feats_siglip,
        "vjepa": feats_vjepa}


def features(name, path, F):
    # "a+b" concatenates two unit-normed spaces, which makes the fused
    # cosine the MEAN of the two cosines - affinity averaging, no
    # weight for me to pick. Tests whether the backbones disagree
    # usefully or just measure the same thing.
    if "+" in name:
        return _norm(np.concatenate(
            [features(p, path, F) for p in name.split("+")], axis=1))
    kind, mid = BACKBONES[name]
    key = hashlib.sha1(
        f"{name}|{path}|{len(F)}|{FPS}".encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{key}.npy"
    if fp.exists():
        return _norm(np.load(fp))
    V = KIND[kind](mid, F)
    np.save(fp, V.astype(np.float32))
    return _norm(V)


def affinity(V, sigma):
    S = V @ V.T
    W = np.exp(-(1.0 - S) / max(sigma, 1e-6))
    np.fill_diagonal(W, 0.0)
    return W


def score(V, gt, dur, sigma, depth):
    W = affinity(V, sigma)
    cuts = recursive_ncut(W, depth=depth)
    if not cuts:
        return 0.0, 0.0, 0
    times = [c[0] / FPS for c in cuts]
    vals = np.array([c[1] for c in cuts])
    tol = 0.05 * dur
    pm = times[:max(len(gt), 1)]
    if len(vals) > 2:
        z = (vals - vals.mean()) / (vals.std() + 1e-8)
        pth = [t for t, zz in zip(times, z) if zz < 0.0]
    else:
        pth = times
    fm, _, _ = f1_at(pm, gt, tol)
    ft, _, _ = f1_at(pth, gt, tol)
    return fm, ft, len(pth)


def self_sigma(V):
    S = V @ V.T
    n = len(V)
    off = ~np.eye(n, dtype=bool)
    return float(np.median(1.0 - S[off]))


def main():
    import encode as E
    from tqdm import tqdm
    corpus = arg("--corpus", "sim")
    limit = arg("--limit", 12, int)
    depth = arg("--depth", 4, int)
    names = arg("--backbones", ",".join(ORDER)).split(",")
    jobs = media_jobs(corpus, limit)

    print(f"{corpus}: {len(jobs)} media @ {FPS} fps, depth {depth}, "
          f"{len(names)} backbones", flush=True)
    frames = {}
    for name, path, gt in tqdm(jobs, desc="decode", unit="media"):
        frames[name] = (E.decode(path, fps=FPS, w=256),
                        E.probe_duration(path), gt)
    nf = sum(len(v[0]) for v in frames.values())
    print(f"decoded {nf} frames\n", flush=True)

    rows = []
    for bn in names:
        if not all(p in BACKBONES for p in bn.split("+")):
            print(f"  skip unknown backbone {bn}")
            continue
        t0 = time.time()
        fix_m, fix_t, slf_m, slf_t, emits = [], [], [], [], []
        try:
            for mn in tqdm(list(frames), desc=f"{bn:<13}", unit="media",
                           leave=False):
                F, dur, gt = frames[mn]
                V = features(bn, mn, F)
                a, b, _ = score(V, gt, dur, 0.25, depth)
                c, d, ne = score(V, gt, dur, self_sigma(V), depth)
                fix_m.append(a), fix_t.append(b)
                slf_m.append(c), slf_t.append(d), emits.append(ne)
        except Exception as e:                       # noqa: BLE001
            print(f"  {bn:<13} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            continue
        rows.append((bn, np.mean(fix_m), np.mean(fix_t),
                     np.mean(slf_m), np.mean(slf_t), np.mean(emits),
                     time.time() - t0))
        print(f"  {bn:<13} fixed {rows[-1][1]:.3f}/{rows[-1][2]:.3f}  "
              f"self {rows[-1][3]:.3f}/{rows[-1][4]:.3f}  "
              f"emits {rows[-1][5]:.1f}  {rows[-1][6]:.0f}s", flush=True)

    print(f"\nSTEP 4 backbone study — {corpus} "
          f"(F1@0.05, matched/emitter)")
    print(f"{'backbone':<14}{'fixed .25':<22}{'self-scaled':<22}"
          f"{'emits':<8}{'sec'}")
    for bn, a, b, c, d, e, s in sorted(rows, key=lambda r: -max(r[3],
                                                               r[1])):
        print(f"{bn:<14}{a:.3f} / {b:.3f}{'':<10}"
              f"{c:.3f} / {d:.3f}{'':<10}{e:<8.1f}{s:.0f}")


if __name__ == "__main__":
    main()
