"""Pretrained LATENT ACTIONS as the unit representation. No text, no names.

The whole build so far encodes what a unit LOOKS like and hopes the
action falls out. Measured, it does not: appearance clusters by scene
(kinematic ratio 0.94-0.98 on drone/car), and on the arm it only works
because a fixed camera makes appearance-change a proxy for action.

LAPA's Latent Action Quantization (Open-X pretrained, released weights,
1.3 GB) learns the opposite thing directly: given two frames, emit
discrete codes for WHAT CHANGED. Trained with an inverse/forward
dynamics objective on unlabelled robot video, so the codebook is a
vocabulary of transitions - never a taxonomy, never words. Exactly the
"actor A acted upon actor B" representation, as a learned token.

INFERENCE ONLY. Nothing is trained here, nothing sees the truthset. The
checkpoint's generalisation is bounded by ITS training distribution
(Open-X = manipulation), which is the honest caveat and precisely what
the drone/car numbers will measure.

Config recovered from the checkpoint's tensor shapes, since it is
undocumented: spatial/temporal depth 8, codebook 8, code_seq_len 4
(pinned by a 3x3 vs 4x4 kernel in vq.cnn_encoder). Loads STRICT.

Unit descriptor: each consecutive frame pair yields 4 codes from a
codebook of 8. Over a span that is a SEQUENCE of code tuples, reduced to
  [per-position histogram (4x8) ; rank-pooled one-hot (4x8)]
so both what happened and its ORDER survive - the same reasoning that
made rank pooling beat mean pooling for appearance.

    python native/laqenc.py --domain sim --limit 72
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))
sys.path.insert(0, "/private/tmp/claude-501/LAPA/laq")

from flowgebd import arg                                       # noqa: E402
from unitenc import _norm                                      # noqa: E402

CKPT = ROOT / "models/laq/laq_openx.pt"
CODEBOOK, CODE_LEN, IMG = 8, 4, 256
PAIRS = 4              # frame pairs sampled per unit
_M = {}


def model():
    if "m" not in _M:
        import torch
        import torch.nn as nn
        from laq_model.latent_action_quantization import \
            LatentActionQuantization
        m = LatentActionQuantization(
            dim=1024, quant_dim=32, codebook_size=CODEBOOK,
            image_size=IMG, patch_size=32, spatial_depth=8,
            temporal_depth=8, dim_head=64, heads=16,
            code_seq_len=CODE_LEN)
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        nn.Module.load_state_dict(m, sd, strict=True)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _M["m"] = m.to(dev).eval()
        _M["dev"], _M["torch"] = dev, torch
    return _M["m"], _M["dev"], _M["torch"]


def _prep(im, torch):
    """RGB, 256x256, [0,1] - exactly LAPA's inference transform."""
    from PIL import Image
    if not isinstance(im, Image.Image):
        im = Image.fromarray(np.asarray(im)[..., :3].astype(np.uint8))
    im = im.convert("RGB").resize((IMG, IMG))
    x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
    return x


def codes_for_pairs(pairs, batch=8):
    """[(imgA, imgB), ...] -> (N, CODE_LEN) int codes."""
    m, dev, torch = model()
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        x = torch.stack([
            torch.stack([_prep(a, torch), _prep(b, torch)], dim=1)
            for a, b in chunk]).to(dev)
        with torch.no_grad():
            ids = m(x, return_only_codebook_ids=True)
        out.append(ids.reshape(len(chunk), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, CODE_LEN), int)


def unit_descriptor(code_seq):
    """(P, CODE_LEN) codes over a unit -> fixed-length vector.

    Histogram keeps WHAT transitions occurred; rank pooling over the
    one-hot sequence keeps their ORDER, which a histogram alone throws
    away - and order is what separates a pick from a place.
    """
    P = len(code_seq)
    if P == 0:
        return np.zeros(2 * CODE_LEN * CODEBOOK, np.float32)
    oh = np.zeros((P, CODE_LEN, CODEBOOK), np.float32)
    for p in range(P):
        for c in range(CODE_LEN):
            oh[p, c, int(code_seq[p, c]) % CODEBOOK] = 1.0
    flat = oh.reshape(P, -1)
    hist = flat.mean(0)
    if P > 1:
        al = (2 * np.arange(1, P + 1) - P - 1).astype(np.float32)
        rk = (al[:, None] * flat).sum(0)
        rk = rk / max(np.linalg.norm(rk), 1e-8)
    else:
        rk = np.zeros_like(hist)
    return np.concatenate([hist, rk]).astype(np.float32)


def units_from_frames(spans, frame_at, fps, pairs=PAIRS):
    """Encode spans given a frame accessor. Frame pairs are CONSECUTIVE
    (a latent action is defined between adjacent frames), sampled evenly
    across the span so the descriptor covers the whole unit."""
    from PIL import Image
    V, keep = [], []
    for sp in spans:
        a, b = sp[0], sp[1]
        starts = np.linspace(a * fps, max(b * fps - 1, a * fps), pairs)
        pr = []
        for s in starts.round().astype(int):
            try:
                i0 = Image.open(frame_at(int(s)))
                i1 = Image.open(frame_at(int(s) + 1))
            except Exception:                            # noqa: BLE001
                continue
            pr.append((i0, i1))
        if len(pr) < 2:
            continue
        V.append(unit_descriptor(codes_for_pairs(pr)))
        keep.append(sp)
    return (_norm(np.stack(V)) if V else np.zeros((0, 2))), keep


def units_from_array(F, spans, fps, pairs=PAIRS, gap=1):
    """Same, for an in-memory decoded array (the sim path)."""
    V, keep = [], []
    for sp in spans:
        a, b = sp[0], sp[1]
        starts = np.linspace(a * fps, max(b * fps - 1, a * fps), pairs)
        pr = []
        for s in starts.round().astype(int):
            # GAP between the two frames. A latent action is defined
            # over a TRANSITION, and its scale is set by how far apart
            # the frames are. LAPA's own inference exposes window_size
            # for exactly this; adjacent frames at 4 fps span 0.25 s,
            # which may be far below the motion scale the codebook was
            # trained on.
            s = int(np.clip(s, 0, len(F) - 1 - gap))
            pr.append((F[s], F[s + gap]))
        if len(pr) < 2:
            continue
        V.append(unit_descriptor(codes_for_pairs(pr)))
        keep.append(sp)
    return (_norm(np.stack(V)) if V else np.zeros((0, 2))), keep


def main():
    import time
    dom = arg("--domain", "sim")
    limit = arg("--limit", 72, int)
    gap = arg("--gap", 1, int)
    pairs = arg("--pairs", 4, int)
    t0 = time.time()

    if dom == "sim":
        import encode as E
        import pyarrow.parquet as pq
        from tqdm import tqdm
        from u6retr import score
        t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
            .to_pydict()
        tmpl = {}
        for e, tm in zip(t["episode"], t["template"]):
            tmpl[int(e)] = tm
        dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                      if p.is_dir() and p.name.startswith("ep"))[:limit]
        seqs = {}
        for d in tqdm(dirs, desc="laq/sim", unit="ep"):
            ei = int(d.name[2:])
            cam = sorted(d.glob("cam*.mp4"))[0]
            dur = E.probe_duration(cam)
            F = E.decode(cam, fps=4.0, w=256)
            sp, x = [], 0.0
            while x + 3.0 <= dur + 1e-6:
                sp.append((x, x + 3.0))
                x += 1.0
            V, _ = units_from_array(F, sp or [(0.0, dur)], 4.0,
                                    pairs=pairs, gap=gap)
            if len(V):
                seqs[ei] = V
        y, p, su, rt = score(seqs, tmpl)
        print(f"\nLAQ latent actions — sim, {len(seqs)} eps, gap={gap} ({gap/4.0:.2f}s), pairs={pairs}")
        print(f"  yield {y:.3f}  prec {p:.3f}  support {su:.1f} "
              f"returned {rt:.1f}")
        print("  reference: siglip2_rank uniform 0.492/0.328, "
              "vjepa2 cpd 0.214/0.136, label oracle 0.942/0.890")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
