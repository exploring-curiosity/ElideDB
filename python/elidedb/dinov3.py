"""DINOv3 — the teacher's substrate. One backbone, three granularities.

Frames (every frame -> scene series), track crops (identity descriptors)
and, later, whatever the FDNN student distills from. Chosen over the
CLIP family for everything appearance-shaped because it has NO text
tower: it was never trained to collapse instances into nameable
categories, which is precisely the failure that made a red and a green
pepper the same object under text-aligned encoders. The instance-level
numbers agree (+10.9 GAP retrieval over DINOv2, which already beat CLIP
at instance top-1).

Gated weights: the user accepted Meta's license on HF (2026-08-01).

No resizing cleverness: everything goes to IMG px square through the
processor, because identity compares crops to crops and frames to
frames — never one to the other — so a shared canonical size keeps
each comparison internally consistent.
"""
from __future__ import annotations

import os

import numpy as np

_M = {}

# ConvNeXt-Tiny, not ViT-S, and the reason is numerical, not taste:
# the DINOv3 ViT variants OVERFLOW FP16 on MPS - every embedding NaN,
# and a NaN loses every >= comparison silently, so the failure mode was
# not an error but "12,583 tracks, 12,583 objects, zero recurrence".
# ViT-S works in fp32 at ~2x its fp16 time, which makes it slower than
# ConvNeXt-Tiny in fp16 - and ConvNeXt measured AUC 0.9994 on the
# corpus's proven pairs (vs yolo26n-reid 0.9047). Faster AND stronger,
# so the 768-d (vs 384) table cost is accepted.
MID = os.environ.get("ELIDEDB_DINOV3",
                     "facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
IMG = int(os.environ.get("ELIDEDB_DINOV3_SZ", "224"))
BATCH = int(os.environ.get("ELIDEDB_DINOV3_BATCH", "64"))


def _load(mid=None):
    mid = mid or MID
    if _M.get("mid") == mid:
        return _M
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from .device import pick
    dev, dtype = pick()
    if "vit" in mid.rsplit("/", 1)[-1]:
        dtype = torch.float32          # fp16 overflow, see above
    _M.clear()
    _M["proc"] = AutoImageProcessor.from_pretrained(mid)
    _M["model"] = AutoModel.from_pretrained(
        mid, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    _M["dev"], _M["dtype"], _M["mid"] = dev, dtype, mid
    _M["torch"] = torch
    return _M


def embed(images, batch=None, mid=None):
    """(N, D) unit-norm float32 embeddings for a list of HWC uint8 arrays.

    Pooled (CLS) output — the global instance/scene vector, the thing
    retrieval compares. Patch tokens exist but are a different product
    for a different consumer; nothing here should quietly average them.
    """
    m = _load(mid)
    torch = m["torch"]
    out = []
    batch = batch or BATCH
    for i in range(0, len(images), batch):
        chunk = [np.ascontiguousarray(x[..., :3]) for x in
                 images[i:i + batch]]
        px = m["proc"](images=chunk, return_tensors="pt",
                       size={"height": IMG, "width": IMG})
        px = px["pixel_values"].to(m["dev"], m["dtype"])
        with torch.no_grad():
            r = m["model"](pixel_values=px)
        v = r.pooler_output.float()
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = v.cpu().numpy().astype(np.float32)
        # FAIL LOUD. The ViT fp16 overflow produced NaN vectors that
        # scored 0.0 AUC without a single exception - every comparison
        # quietly False. Garbage must stop the run, not grade it.
        if np.isnan(v).any():
            raise FloatingPointError(
                f"{_M['mid']}: NaN embeddings (dtype {_M['dtype']}) - "
                "use fp32 for ViT variants")
        out.append(v)
    return (np.concatenate(out) if out
            else np.zeros((0, 384), np.float32))
