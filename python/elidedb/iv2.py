"""InternVideo2-Stage2 1B (Wang et al., ECCV 2024, arXiv 2403.15377)
— the only true VIDEO-native text channel: stage-2 trains video-text
contrastively WITH temporal modeling, which the frame-pooled image
channels (PE, SigLIP2, X-CLIP pooling) structurally lack. Distilled
small variants are reported much weaker on retrieval — 1B or nothing.

One 512-d aligned vector per episode (4-frame clip, the f4
checkpoint's native temporal extent), table iv2_vectors. Weights:
ziyjiang/InternVideo2-1B (fp16 re-serialization of the official
OpenGVLab stage2 checkpoint, towers only); modeling code vendored
from VLM2Vec into models/iv2_stage2_1b by scripts/get_iv2.py (three
MPS patches: flash-attn import guarded, LayerScale gamma naming so
the checkpoint's 80 layer scales actually load, bert config resolved
next to the file)."""
from __future__ import annotations

import numpy as np

_S = {}

MDIR = "models/iv2_stage2_1b"

# InternVideo2's own demo preprocessing (frames2tensor): ImageNet
# stats, 224 square, [B,T,C,H,W]
V_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
V_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model():
    if "model" not in _S:
        from transformers import AutoModel

        from .device import pick, strip_vision
        dev, dtype = pick()
        m = AutoModel.from_pretrained(
            MDIR, trust_remote_code=True, torch_dtype=dtype,
            low_cpu_mem_usage=True).to(dev).eval()
        m = strip_vision(m, "vision_encoder")
        m._config.device = dev      # get_txt_feat routes tokens here
        _S["model"], _S["dev"], _S["dtype"] = m, dev, dtype
    return _S["model"], _S["dev"]


def text_vec(text):
    cache = _S.setdefault("tcache", {})
    if text in cache:
        return cache[text]
    m, _ = load_model()
    v = m.get_txt_feat(text).float().cpu().numpy().reshape(-1)
    if len(cache) > 256:
        cache.clear()
    cache[text] = v
    return v


def clip_vec(frames_hwc):
    """Aligned 512-d vector for a 4-frame clip (HWC uint8 RGB)."""
    import cv2
    import torch
    m, dev = load_model()
    fs = [cv2.resize(f, (224, 224)) for f in frames_hwc]
    x = (np.stack(fs).astype(np.float32) / 255.0 - V_MEAN) / V_STD
    px = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(
        dev, _S["dtype"])
    return m.get_vid_feat(px).float().cpu().numpy().reshape(-1)


def iv2_lookup(store, text):
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "iv2_vectors")
    key = {}
    for r, (s, a) in enumerate(zip(
            tbl.column("stream").to_pylist(),
            (int(v) for v in tbl.column("ts").to_pylist()))):
        key[(str(s), a)] = r
    sc = np.asarray(vecs) @ text_vec(text)

    def lookup(s, a, b):
        r = key.get((str(s), a))
        return float(sc[r]) if r is not None else float("nan")
    return lookup, None
