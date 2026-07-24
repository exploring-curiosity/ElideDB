"""SAM 3.1 concept grounding — the approved detector for the geometric
relational verifier (module named sam3x to avoid clashing with Meta's
`sam3` package).

Promptable Concept Segmentation: a short noun phrase -> ALL instances
(boxes, masks, scores) in the frame, with a presence token that was
built exactly for near-miss text discrimination ("a green object" vs
the yellow one beside it). Replaces FastSAM's prompt-free crops AND the
interim Grounding DINO backend; the containment-change predicate in
grounding.py is detector-agnostic and consumes these boxes unchanged.

Checkpoint: facebook/sam3.1 `sam3.1_multiplex.pt` (user-approved gate).
sam3.model.edt is a triton (CUDA-only, no macOS build) distance-
transform kernel used only for interactive click sampling; a global
fake-triton shim broke torch._dynamo (it probes triton.language.dtype
when the module exists), so instead the edt SUBMODULE is pre-injected
with a cv2.distanceTransform fallback — the docstring of the original
says it mimics exactly that call — and triton is never imported.
"""
from __future__ import annotations

import sys
import types

import numpy as np

_S = {}


def _shim_edt():
    if "sam3.model.edt" in sys.modules:
        return

    def edt_cv2(masks):
        import cv2
        import torch
        m = masks.detach().cpu().numpy().astype("uint8")
        flat = m.reshape(-1, *m.shape[-2:])
        out = np.stack([cv2.distanceTransform(x, cv2.DIST_L2, 0)
                        for x in flat]).reshape(m.shape)
        return torch.from_numpy(out).to(masks.device)

    mod = types.ModuleType("sam3.model.edt")
    mod.edt_triton = edt_cv2
    sys.modules["sam3.model.edt"] = mod


def _patch_pos_enc():
    """PositionEmbeddingSine warm-starts its cache with tensors created
    on a HARDCODED device="cuda" (upstream latency optimization). The
    cache also fills lazily in forward() on the input's device, so on
    non-CUDA hosts the warmup is skipped rather than ported."""
    from sam3.model import position_encoding as pe
    if getattr(pe.PositionEmbeddingSine, "_sdx_patched", False):
        return
    orig = pe.PositionEmbeddingSine.__init__

    def patched(self, *a, **k):
        k["precompute_resolution"] = None
        orig(self, *a, **k)

    pe.PositionEmbeddingSine.__init__ = patched
    pe.PositionEmbeddingSine._sdx_patched = True


def _load(device=None):
    if "proc" in _S:
        return _S
    import torch
    _shim_edt()
    from sam3.model_builder import (build_sam3_image_model,
                                    download_ckpt_from_hf)
    from sam3.model.sam3_image_processor import Sam3Processor
    _patch_pos_enc()
    dev = device or ("mps" if torch.backends.mps.is_available()
                     else "cpu")
    if not torch.cuda.is_available():
        # sam3 sprinkles .pin_memory() through the inference path; on a
        # host with no CUDA pinning is meaningless and on MPS it throws
        # a storage-device error — make it a no-op process-wide
        torch.Tensor.pin_memory = lambda self, *a, **k: self
    # SAM 3.1 is the VIDEO multiplex release (tracking ~7x faster); its
    # only file, sam3.1_multiplex.pt, leaves the image model's neck
    # convs.3 randomly initialized (missing-keys observed). Image PCS
    # weights are unchanged in 3.1, so the image path loads sam3.pt
    # (exact key match); the video predictor uses 3.1 when adopted.
    ckpt = download_ckpt_from_hf(version="sam3")
    model = build_sam3_image_model(device=dev, checkpoint_path=ckpt,
                                   load_from_HF=False)
    # the builder only moves the model when device == "cuda"; fp32
    # everywhere (scripts/patch_sam3_mps.py removes the one hardcoded
    # bf16 fused op that aborted Metal matmuls)
    model = model.to(dev).float()
    _S["proc"] = Sam3Processor(model, device=dev)
    _S["dev"] = dev
    return _S


def segment_concept(image, phrase, threshold=0.3):
    """One PIL image + noun phrase -> list of (box_xyxy, score, mask).
    Masks are HxW bool (None if the head returned none)."""
    st = _load()
    state = st["proc"].set_image(image)
    out = st["proc"].set_text_prompt(state=state, prompt=phrase)

    def _np(t):
        return (t.detach().cpu().numpy() if hasattr(t, "detach")
                else np.asarray(t))
    boxes, scores = _np(out["boxes"]), _np(out["scores"])
    masks = out.get("masks")
    masks = _np(masks) if masks is not None else None
    res = []
    for i in range(len(scores)):
        sc = float(scores[i])
        if sc < threshold:
            continue
        box = np.asarray(boxes[i], dtype=float).reshape(-1)
        m = None
        if masks is not None:
            mi = np.asarray(masks[i])
            m = np.squeeze(mi) > 0.5
        res.append((box, sc, m))
    res.sort(key=lambda r: -r[1])
    return res


def detect_phrases_sam3(images, phrases, threshold=0.3):
    """Same contract as grounding.detect_phrases: per image
    {phrase: (best box, score) | None} — drop-in backend swap."""
    per = []
    for im in images:
        d = {}
        for p in phrases:
            r = segment_concept(im, p, threshold=threshold)
            d[p] = (r[0][0], r[0][1]) if r else None
        per.append(d)
    return per
