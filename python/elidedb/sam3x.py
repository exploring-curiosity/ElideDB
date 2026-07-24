"""SAM 3.1 VIDEO tracker — concept MASKLETS through time, no images.

User rule (2026-07-24): when a model has a video API, the video API is
the only one allowed — processing clips frame-by-frame through an image
model is not video-native. This wrapper runs Meta's intended pipeline:
build_sam3_video_predictor with the sam3.1_multiplex.pt checkpoint
(Object Multiplex: joint multi-object tracking), one SESSION per clip,
one text prompt per concept, propagate_in_video for tracked masklets on
every frame. The tracker's temporal identity is what per-frame
detection could never give: presence, containment, and trajectory of
the SAME object instance across the clip.

MPS port (the upstream stack assumes CUDA end to end):
  - torch.Tensor.cuda / torch.nn.Module.cuda routed to MPS process-wide
    (one shim catches every hardcoded .cuda() call site)
  - torch.cuda memory-stat fns stubbed (session stats logging)
  - pin_memory no-op; sam3.model.edt injected with a cv2 fallback
    (real triton import breaks torch._dynamo probing)
  - model cast fp32 post-load (hardcoded bf16 aborts Metal matmuls;
    scripts/patch_sam3_mps.py removes the fused-op bf16 too)
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
    """PositionEmbeddingSine warm-starts its cache on a hardcoded
    device="cuda"; the cache also fills lazily on the input's device,
    so on non-CUDA hosts the warmup is skipped rather than ported."""
    from sam3.model import position_encoding as pe
    if getattr(pe.PositionEmbeddingSine, "_sdx_patched", False):
        return
    orig = pe.PositionEmbeddingSine.__init__

    def patched(self, *a, **k):
        k["precompute_resolution"] = None
        orig(self, *a, **k)

    pe.PositionEmbeddingSine.__init__ = patched
    pe.PositionEmbeddingSine._sdx_patched = True


def _shim_cuda(dev):
    """Route every hardcoded CUDA call in the sam3 stack to `dev`:
    .cuda() methods, .to("cuda"/torch.device("cuda")) targets, and the
    torch.cuda bookkeeping fns the session machinery touches."""
    import torch
    if torch.cuda.is_available() or getattr(torch, "_sdx_cuda_shim",
                                            False):
        return
    torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(dev)
    torch.Tensor.pin_memory = lambda self, *a, **k: self

    def _is_cuda(x):
        return (isinstance(x, (str, torch.device))
                and str(x).startswith("cuda"))
    for cls in (torch.Tensor, torch.nn.Module):
        orig_to = cls.to

        def make_to(orig):
            def to(self, *a, **k):
                a = tuple(dev if _is_cuda(x) else x for x in a)
                if _is_cuda(k.get("device")):
                    k["device"] = dev
                return orig(self, *a, **k)
            return to
        cls.to = make_to(orig_to)
    for fn, val in (("memory_allocated", 0), ("memory_reserved", 0),
                    ("max_memory_allocated", 0),
                    ("max_memory_reserved", 0),
                    ("current_device", 0)):
        setattr(torch.cuda, fn, lambda *a, _v=val, **k: _v)
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.mem_get_info = lambda *a, **k: (0, 0)
    # factory-level interception: stored torch.device("cuda") objects
    # flow into tensor factories all over the multiplex state machinery
    for name in ("zeros", "ones", "empty", "full", "tensor",
                 "arange", "linspace", "rand", "randn", "eye",
                 "as_tensor", "zeros_like", "ones_like",
                 "empty_like", "full_like"):
        orig_f = getattr(torch, name)

        def make_f(orig):
            def f(*a, **k):
                if _is_cuda(k.get("device")):
                    k["device"] = dev
                return orig(*a, **k)
            return f
        setattr(torch, name, make_f(orig_f))
    torch._sdx_cuda_shim = True


def _load(device=None):
    if "pred" in _S:
        return _S
    import torch
    dev = device or ("mps" if torch.backends.mps.is_available()
                     else "cpu")
    _shim_edt()
    _shim_cuda(dev)
    from sam3.model_builder import build_sam3_predictor
    _patch_pos_enc()
    # the RECOMMENDED 3.1 entry point (multiplex tracker); the base
    # Sam3VideoPredictor cannot load the multiplex checkpoint (key
    # mismatch, observed). FA3 is CUDA-only; async frame loading races
    # with the single MPS queue — both off.
    pred = build_sam3_predictor(version="sam3.1", use_fa3=False,
                                async_loading_frames=False)
    # the base predictor's start_session forwards kwargs the multiplex
    # init_state does not accept (offload_state_to_cpu, ...) — filter
    # to the actual signature instead of chasing upstream drift
    import inspect
    orig = pred.model.init_state
    sig = inspect.signature(orig)

    def init_state(**kw):
        return orig(**{k: v for k, v in kw.items()
                       if k in sig.parameters})
    pred.model.init_state = init_state
    _S["pred"] = pred
    _S["dev"] = dev
    return _S


def track_concepts(store, stream, t0, t1, phrases, n_frames=12,
                   width=480):
    """One clip + concept phrases -> tracked masklets.

    Returns {phrase: {"presence": [0/1 per frame],
                      "boxes": [xyxy|None per frame],
                      "masks": [HxW bool|None per frame]}}
    or None when the clip has too few frames. Frames are written once
    as JPEGs (the tracker's native input) and shared by all phrases.
    """
    import shutil
    import tempfile

    import pyarrow.compute as pc
    from PIL import Image

    from .video import FrameSet
    st = _load()
    frames_tbl = store.table("frames").scan()
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), stream),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), t0),
                pc.less_equal(frames_tbl.column("ts"), t1))))
    if len(sel) < 4:
        return None
    pick = np.linspace(0, len(sel) - 1,
                       min(n_frames, len(sel))).round().astype(int)
    dec = FrameSet(store, "frames", sel.take(pick)).decode(width=width)
    if len(dec) < 4:
        return None
    imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
    tmp = tempfile.mkdtemp(prefix="sdx_sam3_")
    try:
        for i, im in enumerate(imgs):
            im.save(f"{tmp}/{i:05d}.jpg", quality=90)
        out = {}
        for phrase in phrases:
            resp = st["pred"].handle_request(request=dict(
                type="start_session", resource_path=tmp))
            sid = resp["session_id"]
            try:
                st["pred"].handle_request(request=dict(
                    type="add_prompt", session_id=sid, frame_index=0,
                    text=phrase))
                per = {}
                for fr in st["pred"].handle_stream_request(request=dict(
                        type="propagate_in_video", session_id=sid)):
                    per[int(fr["frame_index"])] = fr.get("outputs", fr)
                out[phrase] = _masklets(per, len(imgs))
            finally:
                st["pred"].handle_request(request=dict(
                    type="close_session", session_id=sid))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _masklets(per_frame, n):
    """Collapse the tracker's per-frame outputs (out_obj_ids /
    out_probs / out_boxes_xywh / out_binary_masks) into presence/box/
    mask series for ONE persistent identity — the obj_id with the
    highest total probability across the clip. Per-frame argmax let
    the series hop between same-concept instances (two green toys),
    faking movement and defeating the manipulated-object test
    (audit-bench-caught: zero static kills on attribute queries)."""
    # first pass: total evidence per identity
    totals = {}
    for i in range(n):
        o = per_frame.get(i)
        if not isinstance(o, dict):
            continue
        ids = np.asarray(o.get("out_obj_ids", [])).reshape(-1)
        probs = np.asarray(o.get("out_probs", []), float).reshape(-1)
        for oid, p in zip(ids, probs):
            totals[int(oid)] = totals.get(int(oid), 0.0) + float(p)
    presence = [0.0] * n
    boxes = [None] * n
    masks = [None] * n
    if not totals:
        return {"presence": presence, "boxes": boxes, "masks": masks,
                "any_moved": False}
    best_id = max(totals, key=totals.get)
    per_id_centers = {}
    per_id_best = {}
    for i in range(n):
        o = per_frame.get(i)
        if not isinstance(o, dict):
            continue
        ids = np.asarray(o.get("out_obj_ids", [])).reshape(-1)
        probs = np.asarray(o.get("out_probs", []), float).reshape(-1)
        bxs = np.asarray(o.get("out_boxes_xywh", []),
                         float).reshape(-1, 4)
        for j, oid in enumerate(int(v) for v in ids):
            bx = bxs[j]
            per_id_centers.setdefault(oid, []).append(
                (bx[0] + bx[2] / 2, bx[1] + bx[3] / 2,
                 max(bx[2], bx[3])))
            m = o.get("out_binary_masks")
            mj = (np.squeeze(np.asarray(m[j])) > 0.5
                  if m is not None and len(m) > j else None)
            if oid not in per_id_best \
                    or probs[j] > per_id_best[oid][0]:
                per_id_best[oid] = (float(probs[j]), i, mj)
            if oid != best_id:
                continue
            presence[i] = float(probs[j])
            boxes[i] = np.array([bx[0], bx[1],
                                 bx[0] + bx[2], bx[1] + bx[3]])
            masks[i] = mj
    # "does ANY instance of this concept move" — the manipulated object
    # may be a different identity than the most-visible one (a second
    # green toy); the static-kill must not execute true clips for that.
    # The MOVER's identity is also exposed so attribute checks (color)
    # interrogate the object that actually acted, not the most visible
    any_moved = False
    mover, mover_ratio = None, 0.0
    for oid, pts in per_id_centers.items():
        if len(pts) < 2:
            continue
        cs = np.asarray([(x, y) for x, y, _ in pts])
        size = float(np.median([s for _, _, s in pts])) + 1e-6
        exc = float(np.max(np.linalg.norm(cs - cs.mean(0),
                                          axis=1))) * 2.0
        if exc >= 0.7 * size:
            any_moved = True
            if exc / size > mover_ratio:
                mover, mover_ratio = oid, exc / size
    mover_frame = mover_mask = None
    if mover is not None and mover in per_id_best:
        _, mover_frame, mover_mask = per_id_best[mover]
    return {"presence": presence, "boxes": boxes, "masks": masks,
            "any_moved": any_moved, "mover_frame": mover_frame,
            "mover_mask": mover_mask}
