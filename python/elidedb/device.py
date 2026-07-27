"""One device policy for every model loader.

The store format is portable; the model loaders must be too. Every
loader asks this module instead of probing torch itself, so a cloud
CPU box, an Apple laptop, and a CUDA node run the same code with the
right placement:

  ELIDEDB_DEVICE  force a device ("cpu", "mps", "cuda"); default is
                  mps when available, else cuda, else cpu
  ELIDEDB_DTYPE   force a torch dtype name; default float16 on
                  mps/cuda and bfloat16 on cpu (halves resident
                  memory against fp32, and fp16 matmuls are not a
                  real CPU option)
"""
from __future__ import annotations

import os


def text_only():
    """Serving deployments set ELIDEDB_TEXT_ONLY=1: the query path
    encodes TEXT only (every frame vector is precomputed at ingest),
    so the loaders drop their vision towers after load and roughly
    halve resident memory. Ingest machines leave it unset."""
    return os.environ.get("ELIDEDB_TEXT_ONLY", "") == "1"


def strip_vision(model, *attrs):
    """Release the named submodules when serving text-only."""
    if not text_only():
        return model
    import gc
    for a in attrs:
        if hasattr(model, a):
            setattr(model, a, None)
    gc.collect()
    return model


def pick():
    import torch
    dev = os.environ.get("ELIDEDB_DEVICE", "").strip()
    if not dev:
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    name = os.environ.get("ELIDEDB_DTYPE", "").strip()
    if name:
        dtype = getattr(torch, name)
    else:
        dtype = torch.float16 if dev in ("mps", "cuda") \
            else torch.bfloat16
    return dev, dtype
