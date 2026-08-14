"""One place that decides where tensors live.

PORTABILITY DECISION (2026-08-11): the training stack stays PyTorch,
never MLX, because the target hardware is NVIDIA + AMD + TPU.
  PyTorch  cuda (NVIDIA), rocm (AMD), xla (TPU), mps (Apple) - the
           same model code, one device string.
  MLX      Apple-first; a CUDA backend exists and is Apple-sponsored
           but not all operators are implemented, AMD is "further in
           the future", and there is no TPU path at all.
Weights are portable either way (arrays are arrays); it is the
TRAINING LOOP that gets locked to a vendor, so that is what must stay
neutral. FDNN's ideas are therefore reimplemented in PyTorch rather
than imported from the MLX model.

Override with RELMO_DEVICE=cuda|mps|cpu.
"""
from __future__ import annotations

import os

import torch


def pick():
    d = os.environ.get("RELMO_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and \
            torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick()


def empty_cache():
    """Vendor-neutral cache release (the MPS allocator leak cost a 20x
    slowdown once; the same call must not crash on CUDA or CPU)."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()
