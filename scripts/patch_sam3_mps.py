"""Patch the installed `sam3` package for non-CUDA (Apple MPS) hosts.

Upstream assumes CUDA in a handful of inference-path spots. Re-runnable
(idempotent) so a reinstall of sam3 just needs this script again:

1. decoder.py: TransformerDecoder.__init__ eagerly precomputes a coord
   cache on a HARDCODED device="cuda". The lazy path in _get_rpb_matrix
   already rebuilds it on the INPUT's device when the cache is None —
   so the correct fix is to skip the eager warmup, not to port it.

2. perflib/fused.py: addmm_act (fused linear+activation used by every
   ViT MLP block) hardcodes bfloat16 — the bf16 activations then mix
   with fp32 downstream and abort in a Metal matmul assertion
   ("Destination NDArray and Accumulator NDArray cannot have different
   datatype"), found by forward-pre-hook bisection. On non-CUDA
   devices fall back to plain activation(linear(x)) in input dtype.

Everything else is handled at call sites in elidedb.sam3x: the builder
only moves the model for device == "cuda" (we .to(dev) ourselves), and
Sam3Processor takes an explicit device argument.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    # find_spec locates the package without executing its __init__
    # (importing sam3 would hit the CUDA-only triton import first)
    from importlib.util import find_spec
    root = Path(find_spec("sam3").submodule_search_locations[0])
    p = root / "model/decoder.py"
    src = p.read_text()
    old = ("            if resolution is not None and "
           "stride is not None:\n"
           "                feat_size = resolution // stride\n"
           "                coords_h, coords_w = self._get_coords(\n"
           '                    feat_size, feat_size, device="cuda"\n'
           "                )")
    new = ("            if False:  # SDX-MPS: skip eager CUDA coord "
           "cache; lazy path builds on input device\n"
           "                feat_size = resolution // stride\n"
           "                coords_h, coords_w = self._get_coords(\n"
           '                    feat_size, feat_size, device="cuda"\n'
           "                )")
    if new in src:
        print("decoder.py: already patched")
    elif old in src:
        p.write_text(src.replace(old, new))
        print(f"decoder.py: patched ({p})")
    else:
        print("decoder.py: PATTERN NOT FOUND — upstream changed, "
              "inspect manually", file=sys.stderr)
        sys.exit(1)

    p2 = root / "perflib/fused.py"
    src2 = p2.read_text()
    old2 = ("def addmm_act(activation, linear, mat1):\n"
            "    if torch.is_grad_enabled():\n"
            '        raise ValueError("Expected grad to be disabled.")\n')
    new2 = ("def addmm_act(activation, linear, mat1):\n"
            "    if torch.is_grad_enabled():\n"
            '        raise ValueError("Expected grad to be disabled.")\n'
            '    if mat1.device.type != "cuda":\n'
            "        # SDX-MPS: the fused op below hardcodes bfloat16;"
            " bf16 activations\n"
            "        # abort in a Metal matmul dtype assertion —"
            " plain path instead\n"
            "        if activation in [torch.nn.functional.relu,"
            " torch.nn.ReLU]:\n"
            "            return torch.nn.functional.relu(linear(mat1))\n"
            "        return torch.nn.functional.gelu(linear(mat1))\n")
    if new2 in src2:
        print("perflib/fused.py: already patched")
    elif old2 in src2:
        p2.write_text(src2.replace(old2, new2))
        print(f"perflib/fused.py: patched ({p2})")
    else:
        print("perflib/fused.py: PATTERN NOT FOUND — upstream changed, "
              "inspect manually", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
