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

    p3 = root / "model/sam3_multiplex_base.py"
    src3 = p3.read_text()
    old3 = "if torch.cuda.get_device_properties(0).major >= 8:"
    new3 = ("if torch.cuda.is_available() and "
            "torch.cuda.get_device_properties(0).major >= 8:")
    if new3 in src3:
        print("sam3_multiplex_base.py: already patched")
    elif old3 in src3:
        p3.write_text(src3.replace(old3, new3))
        print(f"sam3_multiplex_base.py: patched ({p3})")
    else:
        print("sam3_multiplex_base.py: PATTERN NOT FOUND",
              file=sys.stderr)
        sys.exit(1)

    src = p.read_text()
    old6 = ("        self.freqs_cis = self.freqs_cis.to(q.device)\n"
            "        if self.freqs_cis.shape[0] != q.shape[-2]:")
    new6 = ("        self.freqs_cis = self.freqs_cis.to(q.device)\n"
            "        if self.use_rope_real:  # SDX-MPS: upstream moves"
            " only the complex tensor;\n"
            "            # the real/imag pair stays on CPU when shapes"
            " match\n"
            "            self.freqs_cis_real = "
            "self.freqs_cis_real.to(q.device)\n"
            "            self.freqs_cis_imag = "
            "self.freqs_cis_imag.to(q.device)\n"
            "        if self.freqs_cis.shape[0] != q.shape[-2]:")
    if new6 in src:
        print("decoder.py rope: already patched")
    elif old6 in src:
        p.write_text(src.replace(old6, new6))
        print("decoder.py rope: real/imag device-follow patched")
    else:
        print("decoder.py rope: PATTERN NOT FOUND", file=sys.stderr)
        sys.exit(1)

    p4 = root / "model/sam3_multiplex_tracking.py"
    src4 = p4.read_text()
    old4 = '        inference_state["device"] = torch.device("cuda")'
    new4 = ('        inference_state["device"] = torch.device(\n'
            '            "cuda" if torch.cuda.is_available() else\n'
            '            "mps" if torch.backends.mps.is_available()\n'
            '            else "cpu")  # SDX-MPS: session device feeds\n'
            "            # every tensor factory downstream\n")
    if new4 in src4:
        print("sam3_multiplex_tracking.py: already patched")
    elif old4 in src4:
        p4.write_text(src4.replace(old4, new4))
        print(f"sam3_multiplex_tracking.py: patched ({p4})")
    else:
        print("sam3_multiplex_tracking.py: PATTERN NOT FOUND",
              file=sys.stderr)
        sys.exit(1)

    p5 = root / "model/sam3_multiplex_detector.py"
    src5 = p5.read_text()
    old5 = "x.to(torch.bfloat16)"
    new5 = ("(x.to(torch.bfloat16) if torch.cuda.is_available() "
            "else x)")
    if new5 in src5:
        print("sam3_multiplex_detector.py: already patched")
    elif old5 in src5:
        p5.write_text(src5.replace(old5, new5))
        print(f"sam3_multiplex_detector.py: patched both bf16 "
              f"all-gather casts ({p5})")
    else:
        print("sam3_multiplex_detector.py: PATTERN NOT FOUND",
              file=sys.stderr)
        sys.exit(1)

    import re
    src4b = p4.read_text()
    # output buffers land on MPS (upstream assumes pinned-CPU);
    # .cpu() is a no-op for CPU tensors so the blanket is safe
    fixed, nsub = re.subn(r"(?<!cpu\(\))\.numpy\(\)",
                          ".cpu().numpy()", src4b)
    if nsub:
        p4.write_text(fixed)
        print(f"sam3_multiplex_tracking.py numpy: {nsub} sites "
              "prefixed with .cpu()")
    else:
        print("sam3_multiplex_tracking.py numpy: already patched")

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
