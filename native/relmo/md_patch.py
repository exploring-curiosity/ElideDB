"""MetaDrive Apple-Silicon compatibility shim.

MEASURED, not guessed. MetaDrive hardcodes 16x MSAA in five places. On
Apple Silicon, Panda3D's GL 4.1 backend fails to allocate a 16x-MSAA
float16 FBO, and `FilterManager.render_scene_into()` returns None
instead of raising - so the first symptom is a bare
`AttributeError: 'NoneType' object has no attribute 'set_shader'`
several frames into engine construction, which points nowhere near the
cause. Probe result on this machine:

    msaa=16  render_scene_into -> None   (FAILS)
    msaa= 8  render_scene_into -> OK
    msaa= 4  render_scene_into -> OK

The Panda3D types are immutable C++ bindings, so this cannot be
monkeypatched at runtime; the installed source must be edited. This
script does that idempotently and is safe to re-run after any
`pip install -U metadrive-simulator`.

Second fix: MetaDrive assumes "Mac don't support offscreen rendering"
and forces an on-screen window, which would make headless/detached
generation impossible. Measured on this machine, Panda3D offscreen
buffers DO work (GraphicsBuffer created, gsg valid), so that override
is disabled.

    python -m relmo.md_patch          # apply
    python -m relmo.md_patch --check  # report only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_MSAA = 4


def targets():
    import metadrive
    root = Path(metadrive.__file__).parent
    return sorted(set(
        list((root / "component" / "sensors").glob("*.py"))
        + [root / "engine" / "core" / "image_buffer.py",
           root / "engine" / "core" / "engine_core.py",
           root / "third_party" / "simplepbr" / "__init__.py"]))


def patch(check_only=False):
    pats = [(re.compile(r"set_multisamples\((\d+)\)"), "set_multisamples"),
            (re.compile(r"msaa_samples\s*=\s*(\d+)"), "msaa_samples")]
    changed, found = [], []
    for f in targets():
        if not f.exists():
            continue
        src = f.read_text()
        out = src
        for rx, name in pats:
            def sub(m):
                n = int(m.group(1))
                if n <= MAX_MSAA:
                    return m.group(0)
                found.append(f"{f.name}: {name}={n}")
                return m.group(0).replace(str(n), str(MAX_MSAA))
            out = rx.sub(sub, out)
        # let headless offscreen work: the mac override forces a window
        out = out.replace(
            "if is_mac() and (self.mode == RENDER_MODE_OFFSCREEN):",
            "if False and is_mac() and (self.mode == RENDER_MODE_OFFSCREEN):")
        if out != src:
            changed.append(f.name)
            if not check_only:
                f.write_text(out)
    return dict(patched=changed, sites=found, applied=not check_only,
                max_msaa=MAX_MSAA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    import json
    print(json.dumps(patch(a.check), indent=1))
