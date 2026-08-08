"""Which DINOv3 is worth the compute? Measured, not assumed.

The incumbent is ConvNeXt-Tiny (29M) - the SMALLEST member of the
family - chosen when the budget was one minute per hour of video. With
52 GB of unified memory and 2 GB in use, that is the wrong point on the
curve.

Every candidate here is DINOv3: self-supervised on LVD-1689M with no
labels, no captions and no codebook, so the whole family is admissible
under the vision-only rule. Nothing else enters - a text-aligned encoder
would be disqualified however well it scored.

    --probe   throughput and feature dim per candidate, no store built
    --run     build a fixed slice with each and grade it identically

Throughput is measured on real frames at the real decode width, because
a cost estimated from the smallest media in the corpus is how the last
cost estimate came out 2x wrong.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "native"))

import vsrc                                              # noqa: E402

# name -> (hf id, needs fp32).  DINOv3 ViT overflows fp16 - this project
# hit that NaN trap before and it is recorded rather than rediscovered.
CANDIDATES = {
    "ct":   ("facebook/dinov3-convnext-tiny-pretrain-lvd1689m", False),
    "cs":   ("facebook/dinov3-convnext-small-pretrain-lvd1689m", False),
    "cb":   ("facebook/dinov3-convnext-base-pretrain-lvd1689m", False),
    "cl":   ("facebook/dinov3-convnext-large-pretrain-lvd1689m", False),
    "vitb": ("facebook/dinov3-vitb16-pretrain-lvd1689m", True),
    "vitl": ("facebook/dinov3-vitl16-pretrain-lvd1689m", True),
    "vithp": ("facebook/dinov3-vith16plus-pretrain-lvd1689m", True),
}


def probe(name, frames, res, batch):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    mid, fp32 = CANDIDATES[name]
    dtype = torch.float32 if fp32 else torch.float16
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(mid)
    m = AutoModel.from_pretrained(mid, dtype=dtype,
                                  low_cpu_mem_usage=True).to("mps").eval()
    load = time.time() - t0
    npar = sum(p.numel() for p in m.parameters())
    out, t0 = [], time.time()
    for i in range(0, len(frames), batch):
        chunk = [np.ascontiguousarray(x[..., :3])
                 for x in frames[i:i + batch]]
        px = proc(images=chunk, return_tensors="pt",
                  size={"height": res, "width": res})["pixel_values"]
        with torch.no_grad():
            r = m(pixel_values=px.to("mps", dtype))
        v = getattr(r, "pooler_output", None)
        if v is None:
            v = r.last_hidden_state.mean(1)
        out.append(v.float().cpu().numpy())
    torch.mps.synchronize()
    dt = time.time() - t0
    V = np.concatenate(out)
    peak = torch.mps.current_allocated_memory() / 1e9
    bad = bool(np.isnan(V).any())
    del m
    torch.mps.empty_cache()
    # frames per second -> minutes of compute per hour of video, at
    # vsrc.FPS sampled frames per second of video
    fps = len(frames) / dt
    minph = (3600 * vsrc.FPS / fps) / 60
    return dict(name=name, par=npar / 1e6, dim=V.shape[1], load=load,
                fps=fps, minph=minph, gb=peak, nan=bad)


def sweep(name, frames, res, batches):
    """Largest batch the machine will take, and what it buys."""
    print(f"\n{name} @ {res}")
    print(f"{'batch':<8}{'frames/s':<11}{'min/h':<9}{'alloc GB':<10}"
          f"{'driver GB'}")
    best = None
    for b in batches:
        try:
            r = probe(name, frames, res, b)
        except Exception as e:                           # noqa: BLE001
            print(f"{b:<8}FAILED {type(e).__name__}: {str(e)[:50]}")
            break
        import torch
        drv = torch.mps.driver_allocated_memory() / 1e9
        print(f"{b:<8}{r['fps']:<11.1f}{r['minph']:<9.1f}{r['gb']:<10.2f}"
              f"{drv:.2f}", flush=True)
        if best is None or r["fps"] > best[1]:
            best = (b, r["fps"])
    return best


def main():
    from flowgebd import arg
    res = arg("--res", 224, int)
    batch = arg("--batch", 32, int)
    n = arg("--frames", 96, int)
    want = arg("--only", ",".join(CANDIDATES)).split(",")

    s = vsrc.sources("bridge", 1)[0]
    F = s.cut(0.0, n / vsrc.FPS)
    if "--sweep" in sys.argv:
        for spec in arg("--sweep", "cl:448,vitl:224").split(","):
            nm, r = spec.split(":")
            sweep(nm, F, int(r), [16, 32, 64, 128, 192, 256])
        return
    print(f"probe on {len(F)} real frames at {F.shape[2]}x{F.shape[1]} "
          f"decode, model input {res}, batch {batch}\n")
    print(f"{'model':<7}{'params M':<10}{'dim':<6}{'load s':<8}"
          f"{'frames/s':<10}{'min/h video':<13}{'GB':<7}{'NaN'}")
    for name in want:
        if name not in CANDIDATES:
            continue
        try:
            r = probe(name, F, res, batch)
        except Exception as e:                           # noqa: BLE001
            print(f"{name:<7}FAILED {type(e).__name__}: {str(e)[:60]}")
            continue
        print(f"{r['name']:<7}{r['par']:<10.0f}{r['dim']:<6}"
              f"{r['load']:<8.1f}{r['fps']:<10.1f}{r['minph']:<13.1f}"
              f"{r['gb']:<7.1f}{'YES' if r['nan'] else '-'}", flush=True)


if __name__ == "__main__":
    main()
