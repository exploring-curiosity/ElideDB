"""STEP 3 - ENCODE. The single shared perception function.

ONE code path for write and read. If a query is encoded by different
code than the corpus, every similarity is measured across a seam and
nothing downstream can be trusted; that seam is a classic silent
failure and this module exists to make it structurally impossible.

Contract:
    frames_for(source, t0, t1)  -> uniform frame sample of a time range
    encode_spans(source, spans) -> (N, d) unit-norm vectors
Both write (native/rawwrite2.py) and read call ONLY these.

Nothing here knows what a corpus is about: input is a media path and
time ranges, output is vectors.

    python native/encode.py --selftest     # grade for step 3
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

MID = "facebook/vjepa2-vitl-fpc64-256"
NF = 8                # frames per span handed to the encoder
DEC_FPS = 4.0         # decode rate
DEC_W = 256           # decode width
BATCH = 4

SCRATCH = Path(os.environ.get(
    "ELIDEDB_SCRATCH",
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad"))

_M = {}


def _model():
    if "m" not in _M:
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        _M["proc"] = AutoVideoProcessor.from_pretrained(MID)
        _M["m"] = AutoModel.from_pretrained(MID, dtype=torch.float16) \
            .to("mps").eval()
        _M["torch"] = torch
    return _M["m"], _M["proc"], _M["torch"]


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


def decode(path, fps=DEC_FPS, w=DEC_W):
    """Whole media at a fixed low rate. Deterministic: the same file
    always yields the same array, which is what makes write and read
    comparable at all."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True)
    W0, H0 = [int(x) for x in probe.stdout.strip().split(",")[:2]]
    H = int(round(H0 * w / W0 / 2) * 2)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
         f"fps={fps},scale={w}:{H}", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "pipe:1"], capture_output=True)
    n = len(r.stdout) // (w * H * 3)
    return np.frombuffer(r.stdout[:n * w * H * 3],
                         np.uint8).reshape(n, H, w, 3)


def frames_for(F, t0, t1, nf=NF, fps=DEC_FPS):
    """Uniform sample of one time range from a decoded array."""
    ia = int(round(t0 * fps))
    ib = int(round(t1 * fps))
    ia = max(0, min(ia, len(F) - 1))
    ib = max(ia + 1, min(ib, len(F) - 1))
    idx = np.linspace(ia, ib, nf).round().astype(int)
    return [F[min(int(j), len(F) - 1)] for j in idx]


def encode_spans(F, spans, batch=BATCH):
    """(N,d) unit-norm vectors for time ranges of one decoded media."""
    model, proc, torch = _model()
    out = []
    for i0 in range(0, len(spans), batch):
        clips = [frames_for(F, a, b) for (a, b) in spans[i0:i0 + batch]]
        inp = proc(clips, return_tensors="pt")
        pv = inp["pixel_values_videos"].to("mps", torch.float16)
        with torch.no_grad():
            o = model(pixel_values_videos=pv)
        V = o.last_hidden_state.mean(1).float().cpu().numpy()
        out.append(V)
    V = np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True),
                          1e-8)


def selftest():
    """STEP 3 GRADE: write-path and read-path encodings of the same
    ranges must be identical, and the function must be deterministic
    across independent decodes."""
    src = sorted((ROOT / "data/sim_chains").glob("ep*/cam*.mp4"))[0]
    dur = probe_duration(src)
    spans = [(0.0, 2.0), (3.0, 7.0), (max(dur - 5, 0.0), dur - 1)]
    F1 = decode(src)
    A = encode_spans(F1, spans)                    # "write path"
    F2 = decode(src)                               # independent decode
    B = encode_spans(F2, spans)                    # "read path"
    same_bytes = bool((F1 == F2).all())
    dif = float(np.abs(A - B).max())
    cos = float(np.min([A[i] @ B[i] for i in range(len(spans))]))
    # batching must not change results
    C = np.concatenate([encode_spans(F1, [s], batch=1) for s in spans])
    dif_b = float(np.abs(A - C).max())
    print(f"decode deterministic: {same_bytes}")
    print(f"write vs read  max|diff| {dif:.2e}   min cos {cos:.6f}")
    print(f"batch invariance max|diff| {dif_b:.2e}")
    ok = same_bytes and cos > 0.9999 and dif_b < 1e-3
    print(f"STEP 3 GRADE: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
