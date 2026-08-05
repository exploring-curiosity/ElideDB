"""SEGMENTATION-FREE write: raw video in, window vectors out.

No dataset metadata of any kind. The old kitchen store read demo
boundaries out of Bridge's meta parquet (from_timestamp/to_timestamp
per demo), so every number measured on it assumed a segmentation a
customer upload does not have. This path is told nothing: it walks
raw files on a uniform multi-scale window grid and embeds each window
with frozen pretrained encoders.

Scales are a ladder, not a tuned choice - the system cannot know how
long an "event" is in an unseen domain, so it indexes several
durations and lets retrieval pick.

    python native/rawwrite.py --corpus sim   [--limit N]
    python native/rawwrite.py --corpus bench [--limit N]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

SCALES = (2.0, 4.0, 8.0)      # window durations, seconds
STRIDE_FRAC = 0.5             # stride = scale * this
NF = 8                        # frames fed per window
FPS_DEC = 4.0                 # decode rate
VER = "v1"

SCRATCH = Path(os.environ.get(
    "ELIDEDB_SCRATCH",
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad"))


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def out_dir(corpus):
    d = SCRATCH / f"raw_{VER}_{corpus}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corpus_files(corpus, limit=0):
    """[(file_id, path, duration_s)] - the ONLY structure used."""
    out = []
    if corpus == "sim":
        eps = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                     if p.is_dir() and p.name.startswith("ep"))
        for d in eps:
            cams = sorted(d.glob("cam*.mp4"))
            if cams:
                out.append((d.name, cams[0]))
    else:
        base = ROOT / "data/bridge/videos/observation.images.image_0/chunk-000"
        for f in (119, 120, 129, 132):
            p = base / f"file-{f:03d}.mp4"
            if p.exists():
                out.append((f"file-{f:03d}", p))
    if limit:
        out = out[:limit]
    res = []
    for fid, p in out:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0",
                            str(p)], capture_output=True, text=True)
        res.append((fid, p, float(r.stdout.strip())))
    return res


def decode_all(path, fps=FPS_DEC, w=256):
    """Whole file at low rate/resolution -> (T,H,W,3) uint8."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
         f"fps={fps},scale={w}:-2", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "pipe:1"], capture_output=True)
    buf = r.stdout
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True)
    W0, H0 = [int(x) for x in probe.stdout.strip().split(",")[:2]]
    H = int(round(H0 * w / W0 / 2) * 2)
    n = len(buf) // (w * H * 3)
    return np.frombuffer(buf[:n * w * H * 3], np.uint8).reshape(n, H, w, 3)


def windows_for(dur, fps=FPS_DEC):
    """Uniform multi-scale grid. No content, no metadata."""
    out = []
    for sc in SCALES:
        step = sc * STRIDE_FRAC
        t = 0.0
        while t + sc <= dur + 1e-6:
            out.append((t, t + sc, sc))
            t += step
        if not out or out[-1][1] < dur - sc:
            pass
    return out


def main():
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    from tqdm import tqdm

    corpus = arg("--corpus", "sim")
    limit = arg("--limit", 0, int)
    files = corpus_files(corpus, limit)
    tot_s = sum(f[2] for f in files)
    d = out_dir(corpus)
    todo = [f for f in files if not (d / f"{f[0]}.npz").exists()]
    print(f"{corpus}: {len(files)} raw files, {tot_s/60:.0f} min video, "
          f"{len(todo)} to index", flush=True)
    if not todo:
        return
    est = sum(len(windows_for(f[2])) for f in todo)
    print(f"  ~{est:,} windows at scales {SCALES}s "
          f"(ETA ~{est*0.42/60:.0f} min)", flush=True)

    MID = "facebook/vjepa2-vitl-fpc64-256"
    proc = AutoVideoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=torch.float16) \
        .to("mps").eval()

    t_all = time.time()
    for fid, path, dur in tqdm(todo, desc=corpus, unit="file"):
        F = decode_all(path)
        wins = windows_for(dur)
        vecs, meta = [], []
        B = 4
        for i0 in range(0, len(wins), B):
            chunk = wins[i0:i0 + B]
            clips = []
            for (a, b, sc) in chunk:
                ia = int(a * FPS_DEC)
                ib = min(int(b * FPS_DEC), len(F) - 1)
                if ib <= ia:
                    ib = min(ia + 1, len(F) - 1)
                idx = np.linspace(ia, ib, NF).round().astype(int)
                clips.append([F[min(j, len(F) - 1)] for j in idx])
            inp = proc(clips, return_tensors="pt")
            pv = inp["pixel_values_videos"].to("mps", torch.float16)
            with torch.no_grad():
                o = model(pixel_values_videos=pv)
            V = o.last_hidden_state.mean(1).float().cpu().numpy()
            for (a, b, sc), v in zip(chunk, V):
                vecs.append(v / (np.linalg.norm(v) + 1e-8))
                meta.append((a, b, sc))
        np.savez_compressed(d / f"{fid}.npz",
                            V=np.stack(vecs).astype(np.float16),
                            win=np.array(meta, np.float32))
    print(f"indexed {len(todo)} files in "
          f"{(time.time()-t_all)/60:.1f} min -> {d}", flush=True)


if __name__ == "__main__":
    main()
