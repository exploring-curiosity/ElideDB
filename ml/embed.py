#!/usr/bin/env python3
"""Embed per-window frames with SigLIP (local, MLX) and write the run files
the C++ vector index consumes.

Window = the retrieval unit: a fixed-length slice of one camera stream's
timeline (default 2 s). One or more frames are sampled per window via the SFI
byte-range path (never a full-file decode), embedded, mean-pooled and
L2-normalized (so cosine == dot downstream).

Outputs under <store>/ml/<run_id>/:
  embeddings.f32   raw float32 [n, d], row-major, L2-normalized
  embeddings.npy   same data for notebooks
  windows.json     {"dim": d, "windows": [{stream_id, t0_ns, t1_ns}, ...]}
  meta.json        model id + embed_text_cmd (queries must use THIS model)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from sfi_reader import read_sfi, read_jpeg_packet  # noqa: E402


def decode_frame(sfi, row: int) -> Image.Image | None:
    """Decode one frame; None if the packet is corrupt (e.g. the torn tail
    frame of a segment that was mid-write when recording stopped)."""
    try:
        img = Image.open(io.BytesIO(read_jpeg_packet(sfi, row)))
        # Draft mode: JPEG DCT-domain downscale (~1/4) makes 4K decode cheap;
        # SigLIP resizes to 384x384 anyway, so nothing of value is lost.
        img.draft("RGB", (img.width // 4, img.height // 4))
        return img.convert("RGB")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--model", default="mlx-community/siglip-so400m-patch14-384")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--frames-per-window", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    store = Path(args.store)
    current = int((store / "CURRENT").read_text().strip())
    manifest = json.loads(
        (store / "manifests" / f"manifest-{current}.json").read_text())

    # ---- Enumerate windows and their sample frame rows ----------------------
    win_ns = int(args.window_s * 1e9)
    windows = []  # (stream_id, t0, t1, [(sfi, row), ...])
    for vs in manifest.get("video_streams", []):
        for seg in vs["segments"]:
            sfi = read_sfi(str(store / seg["sfi_path"]))
            t = (seg["first_pts_ns"] // win_ns) * win_ns
            while t <= seg["last_pts_ns"]:
                t0, t1 = max(t, seg["first_pts_ns"]), min(t + win_ns - 1,
                                                          seg["last_pts_ns"])
                lo = np.searchsorted(sfi.frames["pts_ns"], t0, side="left")
                hi = np.searchsorted(sfi.frames["pts_ns"], t1, side="right")
                if hi > lo:
                    k = min(args.frames_per_window, hi - lo)
                    rows = np.linspace(lo, hi - 1, k).round().astype(int)
                    windows.append((vs["stream_id"], int(t0), int(t1),
                                    [(sfi, int(r)) for r in rows]))
                t += win_ns
    if args.max_windows:
        windows = windows[: args.max_windows]
    print(f"embedding {len(windows)} windows "
          f"({args.frames_per_window} frame(s) each) with {args.model}")

    # ---- Load model (downloads on first run) --------------------------------
    from mlx_embeddings.utils import load
    import mlx.core as mx
    model, processor = load(args.model)

    def embed_images(images):
        inputs = processor(images=images, return_tensors="np")
        out = model.get_image_features(mx.array(inputs["pixel_values"]))
        emb = np.array(out, dtype=np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # ---- Batched embed, mean-pool per window --------------------------------
    dim = None
    vecs = np.zeros((len(windows), 0), dtype=np.float32)
    frame_jobs = [(wi, sfi, row) for wi, (_, _, _, rows) in enumerate(windows)
                  for (sfi, row) in rows]
    pooled: dict[int, list[np.ndarray]] = {}
    t_start = time.time()
    bad_frames = 0
    for i in range(0, len(frame_jobs), args.batch):
        chunk = frame_jobs[i:i + args.batch]
        decoded = [(job, decode_frame(sfi, row)) for job in chunk
                   for (_, sfi, row) in [job]]
        good = [(job, img) for (job, img) in decoded if img is not None]
        bad_frames += len(decoded) - len(good)
        if not good:
            continue
        emb = embed_images([img for (_, img) in good])
        if dim is None:
            dim = emb.shape[1]
            vecs = np.zeros((len(windows), dim), dtype=np.float32)
        for ((wi, _, _), _), e in zip(good, emb):
            pooled.setdefault(wi, []).append(e)
        done = i + len(chunk)
        if done % (args.batch * 8) < args.batch:
            rate = done / (time.time() - t_start)
            print(f"  {done}/{len(frame_jobs)} frames ({rate:.1f}/s)",
                  flush=True)
    if bad_frames:
        print(f"  skipped {bad_frames} undecodable frames (torn tail packets)")
    # Windows whose every sampled frame was corrupt are dropped outright —
    # a zero vector would poison cosine ranking.
    keep = sorted(pooled.keys())
    for wi in keep:
        v = np.mean(pooled[wi], axis=0)
        vecs[wi] = v / np.linalg.norm(v)
    if len(keep) < len(windows):
        print(f"  dropped {len(windows) - len(keep)} windows with no frames")
        windows = [windows[wi] for wi in keep]
        vecs = vecs[keep]

    # ---- Write the run ------------------------------------------------------
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = store / "ml" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    vecs.tofile(run_dir / "embeddings.f32")
    np.save(run_dir / "embeddings.npy", vecs)
    (run_dir / "windows.json").write_text(json.dumps({
        "dim": dim,
        "windows": [{"stream_id": s, "t0_ns": t0, "t1_ns": t1}
                    for (s, t0, t1, _) in windows],
    }))
    embed_text_cmd = (f"{sys.executable} "
                      f"{Path(__file__).parent.resolve()}/embed_text.py "
                      f"--model {args.model}")
    (run_dir / "meta.json").write_text(json.dumps({
        "model": args.model,
        "dim": dim,
        "window_s": args.window_s,
        "frames_per_window": args.frames_per_window,
        "embed_text_cmd": embed_text_cmd,
    }, indent=2))
    print(f"run {run_id}: {len(windows)} windows, dim {dim} -> {run_dir}")
    print(f"next: python3 ml/cluster.py --store {args.store} --run-id {run_id}")


if __name__ == "__main__":
    main()
