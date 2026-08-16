"""Stage A of SSL_TRAINING.md: turn raw downloads into registry corpora.

Each subcommand writes data/relmo/datasets/<name>/ with per-episode mp4s and
a manifest.json in registry format. Resumable: existing mp4s are skipped and
the manifest is rebuilt from DISK at the end, never from intent.

    python3 -m relmo.prep_data bridge|kitti|oxford|drone
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

DATA = R.BASE.parent            # .../data
FF = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]


def probe_dur(p):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)], text=True).strip()
        return float(out)
    except Exception:                                          # noqa: BLE001
        return 0.0


def write_manifest(out, fps, extra=None):
    """Manifest from what is ON DISK. `extra` maps id -> dict merged in."""
    eps = []
    for p in sorted(out.glob("*/frames.mp4")):
        d = probe_dur(p)
        if d <= 0:
            p.unlink()          # broken cut: remove so a rerun redoes it
            continue
        e = dict(id=p.parent.name, video=str(p), fps=fps,
                 T=int(round(d * fps)))
        if extra and p.parent.name in extra:
            e.update(extra[p.parent.name])
        eps.append(e)
    (out / "manifest.json").write_text(json.dumps(
        dict(fps=fps, episodes=eps), indent=1))
    hrs = sum(e["T"] / fps for e in eps) / 3600
    print(f"VERIFIED {out.name}: {len(eps)} episodes on disk, {hrs:.2f} h")
    return len(eps), hrs


def bridge(hours=5.0, seed=0):
    """ONE episode per distinct task string, round-robin to the budget.
    Camera image_0 only (cross-view is barred anyway). Task strings are kept
    in the manifest for the TEXT layer; f never reads them."""
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    src = DATA / "bridge"
    cam = "observation.images.image_0"
    e = pd.read_parquet(sorted((src / "meta" / "episodes").rglob("*.parquet")))
    t = pd.read_parquet(src / "meta" / "tasks.parquet").reset_index()
    fps = float(json.load(open(src / "meta" / "info.json"))["fps"])
    e["task"] = e["tasks"].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) else str(x))
    rng = np.random.default_rng(seed)
    # one random episode per task, then shuffle task order for the budget cut
    picks = (e.sample(frac=1, random_state=seed)
             .groupby("task", sort=False).head(1))
    picks = picks.sample(frac=1, random_state=seed + 1)
    budget = hours * 3600
    rows, tot = [], 0.0
    for _, r in picks.iterrows():
        dur = float(r["length"]) / fps
        if dur < 4.0:           # shorter than one encoder window
            continue
        if tot + dur > budget:
            break
        rows.append(r)
        tot += dur
    out = R.BASE / "datasets" / "bridge_wide"
    out.mkdir(parents=True, exist_ok=True)
    print(f"bridge_wide: {len(rows)} episodes, {tot/3600:.2f} h, "
          f"{len(set(r['task'] for r in rows))} distinct tasks", flush=True)
    extra = {}
    for r in tqdm(rows, unit="ep", desc="bridge-cut"):
        eid = f"ep{int(r['episode_index']):06d}"
        extra[eid] = dict(task=str(r["task"]))
        dst = out / eid / "frames.mp4"
        if dst.exists():
            continue
        dst.parent.mkdir(exist_ok=True)
        chunk = int(r[f"videos/{cam}/chunk_index"])
        fi = int(r[f"videos/{cam}/file_index"])
        t0 = float(r[f"videos/{cam}/from_timestamp"])
        t1 = float(r[f"videos/{cam}/to_timestamp"])
        vid = src / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{fi:03d}.mp4"
        # re-encode: -c copy would snap to keyframes and bleed neighbours
        subprocess.run(FF + ["-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
                             "-i", str(vid), "-c:v", "libx264",
                             "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", str(dst)], check=True)
    write_manifest(out, fps, extra)


def _seq_to_mp4(frames, dst, fps):
    """Image sequence -> mp4 via a concat list (globs are fragile)."""
    lst = dst.parent / ".frames.txt"
    lst.write_text("".join(f"file '{f}'\nduration {1/fps:.6f}\n"
                           for f in frames))
    subprocess.run(FF + ["-f", "concat", "-safe", "0", "-i", str(lst),
                         "-c:v", "libx264", "-preset", "veryfast", "-crf",
                         "20", "-pix_fmt", "yuv420p",
                         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                         str(dst)], check=True)
    lst.unlink()


def kitti(fps=10.0):
    from tqdm import tqdm
    out = R.BASE / "datasets" / "kitti_seq"
    out.mkdir(parents=True, exist_ok=True)
    drives = sorted(p for p in (DATA / "kitti").rglob("image_02")
                    if (p / "data").is_dir())
    print(f"kitti_seq: {len(drives)} drives", flush=True)
    for d in tqdm(drives, unit="drive", desc="kitti"):
        eid = d.parent.name          # e.g. 2011_09_26_drive_0059_sync
        dst = out / eid / "frames.mp4"
        if dst.exists():
            continue
        frames = sorted((d / "data").glob("*.png"))
        if len(frames) < 40:         # < one 4 s window at 10 fps
            continue
        dst.parent.mkdir(exist_ok=True)
        _seq_to_mp4(frames, dst, fps)
    write_manifest(out, fps)


def oxford(fps=16.0):
    out = R.BASE / "datasets" / "oxford_seq"
    out.mkdir(parents=True, exist_ok=True)
    for cam_dir in sorted((DATA / "oxfordDataset").iterdir()):
        if not cam_dir.is_dir():
            continue
        frames = sorted(cam_dir.glob("*.png")) or sorted(cam_dir.glob("*.jpg"))
        if len(frames) < int(4 * fps):
            continue
        dst = out / f"oxford_{cam_dir.name}" / "frames.mp4"
        if dst.exists():
            continue
        dst.parent.mkdir(exist_ok=True)
        print(f"oxford {cam_dir.name}: {len(frames)} frames", flush=True)
        _seq_to_mp4(frames, dst, fps)
    write_manifest(out, fps)


def drone(fps=30.0):
    from tqdm import tqdm
    src = DATA / "drone"
    stage = src / "_extracted"
    stage.mkdir(exist_ok=True)
    for z in sorted(src.glob("*.zip")):
        mark = stage / f".done_{z.stem}"
        if mark.exists():
            continue
        print(f"unzip {z.name}...", flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(stage / z.stem)
        mark.touch()
    out = R.BASE / "datasets" / "drone_fpv"
    out.mkdir(parents=True, exist_ok=True)
    # any dir under _extracted holding >=120 ordered images is a sequence
    seqs = []
    for d in sorted(stage.rglob("*")):
        if not d.is_dir():
            continue
        fr = sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))
        if len(fr) >= int(4 * fps):
            seqs.append((d, fr))
    # drop nested duplicates (keep deepest dirs only)
    seqs = [(d, fr) for d, fr in seqs
            if not any(str(o).startswith(str(d) + "/") for o, _ in seqs
                       if o != d)]
    print(f"drone_fpv: {len(seqs)} sequences", flush=True)
    for d, fr in tqdm(seqs, unit="seq", desc="drone"):
        eid = "_".join(d.relative_to(stage).parts)[:80]
        dst = out / eid / "frames.mp4"
        if dst.exists():
            continue
        dst.parent.mkdir(exist_ok=True)
        _seq_to_mp4(fr, dst, fps)
    write_manifest(out, fps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bridge", "kitti", "oxford", "drone"])
    a = ap.parse_args()
    {"bridge": bridge, "kitti": kitti, "oxford": oxford, "drone": drone}[a.cmd]()
