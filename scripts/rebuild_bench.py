"""Rebuild the bench store FROM RAW, with every stage timed.

The stores were deleted deliberately; this reconstructs everything from
`data/bridge` plus the frozen truthset, and turns on every mechanism
added since the last build: page-index footers, fp16+BSS vectors,
episode-aligned media renditions, EVC1 vector codes, and the learned ts
index.

It orchestrates the proven per-stage scripts rather than reimplementing
them, and reports the numbers a database is judged by:

  ingest    seconds per MB of source video
  embed     seconds per episode, per channel
  index     seconds and bytes per artifact
  retrieve  milliseconds per query, and bytes read per query

Stages (run all, or one at a time with --stage):
  parent    lake/bridge4h from data/bridge   (video ingest + FDNN embed)
  channels  pe, sig2, iv2, xclip, vjepa, act
  bench     filter to lake/bench (truthset episodes only)
  modernize compress vectors, rebuild media, standalone, page index
  index     EVC1 codes + learned ts index
  fit       refit the set weights on the truthset
  measure   retrieval latency, scan bytes, frozen benchmark

  python scripts/rebuild_bench.py [--stage all] [--from <stage>]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

SRC = ROOT / "data/bridge"
CAM = "observation.images.image_0"
TRUTH = ROOT / "eval/truthsets/bridge4h.parquet"
REPORT = ROOT / "eval/rebuild_timings.json"
ELIDE = ROOT / "rust/target/release/elide"

T: dict = {}


def save():
    REPORT.write_text(json.dumps(T, indent=1))


def run(name: str, cmd: list[str], **kw) -> float:
    print(f"\n=== {name} ===\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    dt = time.perf_counter() - t
    T.setdefault("stages", {})[name] = {"seconds": round(dt, 2),
                                        "exit": r.returncode}
    save()
    print(f"--- {name}: {dt:.1f}s (exit {r.returncode})", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{name} failed with exit {r.returncode}")
    return dt


def source_accounting():
    truth = pq.read_table(TRUTH).to_pydict()
    idx = sorted({int(s.rsplit("-", 1)[1]) for s in truth["stream"]})
    used = [SRC / f"videos/{CAM}/chunk-000/file-{i:03d}.mp4" for i in idx]
    missing = [p for p in used if not p.exists()]
    if missing:
        raise SystemExit(f"raw source missing: {missing}")
    b = sum(p.stat().st_size for p in used)
    T["source"] = {"files": [p.name for p in used], "bytes": b,
                   "mb": round(b / 1e6, 1)}
    save()
    print(f"source: {len(used)} files, {b/1e6:.1f} MB")
    return b


def store_size(path: Path) -> dict:
    tables = media = artifacts = 0
    for p in (path / "tables").rglob("*") if (path / "tables").is_dir() else []:
        if p.is_file():
            if p.suffix == ".parquet":
                tables += p.stat().st_size
            elif p.name.startswith(("_cache", "_codes", "_tsidx")) \
                    or p.suffix == ".npy":
                artifacts += p.stat().st_size
    if (path / "media").is_dir():
        media = sum(p.stat().st_size for p in (path / "media").iterdir()
                    if p.is_file() and not p.is_symlink())
    return {"tables": tables, "media": media, "artifacts": artifacts,
            "total": tables + media + artifacts}


STAGES = ["parent", "channels", "bench", "modernize", "index", "fit",
          "measure"]


def main():
    argv = sys.argv
    only = argv[argv.index("--stage") + 1] if "--stage" in argv else "all"
    start = argv[argv.index("--from") + 1] if "--from" in argv else None
    todo = STAGES if only == "all" else [only]
    if start:
        todo = STAGES[STAGES.index(start):]
    py = sys.executable

    src_bytes = source_accounting()

    if "parent" in todo:
        dt = run("parent: bridge4h ingest + FDNN embed",
                 [py, "scripts/bridge4h.py"])
        T["ingest"] = {
            "seconds": round(dt, 2),
            "source_mb": round(src_bytes / 1e6, 1),
            "mb_per_second": round(src_bytes / 1e6 / dt, 2),
            "seconds_per_mb": round(dt / (src_bytes / 1e6), 4),
        }
        save()

    if "channels" in todo:
        for name, script in [("pe", "pe_ingest.py"),
                             ("sig2", "sig2_ingest.py"),
                             ("iv2", "iv2_ingest.py"),
                             ("xclip", "xclip_ingest.py"),
                             ("vjepa", "vjepa_channel.py"),
                             ("act", "action_ingest.py")]:
            run(f"channel: {name}", [py, f"scripts/{script}", "lake/bridge4h"])

    if "bench" in todo:
        run("bench: filter to truthset episodes",
            [py, "scripts/make_bench_store.py"])

    if "modernize" in todo:
        run("modernize: fp16+BSS vectors",
            [py, "scripts/compress_vectors.py", "lake/bench"])
        run("modernize: media rendition rebuild",
            [py, "scripts/recompress_media.py", "lake/bench",
             "--from-store", "lake/bridge4h"])
        run("modernize: standalone + reclaim",
            [py, "scripts/make_standalone.py", "lake/bench"])
        run("modernize: page index", [py, "scripts/add_page_index.py",
                                      "lake/bench"])

    if "index" in todo:
        for tbl in ("frame_vectors", "pe_vectors", "sig2_vectors",
                    "iv2_vectors", "xclip_vectors", "vjepa_vectors",
                    "object_vectors", "motion_vectors"):
            try:
                run(f"index: EVC1 codes {tbl}",
                    [ELIDE, "vindex", "lake/bench", tbl])
            except SystemExit:
                print(f"  (skip {tbl}: not present)")
        for tbl in ("frames", "frame_vectors", "episodes"):
            try:
                run(f"index: learned ts {tbl}",
                    [ELIDE, "tsindex", "lake/bench", tbl])
            except SystemExit:
                print(f"  (skip {tbl})")

    if "fit" in todo:
        run("fit: set weights on the truthset",
            [py, "scripts/fit_set_weights.py"])

    if "measure" in todo:
        run("measure: frozen benchmark + retrieval latency",
            [py, "scripts/bench_truth.py"])
        T["store"] = store_size(ROOT / "lake/bench")
        T["store"]["source_bytes"] = src_bytes
        T["store"]["ratio_to_source"] = round(
            T["store"]["total"] / src_bytes, 4)
        save()

    print("\n===== TIMING REPORT =====")
    print(json.dumps(T, indent=1))
    print(f"\nwritten to {REPORT}")


if __name__ == "__main__":
    main()
