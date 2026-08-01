#!/usr/bin/env python3
"""Register a semantic run as a new manifest snapshot.

Snapshots are immutable: attaching embeddings never edits an existing
manifest — it writes manifest-(N+1) with the semantic_run block and moves
CURRENT. `sdx query --snapshot N` before/after shows exactly which corpus a
result set was computed against (embedding model id included: scores are
meaningless without it).
"""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    store = Path(args.store)
    run_dir = store / "ml" / args.run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    n = len(json.loads((run_dir / "windows.json").read_text())["windows"])

    current = int((store / "CURRENT").read_text().strip())
    manifest = json.loads(
        (store / "manifests" / f"manifest-{current}.json").read_text())
    manifest["snapshot"] = current + 1
    manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["semantic_run"] = {
        "run_id": args.run_id,
        "model": meta["model"],
        "dim": meta["dim"],
        "window_count": n,
    }
    out = store / "manifests" / f"manifest-{current + 1}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite immutable {out}")
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    (store / "CURRENT").write_text(f"{current + 1}\n")
    print(f"snapshot {current + 1}: semantic run {args.run_id} "
          f"({n} windows, {meta['model']})")


if __name__ == "__main__":
    main()
