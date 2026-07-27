"""Build the cloud demo bundle: a STANDALONE copy of the bench store.

The lake's bench store symlinks its media into bridge4h to save local
disk; a cloud deployment cannot follow links out of its own directory.
This copies the store with every symlink materialized, drops derived
caches (mmap sidecars rebuild on first touch) and fitter snapshots,
and verifies the result opens and reports standalone.

  python scripts/build_demo_store.py [src] [dst]
  defaults: lake/bench -> deploy/demo/lake/bench
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

SKIP_NAMES = {"_cache", "_desk_umap.json", "_set_weights.prev.json"}


def copy_tree(src: Path, dst: Path):
    copied = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in SKIP_NAMES for part in rel.parts):
            continue
        out = dst / rel
        if p.is_dir() and not p.is_symlink():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # copy2 follows symlinks: linked media materializes here
        shutil.copy2(p, out)
        copied += p.stat().st_size
    return copied


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "lake/bench")
    dst = Path(sys.argv[2] if len(sys.argv) > 2
               else "deploy/demo/lake/bench")
    if dst.exists():
        shutil.rmtree(dst)
    n = copy_tree(src, dst)
    links = [p for p in dst.rglob("*") if p.is_symlink()]
    assert not links, f"symlinks survived the copy: {links[:3]}"
    from elidedb import Store
    db = Store.open(dst)
    tabs = db.tables()
    assert "episodes" in tabs and "frames" in tabs, tabs
    assert (dst / "_set_weights.json").exists(), "fitted artifact missing"
    total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"demo store: {dst}")
    print(f"  tables {len(tabs)}  bytes {total/1e6:.0f} MB "
          f"(copied {n/1e6:.0f} MB)")
    print("  standalone: no symlinks, media materialized")


if __name__ == "__main__":
    main()
