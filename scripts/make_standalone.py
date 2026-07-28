"""Make a store STANDALONE and reclaim what it no longer needs.

A store that symlinks its media to another store is not a database, it
is a view: delete the other store and this one stops serving frames.
Three reclamations, all safe to re-run:

  media     symlinks replaced by real copies inside the store
  stale     Parquet files no longer referenced by the current log
            version (kept for time travel; dropped on request)
  caches    derived sidecars keyed to a DEAD table version
            (_cache/<col>-v<N>.npy, _codes.v<N>.bin) — the current
            version's artifacts are always kept

Usage: python scripts/make_standalone.py <store> [--keep-stale] [--dry-run]
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.store import Store  # noqa: E402


def materialize_media(store_dir: Path, dry: bool) -> tuple[int, int]:
    mdir = store_dir / "media"
    n = total = 0
    if not mdir.is_dir():
        return 0, 0
    for p in sorted(mdir.iterdir()):
        if not p.is_symlink():
            continue
        target = p.resolve()
        if not target.exists():
            print(f"  BROKEN symlink (target gone): {p.name}")
            continue
        size = target.stat().st_size
        n += 1
        total += size
        print(f"  copy {p.name}  {size/1e6:.1f} MB")
        if not dry:
            tmp = p.with_suffix(p.suffix + ".tmp")
            shutil.copy2(target, tmp)
            p.unlink()
            tmp.rename(p)
    return n, total


def sweep_dead_artifacts(store: Store, dry: bool) -> tuple[int, int]:
    """Drop derived sidecars bound to versions that are no longer
    current. They rebuild from the Parquet, which is the source of
    truth; a sidecar for a dead version can never be used again."""
    n = total = 0
    for name in store.tables():
        t = store.table(name)
        cur = t.state().version
        cands = []
        cache = t.dir / "_cache"
        if cache.is_dir():
            cands += [p for p in cache.iterdir() if p.is_file()]
        cands += [p for p in t.dir.iterdir()
                  if p.name.startswith("_codes.v")]
        for p in cands:
            m = re.search(r"v(\d+)", p.name)
            if not m or int(m.group(1)) == cur:
                continue
            size = p.stat().st_size
            n += 1
            total += size
            print(f"  drop {name}/{p.relative_to(t.dir)}  "
                  f"(v{m.group(1)}, current v{cur})  {size/1e6:.1f} MB")
            if not dry:
                p.unlink()
    return n, total


def sweep_stale_parquet(store: Store, dry: bool) -> tuple[int, int]:
    """Remove Parquet files no longer in the current version. This
    DROPS time travel to earlier versions — the log still records
    them, so a reader asking for an old version gets a clean file-not-
    found rather than wrong data."""
    n = total = 0
    for name in store.tables():
        t = store.table(name)
        live = {f.path for f in t.state().files}
        for p in t.dir.iterdir():
            if p.suffix != ".parquet" or p.name in live:
                continue
            size = p.stat().st_size
            n += 1
            total += size
            print(f"  drop {name}/{p.name}  (superseded)  "
                  f"{size/1e6:.1f} MB")
            if not dry:
                p.unlink()
    return n, total


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        sys.exit(__doc__)
    dry = "--dry-run" in flags
    store_dir = Path(args[0])
    store = Store.open(store_dir)

    print("media:")
    mn, mb = materialize_media(store_dir, dry)
    print("dead artifacts:")
    an, ab = sweep_dead_artifacts(store, dry)
    sn = sb = 0
    if "--keep-stale" not in flags:
        print("superseded parquet:")
        sn, sb = sweep_stale_parquet(store, dry)

    print(f"\n{'WOULD ' if dry else ''}materialize {mn} media files "
          f"({mb/1e9:.3f} GB now inside the store), "
          f"reclaim {an} dead artifacts ({ab/1e9:.3f} GB) + "
          f"{sn} superseded parquet ({sb/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
