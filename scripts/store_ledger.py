"""The byte ledger: store size vs raw size, per store — the headline
number next to elision % (docs/ENGINE.md section 4.4).

Writes <store>/_ledger.json:
  raw_source_bytes   from ingest "original" metas in table history
  store_bytes        physical lstat of the store dir, split into
                     tables (current version) / stale (old versions)
                     / media / artifacts (_cache, _codes, weights)
  symlinked_bytes    media that lives OUTSIDE the store (a standalone
                     store must have zero)
  ratio              store_current / raw  (< 1.0 or the store failed
                     the founding goal)

Usage: python scripts/store_ledger.py <store> [...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.store import Store  # noqa: E402


def du(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.lstat().st_size
    if path.is_dir():
        return sum(du(p) for p in path.iterdir())
    return 0


def ledger(store_path: Path) -> dict:
    store = Store.open(store_path)
    raw = 0
    for name in store.tables():
        for h in store.table(name).history():
            m = h.get("meta", {})
            o = m.get("original", {})
            raw += int(o.get("bytes", 0)) if isinstance(o, dict) else 0
    current = stale = artifacts = 0
    for name in store.tables():
        t = store.table(name)
        live = {f.path for f in t.state().files}
        for p in t.dir.iterdir():
            if p.name.startswith("_log"):
                continue
            if p.name.startswith(("_cache", "_codes")) or p.suffix == ".npy":
                artifacts += du(p)
            elif p.suffix == ".parquet":
                if p.name in live:
                    current += du(p)
                else:
                    stale += du(p)
    media = symlinked = 0
    mdir = store_path / "media"
    if mdir.is_dir():
        for p in mdir.iterdir():
            if p.is_symlink():
                try:
                    symlinked += p.resolve().stat().st_size
                except OSError:
                    pass
            else:
                media += du(p)
    out = {
        "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_source_bytes": raw,
        "tables_current_bytes": current,
        "tables_stale_bytes": stale,
        "artifact_bytes": artifacts,
        "media_bytes": media,
        "media_symlinked_external_bytes": symlinked,
        "store_current_bytes": current + media + artifacts,
        "standalone": symlinked == 0,
        "ratio_store_to_raw": round((current + media + artifacts) / raw, 4)
        if raw else None,
    }
    (store_path / "_ledger.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for sp in sys.argv[1:]:
        r = ledger(Path(sp))
        g = lambda k: f"{r[k] / 1e9:.2f}G"  # noqa: E731
        print(f"{sp}: raw {g('raw_source_bytes')}  store "
              f"{g('store_current_bytes')} (tables {g('tables_current_bytes')}"
              f" + media {g('media_bytes')} + artifacts {g('artifact_bytes')};"
              f" stale {g('tables_stale_bytes')})  standalone="
              f"{r['standalone']}  ratio={r['ratio_store_to_raw']}")


if __name__ == "__main__":
    main()
