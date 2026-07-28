"""R0/R1 parity: the Rust engine must agree with the Python engine, exactly.

R0: `elide stats --json` vs Python TableLog state on every store — version,
kind, files, rows, bytes, min/max ts, per table.
R1: randomized time-window scans — Rust row counts vs Python Table.scan on
real tables (property test: pruned scan ≡ full-filter scan).

Run:  python scripts/parity_rs.py [store ...]   (default: every lake store)
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.store import Store  # noqa: E402

ELIDE = Path(__file__).resolve().parents[1] / "rust/target/release/elide"
LAKE = Path(__file__).resolve().parents[1] / "lake"


def rust_stats(store: Path):
    out = subprocess.run([ELIDE, "stats", str(store), "--json"],
                         capture_output=True, text=True, check=True)
    return {t["table"]: t for t in json.loads(out.stdout)}


def rust_scan(store: Path, table: str, t0=None, t1=None):
    cmd = [str(ELIDE), "scan", str(store), table, "--json"]
    if t0 is not None:
        cmd += ["--t0", str(t0)]
    if t1 is not None:
        cmd += ["--t1", str(t1)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def check_stats(store_path: Path) -> int:
    st = Store.open(store_path)
    rs = rust_stats(store_path)
    bad = 0
    for name in st.tables():
        py = st.table(name).state()
        want = dict(version=py.version, kind=py.kind, files=len(py.files),
                    rows=py.rows, bytes=py.bytes,
                    min_ts=py.min_ts, max_ts=py.max_ts)
        got = rs.get(name)
        if got is None:
            print(f"  MISSING in rust: {name}")
            bad += 1
            continue
        for k, v in want.items():
            if got[k] != v:
                print(f"  MISMATCH {name}.{k}: py={v} rs={got[k]}")
                bad += 1
    extra = set(rs) - set(st.tables())
    for name in extra:
        print(f"  EXTRA in rust: {name}")
        bad += 1
    return bad


def check_scans(store_path: Path, n_windows=8, seed=0) -> int:
    st = Store.open(store_path)
    rng = random.Random(seed)
    bad = 0
    for name in st.tables():
        py_state = st.table(name).state()
        if py_state.rows == 0 or py_state.max_ts <= py_state.min_ts:
            continue
        span = py_state.max_ts - py_state.min_ts
        windows = [(None, None)]
        for _ in range(n_windows):
            a = py_state.min_ts + rng.randrange(span)
            b = min(a + rng.randrange(1, max(span // 20, 2)), py_state.max_ts)
            windows.append((a, b))
        for t0, t1 in windows:
            py_rows = len(st.table(name).scan(t0, t1, columns=["ts"]))
            rs = rust_scan(store_path, name, t0, t1)
            if rs["rows"] != py_rows:
                print(f"  SCAN MISMATCH {name} [{t0},{t1}]: "
                      f"py={py_rows} rs={rs['rows']}")
                bad += 1
    return bad


def main():
    stores = [LAKE / s for s in sys.argv[1:]] or sorted(
        p for p in LAKE.iterdir() if (p / "_store.json").exists())
    total = 0
    for sp in stores:
        print(f"== {sp.name}")
        b = check_stats(sp)
        b += check_scans(sp)
        print(f"   {'OK' if b == 0 else f'{b} FAILURES'}")
        total += b
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
