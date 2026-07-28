"""Rust vs Python engine: time-range scan latency on real store tables.

Same 2-second windows, same tables, warm cache (cold needs `sudo purge`).
Reports p50/p99 per engine plus the Rust engine's true counted bytes.

Run:  python scripts/bench_rs_scan.py
"""
from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.store import Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ELIDE = ROOT / "rust/target/release/elide"

CASES = [
    ("lab", "sensor_108_audio", 2_000_000_000),   # 14M rows, 479M
    ("lab", "frames", 2_000_000_000),
    ("bench", "frames", 2_000_000_000),
    ("bridge-full", "frames", 2_000_000_000),
]
N = 40


def windows(state, span_ns, n, seed=7):
    rng = random.Random(seed)
    lo, hi = state.min_ts, state.max_ts - span_ns
    return [(a, a + span_ns) for a in (rng.randrange(lo, hi) for _ in range(n))]


def bench_python(store, table, wins):
    t = store.table(table)
    lat = []
    rows = 0
    for t0, t1 in wins:
        s = time.perf_counter()
        out = t.scan(t0, t1)
        lat.append(time.perf_counter() - s)
        rows += len(out)
    return lat, rows


def bench_rust(store_path, table, wins):
    lat = []
    rows = 0
    bytes_read = bytes_total = 0
    for t0, t1 in wins:
        s = time.perf_counter()
        out = subprocess.run(
            [ELIDE, "scan", str(store_path), table,
             "--t0", str(t0), "--t1", str(t1), "--json"],
            capture_output=True, text=True, check=True)
        lat.append(time.perf_counter() - s)
        j = json.loads(out.stdout)
        rows += j["rows"]
        bytes_read += j["bytes_read"]
        bytes_total += j["bytes_total"]
    return lat, rows, bytes_read, bytes_total


def pct(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1]


def main():
    for store_name, table, span in CASES:
        sp = ROOT / "lake" / store_name
        store = Store.open(sp)
        st = store.table(table).state()
        if st.rows == 0:
            continue
        wins = windows(st, span, N)
        pl, prows = bench_python(store, table, wins)
        rl, rrows, br, bt = bench_rust(sp, table, wins)
        assert prows == rrows, f"row mismatch {store_name}/{table}: {prows} vs {rrows}"
        print(f"{store_name}/{table}: rows/query ~{rrows//N}")
        print(f"  python  p50 {pct(pl,50)*1e3:7.1f} ms   p99 {pct(pl,99)*1e3:7.1f} ms")
        print(f"  rust    p50 {pct(rl,50)*1e3:7.1f} ms   p99 {pct(rl,99)*1e3:7.1f} ms"
              f"   (includes process spawn)")
        print(f"  rust counted bytes/query: {br//N:,} of {bt//N:,}"
              f"  (elided {100*(1-br/max(bt,1)):.2f}%)")


if __name__ == "__main__":
    main()
