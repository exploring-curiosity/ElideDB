#!/usr/bin/env python3
"""Synthetic sensor stream generator for SDX scale tests (M2).

Writes a CSV with a ts_ns column plus N value columns at a given rate and
duration. Timestamps get realistic jitter but stay strictly non-decreasing
(the SDX writer enforces monotonicity — that invariant is what makes zone
maps sharp).

Example (10M rows: 2 kHz x 5000 s x 1 column):
  python3 scripts/synth_sensors.py --rate 2000 --dur 5000 --cols 1 \
      --start-ns 0 --out /tmp/synth.csv
  sdx ingest-csv /tmp/synth.csv --store store --stream-id synth/imu
"""
import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=1000.0, help="Hz")
    ap.add_argument("--dur", type=float, default=100.0, help="seconds")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--start-ns", type=int, default=0)
    ap.add_argument("--jitter-ns", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.rate * args.dur)
    ts = args.start_ns + (np.arange(n, dtype=np.int64) * int(1e9 / args.rate))
    ts += rng.integers(0, args.jitter_ns, n)  # additive jitter keeps order
    ts = np.maximum.accumulate(ts)            # belt-and-braces monotonicity

    t = ts.astype(np.float64) / 1e9
    cols = [np.sin(2 * np.pi * (0.1 + 0.05 * c) * t) +
            0.1 * rng.standard_normal(n) for c in range(args.cols)]

    header = "ts_ns," + ",".join(f"v{c}" for c in range(args.cols))
    data = np.column_stack([ts] + cols)
    fmt = ["%d"] + ["%.6f"] * args.cols
    np.savetxt(args.out, data, fmt=fmt, delimiter=",", header=header,
               comments="")
    print(f"wrote {n} rows x {args.cols + 1} cols to {args.out}")


if __name__ == "__main__":
    main()
