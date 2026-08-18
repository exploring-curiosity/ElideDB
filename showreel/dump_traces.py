#!/usr/bin/env python3
"""Dump one DTW trace per recording, so nothing has to hold the corpus.

    myenv/bin/python showreel/dump_traces.py

WHY THIS EXISTS, measured. `Store("rcasa")` builds a padded DTW bank over every
recording at once: (3556, 256, 1792) float32, 6.53 GB, and 17.64 GB peak RSS
while loading. That is more memory than a free CPU host has, and all of it to
answer queries that never look at more than 48 recordings.

Stage 1 already narrows the field to a shortlist inside the database. Stage 2
only ever needs those, so the trace each one needs is written here as its own
small file and read on demand: 11 ms to load 48, 30 MB resident. The padding
happens over the shortlist instead of the corpus.

Stored as float16. The DTW cost is a cosine between L2-normalised rows, so the
precision that matters is in the direction rather than the magnitude, and the
fidelity check in `sidecar.py --selfcheck` compares rankings against the
full-precision path rather than trusting that argument.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
OUT = Path(os.environ.get("PRECEDENT_TRACES", ROOT / "showreel" / "traces"))


def main() -> int:
    from tqdm import tqdm

    from relmo.vjeval import l2
    from relmo.vjstore import Store

    OUT.mkdir(parents=True, exist_ok=True)
    print("loading the store once (~60s, ~18 GB) so it never has to be loaded "
          "again ...", flush=True)
    st = Store("rcasa")

    n, total = 0, 0
    for rid in tqdm(st.ids, desc="dump", unit="rec"):
        z = l2(st.Z[rid]).astype(np.float16)
        p = OUT / f"{rid}.npy"
        np.save(p, z)
        total += p.stat().st_size
        n += 1
    print(f"\n{n} traces, {total/1e9:.2f} GB in {OUT}")
    print("the sidecar now needs none of the store: see sidecar.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
