#!/usr/bin/env python3
"""Which reduction of RelMo's trace tells the behaviours apart?

    myenv/bin/python brigade/bench/trace_feature.py

The store's indexed column is a MEAN over the descriptor trace, because a
stage-1 prefilter is one vector per row by definition. This asks what else the
trace carries, since the reasoning head is free to read the trace file directly
and is not bound to the shape the index needs.

Every arm is a nearest-neighbour behaviour match with the OVERLAP GUARD on: the
store indexes a sliding 15 s span on a 5 s hop, so consecutive rows share two
thirds of their video, and an unguarded 1-NN mostly measures whether a clip can
find itself shifted by five seconds. Both columns are printed so the size of
that illusion stays visible — it is the difference between 1.000 and 0.305.

Nothing here is fitted. Each arm is a reduction of an array plus an L2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "brigade"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brigade.memory.traces import blocks_of                      # noqa: E402
from relmo_probe import loo_1nn, overlap_mask, separation        # noqa: E402

LABELS = ROOT / "eval_logs" / "reasoner" / "segments.json"
TRACES = ROOT / "brigade" / "artifacts" / "traces"
OUT = ROOT / "eval_logs" / "reasoner" / "trace_feature.json"

# Single blocks first, then the combinations worth knowing about. The point of
# including the fusions is that they LOSE — see the module note in traces.py.
ARMS = [
    ("mean, V-JEPA + SigLIP2", ("mean_fix", "mean_sig")),
    ("std,  V-JEPA + SigLIP2", ("std_fix", "std_sig")),
    ("dir,  V-JEPA + SigLIP2", ("dir_fix", "dir_sig")),
    ("std,  V-JEPA only", ("std_fix",)),
    ("std,  SigLIP2 only", ("std_sig",)),
    ("mean, SigLIP2 only", ("mean_sig",)),
    ("mean + std", ("mean_fix", "std_fix", "mean_sig", "std_sig")),
    ("std + dir", ("std_fix", "dir_fix", "std_sig", "dir_sig")),
    ("all six blocks", ("mean_fix", "std_fix", "dir_fix",
                        "mean_sig", "std_sig", "dir_sig")),
]


def main() -> int:
    rows = json.load(open(LABELS))
    have = {}
    for r in rows:
        p = TRACES / f"{r['clip_id']}.npz"
        if p.exists():
            z = np.load(p)
            have[r["clip_id"]] = blocks_of(z["fix"], z["sig"])
    rows = [r for r in rows if r["clip_id"] in have]
    y = np.array([r["goal"] for r in rows])
    bar = overlap_mask(rows)
    k = len(set(y.tolist()))
    print(f"{len(rows)} spans, {k} behaviours, chance {1/k:.3f}, "
          f"{int((~bar).sum(1).min())} candidates survive the overlap guard\n")
    print(f"  {'reduction':<28}{'1-NN':>7}{'no-overlap':>13}{'separation':>13}")

    res = {}
    for name, blocks in ARMS:
        X = np.stack([np.concatenate([have[r["clip_id"]][b] for b in blocks])
                      / np.sqrt(len(blocks)) for r in rows])
        M = X @ X.T
        same, diff = separation(M, y)
        res[name] = dict(blocks=list(blocks), naive=loo_1nn(M, y),
                         clean=loo_1nn(M, y, bar), separation=same - diff)
        print(f"  {name:<28}{res[name]['naive']:>7.3f}{res[name]['clean']:>13.3f}"
              f"{same - diff:>+13.4f}")

    best = max(res, key=lambda n: res[n]["clean"])
    print(f"\nbest: {best.strip()} at {res[best]['clean']:.3f} "
          f"({res[best]['clean'] / (1/k):.1f}x chance)")
    print("Fusing blocks LOSES to the best single block. Cosine over concatenated\n"
          "unit blocks is the MEAN of the per-block cosines, so uninformative\n"
          "blocks drown an informative one — selection beats fusion here.")
    json.dump(dict(n=len(rows), k=k, chance=1 / k, arms=res), open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
