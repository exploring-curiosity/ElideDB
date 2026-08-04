"""LONG-CONTEXT retrieval on the sim corpus: retrieve episodes whose
whole CHAIN of events matches the query episodes' chain.

This is the hard benchmark by construction - appearance, placement and
colour are randomised per episode, so two episodes of the same template
share only their temporal structure. Same domain-blind machinery as
native/mem.py (frozen encoders -> sequences -> pooled + alignment ->
all-channel fusion), same frozen protocol as every previous chain
number (5 seeds, k = ceil(1.5 x support), RandomState(0)), so results
sit directly beside the symbolic routes and the channel-selection era.

    python native/simbench.py [--l 12] [--band 0.35]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

import mem                                                     # noqa: E402
from mem import load_channels, build_reps, arg                 # noqa: E402
from memfuse import per_seed_scores, combine                   # noqa: E402

DEV = ("swap", "precarious", "push_then_build",
       "build_unstack_move")
HOLD = ("relocate_build", "two_sites_merge")

STRATS = [("mean", "all_rrf"), ("mean", "all_z"), ("max", "all_z"),
          ("max", "all_rrf")]


def run(reps, eps, tmpl, targets, label, strat):
    rs = np.random.RandomState(0)
    ys, ps = [], []
    for tg in targets:
        pool = sorted(e for e, tm in tmpl.items()
                      if tm == tg and e in set(eps))
        if len(pool) < 6:
            continue
        sd = sorted(int(x) for x in rs.choice(pool, 5, replace=False))
        support = len(pool) - len(sd)
        k = math.ceil(1.5 * support)
        S = per_seed_scores(reps, eps, sd)
        tot = combine(S, sd, eps, strat[0], strat[1])
        pos = {e: i for i, e in enumerate(eps)}
        for x in sd:
            if x in pos:
                tot[pos[x]] = -1e9
        got = [eps[i] for i in np.argsort(-tot)[:k]]
        tr = sum(1 for e in got if tmpl.get(e) == tg)
        ys.append(tr / support)
        ps.append(tr / len(got))
    return float(np.mean(ys)), float(np.mean(ps)), ys


def main():
    from elidedb import Store
    import pyarrow.parquet as pq
    mem.L = arg("--l", 12, int)
    mem.BAND = arg("--band", 0.35, float)
    db = Store.open(str(ROOT / "lake/sim_chains"))
    reps = build_reps(load_channels(db, "sim_chains"))
    eps = sorted({e for R in reps.values() for e in R["pool"]})
    print(f"sim_chains: {len(eps)} episodes, channels {sorted(reps)}, "
          f"L={mem.L} band={mem.BAND}", flush=True)
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    best = None
    for st in STRATS:
        dy, dp, dl = run(reps, eps, tmpl, DEV, "DEV", st)
        hy, hp, hl = run(reps, eps, tmpl, HOLD, "HOLDOUT", st)
        print(f"  {st[0]:<5}{st[1]:<9} DEV {dy:.3f}/{dp:.3f}   "
              f"HOLDOUT {hy:.3f}/{hp:.3f}   "
              f"per-template DEV {[round(x,2) for x in dl]}",
              flush=True)
        if best is None or dy > best[1]:
            best = (st, dy, hy)
    print(f"\nfrozen on DEV: {best[0]} -> DEV {best[1]:.3f}  "
          f"HOLDOUT {best[2]:.3f}")


if __name__ == "__main__":
    main()
