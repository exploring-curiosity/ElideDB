"""The grid-alignment control: does the model retrieve the EVENT or the GRID?

In this corpus every episode file begins at the start of its demonstration, so
the window grid - which starts at t=0 of the file - is locked to event onset.
Every recording of an "open" therefore places the opening at the same phase of
the same grid. That coincidence does not survive contact with the production
input the owner described: "a 1 hr video can contain a lot of smaller episodes",
where an event beginning at 13.7 s lands on whatever phase it lands on.

So the number that matters is not query-and-pool-on-the-same-grid. It is:

    QUERY   re-ingested with the window grid shifted by --offset seconds
    POOL    the ordinary records, grid at 0

Same video, same event, same everything except which 4-second windows the
encoder happened to cut. A model that has learned the event is unmoved. A model
that has learned the grid falls over, and the headline number was never real.

    python -m relmo.vjgrid --tags rk6_ns_s0,rk6_nsph_s0 --suffix _off1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrank, vjrel, vjz  # noqa: E402
from relmo.vjood import recs_for  # noqa: E402
from relmo.vjrankeval import grade, score_matrix  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402
from relmo.vjzeval import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--suffix", default="_off1")
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    sp = load_split()
    meta = vjrel.meta_table(a.dataset)
    cache = {}

    def load(phase_name):
        if phase_name not in cache:
            ph = None
            if phase_name:
                from relmo.vjphase import load as load_phase
                ph = load_phase(phase_name)
            pool = recs_for(a.dataset, phase=ph)
            rec, sig = vjz.dirs("vjrec6", a.dataset, suffix=a.suffix)
            ids = sorted(p.stem for p in rec.glob("*.npz")
                         if not p.name.startswith("."))
            shift = vjz.gather(ids, a.dataset, rec_dir=rec, sig_dir=sig,
                               phase=ph)
            shift = {i: v for i, v in shift.items()
                     if v["sig"] is not None and len(v["sig"]) == len(v["a"])}
            cache[phase_name] = (pool, shift)
        return cache[phase_name]

    print(f"grid-offset control: queries from {a.suffix}, pool from the "
          f"ordinary grid\n")
    print(f"{'arm':16s} {'phase':>6s} {'aligned':>9s} {'shifted':>9s} "
          f"{'delta':>8s}   matcher")
    print("-" * 66)

    q_seen = pool_seen = 0
    chance = float("nan")
    for phase_name, label in (("", "off"), ("rcasa_train", "ON")):
        pool, shift = load(phase_name)
        q = sorted(set(shift) & set(sp["val"]))
        pool_ids = set((sp["val"] | sp["test"]) & set(pool))
        if not q:
            print(f"  no shifted records for phase={label} - skipped")
            continue
        P = {i: v["a"] for i, v in pool.items()}
        Q = {i: v["a"] for i, v in shift.items()}
        al = evaluate(P, set(q), pool_ids, boot=a.boot)
        sh = evaluate(P, set(q), pool_ids, boot=a.boot, qdesc=Q)
        q_seen, pool_seen, chance = len(q), len(pool_ids), al["chance"]
        print(f"{'frozen a_t':16s} {label:>6s} {al['overall']:9.3f} "
              f"{sh['overall']:9.3f} {sh['overall']-al['overall']:+8.3f}   "
              f"subsequence DTW")

    for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
        model, ck = vjrank.load_ckpt(t)
        phase_name = ck.get("phase", "")
        pool, shift = load(phase_name)
        q = sorted(set(shift) & set(sp["val"]))
        pool_ids = sorted((sp["val"] | sp["test"]) & set(pool))
        if not q:
            continue
        al = grade(score_matrix(model, pool, q, pool_ids), q, pool_ids, meta)
        sh = grade(score_matrix(model, pool, q, pool_ids, qrecs=shift),
                   q, pool_ids, meta)
        print(f"{t:16s} {('ON' if phase_name else 'off'):>6s} "
              f"{al['prec']:9.3f} {sh['prec']:9.3f} "
              f"{sh['prec']-al['prec']:+8.3f}   learned z, pooled cosine")

    print(f"\nqueries: {q_seen} val recordings on the shifted grid, "
          f"pool {pool_seen} on the ordinary grid, chance {chance:.3f}")
    R.log("vjgrid", tags=a.tags, suffix=a.suffix)


if __name__ == "__main__":
    main()
