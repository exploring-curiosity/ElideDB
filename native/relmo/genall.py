"""Generate the full corpus: 12 training tasks + 2 held-out eval tasks.

Runs every task over both splits. Ordered so that the cheapest, most
diverse data lands first: pretrain before target (pretrain carries ~105
demos across 40-47 kitchens, target ~500 across 10), and atomic before
composite. If the run is interrupted, what exists on disk is already a
usable, kitchen-diverse corpus rather than a deep sample of one task.

Resumable: an episode whose state.npz exists is skipped, so re-running
picks up where it stopped.

    python -m relmo.daemon genall
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.rcreplay import build, episodes  # noqa: E402

# Prototype budget: the presentation is in a day, so this generates a
# SMALL kitchen-diverse corpus rather than all 4913 demos (~36 h, ~165 GB).
# pretrain split only - ~105 demos each across 40-47 kitchens, versus
# target's ~500 across 10 - so a small sample still sees many kitchens.
ATOMIC = ["OpenDrawer", "CloseDrawer", "OpenCabinet", "CloseCabinet",
          "OpenMicrowave", "CloseMicrowave", "PickPlaceCounterToCabinet",
          "PickPlaceCounterToSink", "PickPlaceCounterToDrawer"]
COMPOSITE = ["PrepareCoffee", "StackBowlsCabinet", "LoadDishwasher"]
OOD = ["OpenFridge", "ArrangeTea"]


def plan(n_atomic=20, n_comp=8, n_ood=8, split="pretrain"):
    out = []
    for t in ATOMIC:
        n = len(episodes(t, split))
        if n:
            out.append(("rcasa", t, split, min(n, n_atomic)))
    for t in COMPOSITE:
        n = len(episodes(t, split))
        if n:
            out.append(("rcasa", t, split, min(n, n_comp)))
    for t in OOD:
        for sp in ("pretrain", "target"):
            n = len(episodes(t, sp))
            if n:
                out.append(("rcasa_eval", t, sp, min(n, n_ood)))
                break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic", type=int, default=20)
    ap.add_argument("--composite", type=int, default=8)
    ap.add_argument("--ood", type=int, default=8)
    ap.add_argument("--cams", type=int, default=2)
    ap.add_argument("--dryrun", action="store_true")
    a = ap.parse_args()
    import relmo.rcreplay as RP
    RP.N_CAMS = a.cams
    P = plan(a.atomic, a.composite, a.ood)
    tot = sum(n for _, _, _, n in P)
    print(f"{len(P)} (task, split) jobs, {tot} demonstrations\n")
    for ds, t, sp, n in P:
        print(f"  {ds:11s} {t:28s} {sp:9s} {n:5d}")
    if a.dryrun:
        raise SystemExit
    R.log("genall_start", jobs=len(P), demos=tot)
    t0 = time.time()
    done = 0
    for ds, t, sp, n in P:
        k = n
        made, rej, rows = build([t], ds, k, sp)
        done += k
        el = time.time() - t0
        R.log("genall_task", dataset=ds, task=t, split=sp, demos=k,
              episodes=len(rows), rejected=len(rej),
              elapsed_min=round(el / 60, 1),
              eta_min=round(el / max(done, 1) * (tot - done) / 60, 1))
        print(f"[{done}/{tot}] {t}/{sp}: {len(rows)} episodes, "
              f"{len(rej)} rejected, {el / 60:.0f} min elapsed, "
              f"ETA {el / max(done, 1) * (tot - done) / 3600:.1f} h",
              flush=True)
    R.log("genall_done", minutes=round((time.time() - t0) / 60, 1))
