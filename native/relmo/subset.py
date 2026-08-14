"""Choose which episodes to track, under a wall-clock budget.

Tracking costs a MEASURED 0.217 s per frame on this machine (MPS, with
grid_sampler_3d on the CPU fallback; pure CPU measured 0.632 s/frame, so
the fallback is already the fast path). The corpus is 143k frames across
both datasets, which is 8.65 h - more than the whole prototype budget,
before a single training step.

The alternative levers are worse. Tracking every second frame halves the
cost but doubles the physical horizon behind h=1..8, which would
invalidate the const-velocity and ceiling references already measured in
gates.json. Truncating episodes cuts the manipulation out of the middle
of a demo. So the lever is WHICH episodes, not which frames.

Selection is cost-aware and quality-aware: an episode's value is its
count of usable windows - spans where the target is visible AND moving,
computed at replay time - and its cost is its frame count. Ranking by
windows-per-frame buys the most trainable signal per second of tracking.
A wrist-cam LoadDishwasher episode with 1125 frames and the target
visible 11% of the time costs 243 s and is worth almost nothing; a
226-frame agentview episode costs 49 s and is dense.

Every task keeps a floor of episodes so no task drops out of the corpus,
and the floor is filled before the ranking spends the rest of the
budget. Selection uses sim-derived visibility, which is training-time
privileged information - the same rule the rest of the pipeline follows:
sim state may pick and score data, never serve a query.

    python -m relmo.subset --name rcasa --hours 2.0
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

SEC_PER_FRAME = 0.217          # measured, this machine, MPS + fallback


def value(e):
    """Usable windows per frame of tracking cost."""
    return e.get("win", 0) / max(e["T"], 1)


def plan(name, hours, floor=6, done=None):
    man = R.read_manifest(name)
    done = done or set()
    eps = [e for e in man["episodes"] if e["id"] not in done]
    budget = hours * 3600 / SEC_PER_FRAME
    by = collections.defaultdict(list)
    for e in eps:
        by[e["task"]].append(e)
    for t in by:
        by[t].sort(key=value, reverse=True)

    picked, spent = [], 0.0
    # floor first: every task survives even if it is expensive
    for t in sorted(by):
        for e in by[t][:floor]:
            picked.append(e)
            spent += e["T"]
    chosen = {e["id"] for e in picked}
    # then the best value-per-frame across everything that is left
    rest = sorted((e for e in eps if e["id"] not in chosen),
                  key=value, reverse=True)
    for e in rest:
        if spent + e["T"] > budget:
            continue
        picked.append(e)
        chosen.add(e["id"])
        spent += e["T"]
    return picked, spent, budget


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rcasa")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--floor", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tdir = R.TRACKS / a.name
    done = {p.stem for p in tdir.glob("*.npz")} if tdir.exists() else set()
    picked, spent, budget = plan(a.name, a.hours, a.floor, done)
    c = collections.Counter(e["task"] for e in picked)
    fr = collections.Counter()
    for e in picked:
        fr[e["task"]] += e["T"]
    print(f"{a.name}: {len(picked)} episodes, {int(spent)} frames, "
          f"{spent * SEC_PER_FRAME / 3600:.2f} h "
          f"(budget {budget:.0f} frames / {a.hours} h), "
          f"{len(done)} already tracked")
    for t in sorted(c):
        print(f"  {t:30s} {c[t]:4d} eps {fr[t]:7d} frames")
    out = Path(a.out) if a.out else (R.dataset_dir(a.name) / "track_subset.txt")
    out.write_text("\n".join(e["id"] for e in picked) + "\n")
    print(f"\nwrote {out}")
