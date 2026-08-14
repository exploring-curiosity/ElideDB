"""A/B report: FDNN oscillatory head vs plain MLP head.

Compared at MATCHED STEP COUNTS on the same data and the same
hash-stable splits, judged by VALIDATION rollout error (the arms
never see val gradients). Wall-clock differs because the two runs
contend for one GPU and started at different times - that affects
speed, not the quality comparison at equal steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402


def series(run):
    p = R.MODELS / run / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def report(a="relmowm_v1", b="relmowm_fdnn"):
    sa, sb = series(a), series(b)
    if not sa or not sb:
        print("not enough data yet")
        return
    stop = min(sa[-1]["step"], sb[-1]["step"])
    print(f"matched at step <= {stop}   (val rollout: lower is better)")
    print(f"{'step':>7s} {a:>18s} {b:>18s}   winner")
    da = {r["step"]: r for r in sa}
    db = {r["step"]: r for r in sb}
    common = sorted(set(da) & set(db))
    wins = {a: 0, b: 0}
    for s in common:
        va, vb = da[s]["val_rollout"], db[s]["val_rollout"]
        w = a if va < vb else b
        wins[w] += 1
        print(f"{s:>7d} {va:>18.5f} {vb:>18.5f}   {w}")
    if not common:
        print("  (no matched steps logged yet)")
        return
    la = [da[s]["val_rollout"] for s in common[-4:]]
    lb = [db[s]["val_rollout"] for s in common[-4:]]
    print(f"\nlast-4 mean val rollout   {a}: {np.mean(la):.5f}"
          f"   {b}: {np.mean(lb):.5f}")
    print(f"best val rollout          {a}: "
          f"{min(r['val_rollout'] for r in sa if r['step'] <= stop):.5f}"
          f"   {b}: "
          f"{min(r['val_rollout'] for r in sb if r['step'] <= stop):.5f}")
    print(f"arrow-of-time (val, last) {a}: {da[common[-1]]['val_aot_acc']:.2f}"
          f"   {b}: {db[common[-1]]['val_aot_acc']:.2f}")
    print(f"step-wise wins            {wins}")
    print("\nCAVEAT: one seed per arm. A gap under ~10% of the value is "
          "not separable from seed noise at this sample size.")


if __name__ == "__main__":
    report(*(sys.argv[1:3] or ["relmowm_v1", "relmowm_fdnn"]))
