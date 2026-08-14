"""The one read of the test split. Aggregated over seeds, never one checkpoint.

WHY SEEDS AND NOT A BEST CHECKPOINT. Five runs of the same config spread 0.516
to 0.634 on ood_val at wrec=0.1. Picking the top one and calling it the result
would have reported +0.08 of seed noise as method. So the config is chosen on
MEAN ood_val across seeds (and its variance), and every number below is a mean
with a spread, per split and per query class.

The selected config is fixed BEFORE this runs; test is read once and never
feeds a decision.

    python -m relmo.vjreport --tags sd_r1.0_s0,...,sd_r1.0_s4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjz  # noqa: E402
from relmo.vjood import recs_for  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402
from relmo.vjzeval import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sp = load_split()
    Rin, Rood = recs_for("rcasa"), recs_for("rcasa_eval")
    base = {i: Rin[i]["a"] for i in Rin}
    base.update({i: Rood[i]["a"] for i in Rood})
    inpool = (sp["val"] | sp["test"]) & set(Rin)
    tst = sp["test"] & set(Rin)

    SPLITS = {
        "test_primary": (tst, inpool),
        "test_deploy": (tst, set(Rin)),
        "ood_val": (set(Rood), inpool | set(Rood)),
    }

    def all_splits(desc):
        return {k: evaluate(desc, q, p, boot=2000)
                for k, (q, p) in SPLITS.items()}

    runs = {"frozen baseline": [all_splits(base)]}
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    acc = []
    for t in tags:
        model, ck = vjz.load_ckpt(t)
        d = vjz.encode(model, ck, Rin)
        d.update(vjz.encode(model, ck, Rood))
        acc.append(all_splits(d))
        print(f"  scored {t}", flush=True)
    runs["learned z (mean of %d seeds)" % len(acc)] = acc

    def agg(rs, split, key=None):
        v = [r[split]["overall"] if key is None
             else r[split]["per_group"].get(key, {}).get("prec", np.nan)
             for r in rs]
        return float(np.nanmean(v)), float(np.nanstd(v))

    print("\n" + "=" * 74)
    print("OVERALL - mean +/- sd over seeds")
    print(f"{'arm':30s} " + " ".join(f"{s:>16s}" for s in SPLITS))
    for name, rs in runs.items():
        cells = []
        for s in SPLITS:
            m, sd = agg(rs, s)
            cells.append(f"{m:.3f} +/-{sd:.3f}" if len(rs) > 1 else f"{m:.3f}")
        print(f"{name:30s} " + " ".join(f"{c:>16s}" for c in cells))
    print(f"{'chance':30s} " + " ".join(
        f"{runs['frozen baseline'][0][s]['chance']:>16.3f}" for s in SPLITS))

    for split in SPLITS:
        keys = sorted({k for rs in runs.values() for r in rs
                       for k in r[split]["per_group"]})
        print(f"\n--- {split} : per query class ---")
        print(f"{'class':20s} {'chance':>7s} {'baseline':>9s} "
              f"{'learned':>16s} {'delta':>8s} {'sup':>5s}   0.70?")
        rows = []
        for k in keys:
            b, _ = agg(runs["frozen baseline"], split, k)
            m, sd = agg(list(runs.values())[1], split, k)
            ch = runs["frozen baseline"][0][split]["per_group"].get(
                k, {}).get("chance", float("nan"))
            su = runs["frozen baseline"][0][split]["per_group"].get(
                k, {}).get("support", 0)
            rows.append((k, ch, b, m, sd, su))
        for k, ch, b, m, sd, su in sorted(rows, key=lambda r: -r[3]):
            hit = "YES" if m >= 0.70 else f"{0.70-m:+.3f}"
            print(f"{k:20s} {ch:7.3f} {b:9.3f} {m:9.3f} +/-{sd:.3f} "
                  f"{m-b:+8.3f} {su:5d}   {hit}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {n: [{s: {kk: vv for kk, vv in r[s].items() if kk != "per_task"}
                  for s in SPLITS} for r in rs] for n, rs in runs.items()},
            indent=1, default=float))
        print(f"\nwrote {a.out}")
    R.log("vjreport", tags=a.tags, n_seeds=len(acc), touched_test=True,
          **{f"learned_{s}": round(agg(acc, s)[0], 4) for s in SPLITS},
          **{f"base_{s}": round(agg(runs['frozen baseline'], s)[0], 4)
             for s in SPLITS})


if __name__ == "__main__":
    main()
