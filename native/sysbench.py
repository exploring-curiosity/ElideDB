"""THE SYSTEM BENCHMARK: yield and precision of the read path exactly
as it runs.

Why this file exists. Every headline number in this project's recent
history came from an EVENT-MATRIX proxy: precomputed span vectors,
scored event-against-event, ranked. The shipped path does something
strictly harder - it slides windows over the whole store, fuses phase
DTW, applies CSLS and diffusion, and collapses overlaps - and it is
handed a clip, not an event id. A proxy number is not the product's
number, so this measures vwm_qbe.WMIndex.search itself.

THE TWO METRICS (unchanged, per the standing definition):
    support = true same-primitive events in the store (own episode
              excluded - leave-episode-out, always)
    k       = ceil(1.5 x support), a MAX BOUND, not a quota
    yield   = true / support        (did we find what exists)
    prec    = true / returned       (was what we returned right)

A returned window counts TRUE when it overlaps a true event of the
query's primitive by IoU >= IOU_HIT in that episode's timeline -
position-verified, because position-blind grading is forbidden here
(it inflated an earlier number by counting the right episode at the
wrong moment).

Truth is EVAL-ONLY: primitives and spans grade the output, they never
enter the store, the query, or the scorer.

    python native/sysbench.py --n 120          # stratified sample
    python native/sysbench.py --n 0            # every event (slow)
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
import vwm_qbe  # noqa: E402

FPS = 10.0
IOU_HIT = 0.3
PRIMS = ("pick", "place", "stack", "unstack", "push")
CORPORA = ("sim_chains", "sim_eval_bal")


def load_truth():
    """mid -> [(prim, t0, t1)] from meta.json. EVAL-ONLY."""
    truth = {}
    for corp in CORPORA:
        for ep in sorted((ROOT / "data" / corp).glob("ep*")):
            f = ep / "meta.json"
            if not f.exists():
                continue
            meta = json.loads(f.read_text())
            evs = [(e["prim"], float(e["t0"]), float(e["t1"]))
                   for e in meta["events"]
                   if e["ok"] and e["t1"] - e["t0"] >= 0.6]
            if evs:
                truth[f"{corp}/{ep.name}"] = evs
    return truth


def build_index():
    """One index spanning both ruler corpora (the store under test)."""
    idx = None
    for corp in CORPORA:
        part = vwm_qbe.WMIndex(root=f"data/{corp}", corpus=corp)
        if idx is None:
            idx = part
        else:
            idx.traj.update(part.traj)
            idx.kin.update(part.kin)
    return idx


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / max(union, 1e-6)


def main():
    n_arg = 120
    if "--n" in sys.argv:
        n_arg = int(sys.argv[sys.argv.index("--n") + 1])

    truth = load_truth()
    idx = build_index()
    # only episodes the store actually holds can be queried or hit
    truth = {m: v for m, v in truth.items() if m in idx.traj}
    all_ev = [(m, p, t0, t1) for m, v in truth.items()
              for p, t0, t1 in v]
    total = Counter(p for _, p, _, _ in all_ev)
    print(f"store {len(idx.traj)} episodes | {len(all_ev)} true events "
          f"| " + "  ".join(f"{p} {total[p]}" for p in PRIMS),
          flush=True)

    # NATURAL-distribution sample, not stratified: the standing
    # numbers average over every event as the corpus actually mixes
    # them, and this corpus is 44% pick. A stratified sample would
    # over-weight the rare hard primitives and report a different
    # (lower) number than the one it is being compared against.
    rng = np.random.default_rng(0)
    if n_arg and n_arg < len(all_ev):
        qs = [all_ev[i] for i in
              rng.choice(len(all_ev), n_arg, replace=False)]
    else:
        qs = all_ev
    print(f"querying {len(qs)} events "
          f"(leave-episode-out, k = ceil(1.5 x support))", flush=True)

    ys, ps, rows = [], [], defaultdict(list)
    yo, po = [], []          # oracle-span variant (ranking only)
    t_start = time.time()
    for qi, (mid, prim, t0, t1) in enumerate(qs):
        # support: same-primitive true events OUTSIDE this episode
        sup = sum(1 for m2, p2, _, _ in all_ev
                  if p2 == prim and m2 != mid)
        if not sup:
            continue
        k = int(math.ceil(1.5 * sup))
        s, e = int(t0 * FPS), int(t1 * FPS)
        views = idx.traj[mid]
        T = min(len(r) for r in views)
        s, e = max(0, min(s, T - 2)), min(e, T)
        if e - s < 4:
            continue
        hq = views[0][s:e]
        out = idx.search(hq, k=k, exclude=(mid, t0, t1))
        hit = 0
        for m2, a, b, _ in out:
            if m2 == mid:
                continue                # own episode never counts
            if any(p2 == prim and iou(a, b, u0, u1) >= IOU_HIT
                   for p2, u0, u1 in truth.get(m2, [])):
                hit += 1
        ret = max(len([o for o in out if o[0] != mid]), 1)
        ys.append(hit / sup)
        ps.append(hit / ret)
        rows[prim].append((hit / sup, hit / ret, sup, ret))

        # DECOMPOSITION: the same scorer handed ORACLE SPANS - only
        # ranking is left to do. The gap to the line above is what
        # LOCALIZATION costs, which every event-matrix number in this
        # project has silently been given for free.
        v = vwm_qbe._l2(idx.dmulti(np.asarray(hq, np.float32),
                                   0, len(hq)))
        cand = []
        for m2, evs2 in truth.items():
            if m2 == mid:
                continue
            vw = idx.traj[m2]
            T2 = min(len(r) for r in vw)
            for p2, u0, u1 in evs2:
                a2, b2 = int(u0 * FPS), int(u1 * FPS)
                if b2 > T2 or b2 - a2 < 4:
                    continue
                sc = max(float(v @ vwm_qbe._l2(idx.dmulti(r, a2, b2)))
                         for r in vw)
                cand.append((sc, p2))
        if cand:
            cand.sort(key=lambda c: -c[0])
            top = cand[:k]
            h2 = sum(1 for _, p2 in top if p2 == prim)
            yo.append(h2 / sup)
            po.append(h2 / max(len(top), 1))
        if (qi + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  {qi + 1}/{len(qs)}  yield {np.mean(ys):.3f}  "
                  f"prec {np.mean(ps):.3f}  ({el / (qi + 1):.1f}s/q)",
                  flush=True)

    print("\n=== THE SYSTEM, AS IT RUNS "
          "(leave-episode-out, position-verified) ===")
    print(f"{'prim':9s} {'n':>4s} {'support':>8s} {'returned':>9s} "
          f"{'yield':>7s} {'prec':>7s}")
    for p in PRIMS:
        if not rows[p]:
            continue
        y = np.mean([r[0] for r in rows[p]])
        pr = np.mean([r[1] for r in rows[p]])
        sup = np.mean([r[2] for r in rows[p]])
        ret = np.mean([r[3] for r in rows[p]])
        print(f"{p:9s} {len(rows[p]):4d} {sup:8.0f} {ret:9.0f} "
              f"{y:7.3f} {pr:7.3f}")
    print(f"{'ALL':9s} {len(ys):4d} {'':8s} {'':9s} "
          f"{np.mean(ys):7.3f} {np.mean(ps):7.3f}")
    if yo:
        print(f"\n--- same scorer, ORACLE SPANS (ranking only, no "
              f"localization) ---")
        print(f"{'ALL':9s} {len(yo):4d} {'':8s} {'':9s} "
              f"{np.mean(yo):7.3f} {np.mean(po):7.3f}")
        print(f"localization cost: yield {np.mean(yo) - np.mean(ys):+.3f}"
              f"  prec {np.mean(po) - np.mean(ps):+.3f}")
    print(f"\nwall clock {time.time() - t_start:.0f}s for {len(ys)} "
          f"queries ({(time.time() - t_start) / max(len(ys),1):.1f}s each)")


if __name__ == "__main__":
    main()
