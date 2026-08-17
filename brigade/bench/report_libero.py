#!/usr/bin/env python3
"""Turn a lerobot-eval run into a per-task coverage table.

    python3 brigade/bench/report_libero.py eval_logs/pi05_full

lerobot-eval writes eval_info.json with per-task success lists but no readable
summary; this joins those to the actual task language strings so the output says
"put the bowl on the plate: 9/10" rather than "libero_goal task 7: 0.9".

Every number printed comes from that file. Nothing here estimates or infers.
"""

from __future__ import annotations

import json
import os
import sys

# Which tasks the demo script actually uses, so the table can mark them.
DEMO_DIRECT = {("libero_goal", i) for i in (0, 1, 3, 5, 6, 7, 8, 9)}
DEMO_MULTI = {("libero_10", i) for i in (0, 2, 3, 4, 9)}


def task_languages() -> dict[tuple[str, int], str]:
    """Ask LIBERO for the real instruction strings."""
    from libero.libero import benchmark

    bd = benchmark.get_benchmark_dict()
    out: dict[tuple[str, int], str] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        if suite not in bd:
            continue
        s = bd[suite]()
        for i in range(s.n_tasks):
            out[(suite, i)] = s.get_task(i).language
    return out


def main(run_dir: str) -> int:
    path = os.path.join(run_dir, "eval_info.json")
    if not os.path.exists(path):
        print(f"no eval_info.json in {run_dir}", file=sys.stderr)
        return 1
    info = json.load(open(path))
    lang = task_languages()

    rows = []
    for entry in info.get("per_task", []):
        suite = entry["task_group"]
        tid = int(entry["task_id"])
        succ = entry["metrics"].get("successes", [])
        n_ok, n = sum(bool(x) for x in succ), len(succ)
        key = (suite, tid)
        tag = "DIRECT" if key in DEMO_DIRECT else ("MULTI" if key in DEMO_MULTI else "")
        rows.append((suite, tid, lang.get(key, "?"), n_ok, n, tag))

    rows.sort(key=lambda r: (r[0], r[1]))
    width = max((len(r[2]) for r in rows), default=20)

    print(f"\n{'suite':<15} {'id':>3}  {'task':<{width}}  {'succ':>7}  {'rate':>6}  demo")
    print("-" * (15 + 5 + width + 24))
    for suite, tid, text, n_ok, n, tag in rows:
        rate = f"{100.0 * n_ok / n:.0f}%" if n else "—"
        print(f"{suite:<15} {tid:>3}  {text:<{width}}  {n_ok:>3}/{n:<3}  {rate:>6}  {tag}")

    print("\nper suite")
    for suite, g in sorted(info.get("per_group", {}).items()):
        print(f"  {suite:<15} {g['pc_success']:.1f}%  over {g['n_episodes']} episodes")

    o = info.get("overall", {})
    if o:
        print(f"\nOVERALL {o.get('pc_success', 0):.1f}%  "
              f"({o.get('n_episodes', 0)} episodes, "
              f"{o.get('eval_ep_s', 0):.1f}s per episode)")

    # Demo-relevant subtotal: the tasks the script actually performs.
    demo = [r for r in rows if r[5]]
    if demo:
        ok = sum(r[3] for r in demo)
        tot = sum(r[4] for r in demo)
        print(f"DEMO TASKS {100.0 * ok / tot:.1f}%  ({ok}/{tot} over {len(demo)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "eval_logs/pi05_full"))
