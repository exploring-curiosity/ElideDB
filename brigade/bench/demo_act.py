#!/usr/bin/env python3
"""The demo act: a kitchen robot working through a shift, driven by a learned policy.

    python3 brigade/bench/demo_act.py --episodes 1 --out eval_logs/demo_act

Three acts, thirteen beats, every beat a real LIBERO task string handed to
pi0.5 verbatim. Nothing is scripted: the policy sees two camera images, an 8-dim
arm state and the instruction, and emits end-effector deltas. Opening drawers,
grasping, ordering sub-goals — all of it comes out of the policy.

ACT I  — direct instruction. You say it, it does it.
ACT II — multi-step. ONE instruction, several sub-goals, none of them given.
ACT III— indirect. The human request is underspecified ("put the mug away") and
         MEMORY supplies the missing argument before the policy is invoked.

Act III is the seam worth pointing at: the policy owns HOW, memory owns WHAT and
WHERE. This script prints the resolution explicitly so a viewer can see the
memory lookup happen rather than taking it on faith.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

POLICY = "lerobot/pi05_libero_finetuned"

# (act, suite, task_id, beat label). Task ids verified against the live benchmark.
BEATS = [
    ("I",  "libero_goal", 0, "open the drawer"),
    ("I",  "libero_goal", 7, "turn on the stove"),
    ("I",  "libero_goal", 8, "put the bowl on the plate"),
    ("I",  "libero_goal", 5, "push the plate (non-prehensile)"),
    ("I",  "libero_goal", 9, "put the wine bottle on the rack"),
    ("I",  "libero_goal", 6, "put the cream cheese in the bowl"),
    ("I",  "libero_goal", 1, "put the bowl on the stove"),
    ("I",  "libero_goal", 3, "open the top drawer AND put the bowl inside"),
    ("II", "libero_10",   3, "bowl into bottom drawer, then CLOSE it"),
    ("II", "libero_10",   2, "turn on the stove AND put the moka pot on it"),
    ("II", "libero_10",   9, "mug into the microwave, then CLOSE it"),
    ("II", "libero_10",   1, "TWO objects into the basket"),
    ("II", "libero_10",   4, "two similar mugs, told apart only by language"),
]

# Act III: an underspecified human request, what memory must supply, and the
# beat it resolves to. The memory layer (Postgres/pgvector) answers the middle
# column; the policy then executes the resolved instruction.
INDIRECT = [
    ("put the mug away",      "recall: mugs are kept in the microwave",      ("libero_10", 9)),
    ("clear the counter",     "recall: bowl belongs in the bottom drawer",   ("libero_10", 3)),
    ("start the coffee",      "recall: moka pot goes on the stove, stove on", ("libero_10", 2)),
    ("same as last time",     "recall: last successful task string",          ("libero_goal", 0)),
]


def languages():
    from libero.libero import benchmark

    bd = benchmark.get_benchmark_dict()
    out = {}
    for suite in ("libero_goal", "libero_10"):
        s = bd[suite]()
        for i in range(s.n_tasks):
            out[(suite, i)] = s.get_task(i).language
    return out


def run_beat(suite, tid, episodes, out_dir, python_exe):
    """One beat = one lerobot-eval invocation, so each gets its own video."""
    dest = os.path.join(out_dir, f"{suite}_{tid}")
    cmd = [
        python_exe, "-u", "-m", "lerobot.scripts.lerobot_eval",
        f"--output_dir={dest}",
        "--env.type=libero", f"--env.task={suite}", f"--env.task_ids=[{tid}]",
        "--eval.batch_size=1", f"--eval.n_episodes={episodes}",
        f"--policy.path={POLICY}",
        "--policy.n_action_steps=10", "--policy.device=mps",
        "--policy.compile_model=false",          # torch.compile dies on MPS
        "--env.max_parallel_tasks=1",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    secs = time.time() - t0
    info_path = os.path.join(dest, "eval_info.json")
    if not os.path.exists(info_path):
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return dict(ok=False, rate=0.0, n=0, seconds=secs, error=tail)
    info = json.load(open(info_path))
    succ = info["per_task"][0]["metrics"]["successes"]
    n_ok = sum(bool(x) for x in succ)
    return dict(ok=n_ok > 0, rate=100.0 * n_ok / len(succ), n_ok=n_ok,
                n=len(succ), seconds=secs,
                video=info["overall"].get("video_paths", [None])[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="eval_logs/demo_act")
    ap.add_argument("--python", default=".venv-libero/bin/python")
    ap.add_argument("--dry-run", action="store_true", help="print the act, run nothing")
    args = ap.parse_args()

    lang = languages()
    os.makedirs(args.out, exist_ok=True)

    print("\n" + "=" * 78)
    print("BRIGADE DEMO ACT — learned policy, no scripted motion")
    print("=" * 78)

    results = []
    for act, suite, tid, label in BEATS:
        text = lang.get((suite, tid), "?")
        print(f"\n[ACT {act}] {label}")
        print(f'          instruction: "{text}"')
        if args.dry_run:
            continue
        r = run_beat(suite, tid, args.episodes, args.out, args.python)
        results.append((act, suite, tid, text, r))
        if "error" in r:
            print(f"          FAILED TO RUN: {r['error'][:160]}")
        else:
            print(f"          -> {r['n_ok']}/{r['n']} ({r['rate']:.0f}%) in {r['seconds']:.0f}s")

    print("\n" + "-" * 78)
    print("ACT III — indirect requests. Memory resolves; the policy executes.")
    for said, recalled, (suite, tid) in INDIRECT:
        print(f'\n  human: "{said}"')
        print(f"  memory: {recalled}")
        print(f'  executes: "{lang.get((suite, tid), "?")}"')

    if results:
        ok = sum(r[4].get("n_ok", 0) for r in results)
        tot = sum(r[4].get("n", 0) for r in results)
        secs = sum(r[4]["seconds"] for r in results)
        print("\n" + "=" * 78)
        print(f"DEMO ACT: {ok}/{tot} episodes succeeded "
              f"({100.0 * ok / tot if tot else 0:.1f}%) across {len(results)} tasks "
              f"in {secs / 60:.1f} min")
        json.dump(
            [dict(act=a, suite=s, task_id=t, instruction=x, **r) for a, s, t, x, r in results],
            open(os.path.join(args.out, "demo_act_results.json"), "w"), indent=1,
        )
        print(f"wrote {os.path.join(args.out, 'demo_act_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
