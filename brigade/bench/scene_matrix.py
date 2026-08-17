#!/usr/bin/env python3
"""Which goals can the robot reach from ONE fixed kitchen?

    python3 brigade/bench/scene_matrix.py --init 4 --episodes 2

LIBERO's `libero_goal` suite is one kitchen with ten goals, but each task also
ships its own *initial object placement*. The published 97% for the suite is
measured with every goal started from its own placement. Brigade does not get
that: the robot stands in a kitchen that is however it was left, and memory
picks the goal — so the honest question is which goals are reachable from a
single fixed start.

This measures exactly that, and it is not a formality. The same discipline
applied to `libero_90` found that pi0.5 scores 5/5 on two KITCHEN_SCENE4 tasks
and 0/5 on three others, because only the first two overlap its training set.
Assuming instead of measuring would have put a task the policy cannot do at the
centre of the demo.

Every number written here comes from an episode that ran.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--init", type=int, default=4, help="task whose init state is the kitchen")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", default="eval_logs/scene_matrix.json")
    args = ap.parse_args()

    from libero.libero import benchmark

    from brigade.agent.pilot import Pilot

    suite = benchmark.get_benchmark_dict()[args.suite]()
    goals = [suite.get_task(i).language for i in range(suite.n_tasks)]

    p = Pilot(device="mps")
    p.load()
    p.open_scene(args.suite, args.init)
    print(f"\nkitchen = {args.suite}/{args.init} init state "
          f"({p.default_instruction!r})\n{'=' * 78}", flush=True)

    rows = []
    t0 = time.time()
    for i, goal in enumerate(goals):
        oks = []
        for ep_i in range(args.episodes):
            ep = p.run(goal, keep_frames=False, seed=10000 + ep_i)
            oks.append(ep.success)
        rate = 100.0 * sum(oks) / len(oks)
        rows.append(dict(task_id=i, instruction=goal, n_ok=sum(oks),
                         n=len(oks), rate=rate))
        print(f"  {i}  {sum(oks)}/{len(oks)}  {rate:>5.0f}%   {goal}", flush=True)

    reachable = [r for r in rows if r["n_ok"] == r["n"]]
    print(f"\n{'-' * 78}")
    print(f"reachable from this one kitchen: {len(reachable)}/{len(rows)} goals "
          f"({time.time() - t0:.0f}s)")
    for r in reachable:
        print(f"    {r['instruction']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(suite=args.suite, init_task=args.init, episodes=args.episodes,
                   rows=rows), open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
