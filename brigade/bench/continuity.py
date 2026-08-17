#!/usr/bin/env python3
"""Is it one continuous kitchen, and does the policy still work in one?

    python3 brigade/bench/continuity.py

Two questions, and the second is the risk. LIBERO resets between episodes by
design, so carrying the world forward is a departure from how pi0.5 was
evaluated: every published number for it starts each episode from one of the
benchmark's 50 stored placements. Starting instead from wherever the last
episode ended is, by definition, out of that distribution.

So this measures both:

  CONTINUITY  does object X stay where the robot put it, across the next
              episode's reset, and across a change of goal (which rebuilds the
              whole env)?
  COMPETENCE  does the policy still succeed from a carried-over start, or does
              persistence cost accuracy?

The honest answer to "what is the use" depends on the second number, so it is
measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def places(obs) -> dict:
    return {o.label: o.place for o in obs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../eval_logs/continuity.json")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    from brigade.agent.pilot import Pilot

    # A shift with dependencies: goal 4 moves the bowl to the cabinet, goal 9
    # moves the bottle to the rack, goal 7 touches neither, goal 6 needs the
    # bowl. If the world is continuous, the bowl is still on the cabinet by the
    # end and the reads below show it; if it is not, everything reads "table".
    SHIFT = [(4, "put the bowl on top of the cabinet"),
             (9, "put the wine bottle on the rack"),
             (7, "turn on the stove"),
             (6, "put the cream cheese in the bowl")]

    p = Pilot(device="mps")
    p.load()

    rows, state = [], None
    print(f"\n{'=' * 78}\nONE CONTINUOUS KITCHEN — state carried across every episode "
          f"and every goal change\n{'=' * 78}", flush=True)

    for i, (goal, instruction) in enumerate(SHIFT * args.repeat):
        p.open_scene("libero_goal", goal)      # rebuilds the env from scratch
        ep = p.run(instruction, keep_frames=False, from_state=state)

        before, after = places(ep.start_obs), places(ep.final_obs)
        state = ep.end_state                   # hand it to the next episode

        print(f"\n[{i + 1}] {instruction}")
        print(f"    started with : {before}")
        print(f"    already true : {ep.already_satisfied}")
        print(f"    -> {'SUCCESS' if ep.success else 'failed '}  ({ep.seconds:.0f}s)")
        print(f"    ended with   : {after}", flush=True)
        rows.append(dict(i=i + 1, goal=goal, instruction=instruction,
                         success=bool(ep.success), already=bool(ep.already_satisfied),
                         seconds=round(ep.seconds, 1), before=before, after=after))

    # CONTINUITY: did each episode start where the previous one ended?
    print(f"\n{'-' * 78}\nCONTINUITY — each episode's start vs the previous end")
    carried = broke = 0
    for a, b in zip(rows, rows[1:]):
        same = a["after"] == b["before"]
        carried += same
        broke += not same
        if not same:
            diff = {k: (a["after"].get(k), b["before"].get(k))
                    for k in a["after"] if a["after"].get(k) != b["before"].get(k)}
            print(f"  episode {a['i']} -> {b['i']}: BROKE  {diff}")
    print(f"  carried {carried}/{carried + broke} transitions"
          f"{' — the kitchen is persistent' if broke == 0 else ''}")

    ok = sum(r["success"] for r in rows)
    triv = sum(r["already"] for r in rows)
    print(f"\nCOMPETENCE — {ok}/{len(rows)} succeeded from carried-over starts")
    print(f"  of which already satisfied before acting: {triv}")
    print("  (published pi0.5 numbers all start from the benchmark's own "
          "placements;\n   a carried start is out of that distribution by "
          "construction)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(rows=rows, carried=carried, broke=broke,
                   succeeded=ok, already=triv), open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
