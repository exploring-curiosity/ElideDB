#!/usr/bin/env python3
"""The whole system, one utterance at a time.

    .venv-libero/bin/python brigade/bench/serve_demo.py
    .venv-libero/bin/python brigade/bench/serve_demo.py --ab      # the A/B only

    human says one thing
      -> the store cuts the robot's LIVE view and indexes it
      -> stage 1: the database ranks the whole memory by RelMo's prefilter
      -> stage 2: RelMo re-ranks that shortlist by DTW over the traces
      -> the head turns retrieved clips into a command (K weights, no words)
      -> pi0.5 executes it as a latent prefix
      -> the kitchen keeps what changed; the cameras never stopped

WHAT IS MEASURED, and why it is the CHOICE rather than the success rate.
LIBERO's `check_success` evaluates the SCENE's stored goal predicate, not the
instruction the robot was given — so an episode driven by a memory-derived
command is scored against whatever that scene happens to want, and a success
there says nothing about whether the memory decided well. The claim this
project makes is about the DECISION: the same words, against different stretches
of the robot's own recorded life, produce different commands. That is what the
A/B measures, and it does not depend on pi0.5's success rate at all.

Execution is still run and still reported, because a command that no policy can
act on is not a command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = "../eval_logs/serve"
REASONER = "../eval_logs/reasoner/reasoner.pt"

# What a person actually says. Half of these are true of several behaviours,
# which is the point: the words cannot decide, so something else must.
SCRIPT = [
    "put the bowl away",
    "open a drawer",
    "get the stove ready",
    "put the wine bottle away",
]


def _dump(name: str, obj) -> None:
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1, default=str)


def run(ab_only: bool = False, scene: int = 0) -> int:
    from brigade.agent.serve import Kitchen

    k = Kitchen(REASONER, scene=scene)
    info = k.open()
    print(f"kitchen open: basis {info['basis']}, {info['behaviours']} behaviours")
    print(f"store: {info['store']}\n")

    try:
        # The cameras have to have seen something before anything can be asked:
        # the query IS the live view, and a span is 15 s of video.
        print("watching the kitchen (the query is the robot's own view) ...",
              flush=True)
        k.watch(18.0)

        results: dict = {}

        if not ab_only:
            # ---- the script -------------------------------------------------
            print(f"\n{'=' * 72}\nA HUMAN SAYS ONE THING\n{'=' * 72}")
            for heard in SCRIPT:
                t = k.hear(heard)
                _report(t)
                k.watch(6.0)          # the cameras keep running between commands
            results["script"] = [t.to_json() for t in k.turns]

            # ---- memory off -------------------------------------------------
            print(f"\n{'=' * 72}\nCONTROL: THE SAME WORDS WITH MEMORY OFF\n{'=' * 72}")
            off = [k.hear(h, memory=False) for h in SCRIPT]
            for t in off:
                _report(t)
            results["memory_off"] = [t.to_json() for t in off]
            print("\nWith no clips there is no command. The robot does not act "
                  "worse — it\ncannot act, which is what 'memory is load-bearing' "
                  "has to mean to be a claim.")

        # ---- the A/B --------------------------------------------------------
        # Same words. Two different stretches of the robot's recorded life.
        # Nothing else differs — same head, same policy, same kitchen.
        print(f"\n{'=' * 72}\nA/B: THE SAME WORDS AGAINST DIFFERENT HISTORY\n{'=' * 72}")
        halves = _halves(k)
        if halves is None:
            print("not enough history to split; run the collection first")
        else:
            (a0, a1), (b0, b1) = halves
            ab = []
            for heard in SCRIPT:
                ta = k.hear(heard, since=a0, until=a1, act=False)
                tb = k.hear(heard, since=b0, until=b1, act=False)
                same = ta.chose == tb.chose
                print(f"\n  {heard!r}")
                print(f"    earlier history -> {ta.chose or ta.note!r}"
                      f"   (top {0.0 if ta.weights is None else ta.weights.max():.2f})")
                print(f"    later history   -> {tb.chose or tb.note!r}"
                      f"   (top {0.0 if tb.weights is None else tb.weights.max():.2f})")
                print(f"    {'SAME' if same else 'DIFFERENT'} command")
                ab.append(dict(heard=heard, earlier=ta.to_json(),
                               later=tb.to_json(), differs=not same))
            n = sum(1 for r in ab if r["differs"])
            print(f"\n  {n}/{len(ab)} requests produced a different command from "
                  f"the same words.")
            print("  The words were identical in every pair. What changed was "
                  "which part of\n  the video the memory was allowed to see.")
            results["ab"] = ab

        _dump("serve.json", results)
        print(f"\nwrote {OUT}/serve.json")
        return 0
    finally:
        k.close()


def _halves(k):
    """Split the store's history in two by time. -> ((a0,a1),(b0,b1)) or None."""
    rows = k.store.db.query(
        "SELECT extract(epoch from t0) AS a FROM clips WHERE kitchen_id = %s "
        "AND embedding IS NOT NULL AND basis_id = %s ORDER BY t0",
        (k.store.kitchen, k.relmo.basis))
    ts = [float(r["a"]) for r in rows]
    if len(ts) < 8:
        return None
    mid = ts[len(ts) // 2]
    return (ts[0] - 1, mid), (mid, ts[-1] + 1e6)


def _report(t) -> None:
    print(f"\n  heard: {t.heard!r}")
    if t.note and not t.acted:
        print(f"    {t.note}")
    if t.clips:
        print(f"    memory returned {len(t.clips)} spans in "
              f"{t.retrieval_ms:.1f} ms "
              f"({'stage 2' if t.clips[0].stage == 'dtw' else 'stage 1 only'})")
        for c in t.clips[:3]:
            print(f"      {c.clip_id}  {c.score:.3f}  "
                  f"[{time.strftime('%H:%M:%S', time.localtime(c.t0))}]")
    if t.chose:
        w = t.weights
        print(f"    command: {t.chose!r}  (top {w.max():.2f}, "
              f"margin {t.margin:.2f}, {t.reason_ms:.1f} ms)")
    if t.acted:
        print(f"    acted: {'goal satisfied' if t.succeeded else 'did not finish'}"
              f"  ({t.act_s:.0f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="only the A/B")
    ap.add_argument("--scene", type=int, default=0)
    a = ap.parse_args()
    return run(ab_only=a.ab, scene=a.scene)


if __name__ == "__main__":
    sys.exit(main())
