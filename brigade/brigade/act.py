"""THE SHIFT — a seven-beat act where every beat leans on the one before it.

    python -m brigade.act              # run it with memory, then without
    python -m brigade.act --arm on     # just the memory arm

The single-request A/B shows that memory supplies a missing *argument*. This
shows the harder thing: that memory supplies **continuity**, and that the beats
compound. Each one is a different kind of remembering, and each is scored.

  1  "put the bowl on the stove"      explicit — the baseline. Nothing recalled.
  2  "where is the bowl?"             SPATIAL. Answered from object_beliefs and
                                      checked against the simulator. No episode
                                      runs; there is nothing to see, only to
                                      recall.
  3  "put it back"                    ANAPHORA + NORM. The sentence contains no
                                      noun. Memory supplies the referent (the
                                      bowl, from beat 1) and then where "back"
                                      is (the cabinet, from the norm). Two reads
                                      composing into one instruction.
  4  "where is the bowl?"             SPATIAL AGAIN — and the answer must have
                                      CHANGED. This is the beat that separates a
                                      memory from a lookup table: the same
                                      question, a different true answer, because
                                      the world moved and the robot noticed.
  5  "and the bottle too"             ELLIPSIS. No verb at all. The verb carries
                                      over from beat 3, the destination from the
                                      bottle's own norm — which is a different
                                      place, proving it is retrieval.
  6  "now get the stove going"        PARAPHRASE. No word in common with the
                                      instruction it resolves to.
  7  "feed the cat"                   ABSTENTION. Nothing resembles it, so the
                                      robot declines instead of inventing. A
                                      system that always answers has not shown
                                      it is reading anything.

Run with memory off and beats 2-7 have nothing to work with: no referent, no
belief, no norm, no history. That is the measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .run import SCENE_SUITE, SEED_HISTORY, Brigade, seed

log = logging.getLogger("brigade.act")

# (beat, what a human says, what memory has to supply, how the beat is scored)
#   'do'      — an episode runs; scored by LIBERO's goal predicate
#   'ask'     — nothing runs; scored against the simulator's ground truth
#   'abstain' — scored by whether the robot correctly declines
BEATS = [
    ("1", "put the bowl on the stove", "nothing — explicit", "do"),
    ("2", "where is the bowl?", "spatial belief written in beat 1", "ask"),
    ("3", "put it back", "referent for 'it' + the norm for bowls", "do"),
    ("4", "where is the bowl?", "the belief, now CHANGED by beat 3", "ask"),
    ("5", "and the bottle too", "the verb from beat 3 + the bottle's own norm", "do"),
    ("6", "now get the stove going", "a paraphrase with no shared words", "do"),
    ("7", "feed the cat", "nothing — and it must say so", "abstain"),
]


def run_arm(bg: Brigade, use_memory: bool) -> list[dict]:
    tag = "MEMORY ON" if use_memory else "MEMORY OFF"
    print(f"\n{'=' * 78}\nTHE SHIFT — {tag}\n{'=' * 78}", flush=True)

    # Both arms start from the same kitchen. Beat 1 is explicit and works either
    # way; everything after it is where they diverge.
    bg.enter(1)   # 'put the bowl on the stove'

    rows = []
    for beat, said, needs, kind in BEATS:
        print(f'\n[{beat}] human: "{said}"')
        print(f"    memory must supply: {needs}")

        if kind == "ask":
            ans = bg.where_is(said, use_memory=use_memory)
            ok = bool(ans.get("correct"))
            print(f"    robot: {ans['text']}")
            if ans.get("answered"):
                print(f"    truth: the {ans['label']} is on the "
                      f"{ans.get('actual_place')}   ->  "
                      f"{'CORRECT' if ok else 'WRONG'}  ({ans['latency_ms']:.1f} ms)")
            rows.append(dict(beat=beat, said=said, kind=kind, memory=use_memory,
                             ok=ok, answer=ans.get("text"),
                             place=ans.get("place"), actual=ans.get("actual_place")))
            continue

        res = bg.resolver.resolve(said, use_memory=use_memory)
        if kind == "abstain":
            # Scored the same way in both arms: declining is right, inventing an
            # action is wrong. An earlier version gave the memory-off arm a free
            # pass here on the grounds that it "has nothing to abstain from",
            # which contradicted the verdict printed on the very next line and
            # flattered the control arm by a whole beat.
            ok = res.abstained
            print(f"    robot: {'ABSTAINED — ' + res.rationale[:88] if res.abstained else res.instruction}")
            print(f"    -> {'CORRECT (declined)' if res.abstained else 'WRONG (invented an action)'}")
            rows.append(dict(beat=beat, said=said, kind=kind, memory=use_memory,
                             ok=ok, answer="abstained" if res.abstained else res.instruction))
            continue

        if res.expanded and res.expanded != said:
            print(f'    rewritten: "{res.expanded}"   ({res.rewrite_note})')
        out = bg.handle(said, use_memory=use_memory,
                        goal=res.task_id if use_memory else None)
        instr = (out.get("resolution") or {}).get("instruction")
        secs = (out.get("episode") or {}).get("seconds", 0.0)
        print(f"    executes: {instr!r}")
        print(f"    -> {'SUCCESS' if out['ok'] else 'failed'}  ({secs:.0f}s)")
        rows.append(dict(beat=beat, said=said, kind=kind, memory=use_memory,
                         ok=bool(out["ok"]), answer=instr, seconds=secs,
                         abstained=bool(out.get("abstained"))))

    ok = sum(1 for r in rows if r["ok"])
    print(f"\n{'-' * 78}\n{tag}: {ok}/{len(rows)} beats\n", flush=True)
    return rows


def report(rows: list[dict]) -> None:
    on = [r for r in rows if r["memory"]]
    off = [r for r in rows if not r["memory"]]
    print(f"\n{'=' * 78}\nTHE SHIFT — beat by beat\n{'=' * 78}")
    print(f"{'#':<3}{'what the human said':<30}{'kind':<10}{'mem ON':<9}{'mem OFF'}")
    print("-" * 78)
    for a in on:
        b = next((x for x in off if x["beat"] == a["beat"]), None)
        print(f"{a['beat']:<3}{a['said'][:29]:<30}{a['kind']:<10}"
              f"{'ok' if a['ok'] else 'fail':<9}{'ok' if (b and b['ok']) else 'fail'}")
    print("-" * 78)
    print(f"{'':<43}{sum(r['ok'] for r in on)}/{len(on):<7}"
          f"{sum(r['ok'] for r in off)}/{len(off)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["on", "off", "both"], default="both")
    ap.add_argument("--seed", action="store_true", help="live a history first")
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument("--no-relmo", action="store_true")
    ap.add_argument("--out", default="../eval_logs/brigade_act.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-15s %(message)s",
                        datefmt="%H:%M:%S")

    bg = Brigade()
    if args.wipe:
        bg.mem.setup()
        bg.mem.wipe()
        bg.mem.db.execute("DELETE FROM relmo_recordings")
        print("memory wiped")
    bg.boot(with_relmo=not args.no_relmo)
    if args.seed:
        seed(bg)

    rows = []
    # The memory arm runs FIRST and the off arm second, because the off arm
    # writes nothing: running it first would leave the on arm with the same
    # empty memory and both would fail for the same reason.
    if args.arm in ("on", "both"):
        rows += run_arm(bg, True)
    if args.arm in ("off", "both"):
        rows += run_arm(bg, False)
    if args.arm == "both":
        report(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(beats=[dict(beat=b, said=s, needs=n, kind=k) for b, s, n, k in BEATS],
                   rows=rows,
                   norms=[dict(x) for x in bg.mem.norms()],
                   beliefs=[dict(x) for x in bg.mem.beliefs()]),
              open(args.out, "w"), indent=1, default=str)
    print(f"\nwrote {args.out}")
    bg.relmo.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
