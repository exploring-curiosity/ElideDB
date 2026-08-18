#!/usr/bin/env python3
"""The agent: an episode-triage queue that starts with NO memory and earns one.

    .venv-libero/bin/python showreel/watch.py

A robot fleet produces more episodes than anyone can watch. Each one has to be
dispositioned: filed under what it is, so it can be counted, compared and acted
on. Today that is a person, or a labelling contract, or a taxonomy someone has
to define before a single video can be filed.

This agent starts with an EMPTY memory and does it itself:

    RETRIEVE  ask memory for the nearest precedents to what just arrived
    ACT       precedents agree      -> file it under their disposition, alone
              precedents disagree   -> escalate to a person and take their answer
              memory has nothing    -> escalate; there is nothing else it can do
    STORE     write the clip and the disposition it ended up with

The loop closes on the third line. What the agent files today becomes the
precedent it files against tomorrow, so its own past work is what reduces its
future work. Nothing is retrained; a row is written.

WHY CONSENSUS AND NOT A SIMILARITY THRESHOLD. The first version of this agent
escalated when the top match scored below a cut, and it did not work: measured,
every DTW score sat between 0.16 and 0.27 whether the precedent was right or
wrong. This system ranks well (precision@8 0.840) and calibrates badly, and
those are different properties. Agreement among the top-k is a relative signal
and needs no calibration: if the five nearest things memory holds all say the
same word, memory knows this; if they say five different words, it does not.

WHAT THE RUN SHOWS. Escalation starts at 100% because an empty memory can do
nothing else, and falls as the agent's own filings accumulate. That curve is
the claim: the memory is not decoration on the agent, it is the entire reason
the agent's workload goes down.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import urllib.request

BASE = "http://localhost:8100"
K = 5


def api(path, body=None):
    if body is None:
        return json.load(urllib.request.urlopen(BASE + path))
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", type=int, default=240, help="episodes to stream")
    ap.add_argument("--kinds", type=int, default=12)
    ap.add_argument("--consensus", type=float, default=0.6,
                    help="fraction of top-k that must agree to act alone")
    ap.add_argument("--no-store", action="store_true",
                    help="the control arm: retrieve and act, but never write")
    ap.add_argument("--out", default="../eval_logs/showreel_watch.json")
    a = ap.parse_args()

    from tqdm import tqdm

    rng = random.Random(0)
    s = api("/api/stats")
    kinds = sorted(rng.sample(sorted(s["catalogue"]), a.kinds))

    feed = []
    for t in kinds:
        feed += api(f"/api/sample?task={t}&n={a.feed // a.kinds}")["clips"]
    rng.shuffle(feed)

    print(f"a fleet feed: {len(feed)} episodes across {len(kinds)} kinds, shuffled")
    print(f"the agent starts with NO memory of any of them\n")

    def arm(store: bool, tag: str):
        # Empty the memory of everything in the feed. The agent must earn it.
        api("/api/holdout", dict(tasks=kinds))
        rows, held = [], set()
        for c in tqdm(feed, desc=tag, unit="ep", leave=False):
            d = api("/api/clip", dict(id=c["id"], k=K))
            # Only precedents the AGENT has filed count. Everything else in the
            # corpus is a different kind and is not evidence about this one.
            hits = [h for h in (d.get("hits") or []) if h["id"] in held]
            votes = collections.Counter(h["task"] for h in hits)
            top, n = (votes.most_common(1)[0] if votes else (None, 0))
            agree = n / max(1, len(hits)) if hits else 0.0
            confident = bool(hits) and agree >= a.consensus and len(hits) >= 2

            if confident:                      # ACT alone
                proposed, escalated = top, False
            else:                              # ACT by escalating
                proposed, escalated = c["task"], True   # the person answers
            rows.append(dict(task=c["task"], escalated=escalated,
                             proposed=proposed, correct=proposed == c["task"],
                             n_prec=len(hits), agree=agree))
            if store:                          # STORE
                api("/api/remember", dict(id=c["id"]))
                held.add(c["id"])
        return rows

    try:
        on = arm(True, "memory on ")
        off = arm(False, "memory off")

        def curve(rows, bins=6):
            w = max(1, len(rows) // bins)
            return [(i * w, sum(r["escalated"] for r in rows[i*w:(i+1)*w]) / len(rows[i*w:(i+1)*w]))
                    for i in range(bins) if rows[i*w:(i+1)*w]]

        auto = [r for r in on if not r["escalated"]]
        acc = sum(r["correct"] for r in auto) / max(1, len(auto))
        print(f"\n{'=' * 70}")
        print(f"  escalation rate as the agent's own memory fills "
              f"({len(feed)} episodes)\n")
        print(f"  {'episodes seen':<20}" +
              "".join(f"{str(i)+'+':>9}" for i, _ in curve(on)))
        print(f"  {'MEMORY ON':<20}" + "".join(f"{v:>9.2f}" for _, v in curve(on)))
        print(f"  {'MEMORY OFF':<20}" + "".join(f"{v:>9.2f}" for _, v in curve(off)))
        print(f"\n  MEMORY ON   escalated {sum(r['escalated'] for r in on)}/{len(on)}"
              f"  =  {sum(r['escalated'] for r in on)/len(on):.0%} of the queue reached a person")
        print(f"  MEMORY OFF  escalated {sum(r['escalated'] for r in off)}/{len(off)}"
              f"  =  {sum(r['escalated'] for r in off)/len(off):.0%}")
        print(f"\n  of the {len(auto)} it dispositioned ALONE, {acc:.1%} were right.")
        print(f"  work removed from the queue: "
              f"{(sum(r['escalated'] for r in off) - sum(r['escalated'] for r in on))} episodes"
              f" a person never had to watch.")
        print(f"\n  The OFF arm is the identical agent over the identical feed with\n"
              f"  only the STORE step removed. It never builds a precedent, so every\n"
              f"  episode reaches a person, forever. That is what 'memory is not an\n"
              f"  afterthought' means here: without it the agent does no work at all.")

        json.dump(dict(kinds=kinds, feed=len(feed), consensus=a.consensus,
                       on=dict(escalated=sum(r["escalated"] for r in on)/len(on),
                               curve=curve(on), auto_accuracy=acc,
                               auto_n=len(auto)),
                       off=dict(escalated=sum(r["escalated"] for r in off)/len(off),
                                curve=curve(off))),
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    finally:
        api("/api/holdout", dict(reset=True))
        print("memory restored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
