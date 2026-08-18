#!/usr/bin/env python3
"""Run the agent over a fleet feed, sweep the one knob, and correct a mistake.

    .venv-libero/bin/python showreel/run_agent.py

Three things, in order, because each depends on the last:

  1. COLD START. Empty memory, N episodes, memory-on vs memory-off. The off arm
     is the identical agent with only the write removed.
  2. THE CURVE. The consensus threshold is the single knob between "acts often
     and is sometimes wrong" and "acts rarely and is always right". One point on
     it is a claim; the curve is the honest picture, and an operator picks the
     point from their own cost of being wrong.
  3. THE CASCADE. Overturn one filing and watch every filing that leaned on it
     reopen, transitively, in one transaction.

The "human" is the corpus's own folder name: the thing a reviewer would type.
It is never read by retrieval and never stored on a clip; it enters memory only
as the answer to an escalation, which is exactly how a real reviewer's answer
would.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent
import db

TRUTH: dict[str, str] = {}


def human(rec_id: str) -> str:
    """The reviewer. Answers an escalation, and nothing else."""
    return TRUTH[rec_id]


def load_feed(kinds: int, per: int, seed: int = 0) -> list[dict]:
    rows = db.q("SELECT DISTINCT task FROM moments WHERE task IS NOT NULL "
                "ORDER BY task")
    pick = random.Random(seed).sample([r["task"] for r in rows], kinds)
    feed = []
    for t in pick:
        feed += db.q("SELECT rec_id, task FROM moments WHERE task=%s "
                     "ORDER BY rec_id LIMIT %s", (t, per))
    random.Random(seed).shuffle(feed)
    for f in feed:
        TRUTH[f["rec_id"]] = f["task"]
    return feed


def one_pass(feed, store: bool, consensus: float, label: str, quiet=False):
    agent.reset()
    agent.enqueue([f["rec_id"] for f in feed])
    rows = []
    from tqdm import tqdm
    for _ in tqdm(range(len(feed)), desc=label, unit="ep", leave=False,
                  disable=quiet):
        item = agent.claim("worker-1")
        if item is None:
            break
        if store:
            r = agent.handle(item, answer=human, consensus=consensus)
        else:
            # THE CONTROL: it retrieves and it decides, it just never writes.
            d = agent.decide(item["rec_id"], consensus=consensus)
            db.q("UPDATE inbox SET state='escalated' WHERE item_id=%s",
                 (item["item_id"],))
            r = dict(escalated=not d["confident"], disposition=d["disposition"])
        truth = TRUTH[item["rec_id"]]
        rows.append(dict(escalated=bool(r["escalated"]),
                         correct=(not r["escalated"]) and r["disposition"] == truth))
    return rows


def summarise(rows):
    esc = sum(r["escalated"] for r in rows)
    auto = [r for r in rows if not r["escalated"]]
    return dict(n=len(rows), escalated=esc / max(1, len(rows)),
                auto_n=len(auto),
                auto_acc=sum(r["correct"] for r in auto) / max(1, len(auto)))


def curve(rows, bins=6):
    w = max(1, len(rows) // bins)
    return [round(sum(r["escalated"] for r in rows[i*w:(i+1)*w]) /
                  len(rows[i*w:(i+1)*w]), 2)
            for i in range(bins) if rows[i*w:(i+1)*w]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", type=int, default=12)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--out", default="../eval_logs/precedent.json")
    a = ap.parse_args()

    db.apply_schema()
    feed = load_feed(a.kinds, a.per)
    print(f"fleet feed: {len(feed)} episodes, {a.kinds} kinds, shuffled")
    print(f"the agent starts with an empty memory and no vocabulary\n")

    # ---- 1. cold start --------------------------------------------------
    on = one_pass(feed, True, agent.CONSENSUS, "memory on ")
    off = one_pass(feed, False, agent.CONSENSUS, "memory off")
    s_on, s_off = summarise(on), summarise(off)
    print("1. COLD START: escalation as the agent's own filings accumulate\n")
    print(f"   {'episodes seen':<16}" + "".join(f"{i*(len(on)//6):>8}"
                                                for i in range(6)))
    print(f"   {'MEMORY ON':<16}" + "".join(f"{v:>8.2f}" for v in curve(on)))
    print(f"   {'MEMORY OFF':<16}" + "".join(f"{v:>8.2f}" for v in curve(off)))
    print(f"\n   ON  {s_on['escalated']:.0%} reached a person   "
          f"({s_on['auto_n']} filed alone, {s_on['auto_acc']:.1%} right)")
    print(f"   OFF {s_off['escalated']:.0%} reached a person   "
          f": identical agent, write removed\n")

    # ---- 2. the curve ---------------------------------------------------
    print("2. THE KNOB: consensus required before the agent acts alone\n")
    print(f"   {'consensus':>10}{'escalated':>12}{'filed alone':>13}"
          f"{'of those, right':>17}")
    sweep = []
    for c in (0.4, 0.5, 0.6, 0.8, 1.0):
        r = summarise(one_pass(feed, True, c, f"consensus {c}", quiet=True))
        sweep.append(dict(consensus=c, **r))
        print(f"   {c:>10.1f}{r['escalated']:>12.0%}{r['auto_n']:>13}"
              f"{r['auto_acc']:>17.1%}")
    print("\n   An operator picks the point from their own cost of being wrong;\n"
          "   quoting one of these rows and hiding the rest would be dishonest.\n")

    # ---- 3. the cascade -------------------------------------------------
    one_pass(feed, True, agent.CONSENSUS, "rebuild", quiet=True)
    print("3. THE CASCADE: one person overturns one filing\n")
    seed = db.q("""SELECT f.filing_id, f.disposition,
                          (SELECT count(*) FROM filing_precedents p
                            WHERE p.precedent_id = f.filing_id) AS leaned_on
                     FROM filings f WHERE f.fleet=%s AND f.source='human'
                    ORDER BY leaned_on DESC LIMIT 1""", (agent.FLEET,))[0]
    before = agent.stats()
    out = agent.cascade(seed["filing_id"], "MISFILED-corrected",
                        note="reviewer overturned the original call")
    after = agent.stats()
    print(f"   overturned 1 filing that {seed['leaned_on']} others cited directly")
    print(f"   reopened {out['reopened']} filings transitively, in one transaction")
    print(f"   live filings {before['filings']} -> {after['filings']}, "
          f"queue re-armed for review")
    print(f"\n   This is the part a vector store cannot do. Similarity finds what\n"
          f"   looks alike; only the precedent graph knows which decisions\n"
          f"   inherited a mistake.\n")

    json.dump(dict(feed=len(feed), kinds=a.kinds, cold_start=dict(
        on=s_on, off=s_off, curve_on=curve(on), curve_off=curve(off)),
        sweep=sweep, cascade=dict(reopened=out["reopened"],
                                  direct=int(seed["leaned_on"]))),
        open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print(f"memory now: {json.dumps(agent.stats())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
