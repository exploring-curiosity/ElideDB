#!/usr/bin/env python3
"""The number on the screen, over the whole corpus instead of one lucky query.

    .venv-libero/bin/python showreel/bench.py          # server must be running

A demo that shows one query proves nothing — the first one might be the one
that worked. This runs BOTH arms over the whole task catalogue and reports the
aggregate, so what the page claims per query is checkable in bulk.

    TYPE IT   one text query per task, built from the task's own name
    SHOW IT   random clips as queries, graded against their own task

Both are precision@k against the folder RoboCasa filed the episode under.
Chance is printed per task because the classes are balanced but not equal in
size, and a 90-of-3402 class cannot be compared to a 2.6% floor by eye.

The text arm is a FAIR one: SigLIP2's text tower against raw mean-pooled
SigLIP2 image embeddings, which is how zero-shot text-to-video retrieval is
actually done. It is not handicapped to lose. It loses.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request

BASE = "http://localhost:8100"
K = 8


def api(path: str, body=None):
    if body is None:
        return json.load(urllib.request.urlopen(BASE + path))
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def phrase(task: str) -> str:
    """CamelCase task name -> something a person would type.

    Derived mechanically from the name so it cannot be tuned per task: the
    text arm gets the most direct possible description of what it should find.
    """
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower().split()
    return " ".join(words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=60, help="clip queries to run")
    ap.add_argument("--out", default="../eval_logs/showreel_bench.json")
    a = ap.parse_args()

    from tqdm import tqdm

    s = api("/api/stats")
    corpus, counts = s["moments"], s["counts"]
    tasks = [t for t in s["catalogue"] if counts.get(t, 0) >= 10]
    print(f"{corpus} clips, {len(tasks)} tasks with >=10 examples, k={K}\n")

    # ---- TYPE IT ----------------------------------------------------------
    text_rows = []
    for t in tqdm(tasks, desc="type it", unit="q"):
        d = api("/api/text", dict(q=phrase(t), k=K, expect=t))
        text_rows.append(dict(task=t, phrase=phrase(t), precision=d["precision"],
                              chance=counts[t] / corpus))

    # ---- SHOW IT ----------------------------------------------------------
    random.seed(0)
    clips = api(f"/api/sample?n={a.clips}")["clips"]
    clip_rows = []
    for c in tqdm(clips, desc="show it", unit="q"):
        d = api("/api/clip", dict(id=c["id"], k=K))
        clip_rows.append(dict(task=d["expect"], precision=d["precision"],
                              chance=counts.get(d["expect"], 0) / corpus,
                              stage1_ms=d["stage1_ms"], stage2_ms=d["stage2_ms"],
                              elided=d["elided"]))

    def mean(rows, key):
        return sum(r[key] for r in rows) / max(1, len(rows))

    tp, tc = mean(text_rows, "precision"), mean(text_rows, "chance")
    cp, cc = mean(clip_rows, "precision"), mean(clip_rows, "chance")
    print(f"\n{'=' * 68}")
    print(f"  {'arm':<26}{'P@' + str(K):>8}{'chance':>10}{'x chance':>11}")
    print(f"  {'TYPE IT  (text query)':<26}{tp:>8.3f}{tc:>10.3f}{tp/max(tc,1e-9):>10.1f}x")
    print(f"  {'SHOW IT  (query by clip)':<26}{cp:>8.3f}{cc:>10.3f}{cp/max(cc,1e-9):>10.1f}x")
    print(f"\n  stage 1 {mean(clip_rows,'stage1_ms'):.1f} ms · "
          f"stage 2 {mean(clip_rows,'stage2_ms'):.1f} ms · "
          f"{mean(clip_rows,'elided')*100:.1f}% of the corpus never scored")

    worst = sorted(text_rows, key=lambda r: r["precision"])[:6]
    print(f"\n  where typing words fails hardest:")
    for r in worst:
        print(f"    {r['phrase']:<34}{r['precision']:>6.3f}  (chance {r['chance']:.3f})")
    below = [r for r in text_rows if r["precision"] <= r["chance"]]
    print(f"\n  {len(below)}/{len(text_rows)} text queries are AT OR BELOW chance.")

    json.dump(dict(corpus=corpus, k=K, text=text_rows, clip=clip_rows,
                   summary=dict(text_p=tp, text_chance=tc, clip_p=cp,
                                clip_chance=cc)),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
