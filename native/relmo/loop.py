"""The autonomous loop, as two concurrent services.

    python -m relmo.loop --role producer   # sim -> episodes -> tracks
    python -m relmo.loop --role consumer   # train -> eval -> promote

The producer keeps extending an immutable versioned dataset. The
consumer rescans it while training, so data generated at 03:00 is
being learned from at 03:05. Neither blocks the other, and either can
be killed and restarted without losing work: datasets are append-only
with atomic renames, training resumes from its last checkpoint, and
every event lands in the append-only ledger.

The robot-arm corpus is never touched by either service except by the
evaluator, which reads its VIDEO ONLY and uses its labels solely to
grade output after the fact.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import evaluate as EV  # noqa: E402
from relmo import generate as GEN  # noqa: E402
from relmo import registry as R  # noqa: E402
from relmo import tracks as TR  # noqa: E402
from relmo import train as TRN  # noqa: E402


def producer(dataset, hours, batch=200, cap=20000):
    t_end = time.time() + hours * 3600
    R.log("producer_start", dataset=dataset, batch=batch, cap=cap)
    while time.time() < t_end:
        try:
            man = R.read_manifest(dataset)
            have = man.get("n_episodes", 0)
            if have >= cap:
                time.sleep(300)
                continue
            GEN.generate(dataset, target=min(have + batch, cap))
            TR.build(dataset)
            R.log("producer_cycle", dataset=dataset,
                  episodes=R.read_manifest(dataset)["n_episodes"],
                  tracks=len(list((R.TRACKS / dataset).glob("ep*.npz"))))
        except Exception:
            R.log("producer_error", error=traceback.format_exc()[-800:])
            time.sleep(30)
    R.log("producer_end", dataset=dataset)


def consumer(dataset, hours, chunk=4000, run_id="relmo_v1"):
    t_end = time.time() + hours * 3600
    state = dict(step=0, best=-1.0, cycle=0)
    R.log("consumer_start", dataset=dataset, run=run_id, chunk=chunk)
    while time.time() < t_end:
        state["cycle"] += 1
        try:
            n = len(list((R.TRACKS / dataset).glob("ep*.npz")))
            if n < 40:
                R.log("consumer_wait", tracks=n)
                time.sleep(120)
                continue
            state["step"] += chunk
            TRN.train(dataset, steps=state["step"], run_id=run_id,
                      bs=6, log_every=500, ckpt_every=chunk)
            rec = EV.evaluate(run_id, tag=f"c{state['cycle']}",
                              max_events=160)
            if rec and rec["yield_at_sup"] > state["best"]:
                state["best"] = rec["yield_at_sup"]
                R.best_pointer(run_id, rec["step"], rec["yield_at_sup"],
                               "yield_at_sup")
                R.log("promote", run=run_id, step=rec["step"],
                      yield_at_sup=rec["yield_at_sup"])
            R.log("consumer_cycle", cycle=state["cycle"],
                  step=state["step"], tracks=n,
                  yield_at_sup=(rec or {}).get("yield_at_sup"),
                  single_group_frac=(rec or {}).get("single_group_frac"),
                  best=state["best"])
        except Exception:
            R.log("consumer_error", error=traceback.format_exc()[-800:])
            time.sleep(30)
    R.log("consumer_end", run=run_id, best=state["best"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["producer", "consumer"],
                    required=True)
    ap.add_argument("--dataset", default="physgen_v1")
    ap.add_argument("--hours", type=float, default=10)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()
    if a.role == "producer":
        producer(a.dataset, a.hours, a.batch)
    else:
        consumer(a.dataset, a.hours, a.chunk)
