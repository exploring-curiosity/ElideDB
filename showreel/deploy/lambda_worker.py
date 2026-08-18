"""The agent as a Lambda. Drains the queue, then stops.

Deployed as a scheduled function: EventBridge fires it, it claims and
dispositions whatever is waiting, and it exits. Nothing is long-running, which
is why this costs nothing. Lambda's always-free tier is 1M requests and 400k
GB-seconds a month, and an agent that wakes on a schedule fits inside it with
room to spare.

WHY LAMBDA AND NOT EC2, on a new AWS account. Accounts created after 2025-07-15
get $200 of credits instead of 12 months of free EC2, so a t3.micro left running
quietly eats the budget while doing nothing most of the time. Lambda and S3 are
always-free within limits and scale to zero, as does CockroachDB Basic. The
whole system therefore idles at $0 rather than at "a small amount per hour".

WHY IN-REGION MATTERS, measured rather than assumed: a vector query from a
laptop to the us-east-2 cluster takes ~706 ms, almost all of it network. The
same query from a Lambda in us-east-2 is a local hop. Retrieval latency here is
a deployment property, not a database one.

WHAT DOES NOT RUN HERE. The V-JEPA 2 / SigLIP 2 encoder is a batch job over new
video and has no place in a request path: it runs offline and writes vectors.
This function only reads them, which is why the package is a few megabytes and
not a few gigabytes.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import db

MAX_PER_INVOCATION = int(os.environ.get("PRECEDENT_BATCH", "25"))


def handler(event, context):
    """One drain. Returns what it did, which is what CloudWatch will show."""
    fleet = (event or {}).get("fleet", agent.FLEET)
    budget = int((event or {}).get("max", MAX_PER_INVOCATION))

    # Recover anything a previous invocation claimed and died holding. A Lambda
    # that times out mid-episode leaves state='working' forever otherwise, and
    # nothing errors: the episode is simply never dispositioned again.
    recovered = agent.reclaim(fleet)

    done, escalated = 0, 0
    while done < budget:
        # Leave headroom: being killed by the timeout is what creates the stuck
        # rows above, so stop early rather than get cut off mid-write.
        if context is not None and context.get_remaining_time_in_millis() < 8000:
            break
        item = agent.claim(f"lambda-{getattr(context, 'aws_request_id', 'local')[:8]}",
                           fleet)
        if item is None:
            break
        # No `answer` callback: a Lambda has no human attached. Episodes with no
        # agreeing precedent are left escalated for the review queue, which is
        # the correct behaviour and not a degraded one.
        r = agent.handle(item, answer=None, fleet=fleet)
        done += 1
        escalated += bool(r.get("escalated"))
        if r.get("escalated") and not r.get("filed"):
            # handle() declined to write without a human. Park it for review.
            db.q("UPDATE inbox SET state='escalated' WHERE item_id=%s",
                 (item["item_id"],))

    out = dict(fleet=fleet, dispositioned=done, escalated=escalated,
               recovered=recovered, health=agent.health(fleet),
               stats=agent.stats(fleet))
    print(json.dumps(out))          # structured line -> CloudWatch Logs Insights
    return out


if __name__ == "__main__":
    print(json.dumps(handler({}, None), indent=1))
