"""Motion channel: one delta-appearance vector per event -> `motion_vectors`.

WHY THIS FILE EXISTS
--------------------
`elidedb.motion.index_motion` had NO CALLER. Grep for it and the only hit
is its own `def`. So the motion channel was live code in the read path -
`scenario.py` scores it, and the transition anchor is built on it - over
a table that nothing ever wrote, in a store where it therefore never
existed. The anchor guards for the missing table and returns `{}`, which
is why this was silent: the channel did not fail, it simply never
contributed.

That is a measurable loss, not a tidiness point. The transition anchor is
what took q04 from 0.72 to 0.92, and it exists because TEXT ERASES
DIRECTION - "open the drawer" and "close the drawer" sit at cosine 0.957
in every text-image space we measured, so no appearance channel can tell
them apart. Motion is the channel that can. Running the six pretrained
channels without it measures a system missing the one signal built for
direction.

It is also the cheapest channel by a wide margin: no model at all. A
motion vector is the difference between the mean appearance vector at the
END of an event span and at its START, normalised - so it rides entirely
on `frame_vectors` the write already produced. Seconds, not minutes,
against tens of minutes for any pretrained encoder.

The stale default was the other half of the bug: `index_motion` defaults
to `events_table="context_events"`, a name no current store carries. Even
a caller that existed would have raised. This passes `events` explicitly.

    python scripts/motion_ingest.py lake/fresh_bench
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                     # noqa: E402
from elidedb.motion import index_motion                       # noqa: E402


def main():
    store = Store.open(sys.argv[1] if len(sys.argv) > 1
                       else "lake/fresh_bench")
    need = {"events", "frame_vectors"}
    missing = need - set(store.tables())
    if missing:
        raise SystemExit(
            f"motion needs {sorted(need)}; {store.dir} is missing "
            f"{sorted(missing)}. It is derived from the write, so run the "
            f"write first - there is no model to fall back on.")

    n_ev = len(store.table("events").scan())
    print(f"motion: {n_ev:,} events over "
          f"{len(store.table('frame_vectors').scan()):,} frame vectors",
          flush=True)
    t0 = time.time()
    index_motion(store, events_table="events")
    secs = time.time() - t0

    # THE ARTIFACT IS THE TEST. index_motion returns {"events": 0} rather
    # than raising when every span is too short or too still, so an exit
    # code proves nothing here.
    rows = (len(store.table("motion_vectors").scan())
            if "motion_vectors" in store.tables() else 0)
    print(json.dumps({"rows": rows, "events": n_ev,
                      "seconds": round(secs, 1)}))
    if not rows:
        raise SystemExit("motion wrote no rows - every event span was "
                         "either shorter than 6 frames or had no "
                         "appearance change")


if __name__ == "__main__":
    main()
