"""Build the context index for the Bridge store.

Window geometry is chosen from the data, not copied from the street corpus:
Bridge episodes run ~4.6 s at 5 Hz, so a 4 s window sits inside a single
episode almost always, and a 2 s stride means every episode is covered by two
or three overlapping windows. Nothing here knows where episode boundaries are
— the index is a plain sliding window over the file timeline, which is the
point: retrieval has to find the clip without being told where clips start.

Only every Nth window is captioned. That is the regime the tower exists for
and the one the street corpus was too small to exercise: most windows get
their context vector from the student, and `scripts/bench_bridge.py` grades
the result against human task labels the database has never seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb import context as C                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="lake/bridge")
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--stride-s", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=3,
                    help="caption every Nth window; the rest are estimated")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--skip-captions", action="store_true")
    ap.add_argument("--prompt", default="manipulation",
                    help="caption prompt preset: scene | manipulation")
    args = ap.parse_args()

    db = Store.open(args.store)
    t0 = time.time()

    windows = C.plan_windows(db, args.window_s, args.stride_s, min_frames=6)
    print(f"{len(windows)} windows ({args.window_s}s / {args.stride_s}s stride)",
          flush=True)

    if not args.skip_captions:
        r = C.caption_windows(db, windows, every=args.every, verbose=True,
                              prompt=args.prompt)
        print("captions:", r, flush=True)

    _, _, m = C.train_context(db, window_s=args.window_s,
                              stride_s=args.stride_s, epochs=args.epochs)
    print("train:", json.dumps({k: v for k, v in m.items()
                                if isinstance(v, (int, float))}), flush=True)

    _, rec = C.prune_context(db)
    print("prune:", rec.get("selected", {}).get("stage"),
          rec.get("after"), flush=True)

    print("build:", C.build_context(db), flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
