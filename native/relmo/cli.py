"""ElideDB command line.

    python -m relmo.cli stores
    python -m relmo.cli add     kitchen /videos/robot_runs
    python -m relmo.cli query   kitchen /clips/spill.mp4 --top 10
    python -m relmo.cli like    kitchen <recording-id>
    python -m relmo.cli stats   kitchen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo.api import Memory  # noqa: E402


def _print(hits, elapsed):
    if not hits:
        print("no results")
        return
    w = max((len(h.label) for h in hits), default=20)
    print(f"\n{'#':>2s}  {'match':>6s}  {'span':>15s}  source")
    print("-" * (32 + w))
    for i, h in enumerate(hits, 1):
        print(f"{i:2d}  {h.score*100:5.1f}%  {h.start:6.1f}-{h.end:6.1f}s  "
              f"{h.label}")
    print(f"\n{len(hits)} hits in {elapsed*1000:.0f} ms")


def main():
    ap = argparse.ArgumentParser(prog="elidedb",
                                 description="video memory: when did "
                                             "something like this happen?")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stores", help="list stores and their sizes")

    a = sub.add_parser("add", help="ingest video into a store")
    a.add_argument("store")
    a.add_argument("source", help="video file or directory")
    a.add_argument("--fps", type=float, default=None)

    q = sub.add_parser("query", help="find moments like a clip")
    q.add_argument("store")
    q.add_argument("clip")
    q.add_argument("--start", type=float, default=None)
    q.add_argument("--end", type=float, default=None)
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--exact", action="store_true",
                   help="exact scan instead of the fast path (~40x slower, "
                        "P@10 0.765 vs 0.752)")
    q.add_argument("--json", action="store_true")

    l = sub.add_parser("like", help="find moments like one already stored")
    l.add_argument("store")
    l.add_argument("recording_id")
    l.add_argument("--top", type=int, default=10)
    l.add_argument("--exact", action="store_true")
    l.add_argument("--json", action="store_true")

    s = sub.add_parser("stats", help="store summary")
    s.add_argument("store")

    args = ap.parse_args()

    if args.cmd == "stores":
        st = Memory.list()
        if not st:
            print("no stores yet - create one with:  elidedb add <name> <dir>")
            return
        print(f"{'store':24s} {'recordings':>10s}")
        print("-" * 36)
        for k, v in st.items():
            print(f"{k:24s} {v:10d}")
        return

    mem = Memory.open(args.store)

    if args.cmd == "add":
        n = mem.add(args.source, fps=args.fps)
        print(f"\nstore {args.store!r} now holds {n} recordings")
    elif args.cmd == "stats":
        print(json.dumps(mem.stats(), indent=1))
    elif args.cmd in ("query", "like"):
        t0 = time.time()
        if args.cmd == "query":
            hits = mem.query(args.clip, args.start, args.end,
                             top_k=args.top, fast=not args.exact)
        else:
            hits = mem.query_recording(args.recording_id, top_k=args.top,
                                       fast=not args.exact)
        el = time.time() - t0
        if args.json:
            print(json.dumps([h.__dict__ for h in hits], indent=1))
        else:
            _print(hits, el)


if __name__ == "__main__":
    main()
