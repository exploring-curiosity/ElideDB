#!/usr/bin/env python3
"""Bring the store up to the current schema after a collection run.

    .venv-libero/bin/python brigade/bench/finalize.py

Idempotent. Adds any column the running collector predated, fills the motion
view from traces already on disk (no video is decoded twice, no model runs), and
reports what is actually in the table — counted from the rows, not from a
manifest, because a manifest is a claim and `count(*)` is a fact.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from brigade.memory.store import VideoStore

    store = VideoStore()
    print(f"schema: {store.setup()}")
    print(f"motion view backfilled for {store.backfill_motion()} spans")

    rows = store.db.query(
        """SELECT basis_id,
                  count(*) AS n,
                  count(embedding) AS with_vec,
                  count(motion) AS with_motion,
                  count(trace_path) AS with_trace,
                  coalesce(avg(steps), 0) AS steps,
                  coalesce(sum(n_frames) / nullif(max(fps), 0), 0) AS seconds
           FROM clips WHERE kitchen_id = %s
           GROUP BY basis_id ORDER BY n DESC""", (store.kitchen,))
    print(f"\n  {'basis':<22}{'rows':>6}{'vec':>6}{'motion':>8}{'trace':>7}"
          f"{'steps':>7}{'video':>9}")
    for r in rows:
        print(f"  {str(r['basis_id']):<22}{r['n']:>6}{r['with_vec']:>6}"
              f"{r['with_motion']:>8}{r['with_trace']:>7}"
              f"{float(r['steps']):>7.0f}{float(r['seconds'] or 0):>8.0f}s")
    stale = [r for r in rows if r["with_trace"] == 0]
    if stale:
        print(f"\n  {sum(r['n'] for r in stale)} rows predate the trace column. They keep"
              f"\n  their old basis id, which excludes them from retrieval — the"
              f"\n  namespace working as designed rather than a leak to clean up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
