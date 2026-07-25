"""Build lake/bench — the STABLE BENCHMARK STORE.

Contains exactly the episodes adjudicated in the frozen truthset
(eval/truthsets/bridge4h.parquet), so every clip the engine can
possibly return has a human verdict: the bench grades itself, no eyes
per iteration, no ungraded holes. All channel tables are FILTERED from
bridge4h (identical vectors — the bench measures retrieval, not
ingest); media is symlinked, raw data untouched.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402

SRC = Path("lake/bridge4h")
DST = Path("lake/bench")

# episode-keyed tables: rows carry the episode's exact (stream, ts)
EPISODE_TABLES = ["episodes", "pe_vectors", "action_probs",
                  "motion_vectors", "xclip_vectors", "vjepa_vectors"]
# frame-keyed tables: rows fall inside an episode's [t0, t1]
FRAME_TABLES = ["frames", "frame_vectors", "embeddings",
                "object_vectors"]


def main():
    truth = pq.read_table("eval/truthsets/bridge4h.parquet").to_pydict()
    cand = pq.read_table("eval/truthsets/candidates.parquet").to_pydict()
    t1_of = {(s, int(a)): int(b) for s, a, b in
             zip(cand["stream"], cand["t0"], cand["t1"])}
    eps = sorted({(s, int(a), t1_of[(s, int(a))])
                  for s, a in zip(truth["stream"], truth["t0"])})
    print(f"{len(eps)} graded episodes")
    ep_keys = {(s, a) for s, a, _ in eps}
    spans = {}
    for s, a, b in eps:
        spans.setdefault(s, []).append((a, b))
    for s in spans:
        spans[s].sort()

    if DST.exists():
        shutil.rmtree(DST)
    db = Store.create(DST, "bench-truthset")
    (DST / "media").mkdir(exist_ok=True)
    for f in (SRC / "media").iterdir():
        (DST / "media" / f.name).symlink_to(f.resolve())
    shutil.copy(SRC / "_channel_weights.json",
                DST / "_channel_weights.json")

    src = Store.open(SRC)

    def kindmeta(name):
        mp = SRC / "tables" / name / "_meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            return m.get("kind", "timeseries"), m.get("meta") or {}
        return "timeseries", {}

    for name in EPISODE_TABLES:
        tbl = src.table(name).scan()
        ss = tbl.column("stream").to_pylist()
        ts = [int(v) for v in tbl.column("ts").to_pylist()]
        mask = np.array([(s, a) in ep_keys for s, a in zip(ss, ts)])
        sub = tbl.filter(pc.field("ts").isin(
            [t for t, m in zip(ts, mask) if m])) \
            if False else tbl.take(np.where(mask)[0])
        kind, meta = kindmeta(name)
        db.table(name).append(sub, kind=kind, meta=meta)
        print(f"{name:16s} {len(sub):7d}/{len(tbl)} rows")

    for name in FRAME_TABLES:
        tbl = src.table(name).scan()
        ss = np.asarray(tbl.column("stream").to_pylist())
        ts = np.asarray([int(v) for v in
                         tbl.column("ts").to_pylist()])
        mask = np.zeros(len(tbl), bool)
        for s, sp in spans.items():
            rows = np.where(ss == s)[0]
            if len(rows) == 0:
                continue
            t = ts[rows]
            starts = np.array([a for a, _ in sp])
            ends = np.array([b for _, b in sp])
            j = np.searchsorted(starts, t, side="right") - 1
            ok = (j >= 0) & (t <= ends[np.clip(j, 0, len(ends) - 1)])
            mask[rows[ok]] = True
        sub = tbl.take(np.where(mask)[0])
        kind, meta = kindmeta(name)
        db.table(name).append(sub, kind=kind, meta=meta)
        print(f"{name:16s} {len(sub):7d}/{len(tbl)} rows")

    print(f"bench store at {DST}")


if __name__ == "__main__":
    main()
