#!/usr/bin/env python3
"""Load RelMo's already-encoded corpus into Postgres, once.

    .venv-libero/bin/python showreel/ingest.py

Runs in the interpreter that has psycopg2. It does NOT need transformers 4.57:
loading a RelMo store reads npz files off disk, and only the ENCODER needs the
newer library. That lives in sidecar.py.

Nothing is encoded here. RelMo has already seen this video and its descriptor
traces are on disk; this reads them and writes two vectors per recording into a
table so the database can do the first stage of retrieval:

    appearance  concat(whitened pooled V-JEPA, whitened pooled SigLIP2)/sqrt(2)
                RelMo's own stage-1 prefilter. The database's cosine over this
                column IS that prefilter, not an approximation of it.
    motion      the per-channel temporal standard deviation of the SigLIP2
                trace, L2'd. What MOVED rather than what was there. Measured on
                a separate corpus at 0.748 behaviour-matching against 0.649 for
                the pooled mean.

`task` is the folder name RoboCasa filed the episode under. It is display and
grading only: the retrieval path never reads it, and it is not what any query
matches against. It exists so a viewer can see a green tick instead of taking
the ranking on faith, and so precision can be computed live rather than quoted
from a README.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

DSN = os.environ.get("SHOWREEL_DSN", "postgresql://localhost:5433/brigade")
STORE = "rcasa"
# The two datasets inside the rcasa store that ship playable video. The other
# two are the same demonstrations at a different bitrate with no files kept.
WITH_VIDEO = ("rcasa_atomic_full", "rcasa_composite_full")

SCHEMA = """
CREATE TABLE IF NOT EXISTS moments (
    rec_id      TEXT PRIMARY KEY,
    store       TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    -- RoboCasa's own folder name. GRADING AND DISPLAY ONLY: no query is matched
    -- against it, and removing this column would not change a single ranking.
    task        TEXT,
    video       TEXT NOT NULL,
    seconds     FLOAT,
    steps       INT,
    -- stage 1, RelMo's prefilter
    appearance  VECTOR(512),
    -- the same trace reduced to what moved
    motion      VECTOR(768),
    -- THE TEXT ARM, and it must be a fair one or the comparison is worthless.
    -- Raw mean-pooled SigLIP2 image embedding, L2'd: the space SigLIP2's TEXT
    -- tower was trained to share. Cosine between a sentence and this column is
    -- exactly how zero-shot text-to-video retrieval is done everywhere, done
    -- properly. Whitened columns are deliberately NOT used here: they live in a
    -- basis the text tower knows nothing about, and comparing against them
    -- would be building a straw man and then knocking it down.
    siglip      VECTOR(768)
);
"""


def task_of(path: str) -> str | None:
    m = re.search(r"/(?:atomic|composite)/([^/]+)/", path or "")
    return m.group(1) if m else None


def main() -> int:
    from tqdm import tqdm

    from relmo import registry as R
    from relmo.vjstore import Store

    print("loading the RelMo store (~85s, no encoding) ...", flush=True)
    st = Store(STORE)
    print(f"store {STORE!r}: {len(st.ids)} recordings", flush=True)

    meta: dict[str, dict] = {}
    for ds in WITH_VIDEO:
        for e in R.read_manifest(ds)["episodes"]:
            v = e.get("video") or ""
            if v and os.path.exists(v):
                fps = float(e.get("fps") or 20)
                meta[e["id"]] = dict(dataset=ds, video=v, task=task_of(v),
                                     seconds=float(e.get("T", 0)) / max(fps, 1))
    have = [i for i in st.ids if i in meta]
    print(f"{len(have)} of them have playable video on disk", flush=True)

    idx = {i: n for n, i in enumerate(st.ids)}
    rows = []
    for rid in tqdm(have, desc="reduce", unit="rec"):
        n = idx[rid]
        app = np.concatenate([st.pf[n], st.ps[n]]) / np.sqrt(2.0)
        sig = st.raw[rid][1].astype(np.float32)
        mot = sig.std(0)
        mot = mot / (np.linalg.norm(mot) + 1e-9)
        raw = sig.mean(0)
        raw = raw / (np.linalg.norm(raw) + 1e-9)
        m = meta[rid]
        vec = lambda a: "[" + ",".join(f"{x:.7f}" for x in a) + "]"
        rows.append((rid, STORE, m["dataset"], m["task"], m["video"],
                     round(m["seconds"], 2), int(sig.shape[0]),
                     vec(app), vec(mot), vec(raw)))

    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(SCHEMA)
    execute_batch(cur, """
        INSERT INTO moments (rec_id, store, dataset, task, video, seconds, steps,
                             appearance, motion, siglip)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (rec_id) DO UPDATE SET
            appearance = EXCLUDED.appearance, motion = EXCLUDED.motion,
            siglip = EXCLUDED.siglip,
            task = EXCLUDED.task, video = EXCLUDED.video""", rows, page_size=200)
    for col in ("appearance", "motion", "siglip"):
        cur.execute(f"CREATE INDEX IF NOT EXISTS moments_{col}_hnsw ON moments "
                    f"USING hnsw ({col} vector_cosine_ops)")
    cur.execute("SELECT count(*), count(DISTINCT task) FROM moments")
    n, k = cur.fetchone()
    print(f"\n{n} moments indexed across {k} tasks, both vector columns HNSW")
    cur.execute("SELECT task, count(*) c FROM moments GROUP BY task "
                "ORDER BY c DESC LIMIT 5")
    for t, c in cur.fetchall():
        print(f"  {t:<32}{c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
