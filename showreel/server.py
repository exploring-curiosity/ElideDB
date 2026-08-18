#!/usr/bin/env python3
"""Showreel — two ways to search 3,400 videos, side by side, live-graded.

    .venv-libero/bin/python showreel/server.py      # then open :8100

LEFT PANEL is how video search works today: you type words. SigLIP2's text
tower embeds the sentence into the space its image tower shares, and the
database ranks every recording by cosine. This is not a straw man — it is the
standard zero-shot text-to-video method, implemented properly, over the raw
SigLIP2 column rather than a whitened one the text tower knows nothing about.

RIGHT PANEL is RelMo: you hand it a clip. The database prefilters by cosine
over RelMo's pooled vector, then RelMo re-ranks the survivors by DTW over their
full descriptor traces.

Every result is graded against the folder RoboCasa filed the episode under, so
the precision on screen is computed from the query you just ran. The grading
column is never matched against and never read by retrieval — deleting it would
not change one ranking.

Measured on this corpus before any of this was built (relmo.vjtextbench):

    query by TEXT      P@support 0.224   chance 0.221     <- at chance
    query by EXAMPLE   P@support 0.481   chance 0.193

    cos("open the drawer", "close the drawer") = 0.9806
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
DSN = os.environ.get("SHOWREEL_DSN", "postgresql://localhost:5433/brigade")
RELMO_PY = os.environ.get("SHOWREEL_RELMO_PYTHON",
                          str(ROOT.parent / "myenv" / "bin" / "python"))
PREFILTER_M = 48          # candidates stage 1 hands to stage 2


# ------------------------------------------------------------------- sidecar

class Sidecar:
    """RelMo in its own interpreter. Start once, ask many."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.ready = False

    def start(self) -> bool:
        self.proc = subprocess.Popen(
            [RELMO_PY, "-u", str(ROOT / "sidecar.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=str(ROOT.parent))
        self.proc.stdout.readline()
        r = self.rpc(dict(cmd="ping"), timeout=300)
        self.ready = bool(r.get("ok"))
        return self.ready

    def rpc(self, req: dict, timeout: float = 120) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            return dict(error="sidecar not running")
        with self.lock:
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, ValueError) as exc:
                return dict(error=str(exc))
        try:
            return json.loads(line) if line else dict(error="no reply")
        except json.JSONDecodeError:
            return dict(error=f"bad reply: {line[:120]}")


SIDE = Sidecar()


# ------------------------------------------------------------------ database

def q(sql: str, args=()) -> list[dict]:
    con = psycopg2.connect(DSN)
    # AUTOCOMMIT, and this is not a preference. psycopg2 opens a transaction
    # implicitly and this helper closes the connection without committing, so
    # every UPDATE issued through it was silently rolled back. Reads were
    # unaffected, which is what made it invisible: the holdout reported success,
    # changed nothing, and the agent went on retrieving the very clips that were
    # supposed to be absent from its memory.
    con.autocommit = True
    try:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, args)
            return [dict(r) for r in cur.fetchall()] if cur.description else []
    finally:
        con.close()


def vec(a) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in a) + "]"


def hits(rows: list[dict], want: str | None) -> list[dict]:
    """Rows -> what the browser draws, graded."""
    return [dict(id=r["rec_id"], task=r["task"], seconds=round(float(r["seconds"] or 0), 1),
                 score=round(float(r["score"]), 4),
                 correct=(want is not None and r["task"] == want))
            for r in rows]


# ----------------------------------------------------------------------- app

app = FastAPI(title="Showreel")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/stats")
def stats():
    n = q("SELECT count(*) n, count(DISTINCT task) k FROM moments")[0]
    tasks = q("SELECT task, count(*) c FROM moments GROUP BY task ORDER BY task")
    return dict(moments=int(n["n"]), tasks=int(n["k"]), ready=SIDE.ready,
                catalogue=[t["task"] for t in tasks],
                counts={t["task"]: int(t["c"]) for t in tasks})


@app.get("/api/video")
def video(id: str = ""):
    r = q("SELECT video FROM moments WHERE rec_id = %s", (id,))
    if not r or not os.path.exists(r[0]["video"]):
        return JSONResponse(dict(error="no such clip"), status_code=404)
    return FileResponse(r[0]["video"], media_type="video/mp4")


@app.post("/api/text")
def by_text(body: dict):
    """THE WAY IT IS DONE TODAY. Words in, cosine over SigLIP2's shared space."""
    text = (body.get("q") or "").strip()
    k = int(body.get("k", 8))
    want = body.get("expect") or None
    if not text:
        return JSONResponse(dict(error="empty query"), status_code=400)
    t = time.perf_counter()
    r = SIDE.rpc(dict(cmd="text", texts=[text]))
    embed_ms = (time.perf_counter() - t) * 1e3
    if "vecs" not in r:
        return JSONResponse(dict(error=r.get("error", "text tower failed")),
                            status_code=503)
    t = time.perf_counter()
    rows = q("""SELECT rec_id, task, seconds, 1 - (siglip <=> %s) AS score
                FROM moments WHERE siglip IS NOT NULL AND NOT held_out
                ORDER BY siglip <=> %s LIMIT %s""",
             (vec(r["vecs"][0]), vec(r["vecs"][0]), k))
    sql_ms = (time.perf_counter() - t) * 1e3
    h = hits(rows, want)
    return dict(mode="text", query=text, hits=h, embed_ms=round(embed_ms, 1),
                sql_ms=round(sql_ms, 2),
                precision=round(sum(x["correct"] for x in h) / max(1, len(h)), 3))


@app.post("/api/clip")
def by_clip(body: dict):
    """RELMO. A clip in, two stages out: SQL prefilter, then DTW."""
    rid = (body.get("id") or "").strip()
    k = int(body.get("k", 8))
    # WHICH VIEW STAGE 1 RANKS BY, decided by measurement and not by preference.
    # Over 24 random query clips, precision@8 against a 0.026 chance:
    #
    #     appearance  0.786        motion  0.391
    #
    # That is the reverse of what the same two columns measured on a corpus of
    # ten behaviours inside ONE kitchen, where appearance was near-useless
    # (0.649) and motion won (0.748). The rule underneath both results: match on
    # appearance when the corpus varies in SCENE, on motion when it varies only
    # in ACTION. Here 57 tasks span different rooms, appliances and objects.
    view = body.get("view", "appearance")
    want = body.get("expect") or None
    col = "motion" if view == "motion" else "appearance"
    src = q(f"SELECT {col} AS v, task FROM moments WHERE rec_id = %s", (rid,))
    if not src:
        return JSONResponse(dict(error="no such clip"), status_code=404)
    if want is None:
        want = src[0]["task"]

    t = time.perf_counter()
    rows = q(f"""SELECT rec_id, task, seconds, 1 - ({col} <=> %s) AS score
                 FROM moments WHERE {col} IS NOT NULL AND rec_id <> %s
                   AND NOT held_out
                 ORDER BY {col} <=> %s LIMIT %s""",
             (src[0]["v"], rid, src[0]["v"], max(k, PREFILTER_M)))
    stage1_ms = (time.perf_counter() - t) * 1e3

    total = q("SELECT count(*) n FROM moments WHERE NOT held_out")[0]["n"]
    stage2_ms, reranked = 0.0, False
    if SIDE.ready:
        r = SIDE.rpc(dict(cmd="rank", query=rid,
                          cand=[x["rec_id"] for x in rows]))
        sc = r.get("scores") or {}
        if sc:
            for x in rows:
                if x["rec_id"] in sc:
                    x["score"] = sc[x["rec_id"]]
            rows.sort(key=lambda x: -float(x["score"]))
            stage2_ms, reranked = float(r.get("ms", 0.0)), True
    h = hits(rows[:k], want)
    return dict(mode="clip", query=rid, expect=want, hits=h,
                stage1_ms=round(stage1_ms, 2), stage2_ms=round(stage2_ms, 2),
                reranked=reranked, candidates=len(rows), corpus=int(total),
                elided=round(1 - len(rows) / max(1, int(total)), 4),
                precision=round(sum(x["correct"] for x in h) / max(1, len(h)), 3))


@app.get("/api/sample")
def sample(task: str = "", n: int = 1):
    """A clip to start from. Random within a task, so a demo is not cherry-picked."""
    rows = q("""SELECT rec_id, task, seconds FROM moments
                WHERE (%s = '' OR task = %s) ORDER BY random() LIMIT %s""",
             (task, task, n))
    return dict(clips=[dict(id=r["rec_id"], task=r["task"],
                            seconds=round(float(r["seconds"] or 0), 1)) for r in rows])


@app.post("/api/holdout")
def holdout(body: dict):
    """Take whole kinds of moment OUT of the memory, or put them back.

    This is how "the agent has never seen this before" is created honestly:
    the rows are not hidden from the answer, they are absent from the corpus
    being searched. The holdout is CONSTRUCTED with labels — that is the
    experiment design — but nothing the agent decides at run time reads one.
    """
    tasks = list(body.get("tasks") or [])
    if body.get("reset"):
        q("UPDATE moments SET held_out = false")
        return dict(held_out=[], live=q("SELECT count(*) n FROM moments")[0]["n"])
    q("UPDATE moments SET held_out = false")
    if tasks:
        q("UPDATE moments SET held_out = true WHERE task = ANY(%s)", (tasks,))
    live = q("SELECT count(*) n FROM moments WHERE NOT held_out")[0]["n"]
    return dict(held_out=tasks, live=live,
                out=q("SELECT count(*) n FROM moments WHERE held_out")[0]["n"])


@app.post("/api/remember")
def remember(body: dict):
    """THE AGENT WRITES. A clip it has just seen enters the memory.

    This is the third verb, and the one that makes the loop a loop: the next
    time something like this arrives it will have a precedent, because the agent
    put one there. Nothing about the clip is described or labelled on the way
    in — the vectors were computed from pixels and the row carries no sentence.
    """
    rid = (body.get("id") or "").strip()
    q("UPDATE moments SET held_out = false WHERE rec_id = %s", (rid,))
    return dict(remembered=rid,
                live=q("SELECT count(*) n FROM moments WHERE NOT held_out")[0]["n"])


@app.post("/api/act")
def act(body: dict):
    """THE AGENT ACTS, durably. Escalations are a queue a person works from."""
    q("""CREATE TABLE IF NOT EXISTS agent_actions (
             id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(),
             run TEXT, saw TEXT, action TEXT, precedent TEXT,
             score FLOAT, note TEXT)""")
    q("""INSERT INTO agent_actions (run, saw, action, precedent, score, note)
         VALUES (%s,%s,%s,%s,%s,%s)""",
      (body.get("run"), body.get("saw"), body.get("action"),
       body.get("precedent"), float(body.get("score") or 0), body.get("note")))
    return dict(logged=True)


@app.get("/api/actions")
def actions(run: str = "", k: int = 40):
    rows = q("""SELECT ts, saw, action, precedent, score, note FROM agent_actions
                WHERE (%s = '' OR run = %s) ORDER BY id DESC LIMIT %s""", (run, run, k))
    return dict(actions=rows)


@app.get("/api/pair")
def pair(a: str = "", b: str = ""):
    """Cosine between two sentences in the text tower. The one-line explanation
    for why the left panel fails, checkable rather than quoted."""
    r = SIDE.rpc(dict(cmd="pair", a=a, b=b))
    return dict(a=a, b=b, cos=round(float(r.get("cos", 0.0)), 4))


def main() -> int:
    import uvicorn

    print(f"starting RelMo ({RELMO_PY}) ...", flush=True)
    if not SIDE.start():
        print("WARNING: RelMo sidecar failed; the text arm and DTW re-rank "
              "will be unavailable", flush=True)
    n = q("SELECT count(*) n FROM moments")[0]["n"]
    print(f"{n} moments indexed. http://localhost:8100", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
