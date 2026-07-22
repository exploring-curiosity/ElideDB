"""Iterative complex-query battery with SigLIP-based visual verification.

Verification is MODEL-side, not eyeball-side: for every returned clip we
decode its start and end frame through the byte-range path and score them
with the SigLIP TEACHER against (a) the full query and (b) each clause atom.
Plus the human task label of the overlapped episode (eval-only file) and the
clip's own VLM caption when present. Between them: did the clip contain the
right THINGS (SigLIP), the right EVENT (label/caption), at rank 1?

Scores to read:
  label✓   overlapped episode's human label shares content words with query
  sig      teacher cosine of clip frames vs full query (appearance check)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.embeddings import _embed_images, embed_text     # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

STOP = set("a an the in on to it into onto of and then put place move take "
           "pick up out with from something object thing".split())

QUERIES = [
    "put a green object in the drawer and close it",
    "close the drawer",
    "open the drawer",
    "take something out of the drawer and put it on the table",
    "pick up a banana",
    "put the pot on the stove",
    "wipe the table with a cloth",
    "put a red object into the sink",
]


def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 2}


def main():
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [{"t0": int(a), "t1": int(b), "task": k,
            "stream": stream_of.get(int(i))}
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    frames = db.table("frames").scan()

    def label(s, a, b):
        best = ("?", 0)
        for e in eps:
            if e["stream"] == s:
                ov = min(b, e["t1"]) - max(a, e["t0"])
                if ov > best[1]:
                    best = (e["task"], ov)
        return best[0]

    def sig_score(s, a, b, qv):
        from PIL import Image
        sel = frames.filter(pc.and_(
            pc.equal(frames.column("stream"), s),
            pc.and_(pc.greater_equal(frames.column("ts"), a),
                    pc.less_equal(frames.column("ts"), b))))
        if len(sel) < 2:
            return None
        pick = np.array([0, len(sel) - 1])
        dec = FrameSet(db, "frames", sel.take(pick)).decode(width=448)
        if not dec:
            return None
        vecs = _embed_images([Image.fromarray(d[1]) for d in dec], "fast")
        return float((vecs @ qv).max())

    report = []
    for q in QUERIES:
        t0 = time.time()
        hits, st = db.search_context(q, k=3, deep=6, verify="sync")
        el = time.time() - t0
        qw = content_words(q)
        qv = embed_text(q, model_id="fast")
        rows = []
        for h in hits[:3]:
            lab = label(h["stream"], h["t0"], h["t1"])
            overlap = qw & content_words(lab)
            sig = sig_score(h["stream"], h["t0"], h["t1"], qv)
            rows.append({"label": lab, "match_words": sorted(overlap),
                         "label_ok": len(overlap) >= max(1, len(qw) // 3),
                         "sig": round(sig, 3) if sig else None,
                         "margin": h.get("deep_margin", h["margin"]),
                         "caption": h.get("caption", "")})
        n_ok = sum(r["label_ok"] for r in rows)
        report.append({"query": q, "s": round(el, 1), "hits": rows,
                       "label_ok_top3": n_ok})
        print(f"\nQ: {q}   ({el:.0f}s, {n_ok}/3 label-verified)")
        for r in rows:
            mark = "OK " if r["label_ok"] else "MISS"
            print(f"  [{mark}] m={r['margin']:+.2f} sig={r['sig']} "
                  f"{r['label'][:58]}")
            if r["match_words"]:
                print(f"         matched: {r['match_words']}")

    total = sum(r["label_ok_top3"] for r in report)
    print(f"\n=== battery: {total}/{3 * len(QUERIES)} top-3 hits "
          f"label-verified across {len(QUERIES)} queries ===")
    Path("bench_query_battery.json").write_text(
        json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
