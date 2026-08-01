"""Build the FROZEN truthset for the 14 bench queries (bridge4h).

Contract (user directive: stable benchmark before any more fixes):
ground truth adjudicated ONCE with dense strips — the ACTION must
visibly happen (end-frame state is not evidence, user-caught bias) —
then never regraded. Labels are candidate HINTS only.

Stages:
  candidates  generous label predicates UNION top-100 fused retrieval
              per query -> eval/truthsets/candidates.parquet
  sheets      dense strips for every ungraded candidate: 4 episodes x
              8 frames per sheet + manifest.json
  compile     verdicts.jsonl -> eval/truthsets/bridge4h.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402

OUT = Path("eval/truthsets")

# generous, recall-first hint predicates (candidates only, NOT truth)
_HINTS = [
    lambda L: "draw" in L and ("green" in L or "cucumber" in L
                               or "broccoli" in L),
    lambda L: "draw" in L and ("yellow" in L or "banana" in L
                               or "cheese" in L or "corn" in L),
    lambda L: "draw" in L and "red" in L,
    lambda L: ("pot" in L or "pan" in L or "bowl" in L
               or "vessel" in L) and ("stove" in L or "burner" in L
                                      or "put" in L or "move" in L),
    lambda L: "draw" in L and ("close" in L or "shut" in L
                               or "ferm" in L),
    lambda L: "draw" in L and ("open" in L or "pull" in L),
    lambda L: "fold" in L,
    lambda L: "lid" in L,
    lambda L: "spoon" in L and ("cloth" in L or "towel" in L),
    lambda L: "eggplant" in L,
    lambda L: "banana" in L,
    lambda L: "draw" in L and ("close" in L or "shut" in L
                               or "push" in L),
    lambda L: "draw" in L and ("toy" in L or "out" in L),
    lambda L: ("pot" in L or "pan" in L) and "burner" in L,
]


def _labels(db):
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    lab = {}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                          t["task"]):
        if k:
            lab[(str(stream_of.get(int(i))), int(a))] = k.lower()
    return lab


def candidates():
    from elidedb.scenario import _episodes, search_set
    db = Store.open("lake/bridge4h")
    lab = _labels(db)
    eps = _episodes(db)
    rows = []
    for qi, q in enumerate(QUERIES):
        keys = set()
        for s, a, b in eps:
            if _HINTS[qi](lab.get((s, a), "")):
                keys.add((s, a, b))
        r = search_set(db, q, purity="fast", k_max=100)
        for c in r["clips"]:
            keys.add((c["stream"], c["t0"], c["t1"]))
        for s, a, b in sorted(keys):
            rows.append((qi, q, s, a, b, lab.get((s, a), "")))
        print(f"q{qi:02d}: {len(keys):4d} candidates  {q}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({
        "query_id": pa.array([r[0] for r in rows], pa.int32()),
        "query": pa.array([r[1] for r in rows]),
        "stream": pa.array([r[2] for r in rows]),
        "t0": pa.array([r[3] for r in rows], pa.int64()),
        "t1": pa.array([r[4] for r in rows], pa.int64()),
        "label": pa.array([r[5] for r in rows]),
    })
    pq.write_table(tbl, OUT / "candidates.parquet")
    print(f"total {len(rows)} (query,episode) pairs")


def _graded():
    done = set()
    vf = OUT / "verdicts.jsonl"
    if vf.exists():
        for ln in vf.read_text().splitlines():
            if ln.strip():
                v = json.loads(ln)
                done.add((int(v["q"]), str(v["stream"]), int(v["t0"])))
    return done


def sheets(per_sheet=4, n_frames=8, width=140):
    from PIL import Image, ImageDraw

    from elidedb.video import FrameSet
    db = Store.open("lake/bridge4h")
    cand = pq.read_table(OUT / "candidates.parquet").to_pydict()
    done = _graded()
    sdir = OUT / "sheets"
    sdir.mkdir(parents=True, exist_ok=True)
    mf = sdir / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    frames_tbl = db.table("frames").scan()
    pend = [(int(q), str(s), int(a), int(b), lb) for q, s, a, b, lb in
            zip(cand["query_id"], cand["stream"], cand["t0"],
                cand["t1"], cand["label"])
            if (int(q), str(s), int(a)) not in done]
    existing = {tuple(e[:3]) for rows in manifest.values()
                for e in rows}
    pend = [p for p in pend if (p[0], p[1], p[2]) not in existing]
    print(f"{len(pend)} pairs need sheets", flush=True)
    batch, made = [], 0
    for qi, s, a, b, lb in pend:
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < n_frames:
            continue
        pick = np.linspace(0, len(sel) - 1,
                           n_frames).round().astype(int)
        dec = FrameSet(db, "frames", sel.take(pick)).decode(width=width)
        if len(dec) < n_frames:
            continue
        batch.append((qi, s, a, b, lb,
                      [Image.fromarray(d[1]) for d in sorted(dec)]))
        if len(batch) == per_sheet:
            made += _emit(sdir, manifest, batch, n_frames, width)
            batch = []
    if batch:
        made += _emit(sdir, manifest, batch, n_frames, width)
    mf.write_text(json.dumps(manifest))
    print(f"made {made} sheets; total {len(manifest)}", flush=True)


def _emit(sdir, manifest, batch, n_frames, width):
    from PIL import Image, ImageDraw
    name = f"s{len(manifest):04d}.png"
    fh = batch[0][5][0].height
    band = 20
    sheet = Image.new("RGB", (n_frames * width,
                              len(batch) * (fh + band)), "black")
    dr = ImageDraw.Draw(sheet)
    rows = []
    for ri, (qi, s, a, b, lb, imgs) in enumerate(batch):
        y = ri * (fh + band)
        for fi, im in enumerate(imgs):
            sheet.paste(im, (fi * width, y))
        dr.text((2, y + fh + 2),
                f"r{ri} q{qi:02d} {lb[:70]}", fill="white")
        rows.append([qi, s, a, b])
    sheet.save(sdir / name)
    manifest[name] = rows
    return 1


def compile_out():
    cand = pq.read_table(OUT / "candidates.parquet").to_pydict()
    qtext = {}
    for q, t in zip(cand["query_id"], cand["query"]):
        qtext[int(q)] = t
    rows = []
    for ln in (OUT / "verdicts.jsonl").read_text().splitlines():
        if ln.strip():
            v = json.loads(ln)
            rows.append((int(v["q"]), qtext.get(int(v["q"]), ""),
                         str(v["stream"]), int(v["t0"]),
                         int(v["v"])))
    tbl = pa.table({
        "query_id": pa.array([r[0] for r in rows], pa.int32()),
        "query": pa.array([r[1] for r in rows]),
        "stream": pa.array([r[2] for r in rows]),
        "t0": pa.array([r[3] for r in rows], pa.int64()),
        "true": pa.array([r[4] for r in rows], pa.int8()),
    })
    pq.write_table(tbl, OUT / "bridge4h.parquet")
    n = len(rows)
    pos = sum(r[4] for r in rows)
    print(f"compiled {n} verdicts ({pos} true) -> "
          f"{OUT / 'bridge4h.parquet'}")


if __name__ == "__main__":
    {"candidates": candidates, "sheets": sheets,
     "compile": compile_out}[sys.argv[1] if len(sys.argv) > 1
                             else "candidates"]()
