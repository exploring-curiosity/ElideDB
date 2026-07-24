"""THE PRODUCT BENCH — per-query precision of what was returned.

User contract (verbatim intent): "if i query for 10 samples per query
it should give me upto 10. not 10 forced videos. but whatever has been
returned should be true to the query." Never aggregated into one
number; never a forced set size; generic queries that ride the corpus
prior prove nothing, so the queries are COMPOSITIONAL. Labels are
printed as context but the grade is visual (contact sheets, graded
vendor-side).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.scenario import search_set                      # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

# the user's seven
QUERIES = [
    "pick up a green object from table and put it into the drawer",
    "pick up a yellow object from table and put it into the drawer",
    "pick up a red object from the drawer and put it on the table",
    "pick up a vessel and put it on the stove",
    "robot arm holds the handle and closes the drawer",
    "robot arm opens the drawer",
    "fold a piece of towel",
    # seven more, same compositional shape
    "put the lid on the pot",
    "place the spoon on top of the cloth",
    "put the eggplant into the drawer",
    "put the banana on top of the drawer",
    "the robot arm pushes the drawer shut",
    "take a toy out of the drawer and place it on the table",
    "move the silver pot onto the burner",
]

N_FRAMES = 4
FRAME_W = 210


def main():
    from PIL import Image, ImageDraw
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1
                   else "scratch_product")
    out_dir.mkdir(parents=True, exist_ok=True)
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    lab = {}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                          t["task"]):
        if k:
            lab[(stream_of.get(int(i)), int(a))] = k.lower()
    mode = sys.argv[2] if len(sys.argv) > 2 else "fast"
    frames_tbl = db.table("frames").scan()
    for qi, q in enumerate(QUERIES):
        r = search_set(db, q, purity=mode, k_max=10)
        clips = r["clips"]
        au = ""
        if r.get("audit"):
            a = r["audit"]
            au = (f"  audit[chk {a['checked']} absent "
                  f"{a['killed_absent']} static "
                  f"{a.get('killed_static', 0)} disjoint "
                  f"{a['killed_disjoint']}"
                  f"{' UNGROUNDABLE' if a.get('ungroundable') else ''}]")
        print(f"[q{qi:02d}] returned {len(clips):2d}  "
              f"dirdrop {r['direction_filtered']:4d}{au}  "
              f"{r['ms']:5.0f}ms  {q}", flush=True)
        if not clips:
            continue
        rows = []
        for c in clips:
            sel = frames_tbl.filter(pc.and_(
                pc.equal(frames_tbl.column("stream"), c["stream"]),
                pc.and_(pc.greater_equal(frames_tbl.column("ts"),
                                         c["t0"]),
                        pc.less_equal(frames_tbl.column("ts"),
                                      c["t1"]))))
            if len(sel) < N_FRAMES:
                continue
            pick = np.linspace(0, len(sel) - 1,
                               N_FRAMES).round().astype(int)
            dec = FrameSet(db, "frames",
                           sel.take(pick)).decode(width=FRAME_W)
            if len(dec) < N_FRAMES:
                continue
            rows.append(([Image.fromarray(d[1]) for d in sorted(dec)],
                         lab.get((c["stream"], c["t0"]), "?")))
        if not rows:
            continue
        fh = rows[0][0][0].height
        band = 22
        sheet = Image.new("RGB", (N_FRAMES * FRAME_W,
                                  len(rows) * (fh + band)), "black")
        dr = ImageDraw.Draw(sheet)
        for ri, (imgs, L) in enumerate(rows):
            y = ri * (fh + band)
            for fi, im in enumerate(imgs):
                sheet.paste(im, (fi * FRAME_W, y))
            dr.text((4, y + fh + 3), f"r{ri} label: {L[:80]}",
                    fill="white")
        sheet.save(out_dir / f"q{qi:02d}.png")
    print(f"sheets -> {out_dir}")


if __name__ == "__main__":
    main()
