"""Vendor-side audit calibration — NO customer labels, ever.

The product claim is 'stop annotating'; a calibration step done by the
customer would refute the pitch. So the calibration is done ONCE, by us,
on our own bench data, with human eyes on the pixels: decode the clips
the set-retrieval path actually delivered for the disputed classes and
render contact sheets (stratified clips x 4 frames). A human (or a
vision-capable reviewer) grades the sheets against the QUERY, which
settles the confound the numbers cannot: does the auditor overclaim, or
do the labels undercount clip-level truth?

Output: one PNG contact sheet per disputed query + a manifest mapping
each row to (rank, stream, t0, truth label, label-pred verdict).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elidedb import Store                                    # noqa: E402
from elidedb.scenario import search_set                      # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402
from regress10 import QUERIES                                # noqa: E402

# the three disputes: auditor-high/label-low, label-high/auditor-kill,
# label-zero (is the set truly junk?)
DISPUTED = [
    "picking something up from the table",
    "put the green object in the drawer",
    "put the lid on a vessel",
    "place a toy on top of the towel",
]
N_CLIPS = 10
N_FRAMES = 4
FRAME_W = 220


def main():
    from PIL import Image, ImageDraw
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1
                   else "scratch_calibrate")
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
    preds = {q: p for q, p in QUERIES}
    frames_tbl = db.table("frames").scan()
    manifest = {}
    for q in DISPUTED:
        r = search_set(db, q, purity="fast")
        clips = r["clips"]
        n = len(clips)
        idx = sorted(set(np.linspace(0, n - 1, min(N_CLIPS, n))
                         .round().astype(int)))
        rows, meta = [], []
        for i in idx:
            c = clips[i]
            sel = frames_tbl.filter(pc.and_(
                pc.equal(frames_tbl.column("stream"), c["stream"]),
                pc.and_(
                    pc.greater_equal(frames_tbl.column("ts"), c["t0"]),
                    pc.less_equal(frames_tbl.column("ts"), c["t1"]))))
            if len(sel) < N_FRAMES:
                continue
            pick = np.linspace(0, len(sel) - 1,
                               N_FRAMES).round().astype(int)
            dec = FrameSet(db, "frames",
                           sel.take(pick)).decode(width=FRAME_W)
            if len(dec) < N_FRAMES:
                continue
            imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
            rows.append((i, imgs))
            L = lab.get((c["stream"], c["t0"]), "?")
            meta.append({"rank": int(i), "stream": c["stream"],
                         "t0": int(c["t0"]), "label": L,
                         "label_pred": bool(preds[q](L))})
        if not rows:
            continue
        fh = rows[0][1][0].height
        band = 26
        sheet = Image.new("RGB", (N_FRAMES * FRAME_W,
                                  len(rows) * (fh + band)), "black")
        dr = ImageDraw.Draw(sheet)
        for ri, (rank, imgs) in enumerate(rows):
            y = ri * (fh + band)
            for fi, im in enumerate(imgs):
                sheet.paste(im, (fi * FRAME_W, y))
            m = meta[ri]
            dr.text((4, y + fh + 4),
                    f"row{ri}  rank {rank}/{n}  label: "
                    f"{m['label'][:70]}  pred:"
                    f"{'Y' if m['label_pred'] else 'N'}",
                    fill="white")
        slug = q.replace(" ", "_")[:40]
        sheet.save(out_dir / f"{slug}.png")
        manifest[q] = {"delivered": n, "rows": meta,
                       "sheet": f"{slug}.png"}
        print(f"{len(rows):2d} rows / {n:3d} delivered  -> {slug}.png"
              f"  ({q})")
    (out_dir / "manifest.json").write_text(json.dumps(manifest,
                                                      indent=1))
    print(f"wrote {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
