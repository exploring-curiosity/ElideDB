"""OBJECT STORE: regions proposed by a detector, not by motion.

What a "crop" used to be: the bounding box of a connected component of
thresholded optical flow, padded by 60%, taken from a 192x144 frame.
Nothing in that chain knows what an object is - it is the box of
whatever MOVED. That is why the name vocabulary filled with "black
object", "white object", "wall", "triangle", "black square": the namer
was describing exactly what it was shown, which was a motion smear with
60% background around it. It is also why "eggplant" never appeared in
397 names despite eggplant episodes existing.

Here a region is proposed by an open-vocabulary DETECTOR on the
full-resolution frame, and cropped tight. The prompt is a single
generic English word, so nothing per-dataset enters: the detector is
asked where the objects are, not whether a drawer is present.

Then identity, which is the point of the store: crops are embedded,
clustered, and every cluster gets ONE id. Naming happens per cluster at
read time instead of per crop at write time - 12.7x fewer namings at
k=400 measured on this corpus, and the name becomes an attribute of the
object rather than of the crop.

  python scripts/build_objects.py --probe 24     propose + contact sheet
  python scripts/build_objects.py --build        full store + clusters
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

PROMPT = "object"          # generic English; no dataset vocabulary
BOX_THR = 0.25
MIN_SIDE = 24              # px, at full resolution
MAX_AREA = 0.45            # a box covering half the frame is the scene
OUT = ROOT / "scratch_objects"


def frames_for(db, keys, per_ep=3):
    """Full-resolution frames, byte-range decoded — the crop has to come
    from real pixels, not from the 192x144 raster geometry runs on."""
    ftab = db.table("frames").scan()
    out = []
    for (s, a, b) in keys:
        sel = ftab.filter(pc.and_(
            pc.equal(ftab.column("stream"), s),
            pc.and_(pc.greater_equal(ftab.column("ts"), a),
                    pc.less_equal(ftab.column("ts"), b))))
        if len(sel) < per_ep:
            continue
        pick = np.unique(np.linspace(0, len(sel) - 1, per_ep)
                         .round().astype(int))
        try:
            dec = sorted(FrameSet(db, "frames", sel.take(pick)).decode())
        except Exception:
            continue
        for t, im in dec:
            out.append((s, a, b, int(t), im))
    return out


def propose(frames, batch=8):
    """Detector boxes on full-res frames. Returns (idx, box, score)."""
    from PIL import Image
    from elidedb.grounding import detect_regions
    got = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        ims = [Image.fromarray(f[4]) for f in chunk]
        try:
            res = detect_regions(ims, PROMPT, threshold=BOX_THR)
        except Exception as e:
            print(f"  detector failed: {type(e).__name__}: {e}")
            return got
        for j, boxes in enumerate(res):
            H, W = chunk[j][4].shape[:2]
            for box, score in boxes[:12]:
                x0, y0, x1, y1 = [int(v) for v in box[:4]]
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, W), min(y1, H)
                if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
                    continue
                if (x1 - x0) * (y1 - y0) > MAX_AREA * W * H:
                    continue
                got.append((i + j, (x0, y0, x1, y1), float(score)))
    return got


def sheet(items, path, cols=8, cell=104, label=None):
    """Contact sheet so the crops can be LOOKED at, not just counted."""
    from PIL import Image, ImageDraw
    if not items:
        return None
    rows = (len(items) + cols - 1) // cols
    lab = 12 if label else 0
    W, H = cols * cell, rows * (cell + lab)
    canvas = Image.new("RGB", (W, H), (17, 21, 28))
    d = ImageDraw.Draw(canvas)
    for i, (im, txt) in enumerate(items):
        r, c = divmod(i, cols)
        p = Image.fromarray(im)
        p.thumbnail((cell - 4, cell - 4))
        canvas.paste(p, (c * cell + 2, r * (cell + lab) + 2))
        if label:
            d.text((c * cell + 3, r * (cell + lab) + cell - 10),
                   str(txt)[:18], fill=(150, 190, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def main():
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    n_ep = int(argv[argv.index("--probe") + 1] if "--probe" in argv
               else 24)
    ep = db.table("episodes").scan().to_pydict()
    idx = np.linspace(0, len(ep["ts"]) - 1, n_ep).round().astype(int)
    keys = [(str(ep["stream"][i]), int(ep["ts"][i]), int(ep["t1"][i]))
            for i in idx]

    t0 = time.time()
    fr = frames_for(db, keys, per_ep=3)
    t_dec = time.time() - t0
    print(f"{len(fr)} full-res frames from {len(keys)} episodes "
          f"in {t_dec:.1f}s")

    t0 = time.time()
    props = propose(fr)
    t_det = time.time() - t0
    print(f"{len(props)} regions proposed in {t_det:.1f}s "
          f"({t_det/max(len(fr),1)*1000:.0f} ms/frame)")
    if not props:
        raise SystemExit("no regions - detector unavailable?")

    crops, meta = [], []
    for fi, (x0, y0, x1, y1), sc in props:
        s, a, b, t, im = fr[fi]
        crops.append(im[y0:y1, x0:x1])
        meta.append({"stream": s, "ep_ts": a, "ep_t1": b, "ts": t,
                     "box": [x0, y0, x1, y1], "score": round(sc, 3)})

    p = sheet([(c, f"{m['score']:.2f}") for c, m in zip(crops, meta)][:64],
              OUT / "proposed_crops.png", label=True)
    print(f"contact sheet -> {p}")

    areas = np.array([(m["box"][2] - m["box"][0]) *
                      (m["box"][3] - m["box"][1]) for m in meta], float)
    print(json.dumps({
        "regions": len(crops),
        "per_frame": round(len(crops) / max(len(fr), 1), 2),
        "median_box_px": int(np.median(areas)),
        "median_side_px": int(np.sqrt(np.median(areas))),
        "detector_ms_per_frame": round(t_det / max(len(fr), 1) * 1000),
    }, indent=1))

    # ---- IDENTITY: embed the regions, cluster, one id per cluster
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from elidedb.device import pick
    from elidedb.sig2 import MID
    dev, dtype = pick()
    proc_ = AutoProcessor.from_pretrained(MID)
    sig = AutoModel.from_pretrained(MID, dtype=dtype,
                                    low_cpu_mem_usage=True).to(dev).eval()
    t0 = time.time()
    V = []
    for i in range(0, len(crops), 128):
        b = crops[i:i + 128]
        with torch.no_grad():
            px = proc_(images=[Image.fromarray(c) for c in b],
                       return_tensors="pt").to(dev)
            F = sig.get_image_features(**px)
            F = F / F.norm(dim=-1, keepdim=True)
        V.append(F.float().cpu().numpy())
    V = np.concatenate(V)
    t_emb = time.time() - t0

    K = int(argv[argv.index("--k") + 1] if "--k" in argv
            else max(4, len(V) // 8))
    rng = np.random.default_rng(0)
    C = V[rng.choice(len(V), min(K, len(V)), replace=False)]
    for _ in range(30):
        a = (V @ C.T).argmax(1)
        for k in range(len(C)):
            m = a == k
            if m.any():
                c = V[m].mean(0)
                C[k] = c / (np.linalg.norm(c) + 1e-8)
    assign = (V @ C.T).argmax(1)

    groups = defaultdict(list)
    for i, k in enumerate(assign):
        groups[int(k)].append(i)
    big = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    # one sheet per cluster: this is what "the same object across demos"
    # has to LOOK like for the idea to be worth anything
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (k, idxs) in enumerate(big[:12]):
        eps = {meta[i]["ep_ts"] for i in idxs}
        sheet([(crops[i], "") for i in idxs[:16]],
              OUT / f"object_{rank:02d}_n{len(idxs)}_ep{len(eps)}.png",
              cols=8, cell=96)
        rows.append({"object_id": rank, "instances": len(idxs),
                     "episodes": len(eps)})
    print(json.dumps({
        "embed_seconds": round(t_emb, 1), "clusters": len(groups),
        "crops": len(V),
        "compression": round(len(V) / max(len(groups), 1), 1),
        "largest": rows[:8]}, indent=1))
    print(f"per-object sheets -> {OUT}/object_*.png")


if __name__ == "__main__":
    main()
