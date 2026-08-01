"""V0: does DETECTION x MOTION identify the manipulated object?

Presence was measured insufficient (GDINO alone on q09's band: true
0.689 vs false 0.526, 7 negatives above the best true) for a stated
reason: an eggplant sitting on the table scores exactly like an
eggplant being put away. The event-teacher's L2 says a participant is
a region that CO-MOVES with the agent - so the probe scores each clip
by the best moment of (detector confidence for the noun) x (motion in
that box relative to the scene's own flow median). Max over ~6
sampled moments is L4-legal: few draws, inside matched structure.

Gate (declared in docs/EVENT_TEACHER_PLAN.md): on the q09/q10/q00
bands, true episodes inside the top-5 by participation. Fail -> the
instrument, not the prompt, gets revisited.

  python scripts/probe_participation.py [--q 9,10,0] [--depth 60]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402

GDINO = "IDEA-Research/grounding-dino-base"
NF = 6


def main():
    import cv2
    import torch
    from PIL import Image
    from transformers import (AutoProcessor,
                              GroundingDinoForObjectDetection)

    from elidedb.sig2 import atoms_of
    from elidedb.video import FrameSet

    argv = sys.argv
    qsel = [int(x) for x in (argv[argv.index("--q") + 1].split(",")
                             if "--q" in argv else ["9", "10", "0"])]
    depth = int(argv[argv.index("--depth") + 1]) if "--depth" in argv \
        else 60

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(GDINO)
    model = GroundingDinoForObjectDetection.from_pretrained(
        GDINO, dtype=torch.float32).to(dev).eval()

    d = np.load(ROOT / "ml/teacher_base.npz", allow_pickle=True)
    B, qids = d["B"], [int(q) for q in d["qids"]]
    keys = list(zip([str(s) for s in d["streams"]],
                    [int(v) for v in d["ts"]], [int(v) for v in d["t1"]]))
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    db = Store.open("lake/bench")
    frames_tbl = db.table("frames").scan()

    def frames_of(i):
        s, a, b = keys[i]
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        pi = np.linspace(0, len(sel) - 1,
                         min(NF, len(sel))).round().astype(int)
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode()
        except Exception:
            return []
        return [f for _, f in sorted(dec)]

    def participation(frames, phrase):
        """(best conf, best conf x motion) over the sampled moments."""
        if len(frames) < 3:
            return 0.0, 0.0
        H, W = frames[0].shape[:2]
        grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        best_p, best_m = 0.0, 0.0
        for fi in range(len(frames)):
            im = Image.fromarray(frames[fi])
            inputs = proc(images=im, text=phrase + " .",
                          return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0].sigmoid()
            qbest = int(logits.max(dim=1).values.argmax())
            conf = float(logits[qbest].max())
            cx, cy, bw, bh = [float(v) for v in out.pred_boxes[0, qbest]]
            x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, W), min(y1, H)
            best_p = max(best_p, conf)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            g0 = grey[max(fi - 1, 0)]
            g1 = grey[min(fi + 1, len(grey) - 1)]
            flow = cv2.calcOpticalFlowFarneback(
                g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2)
            inbox = float(mag[y0:y1, x0:x1].mean())
            # camera-shake guard (L4/plan): motion is what stands out
            # from THIS frame pair's own flow median
            scene = float(np.median(mag)) + 1e-6
            best_m = max(best_m, conf * min(inbox / scene, 8.0))
        return best_p, best_m

    for j, qi in enumerate(qids):
        if qi not in qsel or sup.get(qi, 0) == 0:
            continue
        phrase = (atoms_of(QUERIES[qi].lower())[:1] or [QUERIES[qi]])[0]
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        band = np.argsort(-B[:, j].astype(float))[:depth]
        rows, t0 = [], time.time()
        for i in band:
            fr = frames_of(int(i))
            p, m = participation(fr, phrase)
            rows.append((int(lab[i]), p, m))
        A = np.array(rows, float)
        y = A[:, 0].astype(bool)
        for name, col in (("presence", 1), ("PARTICIPATION", 2)):
            x = A[:, col]
            rr = np.argsort(np.argsort(-x)) + 1
            print(f"q{qi:02d} {name:>13}  true {x[y].mean():6.2f}  "
                  f"false {x[~y].mean():6.2f}  "
                  f"true ranks {sorted(rr[y].astype(int))}  "
                  f"negs above best true {(x[~y] > x[y].max()).sum()}",
                  flush=True)
        print(f"     ({(time.time() - t0) / len(band):.1f}s/clip, "
              f"band {len(band)}, phrase {phrase!r})", flush=True)


if __name__ == "__main__":
    main()
