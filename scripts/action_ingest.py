"""Ingest the ACTION PROBE channel: per-episode SSv2 action posteriors.

V-JEPA 2 ViT-L (already the QbE encoder) + Meta's released SSv2
attentive probe -> 174-d softmax per episode into `action_probs`.
Adoption basis (measured before wiring, 2026-07-24): put-in vs take-out
AUC 0.889 zero-shot with the literal class pair on robot video; close
vs open 0.740 (direction stays owned by the motion channel, AUC 0.98).
Video-native evidence of WHAT HAPPENED; replaces VLM judging for action
queries per user directive. Pretrained weights only — any upload.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.action_probe import (N_CLASSES, PROBE_CKPT,     # noqa: E402
                                  clip_action_probs)
from elidedb.video import FrameSet                           # noqa: E402


def main():
    store = Store.open(sys.argv[1] if len(sys.argv) > 1
                       else "lake/bridge4h")
    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = store.table("frames").scan()
    rows_s, rows_a, rows_b, vecs = [], [], [], []
    t0 = time.time()
    for ri, (s, a, b) in enumerate(recs):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 8:
            continue
        pick = np.linspace(0, len(sel) - 1, 16).round().astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=256)
        if len(dec) < 8:
            continue
        p = clip_action_probs([d[1] for d in sorted(dec)])
        rows_s.append(s); rows_a.append(a); rows_b.append(b)
        vecs.append(p.astype(np.float32))
        if (ri + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {ri + 1}/{len(recs)}  {el:.0f}s "
                  f"(eta {el / (ri + 1) * (len(recs) - ri - 1):.0f}s)",
                  flush=True)
    V = np.stack(vecs)
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V).reshape(-1)), N_CLASSES),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("action_probs").append(
        tbl, kind="embeddings",
        meta={"model": "vjepa2-vitl + ssv2 attentive probe",
              "probe": PROBE_CKPT.name, "classes": N_CLASSES,
              "frames_per_episode": 16})
    print(json.dumps({"rows": len(tbl),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
