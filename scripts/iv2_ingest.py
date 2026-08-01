"""Ingest InternVideo2-Stage2 1B episode vectors: 4 frames per
episode -> ONE video-native 512-d aligned vector -> iv2_vectors.
Unlike every frame-pooled channel, the 4 frames pass through the
model TOGETHER — temporal modeling is the point (arXiv 2403.15377).
Cost gate: hard-aborts (appends nothing) if projected total exceeds
60 min."""
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
from elidedb.iv2 import MDIR, clip_vec                       # noqa: E402


def main():
    from elidedb.video import FrameSet
    store = Store.open(sys.argv[1] if len(sys.argv) > 1
                       else "lake/bench")
    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = store.table("frames").scan()
    rows_s, rows_a, rows_b, vecs = [], [], [], []
    t0 = time.time()
    # PER-ITERATION progress. The old "every 200th recording"
    # print moved a bar three times over a 600-item run, which
    # tells you nothing about whether it is alive between them.
    # Load the model BEFORE the bar exists. clip_vec lazy-loads a 4 GB
    # checkpoint on FIRST USE, i.e. inside iteration one, so the bar
    # sat at 0/600 for minutes and looked hung. The longest step in the
    # run deserves to be announced, not hidden behind a frozen bar.
    print('loading InternVideo2 (~4 GB, first use only)...', flush=True)
    _t = time.time()
    from elidedb.iv2 import load_model
    load_model()
    print(f'  model ready in {time.time() - _t:.0f}s', flush=True)
    from tqdm import tqdm
    # tqdm wraps the ITERABLE, so the count advances when an iteration
    # COMPLETES. Updating at the top of the body instead reports work
    # that has not happened yet.
    _bar = tqdm(recs, desc="iv2", unit="rec", dynamic_ncols=True,
                mininterval=0.3)
    for ri, (s, a, b) in enumerate(_bar):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 4:
            continue
        pick = np.linspace(0, len(sel) - 1, 4).round().astype(int)
        try:
            dec = FrameSet(store, "frames",
                           sel.take(pick)).decode(width=224)
        except Exception:
            continue        # stream-boundary episode in filtered store
        if len(dec) < 4:
            continue
        v = clip_vec([d[1] for d in sorted(dec)])
        rows_s.append(s); rows_a.append(a); rows_b.append(b)
        vecs.append(v.astype(np.float32))
    _bar.close()
    V = np.stack(vecs)
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V).reshape(-1)), V.shape[1]),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("iv2_vectors").append(
        tbl, kind="embeddings",
        meta={"model": MDIR, "dim": int(V.shape[1]),
              "frames_per_episode": 4})
    print(json.dumps({"rows": len(tbl), "dim": int(V.shape[1]),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
