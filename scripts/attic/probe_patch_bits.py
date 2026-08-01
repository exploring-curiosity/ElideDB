"""Can the patch grid be stored as SIGN BITS and still find the object?

The budget sweep settled the shape: the full 27x27 grid is needed (the
smallest object, the eggplant, loses its contrast to any spatial
pooling - 0.157 at 27x27 against 0.030 at 13x13) and PCA to 256 dims
IMPROVES contrast rather than costing it, because whitening removes
the directions every patch shares. That leaves 373 KB per frame, 1.67
GB for the corpus - larger than the entire store, which is not a
storage engine's answer to anything.

So: keep the grid, drop the precision. One sign bit per dimension is
32x smaller than fp32 and turns MaxSim into a Hamming count. The
question is whether the object survives it, and there is no reason to
assume so - the earlier random-projection attempt died exactly here,
because MaxSim maximizes over 729 patches and therefore maximizes
whatever noise the compression introduces.

Also measured: keeping only the top-K patches per frame, ranked by
distance from that frame's own mean patch (query-independent, so it
can be done at write time).

  python scripts/probe_patch_bits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                    # noqa: E402
from probe_patch_align import patch_vectors                  # noqa: E402


def main():
    from PIL import Image

    from elidedb.device import pick
    from elidedb.sig2 import MID, _text_vec
    from elidedb.video import FrameSet
    from transformers import AutoModel, AutoProcessor

    dev, dtype = pick()
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=dtype,
                                      low_cpu_mem_usage=True).to(dev).eval()
    db = Store.open("lake/bench")
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    true_of = {}
    for q, s, a, v in zip(t["query_id"], t["stream"], t["t0"], t["true"]):
        if int(v) == 1:
            true_of.setdefault(int(q), []).append((s, int(a)))
    frames_tbl = db.table("frames").scan()
    ep = db.table("episodes").scan()

    def frames_for(s, a, n=4):
        m = ep.filter(pc.and_(pc.equal(ep.column("stream"), s),
                              pc.equal(ep.column("ts"), a)))
        b = int(m.column("t1")[0].as_py())
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        pi = np.linspace(0, len(sel) - 1, n).round().astype(int)
        return [Image.fromarray(f) for _, f in
                sorted(FrameSet(db, "frames", sel.take(pi)).decode())]

    ep_keys = list(zip(ep.column("stream").to_pylist(),
                       (int(v) for v in ep.column("ts").to_pylist())))
    held = {true_of[q][0] for q in (8, 9, 10)}
    fit_ims = []
    for k in ep_keys:
        if k not in held and len(fit_ims) < 24:
            fit_ims += frames_for(*k, n=2)
    FP = patch_vectors(model, proc, dev, fit_ims).reshape(-1, 1152)
    mu = FP.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(FP - mu, full_matrices=False)

    cases = [("eggplant", 9, "an eggplant", "a banana"),
             ("banana", 10, "a banana", "an eggplant"),
             ("spoon", 8, "a spoon", "a banana")]
    D = 256
    R = Vt[:D].T
    print(f"{'case':>10} {'storage':>22} {'KB/frame':>9} "
          f"{'present':>8} {'absent':>7} {'contrast':>9}")
    for name, qi, pos, neg in cases:
        P = patch_vectors(model, proc, dev, frames_for(*true_of[qi][0]))
        n, PP, _ = P.shape
        X = (P.reshape(-1, 1152) - mu) @ R
        X /= np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
        X = X.reshape(n, PP, D)
        qs = {}
        for lab, txt in (("p", pos), ("a", neg)):
            v = (_text_vec(txt) - mu[0]) @ R
            qs[lab] = v / np.linalg.norm(v)

        def row(tag, kb, sp, sn):
            print(f"{name:>10} {tag:>22} {kb:>9.1f} {sp:>8.3f} "
                  f"{sn:>7.3f} {sp - sn:>9.3f}")

        row("fp16 all patches", PP * D * 2 / 1024,
            float((X @ qs["p"]).max()), float((X @ qs["a"]).max()))
        # 1 bit per dimension: cosine becomes a signed agreement count
        B = np.sign(X)
        sp = float((B @ qs["p"]).max() / np.sqrt(D))
        sn = float((B @ qs["a"]).max() / np.sqrt(D))
        row("1-bit signs", PP * D / 8 / 1024, sp, sn)
        # asymmetric: bits for the stored side, full precision for the
        # query (free - there is one query and millions of patches)
        for K in (256, 128, 64):
            # write-time pruning: distance from the frame's own mean
            # patch, so it needs no query and no labels
            keep = []
            for i in range(n):
                d = np.linalg.norm(X[i] - X[i].mean(0, keepdims=True),
                                   axis=-1)
                keep.append(X[i][np.argsort(-d)[:K]])
            Y = np.stack(keep)
            Bq = np.sign(Y)
            row(f"1-bit, top-{K} patches", K * D / 8 / 1024,
                float((Bq @ qs["p"]).max() / np.sqrt(D)),
                float((Bq @ qs["a"]).max() / np.sqrt(D)))
        print()


if __name__ == "__main__":
    main()
