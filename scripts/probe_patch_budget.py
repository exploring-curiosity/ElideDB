"""How far can the patch grid be compressed before the object dies?

Full grids are 729 patches x 1152 dims per frame - 7.5 GB for the
corpus, more than seven times the whole store. Late interaction is
only worth having if it survives a budget, so this measures the two
compressions that matter, on frames whose contents are known:

  spatial   mean-pool GxG blocks of the patch grid (fewer, wider
            patches). Dilutes a small object by the block area, which
            is the whole risk.
  channel   random projection to d dims. Johnson-Lindenstrauss keeps
            inner products in expectation, so this should be nearly
            free - the point is to confirm "nearly".

Reported as the contrast that matters: the object's own score minus
the score of an object that is NOT in the frame, in units of the
background patch spread. If contrast survives at a budget, that
budget ships.

  python scripts/probe_patch_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from probe_patch_align import patch_vectors                  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))


def pool_grid(P, b):
    """(n, G*G, D) -> mean-pool b x b blocks. Trailing rows/cols that
    do not fill a block are dropped, so g = G // b."""
    n, PP, D = P.shape
    G = int(round(np.sqrt(PP)))
    g = G // b
    X = P.reshape(n, G, G, D)[:, :g * b, :g * b, :]
    X = X.reshape(n, g, b, g, b, D).mean(axis=(2, 4)).reshape(n, g * g, D)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


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

    cases = [("eggplant", 9, "an eggplant", "a banana"),
             ("banana", 10, "a banana", "an eggplant"),
             ("spoon", 8, "a spoon", "a banana")]
    # PCA basis, fitted on patches from UNRELATED episodes so the
    # projection never sees the frames it is scored on. Random
    # projection was measured first and is not an option: MaxSim takes
    # the largest of 729 scores, so JL's per-pair noise becomes the
    # thing being maximized - contrast fell 0.110 -> 0.002 at d=256.
    fit_ims = []
    ep_keys = list(zip(ep.column("stream").to_pylist(),
                       (int(v) for v in ep.column("ts").to_pylist())))
    held = {true_of[q][0] for q in (8, 9, 10)}
    for k in ep_keys:
        if k not in held and len(fit_ims) < 24:
            fit_ims += frames_for(*k, n=2)
    FP = patch_vectors(model, proc, dev, fit_ims).reshape(-1, 1152)
    FP = FP - FP.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(FP[np.random.default_rng(0).choice(
        len(FP), min(20000, len(FP)), replace=False)], full_matrices=False)

    print(f"{'case':>10} {'grid':>6} {'dims':>6} {'bytes/frame':>12} "
          f"{'present':>9} {'absent':>8} {'contrast':>9}")
    for name, qi, pos, neg in cases:
        ims = frames_for(*true_of[qi][0])
        P = patch_vectors(model, proc, dev, ims)
        qp, qn = _text_vec(pos), _text_vec(neg)
        D = P.shape[-1]
        G = int(round(np.sqrt(P.shape[1])))
        for b in (1, 2, 3, 4):
            X = P if b == 1 else pool_grid(P, b)
            g = G // b
            for d in (D, 384, 256, 128):
                if d == D:
                    Y, a, n_ = X, qp, qn
                else:
                    R = Vt[:d].T
                    Y = X @ R
                    Y /= np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-8
                    a = qp @ R; a = a / np.linalg.norm(a)
                    n_ = qn @ R; n_ = n_ / np.linalg.norm(n_)
                sp, sn = float((Y @ a).max()), float((Y @ n_).max())
                print(f"{name:>10} {g:>2}x{g:<3} {d:>6} "
                      f"{g * g * d * 2:>12,} {sp:>9.3f} {sn:>8.3f} "
                      f"{sp - sn:>9.3f}")
        print()


if __name__ == "__main__":
    main()
