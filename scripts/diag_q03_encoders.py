"""#1 GATE: can any available encoder beat iv2 on q03's KIND, clip-to-clip?

q03's fusion (0.74) sits at its oracle single channel (iv2, 0.77), so
the only lever left is a stronger video-semantic space. VideoPrism lost
the TEXT probe (eval/videoprism_probe.json) - but that tested its text
tower; QbE needs video-to-video, which is the representation VideoPrism
exists for. This gate measures exactly that, on a 45-episode bank,
before any 2,097-episode build:

    for q03's 5 protocol seeds + 40 background episodes, embed each
    episode as one clip vector per encoder; AUC of cross-seed similarity
    against seed-to-background. Same statistic that predicted every
    failure this session.

Baseline read straight from the store's iv2_vectors - no recompute.

    python scripts/diag_q03_encoders.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
SCRATCH = Path("/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
               "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3"
               "/scratchpad")
sys.path.insert(0, str(SCRATCH / "videoprism"))
sys.path.insert(0, str(SCRATCH / "tfstub"))

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402

QI = 3
NBG = 40
NFRAMES = 8


def bank(db):
    import pyarrow.parquet as pq
    ep = db.table("episodes").scan()
    keys = [(str(s), int(a), int(b)) for s, a, b in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist(),
                ep.column("t1").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G = {(int(q), int(e)): int(v) for q, e, v in
         zip(t["query_id"], t["episode_index"], t["true"])}
    truths = [i for i, e in enumerate(eidx) if G.get((QI, e)) == 1]
    rs = np.random.RandomState(0)
    seeds = sorted(set(int(x) for x in rs.choice(truths, 5, replace=False)))
    rng = np.random.default_rng(0)
    pool = [i for i in range(len(keys)) if i not in set(seeds)]
    bg = sorted(int(x) for x in rng.choice(pool, NBG, replace=False))
    return keys, seeds, bg


def auc_of(vecs, seeds, bg):
    V = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in vecs.items()}
    cross, back = [], []
    for e in seeds:
        if e not in V:
            continue
        for e2 in seeds:
            if e2 != e and e2 in V:
                cross.append(float(V[e] @ V[e2]))
        for e2 in bg:
            if e2 in V:
                back.append(float(V[e] @ V[e2]))
    lab = np.r_[np.ones(len(cross)), np.zeros(len(back))]
    val = np.r_[np.array(cross), np.array(back)]
    r = val.argsort().argsort()
    auc = (r[lab == 1].mean() - (len(cross) - 1) / 2) / max(len(back), 1)
    return float(auc), float(np.median(cross) - np.median(back))


def decode_ep(db, ft, key, n, size):
    import cv2
    s, a, b = key
    sel = ft.filter(pc.and_(
        pc.equal(ft.column("stream"), s),
        pc.and_(pc.greater_equal(ft.column("ts"), a),
                pc.less_equal(ft.column("ts"), b))))
    if len(sel) < 4:
        return None
    pick = np.unique(np.linspace(0, len(sel) - 1, n).round().astype(int))
    dec = FrameSet(db, "frames", sel.take(pick)).decode()
    if not dec:
        return None
    return [cv2.resize(f, (size, size)) for _, f in sorted(dec)]


def main():
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    keys, seeds, bg = bank(db)
    wanted = seeds + bg
    print(f"q{QI:02d} bank: {len(seeds)} seeds + {len(bg)} background")

    # baseline: the store's own iv2 vectors
    iv = db.table("iv2_vectors").scan().to_pydict()
    pos = {(str(s), int(a)): i for i, (s, a) in
           enumerate(zip(iv["stream"], iv["ts"]))}
    IV = np.asarray(iv["vector"], np.float32).reshape(len(iv["ts"]), -1)
    vecs = {}
    for e in wanted:
        j = pos.get((keys[e][0], keys[e][1]))
        if j is not None:
            vecs[e] = IV[j]
    a, g = auc_of(vecs, seeds, bg)
    print(f"iv2 (store)        AUC {a:.3f}   gap {g:+.3f}")

    ft = db.table("frames").scan()
    from videoprism import models as vp
    import jax
    from huggingface_hub import hf_hub_download
    vp.TEXT_TOKENIZERS['c4_en']['model_path'] = hf_hub_download(
        repo_id="t5-base", filename="spiece.model")
    tok = vp.load_text_tokenizer('c4_en')
    ids, pads = vp.tokenize_texts(tok, ["x"])   # dummy; video tower only
    for cfg, name in (("videoprism_lvt_public_v1_base", "vp_B_248M"),
                      ("videoprism_lvt_public_v1_large", "vp_L_580M")):
        t0 = time.time()
        model = vp.get_model(cfg)
        state = vp.load_pretrained_weights(cfg)

        @jax.jit
        def fwd(v):
            ve, te, _ = model.apply(state, v, ids, pads, train=False)
            return ve

        vecs = {}
        from tqdm import tqdm
        for e in tqdm(wanted, desc=name, unit="ep"):
            fr = decode_ep(db, ft, keys[e], NFRAMES, 288)
            if fr is None:
                continue
            x = (np.stack(fr).astype(np.float32) / 255.0)[None]
            ve = np.asarray(fwd(x))[0]
            vecs[e] = ve.reshape(-1) if ve.ndim == 1 else ve.mean(0).reshape(-1)
        a, g = auc_of(vecs, seeds, bg)
        print(f"{name:<18} AUC {a:.3f}   gap {g:+.3f}   "
              f"({time.time()-t0:.0f}s incl load)", flush=True)


if __name__ == "__main__":
    main()
