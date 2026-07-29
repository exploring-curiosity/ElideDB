"""Head-to-head: VideoPrism LvT-B / LvT-L vs InternVideo2 on OUR truthset.

The published MSR-VTT numbers are generic web video. This corpus is
fixed-camera robot manipulation where the signal is direction and
contact, so the only comparison that decides anything is on the frozen
truthset, over an IDENTICAL candidate pool for every model.

Fairness rules, enforced here:
  - one episode pool, every model ranks the same candidates
  - every truthset-TRUE episode is in the pool (never sample away the
    answers), fillers drawn deterministically
  - each model uses its OWN text tower; no cross-model vector mixing

JAX is CPU-only on this machine while IV2 runs on the GPU, so this
reports QUALITY and SIZE. It deliberately reports no speed comparison:
that would measure the backend, not the model.

  python scripts/videoprism_probe.py [--n 120] [--frames 8] [--model both]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
SCRATCH = Path(os.environ.get(
    "VP_SCRATCH",
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-MyProjects-"
    "StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/scratchpad"))
sys.path.insert(0, str(SCRATCH / "videoprism"))
sys.path.insert(0, str(SCRATCH / "tfstub"))

from elidedb.store import Store  # noqa: E402
from elidedb.video import FrameSet  # noqa: E402

QUERIES = [
    "pick up a green object from table and put it into the drawer",
    "pick up a yellow object from table and put it into the drawer",
    "pick up a red object from the drawer and put it on the table",
    "pick up a vessel and put it on the stove",
    "robot arm holds the handle and closes the drawer",
    "robot arm opens the drawer",
    "fold a piece of towel",
    "put the lid on the pot",
    "place the spoon on top of the cloth",
    "put the eggplant into the drawer",
    "put the banana on top of the drawer",
]
STORE = "lake/_bench_recovered"


def truth_map():
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {}
    for q, s, t0, v in zip(t["query_id"], t["stream"], t["t0"], t["true"]):
        truth[(int(q), s, int(t0))] = int(v)
    return truth


def build_pool(db, truth, n):
    """Every graded-true episode, plus deterministic fillers."""
    ep = db.table("episodes").scan()
    allk = list(zip(ep.column("stream").to_pylist(),
                    [int(v) for v in ep.column("ts").to_pylist()],
                    [int(v) for v in ep.column("t1").to_pylist()]))
    true_keys = {(s, t0) for (q, s, t0), v in truth.items() if v == 1}
    must = [k for k in allk if (k[0], k[1]) in true_keys]
    rest = [k for k in allk if (k[0], k[1]) not in true_keys]
    rng = np.random.default_rng(0)
    take = max(n - len(must), 0)
    idx = rng.choice(len(rest), size=min(take, len(rest)), replace=False)
    pool = must + [rest[i] for i in sorted(idx)]
    return pool, len(must)


# Deterministic ts -> source frame index (scripts/bridge4h.py assigns
# ts = EPOCH_NS + file_index*FILE_STRIDE_NS + frame*1e9/FPS).
EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
SRC_DIR = ROOT / "data/bridge/videos/observation.images.image_0/chunk-000"


def decode(db, k, nframes, size=288):
    """Frames straight from the ORIGINAL source, not through the store's
    byte-range index.

    The recovered store carries the pre-fix frame index whose byte
    offsets were paired with decode-ordered packets (the B-frame bug),
    so byte-range reads there raise on a negative length. A model
    comparison wants source pixels anyway: this reads them from the mp4
    the store was built from, which is also one generation less lossy."""
    import subprocess
    from elidedb.fftools import find
    fidx = int(k[0].rsplit("-", 1)[1])
    base = EPOCH_NS + fidx * FILE_STRIDE_NS
    start_frame = (k[1] - base) / 1e9 * FPS
    t_sec = max(start_frame / FPS, 0.0)
    src = SRC_DIR / f"file-{fidx:03d}.mp4"
    if not src.exists():
        return []
    r = subprocess.run(
        [find("ffmpeg"), "-v", "error", "-ss", f"{t_sec:.3f}", "-i", str(src),
         "-frames:v", str(nframes), "-vf", f"scale={size}:{size}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True)
    fb = size * size * 3
    n = len(r.stdout) // fb
    if n == 0:
        return []
    return [np.frombuffer(r.stdout[i*fb:(i+1)*fb], np.uint8).reshape(size, size, 3)
            for i in range(n)]


def resize(frames, size, nframes):
    # frames already arrive at `size` from the source decode above
    out = list(frames[:nframes])
    while len(out) < nframes:                 # pad short episodes by repeat
        out.append(out[-1] if out else np.zeros((size, size, 3), np.uint8))
    return (np.stack(out).astype(np.float32) / 255.0)


def run_videoprism(cfg, pool, db, nframes, size=288):
    import jax
    from videoprism import models as vp
    from huggingface_hub import hf_hub_download

    # the tokenizer vocab lives on gs://; the identical 32k T5 vocab is
    # mirrored on HF (same 791,656-byte sentencepiece model)
    vp.TEXT_TOKENIZERS['c4_en']['model_path'] = hf_hub_download(
        repo_id="t5-base", filename="spiece.model")

    t0 = time.perf_counter()
    model = vp.get_model(cfg)
    state = vp.load_pretrained_weights(cfg)
    tok = vp.load_text_tokenizer('c4_en')
    load_s = time.perf_counter() - t0

    @jax.jit
    def fwd(v, ids, pads):
        return model.apply(state, v, ids, pads, train=False)

    ids, pads = vp.tokenize_texts(tok, QUERIES)

    vids, per_ep = [], []
    for i, k in enumerate(pool):
        f = decode(db, k, nframes)
        if not f:
            vids.append(np.zeros(1))
            continue
        x = resize(f, size, nframes)[None]
        s = time.perf_counter()
        ve, te, _ = fwd(x, ids, pads)
        # JAX dispatches asynchronously: without block_until_ready the
        # timer measures enqueue time, not compute (it read 0.00 s/ep for
        # a 248M model, which is how the mistake surfaced).
        ve = jax.block_until_ready(ve)
        per_ep.append(time.perf_counter() - s)
        vids.append(np.asarray(ve)[0])
        if i == 0:
            txt = np.asarray(te)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(pool)} episodes "
                  f"({np.median(per_ep):.2f}s/ep)", flush=True)
    V = np.stack([v for v in vids if v.ndim == 1 and v.size > 1])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    T = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)
    return V, T, load_s, float(np.median(per_ep))


def iv2_scores(db, pool):
    from elidedb.iv2 import iv2_lookup
    S = np.zeros((len(QUERIES), len(pool)), np.float32)
    for qi, q in enumerate(QUERIES):
        look, _ = iv2_lookup(db, q)
        S[qi] = [look(*k) for k in pool]
    return S


def precision_at_k(S, pool, truth, k=10):
    """Fraction of the top-k that the truthset marks TRUE (ungraded
    counts as false, exactly as the frozen benchmark does)."""
    out = {}
    tot_t = tot_r = 0
    for qi in range(len(QUERIES)):
        order = np.argsort(-S[qi])[:k]
        t = sum(1 for i in order
                if truth.get((qi, pool[i][0], pool[i][1])) == 1)
        out[qi] = t
        tot_t += t
        tot_r += len(order)
    return tot_t, tot_r, out


def main():
    argv = sys.argv
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 120
    nframes = int(argv[argv.index("--frames") + 1]) if "--frames" in argv else 16
    which = argv[argv.index("--model") + 1] if "--model" in argv else "both"

    db = Store.open(STORE)
    truth = truth_map()
    pool, n_true = build_pool(db, truth, n)
    print(f"pool: {len(pool)} episodes ({n_true} graded-true, rest fillers)\n")

    results = {}
    ti = time.perf_counter()
    S = iv2_scores(db, pool)
    t, r, per = precision_at_k(S, pool, truth)
    results["iv2_1B"] = {"true": t, "ret": r, "prec": round(t / max(r, 1), 3),
                         "per_query": per, "params_m": 1000}
    print(f"iv2 (1B)         true {t:>3}/{r}  prec {t/max(r,1):.3f}   "
          f"({time.perf_counter()-ti:.0f}s)")

    cfgs = []
    if which in ("both", "base"):
        cfgs.append(("videoprism_lvt_public_v1_base", "vp_lvt_B_248M", 248))
    if which in ("both", "large"):
        cfgs.append(("videoprism_lvt_public_v1_large", "vp_lvt_L_580M", 580))
    for cfg, label, pm in cfgs:
        print(f"\n{label}: embedding {len(pool)} episodes on CPU...", flush=True)
        V, T, load_s, per_ep = run_videoprism(cfg, pool, db, nframes)
        Sv = T @ V.T
        t, r, per = precision_at_k(Sv, pool[:V.shape[0]], truth)
        results[label] = {"true": t, "ret": r,
                          "prec": round(t / max(r, 1), 3),
                          "per_query": per, "params_m": pm,
                          "load_s": round(load_s, 1),
                          "cpu_s_per_episode": round(per_ep, 2)}
        print(f"{label:16s} true {t:>3}/{r}  prec {t/max(r,1):.3f}   "
              f"(load {load_s:.0f}s, {per_ep:.2f}s/ep on CPU)")

    (ROOT / "eval/videoprism_probe.json").write_text(json.dumps(results, indent=1))
    print("\n" + json.dumps({k: {kk: vv for kk, vv in v.items()
                                 if kk != "per_query"}
                             for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
