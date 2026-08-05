"""WINDOWED descriptions -> sequence alignment. The oracle's shape.

One summary per episode loses the chain: 8 frames over 30 s described
1-2 of 3-8 manipulations and scored 0.13/0.13 (worse than embeddings).
The oracle that scores 0.992 is not a summary - it is a SEQUENCE of
event descriptions. So describe each short window, embed each
description, and align the sequences.

Domain-blind: the prompt asks what happens in a window, nothing about
arms or blocks; the same call works on driving or drone footage.

    python native/vseq.py --store lake/sim_chains --describe
    python native/vseq.py --store lake/sim_chains --bench
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

MID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
WIN_S = 4.0
NF = 4
PROMPT = ("These frames are in time order from a few seconds of video. "
          "In ONE short sentence, say what changes: what moves and "
          "where it ends up. If nothing moves, say 'no change'.")


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def cpath(store_name):
    from track import SCRATCH
    d = SCRATCH / "vseq"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{store_name}.json"


def describe(store_name):
    import chain_delta as cd
    from elidedb import Store
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template
    from vlmrank import to_pil
    from tqdm import tqdm

    db = Store.open(str(ROOT / f"lake/{store_name}"))
    views = {}
    for e, sv, sl in cd.episode_views(db):
        views.setdefault(e, sl)
    p = cpath(store_name)
    out = json.loads(p.read_text()) if p.exists() else {}
    todo = [e for e in sorted(views) if str(e) not in out]
    print(f"{store_name}: {len(todo)} episodes to window-describe",
          flush=True)
    if not todo:
        return out
    model, proc = load(MID)
    cfg = load_config(MID)
    t0 = time.time()
    for i, e in enumerate(tqdm(todo, desc="windows", unit="ep")):
        ts, F = cd.decode_view(db, views[e])
        dur = (int(ts[-1]) - int(ts[0])) / 1e9
        nw = max(int(round(dur / WIN_S)), 2)
        bounds = np.linspace(0, len(F) - 1, nw + 1).astype(int)
        steps = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = np.linspace(a, max(b, a + 1), NF).round().astype(int)
            imgs = to_pil([F[min(j, len(F) - 1)] for j in idx], w=200)
            pr = apply_chat_template(proc, cfg, PROMPT,
                                     num_images=len(imgs))
            r = generate(model, proc, pr, image=imgs, max_tokens=28,
                         verbose=False)
            steps.append((r.text if hasattr(r, "text")
                          else str(r)).strip())
        out[str(e)] = steps
        if (i + 1) % 5 == 0:
            p.write_text(json.dumps(out))
    p.write_text(json.dumps(out))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    return out


def bench(store_name):
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer
    from chain_channels import dtw_sim
    desc = json.loads(cpath(store_name).read_text())
    eps = sorted(int(k) for k in desc)
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    seqs = {}
    for e in eps:
        steps = desc[str(e)] or ["no change"]
        V = m.encode(steps, normalize_embeddings=True,
                     show_progress_bar=False)
        seqs[e] = np.asarray(V, np.float32)
    print(f"{len(eps)} episodes, mean {np.mean([len(seqs[e]) for e in eps]):.1f} steps",
          flush=True)
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    lab = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    groups = {}
    for e, tm in lab.items():
        if e in seqs:
            groups.setdefault(tm, []).append(e)
    for mode in ("dtw", "pool"):
        for KM in (1.0, 1.5):
            rs = np.random.RandomState(0)
            ys, ps = [], []
            for tm, pool in sorted(groups.items()):
                if len(pool) < 6:
                    continue
                sd = sorted(int(x) for x in rs.choice(pool, 5,
                                                      replace=False))
                support = len(pool) - len(sd)
                k = math.ceil(KM * support)
                sc = {}
                for e in eps:
                    if e in sd:
                        continue
                    best = -9
                    for s in sd:
                        if mode == "dtw":
                            v = dtw_sim(seqs[s], seqs[e], band=0.5)
                        else:
                            v = float(seqs[s].mean(0) @ seqs[e].mean(0)
                                      / ((np.linalg.norm(seqs[s].mean(0))
                                          * np.linalg.norm(seqs[e].mean(0)))
                                         + 1e-8))
                        best = max(best, v)
                    sc[e] = best
                got = sorted(sc, key=lambda x: -sc[x])[:k]
                tr = sum(1 for e in got if lab.get(e) == tm)
                ys.append(tr / support)
                ps.append(tr / len(got))
            print(f"  {mode:<5} k={KM}x  yield {np.mean(ys):.3f}  "
                  f"prec {np.mean(ps):.3f}", flush=True)


def main():
    store = arg("--store", "lake/sim_chains").split("/")[-1]
    if "--describe" in sys.argv:
        describe(store)
    if "--bench" in sys.argv:
        bench(store)


if __name__ == "__main__":
    main()
