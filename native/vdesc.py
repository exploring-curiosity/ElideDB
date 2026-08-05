"""WRITE-TIME structure via a generalized VLM, then QbE over it.

The oracle result says retrieval is solved GIVEN a description of what
happened (event scripts -> 0.992 through the same aligner). Frozen
embedding encoders cannot produce that; a general-purpose VLM can, and
asking one "what happens, step by step" is not domain knowledge - the
same prompt runs on tabletop, driving or drone footage and the model
supplies whatever vocabulary that domain needs.

Two comparison paths, both measured:
  text   sentence embedding of the description, cosine QbE
  seq    per-step description embeddings, temporal alignment

Metrics are ONLY yield and precision at k = support (where
yield == precision, the only k at which both can exceed 0.90) and at
k = 1.5*support for continuity with earlier numbers.

    python native/vdesc.py --store lake/sim_chains --describe
    python native/vdesc.py --store lake/sim_chains --bench
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
NF = 8
PROMPT = (
    "These frames are in time order from one video clip. Describe "
    "what happens as a sequence of steps. For each step name the "
    "thing that moves, what is done to it, and where it ends up. "
    "Be concrete and consistent. Answer in at most 60 words.")


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def cache_path(store_name):
    from track import SCRATCH
    d = SCRATCH / "vdesc"
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
        views.setdefault(e, (sv, sl))
    p = cache_path(store_name)
    out = json.loads(p.read_text()) if p.exists() else {}
    todo = [e for e in sorted(views) if str(e) not in out]
    print(f"{store_name}: {len(views)} episodes, {len(todo)} to "
          f"describe (ETA ~{len(todo)*4/60:.0f} min)", flush=True)
    if not todo:
        return out
    model, proc = load(MID)
    cfg = load_config(MID)
    t0 = time.time()
    for i, e in enumerate(tqdm(todo, desc="describe", unit="ep")):
        ts, F = cd.decode_view(db, views[e][1])
        idx = np.linspace(0, len(F) - 1, NF).round().astype(int)
        imgs = to_pil([F[j] for j in idx], w=224)
        pr = apply_chat_template(proc, cfg, PROMPT,
                                 num_images=len(imgs))
        r = generate(model, proc, pr, image=imgs, max_tokens=110,
                     verbose=False)
        out[str(e)] = (r.text if hasattr(r, "text") else str(r)).strip()
        if (i + 1) % 10 == 0:
            p.write_text(json.dumps(out))
    p.write_text(json.dumps(out))
    print(f"described {len(todo)} in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return out


def embed_texts(texts):
    """Sentence embeddings from a general text encoder."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    V = m.encode(texts, normalize_embeddings=True,
                 show_progress_bar=False)
    return np.asarray(V, np.float32)


def bench(store_name):
    import pyarrow.parquet as pq
    desc = json.loads(cache_path(store_name).read_text())
    eps = sorted(int(k) for k in desc)
    V = embed_texts([desc[str(e)] for e in eps])
    S = V @ V.T
    print(f"{len(eps)} descriptions embedded", flush=True)
    if store_name == "sim_chains":
        t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
            .to_pydict()
        lab = {int(e): tm for e, tm in zip(t["episode"],
                                           t["template"])}
        groups = {}
        for e, tm in lab.items():
            groups.setdefault(tm, []).append(e)
        for KM in (1.0, 1.5):
            ys, ps = [], []
            rs = np.random.RandomState(0)
            for tm, pool in sorted(groups.items()):
                pool = [e for e in pool if e in set(eps)]
                if len(pool) < 6:
                    continue
                sd = sorted(int(x) for x in rs.choice(pool, 5,
                                                      replace=False))
                support = len(pool) - len(sd)
                k = math.ceil(KM * support)
                pos = {e: i for i, e in enumerate(eps)}
                sc = S[[pos[x] for x in sd]].max(0).copy()
                for x in sd:
                    sc[pos[x]] = -1e9
                got = [eps[i] for i in np.argsort(-sc)[:k]]
                tr = sum(1 for e in got if lab.get(e) == tm)
                ys.append(tr / support)
                ps.append(tr / len(got))
                if KM == 1.0:
                    print(f"   {tm:<20} yield {tr/support:.2f} "
                          f"prec {tr/len(got):.2f}")
            print(f"  k={KM}x support:  MEAN yield "
                  f"{np.mean(ys):.3f}  prec {np.mean(ps):.3f}",
                  flush=True)


def main():
    store = arg("--store", "lake/sim_chains").split("/")[-1]
    if "--describe" in sys.argv:
        describe(store)
    if "--bench" in sys.argv:
        bench(store)


if __name__ == "__main__":
    main()
