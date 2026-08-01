"""The metric stage B actually trained for: text -> ctx retrieval.

The centroid probe (close_vs_open 0.633) compares CLIP-side centroids and
never touches the TextAdapter — it cannot see what L3 learned. The product
question is: does adapter(embed_text("close the drawer")) rank held-out
CLOSE episodes above OPEN episodes in ctx space? Measured as pairwise AUC
per query, plus the cosine gap, on held-out time only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402

from elidedb.context import embed_texts                      # noqa: E402
from elidedb.fdnnv2 import load_v2, pool_event               # noqa: E402
from fdnnv2_train import CACHE, OUT, episode_groups, load_cache  # noqa: E402

QUERIES = {
    "close": ["close the drawer", "closing the drawer",
              "the robot closes the drawer"],
    "open": ["open the drawer", "opening the drawer",
             "the robot opens the drawer"],
    "putin": ["put something in the drawer", "place an object in the drawer"],
}


def auc(pos, neg):
    return float(np.mean([[p > n for n in neg] for p in pos]))


def main():
    model, adapter, _ = load_v2(OUT)
    px, vec, sid, ts, val = load_cache()
    streams = list(np.load(CACHE / "streams.npy"))
    groups = episode_groups(sid, ts, streams)

    # held-out episodes -> novelty-pooled ctx (the same pooling retrieval uses)
    ep_ctx = {}
    for name, eps in groups.items():
        held = [m for m in eps if val[m].mean() > 0.5][:60]
        vs = []
        for m in held:
            x = px[m][None].astype(np.float32) / 127.5 - 1.0
            out = model.run(mx.array(x))
            vs.append(pool_event(np.array(out["ctx"][0]),
                                 np.array(out["gates"][0]), 0, len(m)))
        ep_ctx[name] = np.stack(vs) if vs else np.zeros((0, 256))
        print(f"{name}: {len(vs)} held-out episodes", flush=True)

    # query vectors through the adapter
    report = {}
    for qname, qtexts in QUERIES.items():
        z = np.array(adapter(mx.array(embed_texts(qtexts))))
        for qt, q in zip(qtexts, z):
            scores = {n: ep_ctx[n] @ q for n in ep_ctx if len(ep_ctx[n])}
            row = {}
            for other in scores:
                if other == qname:
                    continue
                row[f"auc_vs_{other}"] = round(
                    auc(scores[qname], scores[other]), 3)
            row["mean_cos"] = {n: round(float(s.mean()), 4)
                               for n, s in scores.items()}
            report[qt] = row
            print(f"{qt!r:40s} {row}", flush=True)

    Path("bench") / "bench_fdnnv2_text.json".write_text(json.dumps(report, indent=2))
    print("saved -> bench_fdnnv2_text.json")


if __name__ == "__main__":
    main()
