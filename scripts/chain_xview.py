"""CROSS-VIEW contrastive episode encoder: the corpus supervises.

Every measured route dies the same death: chain retrieval needs
near-oracle SEQUENCE fidelity, and both symbolic pipelines (stage
recalls compound: 0.85-good stages -> 0.2-0.3 bench) and fixed
continuous channels (novelty rhythm caps at 0.45 - rhythm carries no
slot structure) fall short. The one supervision signal nobody hand-
built is the corpus's own two cameras: simA and simB film THE SAME
chain. An encoder trained so the two views of an episode embed
together against 149 other episodes must discard camera nuisance and
keep what both views share - the manipulation structure itself.

Self-supervised (no truth anywhere), vision-native (inputs are the
scene-novelty rhythm + detector event series), tiny (a 1-D conv+GRU,
~60k params, minutes on CPU).

    python scripts/chain_xview.py [--store lake/sim_chains]
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
from chain_channels import (episode_spans, by_episode, bench_S,  # noqa: E402
                            DEV, HOLD, norm)

BIN_S = 0.5
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad")


def view_series(db, spans):
    """{(episode, stream) -> (T, C) float32 series}: scene novelty
    profile + detector event indicators on a common 0.5s grid."""
    ep_t0 = {e: t0 for t0, _, e in spans}
    ep_t1 = {e: t1 for _, t1, e in spans}
    out = {}
    for sv in ("simA", "simB"):
        vecs = by_episode(db, "scene_vectors", spans, stream=sv)
        for e, V in vecs.items():
            if len(V) < 8:
                continue
            d = 1.0 - (V[1:] * V[:-1]).sum(1)
            d = np.convolve(d, np.ones(5) / 5, mode="valid")
            nb = int((ep_t1[e] - ep_t0[e]) / (BIN_S * 1e9)) + 1
            # frame series -> bin grid (frames are uniform in time)
            idx = np.linspace(0, len(d) - 1, nb).round().astype(int)
            prof = d[idx]
            out[(e, sv)] = prof
    # detector events from the chain_delta cache
    z = np.load(os.environ.get("ELIDEDB_DELTA_CACHE",
                               str(SCRATCH / "delta_events.npz")),
                allow_pickle=True)
    events = [list(x) for x in z["events"]]
    import chain_delta as cd
    dirs, junk = cd.classify_events(events)
    mags = np.array([x[5] for x in events])
    mq = np.percentile(mags, [50])
    ev = defaultdict(list)
    for x, dr, jk in zip(events, dirs, junk):
        if not jk:
            ev[(int(x[0]), str(x[1]))].append(
                (int(x[2]), int(dr), float(x[5])))
    series = {}
    for (e, sv), prof in out.items():
        nb = len(prof)
        M = np.zeros((nb, 5), np.float32)
        M[:, 0] = prof / (prof.std() + 1e-6)
        for t, dr, mg in ev.get((e, sv), []):
            b = int((t - ep_t0[e]) / (BIN_S * 1e9))
            if 0 <= b < nb:
                M[b, 1 + (dr + 1)] = 1.0        # dep/push/arr one-hot
                M[b, 4] = max(M[b, 4],
                              float(mg > mq[0]) + 1.0)
        series[(e, sv)] = M
    return series


def train(series, dim=64, epochs=300, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    keys = sorted(series)
    eps = sorted({e for e, _ in keys})
    C = next(iter(series.values())).shape[1]

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(C, 32, 5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
            self.gru = nn.GRU(32, dim, batch_first=True,
                              bidirectional=True)
            self.out = nn.Linear(2 * dim, dim)

        def forward(self, xs):
            embs = []
            for x in xs:
                h = self.conv(x.T[None])            # 1,C,T -> 1,32,T
                o, _ = self.gru(h.transpose(1, 2))
                embs.append(self.out(o.mean(1))[0])
            E = torch.stack(embs)
            return E / E.norm(dim=1, keepdim=True).clamp_min(1e-8)

    enc = Enc()
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    tens = {k: torch.tensor(v) for k, v in series.items()}
    rng = np.random.default_rng(seed)
    tau = 0.1
    for it in range(epochs):
        # augment: random temporal crop per sample
        xs, owner = [], []
        for e in eps:
            for sv in ("simA", "simB"):
                x = tens.get((e, sv))
                if x is None:
                    continue
                T = len(x)
                c0 = int(rng.integers(0, max(T // 5, 1)))
                c1 = T - int(rng.integers(0, max(T // 5, 1)))
                xs.append(x[c0:c1])
                owner.append(e)
        E = enc(xs)
        owner = torch.tensor([eps.index(o) for o in owner])
        S = E @ E.T / tau
        S.fill_diagonal_(-1e9)
        pos = owner[:, None] == owner[None, :]
        pos.fill_diagonal_(False)
        loss = 0.0
        n = 0
        lsm = torch.log_softmax(S, dim=1)
        for i in range(len(xs)):
            j = pos[i].nonzero()
            if len(j):
                loss = loss - lsm[i, j[0, 0]]
                n += 1
        loss = loss / max(n, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            with torch.no_grad():
                E = enc([tens[k] for k in keys])
                acc = 0
                for i, (e, sv) in enumerate(keys):
                    o = (e, "simB" if sv == "simA" else "simA")
                    if o not in series:
                        continue
                    sims = E[i] @ E.T
                    sims[i] = -1e9
                    acc += int(keys[int(sims.argmax())][0] == e)
                print(f"  epoch {it+1}: loss {float(loss):.3f}  "
                      f"cross-view top1 {acc}/{len(keys)}",
                      flush=True)
    with torch.no_grad():
        E = enc([tens[k] for k in keys]).numpy()
    emb = defaultdict(list)
    for (e, sv), v in zip(keys, E):
        emb[e].append(v)
    return {e: norm(np.mean(v, 0)) for e, v in emb.items()}


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    spans = episode_spans(db)
    eps = [e for _, _, e in spans]
    print("building view series...", flush=True)
    series = view_series(db, spans)
    print(f"{len(series)} view-series", flush=True)
    emb = train(series)
    n = len(eps)
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = emb.get(eps[i]), emb.get(eps[j])
            if a is None or b is None:
                continue
            S[i, j] = S[j, i] = float(a @ b)
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    print("-- cross-view contrastive channel")
    bench_S(S, eps, tmpl, DEV, "xview DEV")
    bench_S(S, eps, tmpl, HOLD, "xview HOLDOUT")
    np.savez(SCRATCH / "xview_S.npz", eps=np.array(eps), S=S)


if __name__ == "__main__":
    main()
