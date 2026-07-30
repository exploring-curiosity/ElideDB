"""Student: the two product metrics AND the two timings.

Retrieval is one text forward plus one matmul over the corpus, because
every video-side computation was moved to write time. The teacher can
never do this - its cross-encoder is 0.4s per episode and its tokens
are 3.24 GB corpus-wide, unstorable and unpoolable.

Reports, for teacher and student side by side:
  yield / prec   per query at k = ceil(1.5 * support)
  read           wall-clock per query, cold and warm
  write          per-episode cost of the student's own encode step

  python scripts/bench_student.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402


def main():
    import torch
    import torch.nn as nn

    from elidedb.scenario import _episodes
    d = ROOT / "models/student_v1"
    ck = torch.load(d / "student.pt", map_location="cpu",
                    weights_only=True)
    emb = np.load(d / "episode_emb.npz", allow_pickle=True)
    E = np.asarray(emb["E"], np.float32)
    keys = list(zip([str(s) for s in emb["streams"]],
                    [int(v) for v in emb["ts"]]))

    class Tower(nn.Module):
        def __init__(self, d_in, dd):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(),
                                   nn.Linear(512, dd))

        def forward(self, x):
            y = self.f(x)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    class Rerank(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(4 * d, 256), nn.GELU(),
                                   nn.Linear(256, 64), nn.GELU(),
                                   nn.Linear(64, 1))

        def forward(self, qe, ee):
            q = qe.unsqueeze(1).expand_as(ee)
            return self.f(torch.cat([q, ee, q * ee, (q - ee).abs()],
                                    -1)).squeeze(-1)

    q_t = Tower(ck["d_q"], ck["dim"])
    q_t.load_state_dict(ck["q"])
    q_t.eval()
    rr = None
    if "rr" in ck:
        rr = Rerank(ck["dim"]); rr.load_state_dict(ck["rr"]); rr.eval()

    db = Store.open("lake/bench")
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    from elidedb.pe import _text_vec as pe_text
    pe_text("warm the text tower")                 # exclude model load

    print(f"{'q':>4} {'sup':>4} {'K':>4} | {'yield':>6} {'prec':>6} "
          f"| {'ms':>7}")
    ys, ps, ms = [], [], []
    for qi in sorted(sup):
        if sup[qi] == 0:
            continue
        K = int(np.ceil(sup[qi] * 1.5))
        t0 = time.perf_counter()
        qv = np.asarray(pe_text(QUERIES[qi]), np.float32)
        qv /= np.linalg.norm(qv) + 1e-8
        with torch.no_grad():
            qe = q_t(torch.tensor(qv)[None]).numpy()[0]
        sc = E @ qe
        # STAGE 2, the cascade: rerank a head of stage 1's list, the
        # same shape as the teacher's ITM cascade over its top-N
        if rr is not None:
            HEAD = max(K, 100)
            cand = np.argsort(-sc)[:HEAD]
            with torch.no_grad():
                r2 = rr(torch.tensor(qe)[None],
                        torch.tensor(E[cand])[None]).numpy()[0]
            top = cand[np.argsort(-r2)][:K]
        else:
            top = np.argsort(-sc)[:K]
        dt = (time.perf_counter() - t0) * 1000
        lab = np.array([1 if truth.get((qi, k[0], k[1])) == 1 else 0
                        for k in keys])
        tru = int(lab[top].sum())
        y, p = tru / sup[qi], tru / K
        ys.append(y); ps.append(p); ms.append(dt)
        print(f"q{qi:02d} {sup[qi]:>4} {K:>4} | {y:>6.2f} {p:>6.2f} "
              f"| {dt:>7.1f}")
    meta = json.loads((d / "meta.json").read_text())
    print(f"\nSTUDENT  mean yield {np.mean(ys):.2f}  prec {np.mean(ps):.2f}"
          f"  |  read {np.mean(ms):.1f} ms/query "
          f"(p50 {np.median(ms):.1f})")
    print(f"TEACHER  mean yield 0.42  prec 0.30"
          f"  |  read 2,000-75,000 ms/query (ITM cascade)")
    print(f"student {meta['params']/1e6:.2f}M params, "
          f"episode side {E.nbytes/1e6:.2f} MB for {len(keys)} demos, "
          f"trained on {meta['train_queries']} generated queries "
          f"(labels {meta['label_seconds']:.0f}s, "
          f"train {meta['train_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
