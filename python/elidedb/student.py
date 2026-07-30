"""The student, as a servable search path.

This existed only inside scripts/bench_student.py, which meant the 27 ms
number was real but unreachable: the Desk called search_set - the
TEACHER - so every query in the UI paid the teacher's cost. Measured on
lake/bench:

    teacher, cold (loads PE + SigLIP2 + IV2 + V-JEPA + X-CLIP)  59,124 ms
    teacher, warm, repeat query (cache hit)                        116 ms
    teacher, warm, NEW query                                     8,394 ms
    student                                                         27 ms

The teacher is not the serving path and never was - it costs 0.4 s per
episode in its cross-encoder alone. It exists to produce labels. This
module is what should answer a user.

Four stages, the same ones the teacher runs, but every video-side
computation was moved to write time:

    1  bi-encoder recall   text tower -> matmul over episode vectors
    2  PRF                 the head of pass 1 re-queries the corpus
    3  structural gate     transition anchor + event membership +
                           motion-space density, all precomputed columns
    4  cascade rerank      small listwise head over the top-100

Stage order is PRF -> gate -> cascade, which was MEASURED rather than
inherited: copying the teacher's order (cascade before gate) scored
0.30 -> 0.26. The student's cascade is a small head, not a 1B
cross-encoder, and it does better consuming the gate's evidence than
overriding it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

_S: dict = {}
DIR = Path(__file__).resolve().parents[2] / "models/student_v1"


def available() -> bool:
    return (DIR / "student.pt").exists() and (DIR / "episode_emb.npz").exists()


def _load():
    """Weights + episode matrix, once per process."""
    if _S:
        return _S
    import torch
    import torch.nn as nn

    ck = torch.load(DIR / "student.pt", map_location="cpu",
                    weights_only=True)
    emb = np.load(DIR / "episode_emb.npz", allow_pickle=True)

    class Tower(nn.Module):
        def __init__(self, d_in, dd):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(),
                                   nn.Linear(512, dd))

        def forward(self, x):
            y = self.f(x)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    class Rerank(nn.Module):
        def __init__(self, dd):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(4 * dd, 256), nn.GELU(),
                                   nn.Linear(256, 64), nn.GELU(),
                                   nn.Linear(64, 1))

        def forward(self, qe, ee):
            q = qe.unsqueeze(1).expand_as(ee)
            return self.f(torch.cat([q, ee, q * ee, (q - ee).abs()],
                                    -1)).squeeze(-1)

    qt = Tower(ck["d_q"], ck["dim"])
    qt.load_state_dict(ck["q"])
    qt.eval()
    rr = None
    if "rr" in ck:
        rr = Rerank(ck["dim"])
        rr.load_state_dict(ck["rr"])
        rr.eval()
    _S.update(torch=torch, q=qt, rr=rr,
              E=np.asarray(emb["E"], np.float32),
              keys=list(zip([str(s) for s in emb["streams"]],
                            [int(v) for v in emb["ts"]])),
              dim=ck["dim"])
    return _S


def _episode_spans(store, keys):
    """(stream, ts) -> t1, so a hit can be returned as a real clip."""
    ck = ("spans", str(store.dir), store.table("episodes").state().version)
    if ck not in _S:
        ep = store.table("episodes").scan().to_pydict()
        _S[ck] = {(str(s), int(a)): int(b) for s, a, b in
                  zip(ep["stream"], ep["ts"], ep["t1"])}
    return _S[ck]


def search_student(store, text: str, k_max: int = 10) -> dict:
    """Same result shape as scenario.search_set, ~300x faster."""
    from .pe import _text_vec as pe_text
    from .scenario import (_anchor_boost, _query_transitions,
                           _demo_transitions, _motion_density)
    t0 = time.perf_counter()
    S = _load()
    torch = S["torch"]
    E, keys = S["E"], S["keys"]

    qv = np.asarray(pe_text(text), np.float32)
    qv /= np.linalg.norm(qv) + 1e-8
    with torch.no_grad():
        qe = S["q"](torch.tensor(qv)[None]).numpy()[0]

    sc = E @ qe                                            # 1 recall
    seed = np.argsort(-sc)[:max(25, k_max // 4)]           # 2 PRF
    c = E[seed].mean(0)
    c /= np.linalg.norm(c) + 1e-8
    sc = sc + 0.5 * (E @ c)

    stages = ["recall", "prf"]
    need = _query_transitions(text.lower())                # 3 structural
    try:
        sc, anc = _anchor_boost(store, keys, need, sc)
        if anc is not None:
            stages.append("anchor")
        if need:
            kinds = _demo_transitions(store, keys)
            have = np.array([1.0 if (kinds.get(i) or set()) & need else 0.0
                             for i in range(len(keys))])
            head = np.argsort(-sc)[:max(k_max, 20)]
            if float(have[head].mean()) >= 0.35:
                sc = sc + 0.5 * have * float(sc.std())
                stages.append("evk")
        dens = _motion_density(store, keys, sc, k_max)
        if dens is not None:
            sc = sc + 0.5 * dens * float(sc.std())
            stages.append("dens")
    except Exception:
        pass

    if S["rr"] is not None:                                # 4 cascade
        cand = np.argsort(-sc)[:max(k_max, 100)]
        with torch.no_grad():
            r2 = S["rr"](torch.tensor(qe)[None],
                         torch.tensor(E[cand])[None]).numpy()[0]
        top = cand[np.argsort(-r2)][:k_max]
        stages.append("cascade")
    else:
        top = np.argsort(-sc)[:k_max]

    # NO-MATCH GATE. The teacher abstains when the corpus does not
    # contain the action at all, and the product contract is "whatever
    # comes back is true to the query" - a fast path that always returns
    # k clips would quietly break that. Costs 24-76 ms and it is the
    # difference between answering and guessing.
    try:
        from .scenario import _auto_action_support
        g = _auto_action_support(store, text)
        if g is not None and g["max_p"] < 0.05:
            return {"clips": [], "borderline": [], "audit": None,
                    "direction_filtered": 0,
                    "channels": stages + ["no-match gate"],
                    "channels_failed": {}, "degraded": [],
                    "engine": "student", "no_match": True,
                    "scored": len(keys),
                    "ms": round((time.perf_counter() - t0) * 1e3, 1)}
    except Exception:
        pass

    spans = _episode_spans(store, keys)
    clips = [{"stream": keys[i][0], "t0": keys[i][1],
              "t1": spans.get(keys[i], keys[i][1]),
              "score": float(sc[i])} for i in top]
    return {"clips": clips, "borderline": [], "audit": None,
            "direction_filtered": 0, "channels": stages,
            "channels_failed": {}, "degraded": [],
            "engine": "student", "scored": len(keys),
            "ms": round((time.perf_counter() - t0) * 1e3, 1)}
