#!/usr/bin/env python3
"""RelMo, held open. Runs in RelMo's own interpreter; speaks JSON lines.

    {"cmd":"text","texts":[...]}            -> SigLIP2 text-tower vectors
    {"cmd":"rank","query":id,"cand":[ids]}  -> stage-2 DTW over the traces
    {"cmd":"pair","a":"...","b":"..."}      -> cosine between two sentences

Two jobs, and they are the two halves of the demo.

THE TEXT ARM is `text`. SigLIP2's text tower embeds a sentence into the space
its image tower shares, which is the only way words can reach video without
captioning every frame first. This is not a weakened baseline built to lose:
it is how zero-shot text-to-video retrieval is done, and the `pair` command
exists so anyone can check the thing that makes it fail: "open the drawer" and
"close the drawer" come out of that tower nearly parallel.

THE EXAMPLE ARM is `rank`. The database has already pruned the corpus by
cosine over RelMo's prefilter; this ranks the survivors by DTW over their full
descriptor traces, which is the part that sees the order things happened in.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import blob                                                        # noqa: E402

STORE = "rcasa"
SIGLIP = "google/siglip2-base-patch16-224"


class Engine:
    """Holds a text tower and nothing else.

    It used to open the whole RelMo store, which is 17.64 GB of peak RSS and a
    6.53 GB padded DTW bank over all 3,556 recordings, to answer queries that
    never touch more than 48. Traces are read from disk per query instead: 11 ms
    for a shortlist, 30 MB resident, and the process now fits on a free CPU host
    with room to spare. `dump_traces.py` writes them.
    """

    def __init__(self):
        self._txt = None
        self._cache: dict = {}
        n = blob.stats()["cached_traces"]
        where = f"s3://{blob.BUCKET}/traces + cache" if blob.enabled() else str(blob.CACHE)
        print(f"[showreel] {n} traces cached, source {where}",
              file=sys.stderr, flush=True)
        if not n and not blob.enabled():
            print("[showreel] no traces: run dump_traces.py, or stage 2 is skipped",
                  file=sys.stderr, flush=True)
        self.n_traces = n

    # ---- words --------------------------------------------------------------

    def _tower(self):
        if self._txt is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._txt = (AutoModel.from_pretrained(SIGLIP, dtype=torch.float32).eval(),
                         AutoProcessor.from_pretrained(SIGLIP), torch)
        return self._txt

    def text(self, texts: list[str]) -> list[list[float]]:
        m, pr, torch = self._tower()
        with torch.no_grad():
            t = pr(text=list(texts), padding="max_length", max_length=64,
                   return_tensors="pt")
            e = m.get_text_features(**t).float().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        return [[round(float(x), 7) for x in row] for row in e]

    def pair(self, a: str, b: str) -> float:
        v = np.asarray(self.text([a, b]), np.float32)
        return float(v[0] @ v[1])

    # ---- the trace ----------------------------------------------------------

    def _zs(self, rid):
        """One recording's DTW representation, off disk, memoised.

        The cache is bounded: a demo asks about a few hundred recordings, and an
        unbounded dict here would slowly reintroduce exactly the resident corpus
        this design removed. `blob` decides whether the file is already on disk
        or has to come from S3 first; this function does not care which.
        """
        z = self._cache.get(rid)
        if z is None:
            p = blob.trace_path(rid)
            if p is None:
                return None
            z = np.load(p).astype(np.float32)
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[rid] = z
        return z

    def rank(self, query: str, cand: list[str], band: float = 0.25) -> dict:
        from relmo.vjmatch import dtw
        from relmo.vjzeval import PAD_COST, _pad

        # One parallel fetch before the loop, not 49 serial ones inside it.
        blob.prefetch([query] + list(cand))
        q = self._zs(query)
        if q is None:
            return {}
        pairs = [(c, self._zs(c)) for c in cand]
        pairs = [(c, z) for c, z in pairs if z is not None]
        if not pairs:
            return {}
        cand = [c for c, _ in pairs]
        Z = [z for _, z in pairs]
        P, ok = _pad(Z)
        L = np.array([len(z) for z in Z])
        C = 1.0 - np.einsum("sd,nkd->nsk", q, P)
        C = np.where(ok[:, None, :], C, PAD_COST)
        s = -dtw(C, False, L, band)
        return {c: round(max(0.0, 1.0 + float(v)), 6) for c, v in zip(cand, s)}

    # `vec` is gone with the store. The three vectors it returned are columns in
    # the database, written once by ingest.py, and reading them from there costs
    # a SELECT instead of 18 GB.


def main() -> int:
    # The protocol owns stdout; RelMo prints on first use and would corrupt it.
    out, sys.stdout = sys.stdout, sys.stderr
    reply = lambda o: print(json.dumps(o), file=out, flush=True)

    eng = None
    reply(dict(booting=True))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if eng is None:
                eng = Engine()
            c = req.get("cmd")
            if c == "ping":
                reply(dict(ok=True, n=eng.n_traces, **blob.stats()))
            elif c == "text":
                reply(dict(ok=True, vecs=eng.text(req["texts"])))
            elif c == "pair":
                reply(dict(ok=True, cos=eng.pair(req["a"], req["b"])))
            elif c == "rank":
                t = time.time()
                s = eng.rank(req["query"], req["cand"], float(req.get("band", 0.25)))
                reply(dict(ok=True, scores=s, ms=round((time.time()-t)*1e3, 1)))
            elif c == "quit":
                return 0
            else:
                reply(dict(error=f"unknown cmd {c!r}"))
        except Exception as exc:                                  # noqa: BLE001
            import traceback

            traceback.print_exc(file=sys.stderr)
            reply(dict(error=f"{type(exc).__name__}: {exc}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
