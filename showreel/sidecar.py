#!/usr/bin/env python3
"""RelMo, held open. Runs in RelMo's own interpreter; speaks JSON lines.

    {"cmd":"text","texts":[...]}            -> SigLIP2 text-tower vectors
    {"cmd":"rank","query":id,"cand":[ids]}  -> stage-2 DTW over the traces
    {"cmd":"vec","id":rec_id}               -> a recording's three vectors
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
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

STORE = "rcasa"
SIGLIP = "google/siglip2-base-patch16-224"


class Engine:
    def __init__(self):
        from relmo.vjstore import Store

        t0 = time.time()
        print(f"[showreel] loading {STORE} ...", file=sys.stderr, flush=True)
        self.st = Store(STORE)
        print(f"[showreel] {len(self.st.ids)} recordings, {time.time()-t0:.0f}s",
              file=sys.stderr, flush=True)
        self._txt = None
        self._z = None

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
        from relmo.vjeval import l2
        from relmo.vjstore import zs

        if self._z is None:
            self._z = {}
        if rid not in self._z:
            fix, sig = self.st.raw[rid]
            self._z[rid] = l2(np.concatenate(
                [zs(fix.astype(np.float32)), zs(sig.astype(np.float32))], -1))
        return self._z[rid]

    def rank(self, query: str, cand: list[str], band: float = 0.25) -> dict:
        from relmo.vjmatch import dtw
        from relmo.vjzeval import PAD_COST, _pad

        cand = [c for c in cand if c in self.st.raw]
        if query not in self.st.raw or not cand:
            return {}
        q = self._zs(query)
        Z = [self._zs(c) for c in cand]
        P, ok = _pad(Z)
        L = np.array([len(z) for z in Z])
        C = 1.0 - np.einsum("sd,nkd->nsk", q, P)
        C = np.where(ok[:, None, :], C, PAD_COST)
        s = -dtw(C, False, L, band)
        return {c: round(max(0.0, 1.0 + float(v)), 6) for c, v in zip(cand, s)}

    def vec(self, rid: str) -> dict:
        if rid not in self.st.raw:
            return {}
        n = self.st.ids.index(rid)
        fix, sig = self.st.raw[rid]
        app = np.concatenate([self.st.pf[n], self.st.ps[n]]) / np.sqrt(2.0)
        mot = sig.std(0); mot /= np.linalg.norm(mot) + 1e-9
        raw = sig.mean(0); raw /= np.linalg.norm(raw) + 1e-9
        r = lambda a: [round(float(x), 7) for x in a]
        return dict(appearance=r(app), motion=r(mot), siglip=r(raw))


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
                reply(dict(ok=True, n=len(eng.st.ids)))
            elif c == "text":
                reply(dict(ok=True, vecs=eng.text(req["texts"])))
            elif c == "pair":
                reply(dict(ok=True, cos=eng.pair(req["a"], req["b"])))
            elif c == "rank":
                t = time.time()
                s = eng.rank(req["query"], req["cand"], float(req.get("band", 0.25)))
                reply(dict(ok=True, scores=s, ms=round((time.time()-t)*1e3, 1)))
            elif c == "vec":
                reply(dict(ok=True, **eng.vec(req["id"])))
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
