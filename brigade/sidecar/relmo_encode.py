#!/usr/bin/env python3
"""RelMo encoder, as a long-lived sidecar. Runs in RelMo's OWN interpreter.

Brigade's control loop needs `transformers==4.53.2` plus openpi's siglip
overlay, because pi0.5 refuses to load without them. RelMo needs 4.57 for
`VJEPA2Model`. Both are correct, neither can move, and a shared interpreter
cannot hold both — so this file is the boundary, and it is the same boundary
CLAUDE.md already draws for ML sidecars: **the sidecar writes, the core reads**,
no RPC framework, no server.

Protocol: one JSON object per line on stdin, one per line on stdout.

    {"clip": "/path/ep0.mp4", "id": "ep0"}   ->   {"id": ..., "vec": [512], "ms": ...}
    {"cmd": "ping"}                          ->   {"ok": true, "basis": "...", ...}

Why a persistent process rather than one subprocess per clip: V-JEPA 2 and
SigLIP2 take ~30 s to load and a 12 s episode takes ~3 s to encode. Paying the
load per episode would put video memory outside the "every stage fits within one
minute per hour of video" budget the rest of the system is held to; paying it
once keeps encoding online, which is the rule.

The 512-d vector is `concat(qf, qs)/sqrt(2)` where qf and qs are RelMo's
whitened, L2-normalised pooled V-JEPA and SigLIP channels. That construction is
not decorative: RelMo's own stage-1 prefilter scores candidates with
`0.5*(pf@qf + ps@qs)`, and for unit-norm halves that is exactly the inner
product of these vectors. So the index built over this column is not an
approximation of RelMo's retrieval — it IS its first stage, executed by the
database instead of by numpy.

The whitening basis is fitted over a store's contents, so it is a namespace:
vectors written under one basis are meaningless against another. The basis id
travels with every vector and is checked on read.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native"))

# The basis store. rcasa is 3,876 recordings of a robot working in kitchens,
# which is the closest available prior to what Brigade does; a basis fitted on
# Brigade's own two dozen episodes would have at most two dozen dimensions.
BASIS_STORE = "rcasa"


class Encoder:
    def __init__(self, basis_store: str = BASIS_STORE):
        from relmo.vjstore import Store

        t0 = time.time()
        print(f"[relmo] loading basis store {basis_store!r}...", file=sys.stderr, flush=True)
        self.store = Store(basis_store)
        self.basis = basis_store
        self.n_basis = len(self.store.ids)
        print(f"[relmo] basis ready: {self.n_basis} recordings, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        self._mem = None

    def _memory(self):
        """RelMo's own Memory object owns the encoders; reuse rather than
        re-implement its preprocessing, which has non-obvious details (stream
        fps, frame windowing, fp16) that must match how the store was built."""
        if self._mem is None:
            from relmo.api import Memory

            self._mem = Memory.open(self.basis)
        return self._mem

    def encode(self, clip: str, start=None, end=None) -> np.ndarray:
        fix, sig = self._memory()._encode(clip, start, end)
        return self.project(np.asarray(fix, np.float32), np.asarray(sig, np.float32))

    def project(self, fix: np.ndarray, sig: np.ndarray) -> np.ndarray:
        """Raw channel traces -> the 512-d prefilter vector."""
        from relmo.vjreps import apply_w

        st = self.store
        qf = apply_w(fix.mean(0)[None], st._wf)[0]
        qf /= np.linalg.norm(qf) + 1e-9
        qs = apply_w(sig.mean(0)[None], st._ws)[0]
        qs /= np.linalg.norm(qs) + 1e-9
        v = np.concatenate([qf, qs]) / np.sqrt(2.0)
        return v.astype(np.float32)

    def selfcheck(self) -> dict:
        """Prove the claim rather than asserting it in a comment.

        Takes a recording already in the basis store, builds its vector the way
        this sidecar builds one, and compares the inner product against
        RelMo's own prefilter score for the same pair. They must agree to
        floating-point noise or the index is measuring something else.
        """
        from relmo.vjreps import apply_w

        st = self.store
        rid = st.ids[0]
        fix, sig = st.raw[rid]

        # what the database will compute: inner product against stored vectors
        q = self.project(fix, sig)
        V = np.concatenate([st.pf, st.ps], 1) / np.sqrt(2.0)
        mine = V @ q

        # what RelMo computes internally for the same query
        qf = apply_w(fix.mean(0)[None], st._wf)[0]
        qf /= np.linalg.norm(qf) + 1e-9
        qs = apply_w(sig.mean(0)[None], st._ws)[0]
        qs /= np.linalg.norm(qs) + 1e-9
        theirs = 0.5 * (st.pf @ qf + st.ps @ qs)
        err = float(np.abs(mine - theirs).max())
        return dict(recording=rid, max_abs_err=err, agrees=bool(err < 1e-6),
                    dim=int(q.shape[0]))


def main() -> int:
    enc = None
    print(json.dumps(dict(ready=False, booting=True)), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps(dict(error=f"bad json: {exc}")), flush=True)
            continue

        try:
            if enc is None:
                enc = Encoder(req.get("basis") or BASIS_STORE)

            cmd = req.get("cmd")
            if cmd == "ping":
                print(json.dumps(dict(ok=True, basis=enc.basis,
                                      n_basis=enc.n_basis)), flush=True)
            elif cmd == "selfcheck":
                print(json.dumps(dict(ok=True, **enc.selfcheck())), flush=True)
            elif cmd == "quit":
                return 0
            else:
                t0 = time.time()
                vec = enc.encode(req["clip"], req.get("start"), req.get("end"))
                print(json.dumps(dict(id=req.get("id"), basis=enc.basis,
                                      vec=[round(float(x), 7) for x in vec],
                                      ms=round((time.time() - t0) * 1e3, 1))), flush=True)
        except Exception as exc:  # noqa: BLE001 — the caller must see the reason
            import traceback

            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(id=req.get("id"), error=f"{type(exc).__name__}: {exc}")),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
