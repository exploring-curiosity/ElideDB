#!/usr/bin/env python3
"""RelMo, as a long-lived sidecar. Runs in RelMo's OWN interpreter.

Brigade's control loop needs `transformers==4.53.2` plus openpi's siglip
overlay, because pi0.5 refuses to load without them. RelMo needs 4.57 for
`VJEPA2Model`. Both are correct, neither can move, and a shared interpreter
cannot hold both — so this file is the boundary, and it is the same boundary
CLAUDE.md already draws for ML sidecars: **the sidecar writes, the core reads**,
no RPC framework, no server.

Protocol: one JSON object per line on stdin, one per line on stdout.

    {"clip": "/c.mp4", "id": "c"}     -> {vec: [512], trace: "/c.npz", steps: n}
    {"cmd": "fit"}                    -> fit this store's OWN whitening basis
    {"cmd": "project", "ids": [...]}  -> re-project traces under the current basis
    {"cmd": "rank", "query": id, "candidates": [...]}  -> stage-2 DTW scores
    {"cmd": "ping"} / {"cmd": "selfcheck"}

WHAT CHANGED IN v2, and why it had to. The first version stored one 512-d
vector per clip and called that the memory. That vector is RelMo's stage-ONE
PREFILTER — the thing that picks candidates cheaply. RelMo's actual read path
(`vjstore.Store.query`) then ranks those candidates by **DTW over the full
descriptor trace**, because a mean over time cannot tell apart two behaviours
that visit the same pixels in a different order. Shipping stage 1 alone and
calling it retrieval is like shipping an IVF coarse quantiser without the
residual scan. Measured on 30 labelled kitchen segments (bench/relmo_probe.py),
1-NN behaviour match:

    raw pooled, no whitening       0.200
    pooled, RoboCasa basis         0.300      <- what the store held
    pooled, basis refit here       0.400
                                              chance 0.100

Two defects, both structural:

  BASIS. Whitening is what removes the variance a corpus SHARES, and RelMo
  fits it on the corpus being searched precisely because that variance is a
  property of the deployment. Brigade borrowed RoboCasa's. One static kitchen
  filmed from one fixed camera shares far more than RoboCasa does, so the
  borrowed basis left most of it in. Refitting is not tuning — it is the
  documented way RelMo is meant to be deployed on a new site, and it is
  label-free.

  LENGTH. RelMo tiles 4.0 s encoder windows on a 2.0 s hop. A 5 s segment
  therefore yields ONE window — eight 0.25 s descriptor steps covering 2.0 s —
  so every trace was effectively a single glance and DTW had nothing to align.
  The store now indexes a sliding SPAN, not a tile. See memory/store.py.

The 512-d vector is `concat(qf, qs)/sqrt(2)` where qf and qs are the whitened,
L2-normalised pooled V-JEPA and SigLIP channels. That construction is not
decorative: RelMo's prefilter scores candidates with `0.5*(pf@qf + ps@qs)`, and
for unit-norm halves that is exactly the inner product of these vectors. So the
index built over this column is not an approximation of RelMo's retrieval — it
IS its first stage, executed by the database instead of by numpy. `selfcheck`
measures that rather than asserting it.

The basis is a NAMESPACE: vectors written under one are meaningless against
another. Its id travels with every vector and is checked on read.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native"))

# Where traces and the fitted basis live. Traces are RelMo's representation of
# the clip, not a description of it: the file holds two float arrays and no
# name for anything in the kitchen.
ART = ROOT / "brigade" / "artifacts"
TRACES = ART / "traces"
BASIS = ART / "basis.npz"

# The bootstrap basis. rcasa is 3,556 recordings of a robot working in
# kitchens — the closest available prior — and it is what the store uses until
# it has enough of its OWN video to fit on. `fit` then replaces it.
BOOTSTRAP = "rcasa"
MIN_FIT = 24          # below this a fitted basis has too few dims to be worth it
HALF = 256            # per-channel width of the stored vector; 2*HALF = VECTOR(512)


class Encoder:
    def __init__(self, basis_store: str = BOOTSTRAP):
        t0 = time.time()
        print(f"[relmo] loading bootstrap basis {basis_store!r}...",
              file=sys.stderr, flush=True)
        from relmo.vjstore import Store

        self.store = Store(basis_store)
        self.bootstrap = basis_store
        self.n_basis = len(self.store.ids)
        print(f"[relmo] bootstrap ready: {self.n_basis} recordings, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        self._mem = None
        TRACES.mkdir(parents=True, exist_ok=True)
        self._load_basis()

    # ---- the basis ----------------------------------------------------------

    def _load_basis(self) -> None:
        """Prefer this store's own basis; fall back to the bootstrap."""
        if BASIS.exists():
            z = np.load(BASIS, allow_pickle=True)
            self.wf = (z["wf_mu"], z["wf_W"])
            self.ws = (z["ws_mu"], z["ws_W"])
            self.basis = str(z["basis_id"])
            self.n_fit = int(z["n"])
        else:
            self.wf, self.ws = self.store._wf, self.store._ws
            self.basis, self.n_fit = self.bootstrap, self.n_basis

    def reset_basis(self) -> dict:
        """Go back to the bootstrap basis. -> the basis now in force.

        MEASURED, and it reverses what fitting looked like on a shorter corpus.
        Nearest-neighbour behaviour match over 111 spans, chance 0.100, with
        overlapping spans barred:

            RoboCasa basis (3,556 recordings)      0.685
            refit on this kitchen (111 spans)      0.324

        Whitening removes the variance a corpus SHARES, which is exactly right
        when the corpus is diverse and exactly wrong when it is not. 111 heavily
        overlapping spans of one room share the very thing that distinguishes
        the behaviours, so fitting on them whitens the signal away. The earlier
        5 s corpus said the opposite (0.300 -> 0.400) and it was measuring
        near-duplicate matching, not generalisation.

        The lesson is not "never refit" — it is that a basis needs a corpus
        wider than the question being asked of it.
        """
        if BASIS.exists():
            BASIS.unlink()
        self.wf, self.ws = self.store._wf, self.store._ws
        self.basis, self.n_fit = self.bootstrap, self.n_basis
        return dict(basis=self.basis, n=self.n_fit, dim=2 * HALF)

    def fit(self, ids: list[str] | None = None) -> dict:
        """Fit the whitening on THIS store's video. Label-free, by construction.

        RelMo fits per store (`vjstore.Store.rawpool`) because whitening is a
        property of the corpus being searched. Kept, and NOT the default here —
        see `reset_basis` for the measurement that decided it.
        """
        from relmo.vjreps import fit_whiten

        paths = ([TRACES / f"{i}.npz" for i in ids] if ids
                 else sorted(TRACES.glob("*.npz")))
        paths = [p for p in paths if p.exists()]
        if len(paths) < MIN_FIT:
            return dict(error=f"only {len(paths)} traces; need {MIN_FIT} to fit "
                               f"a basis worth having")
        F, S = [], []
        for p in paths:
            z = np.load(p)
            F.append(z["fix"].astype(np.float32).mean(0))
            S.append(z["sig"].astype(np.float32).mean(0))
        F, S = np.stack(F), np.stack(S)
        k = min(256, len(paths) - 1)
        wf, ws = fit_whiten(F, k), fit_whiten(S, k)
        # The id is a digest of what was fitted, so two bases built from
        # different video can never silently share a namespace.
        bid = "kitchen-" + hashlib.sha1(
            ("|".join(sorted(p.stem for p in paths))).encode()).hexdigest()[:10]
        np.savez(BASIS, wf_mu=wf[0], wf_W=wf[1], ws_mu=ws[0], ws_W=ws[1],
                 basis_id=bid, n=len(paths))
        self.wf, self.ws, self.basis, self.n_fit = wf, ws, bid, len(paths)
        return dict(basis=bid, n=len(paths), dim=int(2 * k))

    # ---- encode -------------------------------------------------------------

    def _memory(self):
        """RelMo's own Memory owns the encoders; reuse rather than re-implement
        its preprocessing, which has non-obvious details (stream fps, frame
        windowing, fp16) that must match how the basis store was built."""
        if self._mem is None:
            from relmo.api import Memory

            self._mem = Memory.open(self.bootstrap)
        return self._mem

    def encode(self, clip: str, clip_id: str, start=None, end=None) -> dict:
        fix, sig = self._memory()._encode(clip, start, end)
        fix = np.asarray(fix, np.float32)
        sig = np.asarray(sig, np.float32)
        tp = TRACES / f"{clip_id}.npz"
        np.savez_compressed(tp, fix=fix, sig=sig)
        return dict(vec=self.project(fix, sig), trace=str(tp),
                    steps=int(fix.shape[0]))

    def project(self, fix: np.ndarray, sig: np.ndarray) -> np.ndarray:
        """Raw channel traces -> the 512-d stage-1 prefilter vector.

        Each half is L2-normalised and then zero-padded to HALF. The padding is
        not cosmetic: a whitening basis fitted on N clips has rank N-1, so this
        kitchen's basis is 118-d where RoboCasa's is 512-d, and a vector column
        has one fixed width. Padding after the normalisation leaves both norms
        and every inner product untouched, so `0.5*(pf@qf + ps@qs)` — the
        identity the database's cosine relies on — still holds exactly.
        """
        from relmo.vjreps import apply_w

        def half(x, w):
            q = apply_w(x.mean(0)[None], w)[0]
            q = q / (np.linalg.norm(q) + 1e-9)
            out = np.zeros(HALF, np.float32)
            out[:min(HALF, len(q))] = q[:HALF]
            return out

        v = np.concatenate([half(fix, self.wf), half(sig, self.ws)])
        return (v / np.sqrt(2.0)).astype(np.float32)

    def project_ids(self, ids: list[str]) -> dict:
        """Re-project stored traces. Needed after `fit`: a new basis is a new
        namespace, so every vector written under the old one is stale."""
        out = {}
        for i in ids:
            p = TRACES / f"{i}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            out[i] = [round(float(x), 7)
                      for x in self.project(z["fix"], z["sig"])]
        return out

    # ---- the text channel ---------------------------------------------------

    def text(self, texts: list[str]) -> list[list[float]]:
        """Embed what the human said. Lives here for two reasons.

        Practical: SigLIP2's text tower needs transformers 4.57, which is this
        interpreter and not pi0.5's. Architectural: the words are embedded,
        used, and dropped — nothing is written, so the sidecar is the right
        place for the one component that ever sees language at serve.

        Note what this is NOT for. Text->video retrieval is measured weak in
        RelMo (text erases direction: "open" and "close" sit at cosine 0.957),
        so the request never touches the index. It is an input to the reasoning
        head, which decides among behaviours the VIDEO has already narrowed.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        if getattr(self, "_txt", None) is None:
            mid = "google/siglip2-base-patch16-224"
            self._txt = (AutoModel.from_pretrained(mid, dtype=torch.float32).eval(),
                         AutoProcessor.from_pretrained(mid))
        m, pr = self._txt
        with torch.no_grad():
            t = pr(text=list(texts), padding="max_length", max_length=64,
                   return_tensors="pt")
            e = m.get_text_features(**t).float().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        return [[round(float(x), 7) for x in row] for row in e]

    # ---- stage 2 ------------------------------------------------------------

    def rank(self, query: str, candidates: list[str], band: float = 0.25) -> dict:
        """RelMo's exact stage: DTW over the traces of the survivors.

        The database has already done stage 1 — this reorders its shortlist by
        how the two behaviours actually unfolded in time, which is the part a
        pooled vector cannot see.
        """
        from relmo.vjeval import l2
        from relmo.vjmatch import dtw
        from relmo.vjstore import zs
        from relmo.vjzeval import PAD_COST, _pad

        def load(i):
            z = np.load(TRACES / f"{i}.npz")
            return l2(np.concatenate([zs(z["fix"].astype(np.float32)),
                                      zs(z["sig"].astype(np.float32))], -1))

        qp = TRACES / f"{query}.npz"
        cand = [c for c in candidates if (TRACES / f"{c}.npz").exists()]
        if not qp.exists() or not cand:
            return dict(scores={})
        q = load(query)
        Z = [load(c) for c in cand]
        P, ok = _pad(Z)
        L = np.array([len(z) for z in Z])
        C = 1.0 - np.einsum("sd,nkd->nsk", q, P)
        C = np.where(ok[:, None, :], C, PAD_COST)
        s = -dtw(C, False, L, band)
        # A similarity, not an internal distance with a sign flip on it: the
        # matcher returns negated length-normalised DTW cost of mean (1-cos).
        return dict(scores={c: round(max(0.0, 1.0 + float(v)), 6)
                            for c, v in zip(cand, s)})

    # ---- proof --------------------------------------------------------------

    def selfcheck(self) -> dict:
        """Prove the prefilter claim rather than asserting it in a comment.

        Takes a recording in the bootstrap store, builds its vector the way
        this sidecar builds one, and compares the inner product against RelMo's
        own prefilter score for the same pair. They must agree to floating-point
        noise or the database index is measuring something else. Runs against
        the BOOTSTRAP basis because that is the one RelMo also holds — under a
        refitted kitchen basis there is no second implementation to disagree
        with, which is the point of checking it here.
        """
        from relmo.vjreps import apply_w

        st = self.store
        rid = st.ids[0]
        fix, sig = st.raw[rid]
        qf = apply_w(fix.mean(0)[None], st._wf)[0]
        qf /= np.linalg.norm(qf) + 1e-9
        qs = apply_w(sig.mean(0)[None], st._ws)[0]
        qs /= np.linalg.norm(qs) + 1e-9
        q = np.concatenate([qf, qs]) / np.sqrt(2.0)
        mine = (np.concatenate([st.pf, st.ps], 1) / np.sqrt(2.0)) @ q
        theirs = 0.5 * (st.pf @ qf + st.ps @ qs)
        err = float(np.abs(mine - theirs).max())
        return dict(recording=rid, max_abs_err=err, agrees=bool(err < 1e-6),
                    dim=int(q.shape[0]))


def main() -> int:
    # THE PROTOCOL OWNS STDOUT, and RelMo does not know that. `Memory._encode`
    # prints "loading encoders onto mps..." on its first call, which lands
    # between two JSON replies and makes the client's readline return prose.
    # It cost exactly one clip per session — the first encode failed with
    # "unparseable reply" and every later one worked, which reads like a flake
    # rather than a protocol bug. Everything that is not a reply now goes to
    # stderr; replies go to the handle captured here.
    out = sys.stdout
    sys.stdout = sys.stderr

    def reply(obj) -> None:
        print(json.dumps(obj), file=out, flush=True)

    enc = None
    reply(dict(ready=False, booting=True))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            reply(dict(error=f"bad json: {exc}"))
            continue

        try:
            if enc is None:
                enc = Encoder(req.get("bootstrap") or BOOTSTRAP)
            cmd = req.get("cmd")
            if cmd == "ping":
                reply(dict(ok=True, basis=enc.basis, n_basis=enc.n_fit,
                                 fitted=enc.basis != enc.bootstrap))
            elif cmd == "fit":
                r = enc.reset_basis() if req.get("reset") else enc.fit(req.get("ids"))
                reply(dict(ok="error" not in r, **r))
            elif cmd == "project":
                reply(dict(ok=True, basis=enc.basis,
                                 vecs=enc.project_ids(req["ids"])))
            elif cmd == "text":
                reply(dict(ok=True, vecs=enc.text(req["texts"])))
            elif cmd == "rank":
                t0 = time.time()
                r = enc.rank(req["query"], req["candidates"],
                             float(req.get("band", 0.25)))
                reply(dict(ok=True, ms=round((time.time() - t0) * 1e3, 1), **r))
            elif cmd == "selfcheck":
                reply(dict(ok=True, **enc.selfcheck()))
            elif cmd == "quit":
                return 0
            else:
                t0 = time.time()
                r = enc.encode(req["clip"], req["id"],
                               req.get("start"), req.get("end"))
                reply(dict(id=req["id"], basis=enc.basis,
                                 vec=[round(float(x), 7) for x in r["vec"]],
                                 trace=r["trace"], steps=r["steps"],
                                 ms=round((time.time() - t0) * 1e3, 1)))
        except Exception as exc:  # noqa: BLE001 — the caller must see the reason
            import traceback

            traceback.print_exc(file=sys.stderr)
            reply(dict(id=req.get("id"), error=f"{type(exc).__name__}: {exc}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
