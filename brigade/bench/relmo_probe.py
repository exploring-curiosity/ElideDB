#!/usr/bin/env python3
"""Which RelMo representation separates the kitchen's behaviours?

    myenv/bin/python brigade/bench/relmo_probe.py

BACKGROUND. The reasoning head failed on the store's clip vectors: 1-NN
behaviour match 0.200 against a 0.100 chance, within-behaviour cosine +0.4524
vs across +0.4305. I called that a representation problem. This locates it.

The store keeps ONE number per clip: RelMo's 512-d prefilter vector, which is
`concat(whitened pooled V-JEPA, whitened pooled SigLIP2)/sqrt(2)`. That is
stage ONE of RelMo's retrieval. RelMo's own read path (vjstore.Store.query)
uses it to pick M candidates and then ranks them by **DTW over the full
descriptor traces**. Brigade shipped the prefilter and called it the memory.

Two things could therefore be wrong, and they are separable without training:

  POOLING   a mean over time cannot distinguish two behaviours that visit the
            same pixels in a different order. DTW can. This is the whole
            reason RelMo has a second stage.
  BASIS     the whitening is fitted on rcasa (3,556 RoboCasa recordings).
            Whitening is what removes the variance a corpus shares, and the
            variance THIS corpus shares — one static kitchen, ten behaviours —
            is not the variance RoboCasa shares.

Four representations, one encode pass, leave-one-out 1-NN behaviour match:

    raw       pooled, no whitening          — the floor
    rcasa     pooled, RoboCasa basis        — what the store holds today
    libero    pooled, basis refit here      — tests BASIS
    dtw       full trace, RelMo's stage 2   — tests POOLING

Runs in RelMo's interpreter. No database, no pi0.5, no training.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native"))

LABELS = ROOT / "eval_logs" / "reasoner" / "segments.json"
CLIPS = ROOT / "brigade" / "artifacts" / "clips"
CACHE = ROOT / "eval_logs" / "reasoner" / "traces.npz"
OUT = ROOT / "eval_logs" / "reasoner" / "probe.json"


# ------------------------------------------------------------------- encode

def encode_all(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """clip_id -> (fix, sig) traces, cached so the variants are free to re-run."""
    # The store now keeps traces beside the video, so after a collection run
    # there is nothing to encode: read what the write path already produced.
    # Re-encoding would also be a subtly different measurement — it would test
    # the encoder, not the artefact retrieval actually uses.
    have = {p.stem: p for p in (ROOT / "brigade" / "artifacts" / "traces").glob("*.npz")}
    want = [r["clip_id"] for r in rows if r["clip_id"] in have]
    if len(want) >= 0.9 * len(rows) and want:
        out = {}
        for cid in want:
            z = np.load(have[cid])
            out[cid] = (z["fix"].astype(np.float32), z["sig"].astype(np.float32))
        print(f"traces: {len(out)} read from the store's write path")
        return out
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        d = {k: (z[f"{k}.fix"], z[f"{k}.sig"]) for k in z["ids"].tolist()}
        print(f"cache: {len(d)} traces from {CACHE.name}")
        return d

    from tqdm import tqdm

    from relmo.api import Memory

    # Load the encoders BEFORE the bar exists. A lazy multi-gigabyte load
    # inside iteration one hides behind a bar reading 0/N and looks hung.
    mem = Memory.open("rcasa")
    print("loading V-JEPA 2 + SigLIP 2 (~30s) ...", flush=True)
    t0 = time.time()
    probe = rows[0]
    mem._encode(str(CLIPS / f"{probe['clip_id']}.mp4"), None, None)
    print(f"encoders ready in {time.time() - t0:.0f}s", flush=True)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in tqdm(rows, desc="encode", unit="clip"):
        p = CLIPS / f"{r['clip_id']}.mp4"
        if not p.exists():
            continue
        try:
            out[r["clip_id"]] = mem._encode(str(p), None, None)
        except Exception as exc:                                  # noqa: BLE001
            tqdm.write(f"  {r['clip_id']}: {exc}")
    flat = {"ids": np.array(list(out))}
    for k, (f, s) in out.items():
        flat[f"{k}.fix"], flat[f"{k}.sig"] = f, s
    np.savez_compressed(CACHE, **flat)
    return out


# ------------------------------------------------------------------ scoring

def loo_1nn(S: np.ndarray, y: np.ndarray, bar: np.ndarray | None = None) -> float:
    """Leave-one-out nearest-neighbour behaviour match under a similarity.

    `bar` masks candidates that must not count. It is not optional in spirit:
    the store indexes a sliding span, so consecutive rows share two thirds of
    their video and a plain 1-NN is mostly asking whether a clip can find
    ITSELF shifted by five seconds. That question has a trivially correct
    answer and the same label for free.
    """
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    if bar is not None:
        S = np.where(bar, -np.inf, S)
    return float((y[S.argmax(1)] == y).mean())


def overlap_mask(rows: list[dict], span: float = 15.0, hop: float = 5.0) -> np.ndarray:
    """True where two spans share video. Same behaviour block, within span/hop
    positions of each other — which is how the collector writes them."""
    n = len(rows)
    reach = int(round(span / hop))
    pos, last, run = [], None, -1
    for r in rows:                       # position within this behaviour's block
        run = run + 1 if r["goal"] == last else 0
        last = r["goal"]
        pos.append(run)
    m = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if rows[i]["goal"] == rows[j]["goal"] and abs(pos[i] - pos[j]) < reach:
                m[i, j] = True
    return m


def separation(S: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    same = (y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool)
    diff = y[:, None] != y[None, :]
    return float(S[same].mean()), float(S[diff].mean())


def pooled(traces, ids, whiten: str) -> np.ndarray:
    """Pooled channel vectors, optionally whitened. Mirrors the sidecar."""
    from relmo.vjreps import apply_w, fit_whiten

    halves = []
    for key in (0, 1):
        X = np.stack([traces[i][key].astype(np.float32).mean(0) for i in ids])
        if whiten == "rcasa":
            from relmo.vjstore import Store

            st = _store()
            X = apply_w(X, st._wf if key == 0 else st._ws)
        elif whiten == "libero":
            # Fitted on THIS corpus. n-1 components, the same rule vjstore
            # uses, so the comparison isolates the corpus and not the rank.
            X = apply_w(X, fit_whiten(X, min(256, len(X) - 1)))
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        halves.append(X)
    return np.concatenate(halves, 1) / np.sqrt(2.0)


_ST = None


def _store():
    global _ST
    if _ST is None:
        from relmo.vjstore import Store

        print("loading the rcasa basis (~85s) ...", flush=True)
        _ST = Store("rcasa")
    return _ST


def dtw_matrix(traces, ids, band: float = 0.0) -> np.ndarray:
    """RelMo's stage 2, every pair. -> negated length-normalised DTW cost."""
    from relmo.vjeval import l2
    from relmo.vjmatch import dtw
    from relmo.vjstore import zs
    from relmo.vjzeval import PAD_COST, _pad

    Z = [l2(np.concatenate([zs(traces[i][0].astype(np.float32)),
                            zs(traces[i][1].astype(np.float32))], -1))
         for i in ids]
    P, ok = _pad(Z)
    L = np.array([len(z) for z in Z])
    S = np.zeros((len(ids), len(ids)), np.float32)
    for i, q in enumerate(Z):
        C = 1.0 - np.einsum("sd,nkd->nsk", q, P)
        C = np.where(ok[:, None, :], C, PAD_COST)
        S[i] = -dtw(C, False, L, band)
    return S


# --------------------------------------------------------------------- main

def main() -> int:
    rows = json.load(open(LABELS))
    traces = encode_all(rows)
    rows = [r for r in rows if r["clip_id"] in traces]
    ids = [r["clip_id"] for r in rows]
    y = np.array([r["goal"] for r in rows])
    k = len(set(y.tolist()))
    lens = [traces[i][0].shape[0] for i in ids]
    print(f"\n{len(ids)} clips, {k} behaviours, chance {1/k:.3f}")
    print(f"trace length: min {min(lens)}  median {int(np.median(lens))}  "
          f"max {max(lens)} descriptor steps of 0.25 s")
    if max(lens) <= 8:
        print("  NOTE: <=8 steps is ONE encoder window. RelMo tiles 4.0 s windows\n"
              "  on a 2.0 s hop, so a 5 s segment carries 2.0 s of descriptor and\n"
              "  DTW has almost nothing to align. See the length sweep below.")

    arms = [("raw    (pooled, no whitening)", "none"),
            ("libero (pooled, basis refit here)", "libero")]
    if "--no-rcasa" not in sys.argv:
        arms.insert(1, ("rcasa  (pooled, RoboCasa basis)", "rcasa"))

    bar = overlap_mask(rows)
    n_ok = int((~bar).sum(1).min())
    print(f"\n{'':38}{'1-NN':>6}{'no-overlap':>12}   separation")
    print(f"{'':38}{'':>6}{'':>12}   (within - across)")

    res = {}
    for name, how in ((n, pooled(traces, ids, w)) for n, w in arms):
        S = how
        M = S @ S.T
        same, diff = separation(M, y)
        res[name.split()[0]] = dict(acc=loo_1nn(M, y), clean=loo_1nn(M, y, bar),
                                    same=same, diff=diff)
        r = res[name.split()[0]]
        print(f"  {name:<36}{r['acc']:>6.3f}{r['clean']:>12.3f}   {same-diff:+.4f}")

    M = dtw_matrix(traces, ids)
    same, diff = separation(M, y)
    res["dtw"] = dict(acc=loo_1nn(M, y), clean=loo_1nn(M, y, bar),
                      same=same, diff=diff)
    print(f"  {'dtw    (full trace, RelMo stage 2)':<36}"
          f"{res['dtw']['acc']:>6.3f}{res['dtw']['clean']:>12.3f}   {same-diff:+.4f}")

    print(f"\nchance {1/k:.3f}. The no-overlap column bars every span that shares\n"
          f"video with the query — its own sliding neighbours — so it is asking\n"
          f"whether a DIFFERENT performance of the same behaviour is found. That\n"
          f"is the number that means anything; at least {n_ok} candidates remain\n"
          f"for every query.")
    best = max(res, key=lambda r: res[r]["clean"])
    print(f"best (no-overlap): {best} at {res[best]['clean']:.3f}")
    json.dump(dict(n=len(ids), k=k, chance=1 / k, steps=dict(
        min=min(lens), median=int(np.median(lens)), max=max(lens)), variants=res),
        open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
