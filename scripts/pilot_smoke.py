"""Pilot smoke test: is this install healthy?

Checks every layer a pilot depends on, in dependency order, and prints
PASS/FAIL per check with the number that justifies it. Exit code 0 only
if every REQUIRED check passes. Model-download checks run last so a
fresh install fails fast on the cheap layers first.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

RESULTS = []


def check(name, required=True):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            ok = False
        ms = (time.perf_counter() - t0) * 1e3
        RESULTS.append((name, ok, required, detail, ms))
        mark = "PASS" if ok else ("FAIL" if required else "warn")
        print(f"[{mark}] {name:34s} {detail} ({ms:.0f} ms)", flush=True)
        return ok
    return deco


def main():
    @check("imports (pyarrow, numpy, mlx)")
    def _():
        import mlx.core  # noqa: F401
        import numpy  # noqa: F401
        import pyarrow  # noqa: F401
        return "ok"

    @check("ffmpeg on PATH")
    def _():
        from elidedb.fftools import find
        return find("ffmpeg")

    stores = sorted(p.name for p in Path("lake").iterdir()
                    if (p / "_store.json").exists()) if Path("lake").exists() \
        else []

    @check("stores discoverable")
    def _():
        if not stores:
            raise RuntimeError("no stores under lake/ — load data first")
        return ", ".join(stores)

    from elidedb import Store
    ref = None
    for s in ("bridge4h", *stores):
        if s in stores:
            ref = Store.open(f"lake/{s}")
            break

    @check("store opens + tables scan")
    def _():
        n = {t: len(ref.table(t).scan()) for t in ("frames",)}
        return f"{ref.name}: frames={n['frames']:,}"

    @check("time-window read (byte-range decode)")
    def _():
        fr = ref.table("frames").scan()
        t0 = int(fr.column("ts")[0].as_py())
        out = ref.window(t0, t0 + 2_000_000_000)
        w = out[0] if isinstance(out, tuple) else out
        n = sum(len(v) for v in w.values() if hasattr(v, "__len__"))
        return f"{len(w)} tables, {n} rows"

    @check("text tower loads (first run downloads)")
    def _():
        from elidedb.context import embed_texts
        v = embed_texts(["smoke test"])
        return f"dim {v.shape[1]}"

    @check("context search end-to-end")
    def _():
        t0 = time.perf_counter()
        hits, stats = ref.search_context("something moving", k=3)
        ms = (time.perf_counter() - t0) * 1e3
        return f"{len(hits)} hits, {ms:.0f} ms, {stats['candidates']} cand"

    @check("warm query under 100 ms", required=False)
    def _():
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            ref.search_context("something moving", k=3)
            best = min(best, (time.perf_counter() - t0) * 1e3)
        if best > 100:
            raise RuntimeError(f"{best:.0f} ms")
        return f"{best:.0f} ms"

    @check("motion channel present", required=False)
    def _():
        n = len(ref.table("motion_vectors").scan())
        return f"{n:,} motion vectors"

    @check("verifier VLM loads (background tier)", required=False)
    def _():
        from elidedb.rerank import DEFAULT_VLM, _load
        _load(DEFAULT_VLM)
        return DEFAULT_VLM.split("/")[-1]

    hard_fail = [r for r in RESULTS if not r[1] and r[2]]
    soft_fail = [r for r in RESULTS if not r[1] and not r[2]]
    print(f"\n{len(RESULTS) - len(hard_fail) - len(soft_fail)}/{len(RESULTS)}"
          f" passed"
          + (f", {len(soft_fail)} warnings" if soft_fail else "")
          + (f", {len(hard_fail)} FAILURES" if hard_fail else ""))
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
