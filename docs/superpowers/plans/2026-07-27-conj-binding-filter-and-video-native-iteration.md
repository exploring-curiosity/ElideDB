# ElideDB Iterative Improvement: Binding Filters + Video-Native Channels

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the three compositional-binding queries (q08/q09/q10, all at 0.00 precision) off the floor and raise the ledger mean beyond 0.25/0.26 — without violating the no-hardwire rule, with every change arbitrated by the frozen-truthset ledger.

**Architecture:** The set path fuses 8 channels (pe, act, vid, obj, mot, prf, sig2, conj) via weighted RRF over the 1,122-episode `lake/bench` store, with a fitted contrast filter for directional queries. This plan (1) extracts one shared fit/live selection code path, (2) makes filter membership a *fitted* artifact so the conjunctive atom channel can act as a hard binding constraint instead of a drowned vote, (3) replaces the last hand dictionary (`_HYPONYMS`) with corpus-attested WordNet vocabulary, (4) evaluates InternVideo2-Stage2 1B as a true video-native text channel, and (5) wires the measured path into the Desk product surface.

**Tech Stack:** Python 3 (`./myenv/bin/python`), numpy, pyarrow, transformers 4.57.6 on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1` on every model-touching command), SigLIP 2 (`google/siglip2-so400m-patch14-384`), nltk/WordNet (new dep), InternVideo2-Stage2 1B (cost-gated), frozen truthset `eval/truthsets/bridge4h.parquet`, ledger `scripts/bench_truth.py` → `BENCHMARKS.md`.

---

## Research thesis (why these tasks, in this order)

**T1 — Consensus fusion structurally drowns decisive minority channels.** Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) is robust *because* it rewards agreement across rankers; a single channel that alone knows the answer is outvoted by construction. Our measurement confirms it: `conj` was added as a vote (fitted weight 1.0 among eight channels) and q08/q09/q10 stayed at 0.00. The IR-correct role for a constraint is a **filter** (candidate pruning), not a ranker. The codebase already states this principle for direction ("contrast channels FILTER, content channels ORDER", scenario.py); this plan extends it to binding.

**T2 — Contrastive image-text encoders cannot rank binding; they must be constrained.** Winoground (Thrush et al., CVPR 2022) showed SOTA image-text models at or below chance on compositional matching; ARO (Yuksekgonul et al., ICLR 2023, "When and Why Vision-Language Models Behave like Bags-of-Words") showed CLIP-family scores are nearly invariant to attribute/relation shuffling; SugarCrepe (Hsieh et al., NeurIPS 2023) confirmed it survives hard-negative cleanup. Consequence: no reweighting of pe/sig2/vid can fix "spoon **on** cloth" — a bag-of-concepts score for the full sentence is the same for spoon-near-cloth. What CAN work: decompose the query into atoms, demand each atom independently finds evidence (a hard MIN/veto), and verify the *relation* with pixel geometry (the SAM 3.1 tracker audit, already built). This is the mechanical analog of training-free composed retrieval by decomposition (CIReVL, Karthik et al., ICLR 2024) with a regex instead of an LLM — LLM-free per the no-VLM-judges directive.

**T3 — Fit and live must be one code path.** Twice we measured divergence regressions (knee reshaping the fitted cut: fit LOQO 0.21 vs live 0.16; obj-boost overriding fitted weights). Anything the fitter searches must execute through the same function the query executes. This is standard ML-systems hygiene (train/serve skew), and here it is *ledger-measured*, not theoretical.

**T4 — Vocabulary must be dictionary × corpus, never dictionary-as-dataset.** The no-hardwire rule's test: English lexicon is allowed; which concepts matter in THIS corpus is data. `_HYPONYMS = {vessel: pot/pan/bowl, ...}` encodes a kitchen. WordNet hypernymy (Miller, CACM 1995) is the dictionary half; scoring candidate lemmas against the store's SigLIP 2 frame space is the corpus half. A street-scene store would attest "boat" for vessel; nothing in code prefers either.

**T5 — Video-native text alignment needs a video-native encoder.** All current text channels are image-frame models (PE, SigLIP 2, X-CLIP pooled). InternVideo2 (Wang et al., ECCV 2024, arXiv 2403.15377) stage-2 trains video-text contrastively with temporal modeling; its small distilled variants are reported substantially weaker on retrieval, so it is the 1B or nothing. This is the only *new-encoder* bet in the plan and is cost-gated with explicit bailout criteria — it is a research candidate, and a clean negative result is an acceptable outcome recorded in the ledger notes.

**Validation discipline throughout:** leave-one-query-out (10 queries cannot be memorized), ledger row per commit, regressions reported not hidden, per-query precision-of-returned with abstention (never x/140). Iteration loop is ~4 min (fit ~2 min + bench ~1.5 min), run in the **foreground** — no background watchers (a self-matching poll pattern already burned us once).

## Baseline (2026-07-26 05:05, uncommitted work on top of d88ca01)

| metric | value |
|---|---|
| ledger | mean prec **0.25**, mean yield **0.26**, 25/100 returned true |
| per query | q00 2/10, q01 2/10, q02 5/10, q03 5/10, q04 5/10, q05 3/10, q07 3/10, q08 **0/10**, q09 **0/10**, q10 **0/10**, q06 gate PASS |
| fit | LOQO 0.198; dir {pe 6, act 1, vid 1, obj 0, mot 1, prf 1, sig2 0, conj 1} fq 0.33; con {pe 2, act 0.5, vid 1, obj 1, mot 1, prf 1, sig2 1, conj 1} fq 0.33 |
| known wart | numpy "All-NaN slice" RuntimeWarning leaks the source line `ch["sig2"] = (np.nanmax(...` into bench stdout (one episode of 1,122 has no sig2 rows — a stream-boundary skip at ingest; 8,968 = 1,121 × 8) |

## Ground rules for the executor

- Python is ALWAYS `./myenv/bin/python`; every command that touches a model gets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Tests follow the repo's plain-python runner pattern (see `tests/test_fusion.py`): no pytest installed; each file ends with a `__main__` block that runs every `test_*` function.
- `eval/truthsets/bridge4h.parquet` is FROZEN. Never regrade, never edit.
- Never run `scripts/sig2_ingest.py` or any ingest again unless a task says so — the vectors exist.
- Line numbers below are as of this writing and will drift; every edit is anchored to an exact unique string.
- The ledger command is `PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py` — it appends a row to BENCHMARKS.md by itself. Never hand-write a ledger row.
- Fit command is `PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/fit_set_weights.py` (~2 min, prints `captured qNN`, LOQO lines, final weights, writes `lake/bench/_set_weights.json`).
- Run fit and bench in the foreground. Do not create background watchers.

---

### Task 0: Commit the measured sig2+conj baseline

The 0.25/0.26 ledger row exists but the code that produced it is uncommitted. Commit it verbatim so every later diff is attributable.

**Files:**
- Commit (no edits): `python/elidedb/sig2.py`, `scripts/sig2_ingest.py`, `python/elidedb/scenario.py`, `scripts/fit_set_weights.py`, `BENCHMARKS.md`

- [ ] **Step 1: Verify the tree matches the summary**

Run: `git status --short`
Expected: exactly ` M BENCHMARKS.md`, ` M python/elidedb/scenario.py`, ` M scripts/fit_set_weights.py`, `?? python/elidedb/sig2.py`, `?? scripts/sig2_ingest.py`. If anything else is modified, STOP and report.

- [ ] **Step 2: Commit**

```bash
git add BENCHMARKS.md python/elidedb/scenario.py scripts/fit_set_weights.py python/elidedb/sig2.py scripts/sig2_ingest.py
git commit -m "SigLIP2 channel + mechanical conjunctive atoms: ledger 0.25/0.26 (yield high); binding queries still 0 — conj drowned as an RRF vote, next: fitted filter membership"
```

---

### Task 1: `variant_max` helper + binding-primitive tests

Fixes the All-NaN warning leak (T-baseline wart) and puts the pure query-decomposition logic under test before we build on it.

**Files:**
- Create: `tests/test_binding.py`
- Modify: `python/elidedb/fusion.py` (append function)
- Modify: `python/elidedb/scenario.py` (3 nanmax sites + 1 import, lines ~198–255)
- Modify: `scripts/fit_set_weights.py` (3 nanmax sites in `capture`, lines ~54–67)

- [ ] **Step 1: Write the failing test**

Create `tests/test_binding.py`:

```python
"""Binding primitives: mechanical atom decomposition + variant-max.

atoms_of is the LLM-free decomposition (regex over determiner phrases)
that the conjunctive channel and the binding filter depend on; its
contract is one atom per noun phrase, >=2 atoms or conj abstains.
variant_max replaces the bare np.nanmax over query-variant score
arrays: all-NaN rows (episodes a channel cannot score) must stay NaN
WITHOUT numpy's All-NaN-slice warning polluting bench stdout.

Run: ./myenv/bin/python tests/test_binding.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.fusion import variant_max      # noqa: E402
from elidedb.sig2 import atoms_of           # noqa: E402


def test_atoms_two_atoms_spoon_cloth():
    a = atoms_of("place the spoon on top of the cloth")
    assert len(a) == 2
    assert a[0].startswith("the spoon") and a[1] == "the cloth"


def test_atoms_attribute_kept():
    a = atoms_of("put the green object into the drawer")
    assert "the green object" in a and "the drawer" in a


def test_atoms_single_noun_means_abstain():
    assert len(atoms_of("close the drawer")) < 2


def test_variant_max_all_nan_column_silent():
    vs = [np.array([1.0, np.nan]), np.array([2.0, np.nan])]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = variant_max(vs)
    assert out[0] == 2.0 and np.isnan(out[1])
    assert not any("All-NaN" in str(x.message) for x in w)


def test_variant_max_single_variant_identity():
    out = variant_max([np.array([1.0, np.nan, 3.0])])
    assert out[0] == 1.0 and np.isnan(out[1]) and out[2] == 3.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `./myenv/bin/python tests/test_binding.py`
Expected: `ImportError: cannot import name 'variant_max' from 'elidedb.fusion'`

- [ ] **Step 3: Implement `variant_max`**

Append to `python/elidedb/fusion.py`:

```python
def variant_max(vs):
    """Row-wise max over query-variant score arrays. All-NaN rows
    (episodes absent from a channel's table — they abstain, they are
    not errors) stay NaN without numpy's All-NaN-slice RuntimeWarning,
    which was leaking source lines into bench stdout."""
    m = np.stack(vs)
    out = np.full(m.shape[1], np.nan)
    fin = np.isfinite(m).any(0)
    if fin.any():
        out[fin] = np.nanmax(m[:, fin], 0)
    return out
```

(`fusion.py` already imports numpy as `np`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./myenv/bin/python tests/test_binding.py`
Expected: `5 passed`

- [ ] **Step 5: Use it at all six call sites**

In `python/elidedb/scenario.py`, change the import inside `search_set`:

```python
    from .fusion import rrf
```
→
```python
    from .fusion import rrf, variant_max
```

Then three replacements:

```python
        ch["pe"] = np.nanmax(np.stack(vs), 0) if len(vs) > 1 else vs[0]
```
→
```python
        ch["pe"] = variant_max(vs)
```

```python
        ch["sig2"] = (np.nanmax(np.stack(vs), 0)
                      if len(vs) > 1 else vs[0])
```
→
```python
        ch["sig2"] = variant_max(vs)
```

```python
        ch["vid"] = np.nanmax(np.stack(vs), 0) if len(vs) > 1 else vs[0]
```
→
```python
        ch["vid"] = variant_max(vs)
```

In `scripts/fit_set_weights.py`, extend the module import:

```python
from elidedb.fusion import rrf                               # noqa: E402
```
→
```python
from elidedb.fusion import rrf, variant_max                  # noqa: E402
```

and in `capture()` make the same three replacements:

```python
    out["pe"] = np.nanmax(np.stack(vs), 0) if len(vs) > 1 else vs[0]
```
→
```python
    out["pe"] = variant_max(vs)
```

```python
    out["vid"] = np.nanmax(np.stack(vs), 0) if len(vs) > 1 else vs[0]
```
→
```python
    out["vid"] = variant_max(vs)
```

```python
    out["sig2"] = np.nanmax(np.stack(vs), 0) if len(vs) > 1 else vs[0]
```
→
```python
    out["sig2"] = variant_max(vs)
```

- [ ] **Step 6: Behavior-preservation check (bench only, fit unchanged)**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py`
Expected: same per-query numbers as baseline (q00 2/10 … q08/q09/q10 0/10, mean prec 0.25, mean yield 0.26) and **no** stray `ch["sig2"] = …` line in stdout. Small ± on individual queries means something changed — investigate before committing (variant_max must be numerically identical to nanmax on rows with any finite value).

- [ ] **Step 7: Commit**

```bash
git add tests/test_binding.py python/elidedb/fusion.py python/elidedb/scenario.py scripts/fit_set_weights.py BENCHMARKS.md
git commit -m "variant_max: all-NaN variant rows abstain silently; binding primitives under test (atoms contract: one atom per NP, <2 => conj abstains)"
```

---

### Task 2: One selection code path for fit and live (`setpath.py`)

Fit-live skew is a measured regression class here (knee vs fitted cut: LOQO 0.21 vs live 0.16). Extract `rankfrac` + the contrast filter into a shared module both callers use. Behavior-preserving: bench must reproduce the baseline row.

**Files:**
- Create: `python/elidedb/setpath.py`
- Create: `tests/test_setpath.py`
- Modify: `python/elidedb/scenario.py` (filter block, lines ~361–381)
- Modify: `scripts/fit_set_weights.py` (`rankfrac` + `score_query`, lines ~95–124)

- [ ] **Step 1: Write the failing test**

Create `tests/test_setpath.py`:

```python
"""Shared set-path selection — the fit-live contract.

Everything the fitter searches over must execute through these
functions and only these; the knee/obj-boost divergences (fit LOQO
0.21 vs live 0.16, measured) are the regression class this kills.

Run: ./myenv/bin/python tests/test_setpath.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.setpath import filter_mask, rankfrac   # noqa: E402


def test_rankfrac_nan_votes_neutral():
    r = rankfrac(np.array([3.0, np.nan, 1.0]))
    assert r[0] == 1.0 and r[1] == 0.5 and r[2] == 0.0


def test_filter_bottom_quantile_dies():
    ch = {"mot": np.array([0.9, 0.1, 0.5, 0.7])}
    alive = filter_mask(ch, ["mot"], 1 / 3)
    assert list(alive) == [True, False, True, True]


def test_filter_median_disagreement_cancels():
    ch = {"mot": np.array([0.9, 0.1, 0.5, 0.7]),
          "prf": np.array([0.1, 0.9, 0.5, 0.7])}
    alive = filter_mask(ch, ["mot", "prf"], 1 / 3)
    assert alive.all()


def test_filter_no_evidence_all_alive():
    ch = {"pe": np.array([0.9, 0.1]),
          "conj": np.array([np.nan, np.nan])}
    alive = filter_mask(ch, ["conj"], 1 / 3)
    assert alive.all()


def test_filter_zero_quantile_all_alive():
    ch = {"mot": np.array([0.9, 0.1, 0.5])}
    assert filter_mask(ch, ["mot"], 0.0).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `./myenv/bin/python tests/test_setpath.py`
Expected: `ModuleNotFoundError: No module named 'elidedb.setpath'`

- [ ] **Step 3: Implement `setpath.py`**

Create `python/elidedb/setpath.py`:

```python
"""Shared set-path selection: ONE code path for the live query
(scenario.search_set) and the fitter (scripts/fit_set_weights.py).

The fit-live divergences of the acceptance sprint (the knee reshaping
the fitted cut, the obj-boost overriding fitted weights) were measured
regressions: what the fit optimized was not what the query executed.
Anything the fit searches over must run through these functions."""
from __future__ import annotations

import numpy as np


def rankfrac(v):
    """Rank fraction in [0,1]; NaN (abstaining) entries vote a neutral
    0.5 so a channel that cannot score an episode never executes it."""
    v = np.asarray(v, float)
    r = np.full(len(v), 0.5)
    fin = np.isfinite(v)
    if fin.sum() > 1:
        r[fin] = np.argsort(np.argsort(v[fin])) / (fin.sum() - 1)
    return r


def filter_mask(ch, names, q):
    """Median rank-fraction over the named filter channels; episodes
    in the bottom-q die. Quantile, never sign: AUC-validated channels
    have uncalibrated zero points (a sign test executed 130/183 true
    closes, measured). All-alive when no named channel has evidence or
    q is 0."""
    con = [rankfrac(ch[c]) for c in names
           if c in ch and np.isfinite(ch[c]).any()]
    size = len(next(iter(ch.values())))
    if not con or q <= 0:
        return np.ones(size, bool)
    return ~(np.median(np.stack(con), 0) < q)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./myenv/bin/python tests/test_setpath.py`
Expected: `5 passed`

- [ ] **Step 5: Switch `search_set` to the shared filter**

In `python/elidedb/scenario.py`, replace the whole block (the inner `_rankfrac` def plus the contrast filter — anchor on this exact text):

```python
    def _rankfrac(v):
        r = np.full(len(v), 0.5)
        fin = np.isfinite(v)
        if fin.sum() > 1:
            order = np.argsort(np.argsort(v[fin]))
            r[fin] = order / (fin.sum() - 1)
        return r

    dropped = 0
    alive = np.ones(len(keys), bool)
    if contrast_ch:
        # unified contrast filter: mean rank fraction over EVERY
        # available contrast channel (motion delta, action-class
        # contrast, corpus-derived PRF anchors), bottom third dies.
        # Quantile not sign (uncalibrated zeros execute true clips —
        # 130/183 measured); abstaining channels vote neutral 0.5.
        cf = np.median(np.stack([_rankfrac(v) for v in
                                 contrast_ch.values()]), 0)
        bad = cf < filter_q
        alive &= ~bad
        dropped = int(bad.sum())
```

with:

```python
    from .setpath import filter_mask
    alive = (filter_mask(contrast_ch, list(contrast_ch), filter_q)
             if contrast_ch else np.ones(len(keys), bool))
    dropped = int((~alive).sum())
```

- [ ] **Step 6: Switch the fitter to the shared filter**

In `scripts/fit_set_weights.py`, delete the local `rankfrac` function:

```python
def rankfrac(v):
    r = np.full(len(v), 0.5)
    fin = np.isfinite(v)
    if fin.sum() > 1:
        r[fin] = np.argsort(np.argsort(v[fin])) / (fin.sum() - 1)
    return r
```

add to module imports (next to the `rrf` import):

```python
from elidedb.setpath import filter_mask                      # noqa: E402
```

and replace `score_query` in full:

```python
def score_query(case, w, fq):
    ch = {c: case["ch"][c] for c in CH
          if np.isfinite(case["ch"][c]).any() and w.get(c, 0) > 0}
    if not ch:
        return 0.0
    fused = rrf(ch, weights=w)
    alive = np.ones(len(fused), bool)
    if case["dir"] and fq > 0:
        alive = filter_mask(case["ch"], ["mot", "act", "prf"], fq)
    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])][:K]
    lab = case["lab"][order]
    n = len(order)
    if n == 0:
        return 0.0
    tru = int((lab == 1).sum())
    prec = tru / n
    yld = tru / min(K, case["sup"]) if case["sup"] else 0.0
    return prec + 0.5 * yld
```

- [ ] **Step 7: Behavior-preservation check — fit AND bench must reproduce baseline**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/fit_set_weights.py
```
Expected: LOQO 0.198 and the Baseline table's dir/con weights, fq 0.33/0.33 (coordinate ascent over identical channel arrays is deterministic; a single differing grid entry means borderline MPS float16 jitter in a re-encoded text vector — acceptable only if the bench row below still reproduces).

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py
```
Expected: baseline row reproduced (mean prec 0.25, mean yield 0.26). Known intentional corner: `filter_mask` uses the FITTER's semantics (no-evidence channels excluded from the median) while the old live path included them as neutral 0.5 rows — they differ only when a contrast channel is entirely NaN for a query. If the row shifts, this corner is why; report the delta and keep the unified semantics (unification IS the task), but any shift larger than ±1 clip on one query means something else broke — investigate.

- [ ] **Step 8: Commit**

```bash
git add python/elidedb/setpath.py tests/test_setpath.py python/elidedb/scenario.py scripts/fit_set_weights.py BENCHMARKS.md lake/bench/_set_weights.json
git commit -m "setpath: one selection code path for fit and live (rankfrac + filter_mask) — behavior-preserving, ledger row reproduced; kills the fit-live skew regression class"
```

---

### Task 3: Fitted filter membership — conj becomes a binding FILTER (the central task)

Thesis T1+T2. Today the filter set is hardcoded (`mot/act/prf`) and only exists for directional queries; binding queries have NO filter stage, so conj can only vote inside RRF where seven bag-of-concepts channels outvote it. Make filter membership a fitted, per-query-type artifact. The fitter may then discover `conj` as a filter for non-directional queries — or reject it, which is an honest negative result that Task 4's diagnostics will explain.

**Files:**
- Modify: `scripts/fit_set_weights.py` (`score_query`, `fit`, `main`)
- Modify: `python/elidedb/scenario.py` (cfg read ~lines 311–338, filter application from Task 2)

- [ ] **Step 1: Fit-side — thread the filter set `fc` through scoring**

In `scripts/fit_set_weights.py`, replace `score_query` in full (this supersedes Task 2's version — the `case["dir"]` condition is absorbed into per-type fitting because `fit()` is always called on a single-type subset):

```python
def score_query(case, w, fq, fc):
    ch = {c: case["ch"][c] for c in CH
          if np.isfinite(case["ch"][c]).any() and w.get(c, 0) > 0}
    if not ch:
        return 0.0
    fused = rrf(ch, weights=w)
    alive = (filter_mask(case["ch"], fc, fq) if fc and fq > 0
             else np.ones(len(fused), bool))
    idx = np.where(alive)[0]
    order = idx[np.argsort(-fused[idx])][:K]
    lab = case["lab"][order]
    n = len(order)
    if n == 0:
        return 0.0
    tru = int((lab == 1).sum())
    prec = tru / n
    yld = tru / min(K, case["sup"]) if case["sup"] else 0.0
    return prec + 0.5 * yld
```

- [ ] **Step 2: Fit-side — search membership by greedy toggle**

Replace the `fit` inner function in `main()` in full:

```python
    def fit(subset, fc0):
        w = {c: 1.0 for c in CH}
        fq = 1 / 3
        fc = list(fc0)
        best = sum(score_query(c, w, fq, fc) for c in subset)
        for _ in range(4):
            improved = False
            for c in CH:
                for g in grid:
                    w2 = dict(w); w2[c] = g
                    s = sum(score_query(x, w2, fq, fc) for x in subset)
                    if s > best + 1e-9:
                        best, w, improved = s, w2, True
            for f2 in fqs:
                s = sum(score_query(x, w, f2, fc) for x in subset)
                if s > best + 1e-9:
                    best, fq, improved = s, f2, True
            # membership toggle: any channel may join or leave the
            # filter set — the fitter, not code, decides which
            # channels have veto authority for this query type
            for c in CH:
                fc2 = ([x for x in fc if x != c] if c in fc
                       else fc + [c])
                s = sum(score_query(x, w, fq, fc2) for x in subset)
                if s > best + 1e-9:
                    best, fc, improved = s, fc2, True
            if not improved:
                break
        return w, fq, fc
```

- [ ] **Step 3: Fit-side — starts, LOQO, and output schema**

Still in `main()`, replace the LOQO block and the final fit/output block. Anchor: from `    # leave-one-query-out: the honest generalization estimate.` through the final two `print` lines. New code:

```python
    # starts = today's live behavior, so the fitted result can only
    # move away from it by measured improvement
    DIR0 = ["mot", "act", "prf"]
    CON0 = []

    # leave-one-query-out: the honest generalization estimate.
    # Weights AND filter membership fitted PER QUERY TYPE (directional
    # vs not) — routing is a lexicon property (swap exists?), roles
    # are data.
    loqo = []
    for i in range(len(cases)):
        train = [c for j, c in enumerate(cases) if j != i]
        same = [c for c in train if c["dir"] == cases[i]["dir"]]
        w, fq, fc = fit(same or train,
                        DIR0 if cases[i]["dir"] else CON0)
        loqo.append(score_query(cases[i], w, fq, fc))
        print(f"LOQO holdout {cases[i]['q'][:44]:44s} "
              f"score {loqo[-1]:.2f}", flush=True)
    print(f"LOQO mean objective: {np.mean(loqo):.3f}")

    w_dir, fq_dir, fc_dir = fit([c for c in cases if c["dir"]]
                                or cases, DIR0)
    w_con, fq_con, fc_con = fit([c for c in cases if not c["dir"]]
                                or cases, CON0)
    out = {"set_weights_dir": w_dir, "filter_quantile_dir": fq_dir,
           "filter_channels_dir": fc_dir,
           "set_weights": w_con, "filter_quantile": fq_con,
           "filter_channels": fc_con,
           "fitted_on": "eval/truthsets/bridge4h.parquet",
           "loqo_mean": round(float(np.mean(loqo)), 3)}
    p = Path("lake/bench/_set_weights.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"dir {json.dumps(w_dir)} fq={fq_dir:.2f} fc={fc_dir}")
    print(f"con {json.dumps(w_con)} fq={fq_con:.2f} fc={fc_con}")
```

- [ ] **Step 4: Live-side — read and apply fitted membership**

In `python/elidedb/scenario.py`:

(a) Anchor on the weight-defaults line before the cfg try-block and add the membership default:

```python
    directional = sq is not None
    weights = {c: 1.0 for c in ch}
    filter_q = 1 / 3
```
→
```python
    directional = sq is not None
    weights = {c: 1.0 for c in ch}
    filter_q = 1 / 3
    fnames = None       # None => legacy: every contrast channel
```

(b) Inside the fitted branch, after the `filter_q = float(cfg.get(fk, 1 / 3))` line, add:

```python
            ck = ("filter_channels_dir" if directional and
                  "filter_channels_dir" in cfg else "filter_channels")
            if ck in cfg:
                fnames = list(cfg[ck])
```

(c) Replace Task 2's filter application:

```python
    from .setpath import filter_mask
    alive = (filter_mask(contrast_ch, list(contrast_ch), filter_q)
             if contrast_ch else np.ones(len(keys), bool))
    dropped = int((~alive).sum())
```

with:

```python
    # FITTED VETO AUTHORITY: which channels filter is a per-store
    # learned artifact, not code. For binding queries this lets conj
    # act as a hard constraint (each query atom must find its own
    # frame evidence) instead of a drowned RRF vote — consensus
    # fusion structurally outvotes a decisive minority channel
    # (Cormack et al. 2009), and bag-of-concepts encoders cannot
    # rank binding (Winoground/ARO), so the constraint must prune.
    from .setpath import filter_mask
    fsrc = dict(ch)
    fsrc.update(contrast_ch)
    if fnames is None:
        fnames = list(contrast_ch)
    alive = (filter_mask(fsrc, fnames, filter_q)
             if fnames else np.ones(len(keys), bool))
    dropped = int((~alive).sum())
```

- [ ] **Step 5: Refit**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/fit_set_weights.py
```
Expected: runs ~3 min (extra toggle loop); prints `fc=` lists for dir and con; LOQO mean printed. Record the LOQO delta vs 0.198 — if LOQO drops materially (>0.02), the membership search is overfitting the 10 queries; report it, don't hide it.

- [ ] **Step 6: Bench**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py
```
Expected outcomes, all acceptable, in order of preference:
1. `fc_con` contains `"conj"` and q08/q09/q10 move above 0/10 with mean prec ≥ 0.25 — the thesis confirmed.
2. `fc_con` contains `"conj"` but binding queries stay 0 — the conj scores themselves are wrong; Task 4 diagnoses.
3. Fitter rejects conj from `fc_con` (empty or other channels) — negative result; Task 4 diagnoses.
Regression on q00–q07 beyond noise (±1 clip) is NOT acceptable — if it happens, inspect which fitted fc caused it and report the tradeoff.

- [ ] **Step 7: Commit (whatever the outcome — the ledger row is the record)**

```bash
git add scripts/fit_set_weights.py python/elidedb/scenario.py lake/bench/_set_weights.json BENCHMARKS.md
git commit -m "Fitted filter membership: veto authority per query type is learned, not coded; conj eligible as binding constraint (consensus fusion drowns minority channels — filter, don't vote)"
```

---

### Task 4: Binding diagnostics instrument

The diagnose→fix→ledger microscope for q08/q09/q10: per-channel AUC against the truthset, top-10 hit counts, and per-atom SigLIP 2 max-frame scores on true vs returned episodes. This tells us whether binding failure lives in (a) atom parsing, (b) the encoder (atoms score high on false episodes too — true bag-of-concepts failure needing the tracker audit), or (c) fusion/filtering.

**Files:**
- Create: `scripts/diag_binding.py`

- [ ] **Step 1: Write the script**

Create `scripts/diag_binding.py`:

```python
"""Binding diagnostics: WHERE do the binding queries die?

Per-channel AUC vs the frozen truthset, top-10 label hits, and
per-atom SigLIP2 max-frame scores on TRUE vs RETURNED episodes.
Instrument only — never a shipping path.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python \
     scripts/diag_binding.py [qid ...]     (default: 8 9 10)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402
from fit_set_weights import capture                          # noqa: E402


def auc(sc, lab):
    """Mann-Whitney AUC from 0-indexed ranks:
    (mean rank of positives - (P-1)/2) / N_negative."""
    fin = np.isfinite(sc)
    s, la = sc[fin], lab[fin]
    if la.sum() in (0, len(la)):
        return float("nan")
    r = np.argsort(np.argsort(s))
    return float((r[la == 1].mean() - (la.sum() - 1) / 2)
                 / (len(la) - la.sum()))


def main():
    db = Store.open("lake/bench")
    from elidedb.scenario import _episodes
    from elidedb.sig2 import _frame_scores, _index, atoms_of
    keys = _episodes(db)
    t = pq.read_table("eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(t0)): int(v) for q, s, t0, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    for qi in [int(a) for a in sys.argv[1:]] or [8, 9, 10]:
        text = QUERIES[qi]
        print(f"\n=== q{qi:02d} {text}")
        ch, _ = capture(db, keys, text)
        lab = np.array([truth.get((qi, s, a), -1) for s, a, b in keys])
        lab = np.where(lab == 1, 1, 0) * (lab >= 0)
        print(f"support {int(lab.sum())}")
        for c in sorted(ch):
            v = ch[c]
            if np.isfinite(v).any():
                top = np.argsort(-np.nan_to_num(v, nan=-9e9))[:10]
                print(f"  {c:5s} AUC {auc(v, lab):+.3f} "
                      f"top10 {int(lab[top].sum())}/10")
        atoms = atoms_of(text)
        if len(atoms) < 2:
            print("  (single atom — conj abstains)")
            continue
        idx = _index(db)
        per = {a: _frame_scores(db, a) for a in atoms}
        v = np.nan_to_num(ch["conj"], nan=-9e9)
        groups = (("TRUE", [k for k, la in zip(keys, lab)
                            if la == 1][:5]),
                  ("RET ", [keys[i] for i in np.argsort(-v)[:5]]))
        for tag, group in groups:
            for s, a, b in group:
                lst = idx.get(str(s))
                if not lst:
                    continue
                starts = [x[0] for x in lst]
                j = int(np.searchsorted(starts, a,
                                        side="right")) - 1
                if j < 0 or lst[j][0] != a:
                    continue
                rows = lst[j][1]
                parts = " ".join(
                    f"{at.split()[-1]}={per[at][rows].max():.3f}"
                    for at in atoms)
                print(f"  {tag} {str(s)[-18:]} {a}: {parts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/diag_binding.py 8 9 10
```
Expected: for each query — support count, one AUC/top10 line per channel, then TRUE/RET lines like `spoon=0.183 cloth=0.152`. Read them:
- conj AUC near +0.5 (chance) with TRUE and RET atom scores overlapping ⇒ encoder can't separate — escalate to the tracker binding audit (`purity="audited"`) for these queries or the IV2 channel (Task 7).
- conj AUC well above chance but top10 0/10 ⇒ fusion/filter problem — revisit Task 3's fitted config.
- an atom scoring ~equal on everything (e.g. an abstract head noun) ⇒ vocabulary problem — Task 6 helps.

- [ ] **Step 3: Record the finding and commit**

Append 2–4 sentences to BENCHMARKS.md under a `### Binding diagnosis (2026-07-27)` heading stating which of the three failure modes the numbers show — actual numbers, no adjectives.

```bash
git add scripts/diag_binding.py BENCHMARKS.md
git commit -m "diag_binding: per-channel AUC + per-atom frame scores for binding queries — locates failure in parsing vs encoder vs fusion"
```

---

### Task 5: Model the no-match gate inside the fit

Last fit-live divergence: live kills gated queries (`_auto_action_support` max_p < 0.05 ⇒ empty return) but the fitter scores them as normal cases, so weights can be tuned to please episodes the gate will discard.

**Files:**
- Modify: `scripts/fit_set_weights.py` (`score_query` top, capture loop in `main`)

- [ ] **Step 1: Capture the gate per query**

In `main()`, replace the case-construction block:

```python
    for qi in qids:
        text = QUERIES[qi]
        ch, isdir = capture(db, keys, text)
        lab = np.array([truth.get((qi, s, a), -1) for s, a, b in keys])
        # strict: ungraded counts false, matching the ledger
        lab = np.where(lab == 1, 1, 0) * (lab >= 0)
        cases.append({"ch": ch, "lab": lab, "dir": isdir,
                      "sup": int((lab == 1).sum()), "q": text})
        print(f"captured q{qi:02d} sup={int((lab == 1).sum())}",
              flush=True)
```

with:

```python
    from elidedb.scenario import _auto_action_support
    for qi in qids:
        text = QUERIES[qi]
        ch, isdir = capture(db, keys, text)
        lab = np.array([truth.get((qi, s, a), -1) for s, a, b in keys])
        # strict: ungraded counts false, matching the ledger
        lab = np.where(lab == 1, 1, 0) * (lab >= 0)
        g = _auto_action_support(db, text)
        gated = bool(g is not None and g["max_p"] < 0.05)
        cases.append({"ch": ch, "lab": lab, "dir": isdir, "gated":
                      gated, "sup": int((lab == 1).sum()), "q": text})
        print(f"captured q{qi:02d} sup={int((lab == 1).sum())}"
              f"{' GATED' if gated else ''}", flush=True)
```

- [ ] **Step 2: Make gated queries a constant for the ascent**

At the top of `score_query` (Task 3's version), insert as the first lines:

```python
    if case.get("gated"):
        # live returns empty for gated queries regardless of weights;
        # a constant removes them from the ascent so weights are never
        # tuned to please episodes the gate will kill
        return 0.0
```

- [ ] **Step 3: Refit + bench, verify no regression**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/fit_set_weights.py
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py
```
Expected: `captured q06 … GATED` in fit output; q06 gate still PASS in bench; other queries within ±1 clip of the Task 3 row (weights may shift slightly since q06 no longer contributes gradient).

- [ ] **Step 4: Commit**

```bash
git add scripts/fit_set_weights.py lake/bench/_set_weights.json BENCHMARKS.md
git commit -m "Gate modeled in fit: gated queries are constants for the ascent — the last fit-live divergence closed"
```

---

### Task 6: Corpus-attested vocabulary replaces `_HYPONYMS`

Thesis T4. WordNet hypernymy = dictionary (allowed); which hyponyms THIS corpus contains = data, attested by SigLIP 2 frame scores. Deletes the last hand table.

**Files:**
- Create: `python/elidedb/vocab.py`
- Create: `tests/test_vocab.py`
- Modify: `python/elidedb/scenario.py` (delete `_HYPONYMS` ~line 43, replace variants block ~lines 211–215)
- Modify: `scripts/fit_set_weights.py` (replace variants block in `capture`)

- [ ] **Step 1: Install the dictionary**

```bash
./myenv/bin/pip install nltk
./myenv/bin/python -c "import nltk; nltk.download('wordnet'); from nltk.corpus import wordnet as wn; print('vessel senses:', len(wn.synsets('vessel', pos='n')))"
```
Expected: prints a nonzero sense count. WordNet data lands in `~/nltk_data` (~35 MB, one-time).

- [ ] **Step 2: Write the failing test**

Create `tests/test_vocab.py`:

```python
"""Corpus vocabulary — the dictionary half is deterministic WordNet.

The no-hardwire split: WordNet hypernymy is English (allowed);
which hyponyms the corpus attests is data (integration-tested against
lake/bench when present, skipped otherwise).

Run: ./myenv/bin/python tests/test_vocab.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.vocab import hyponym_lemmas    # noqa: E402


def test_vessel_covers_cookware():
    lems = hyponym_lemmas("vessel")
    # dictionary fact, corpus-independent; if this fails, raise the
    # depth argument — do NOT hand-add lemmas
    assert "pot" in lems


def test_lemmas_single_word_lowercase():
    for lem in hyponym_lemmas("container")[:50]:
        assert lem.isalpha() and lem == lem.lower()


def test_unknown_word_empty():
    assert hyponym_lemmas("zzzzqqq") == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `./myenv/bin/python tests/test_vocab.py`
Expected: `ModuleNotFoundError: No module named 'elidedb.vocab'`

- [ ] **Step 4: Implement `vocab.py`**

Create `python/elidedb/vocab.py`:

```python
"""Corpus-attested vocabulary: the self-recognized replacement for the
_HYPONYMS hand table (the last dictionary-of-the-dataset in the code).

Split per the no-hardwire rule's dictionary-vs-dataset test:
  - WordNet hypernymy is ENGLISH (corpus-independent, allowed);
  - WHICH hyponyms exist in THIS corpus is DATA — attested by scoring
    each candidate lemma against the store's SigLIP2 frame space and
    keeping only lemmas the corpus matches better than the query's own
    word. A kitchen store attests pot/pan for "vessel"; a street store
    would attest boat — nothing in code prefers either.
Attestations cache per store in _vocab.json (first query per new word
pays ~2s of text encodes, then free)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def hyponym_lemmas(word, depth=2):
    """Single-word WordNet hyponym lemmas over every noun sense —
    dictionary knowledge only, no corpus involved."""
    from nltk.corpus import wordnet as wn
    try:
        senses = wn.synsets(word, pos="n")
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
        senses = wn.synsets(word, pos="n")
    out = []
    for syn in senses:
        frontier = [syn]
        for _ in range(depth):
            frontier = [h for s in frontier for h in s.hyponyms()]
            for s in frontier:
                for lem in s.lemma_names():
                    if ("_" not in lem and lem.isalpha()
                            and lem.lower() != word):
                        out.append(lem.lower())
    return list(dict.fromkeys(out))


def _top5(sc):
    k = min(5, len(sc))
    return float(np.sort(sc)[-k:].mean())


def _attested(store, word, cap):
    """Hyponyms of `word` that THIS corpus scores above the word
    itself — the corpus prefers the specific term or the category
    word stands."""
    from .sig2 import _frame_scores
    cands = hyponym_lemmas(word)[:40]
    if not cands:
        return []
    base = _top5(_frame_scores(store, f"a photo of a {word}"))
    keep = []
    for c in cands:
        sc = _top5(_frame_scores(store, f"a photo of a {c}"))
        if sc > base:
            keep.append((sc, c))
    return [c for _, c in sorted(keep, reverse=True)[:cap]]


def corpus_variants(store, text, cap=3):
    """Query variants substituting the first noun the corpus attests
    better hyponyms for; [text] alone when nothing is attested.
    Mirrors the one-substituted-noun shape of the old _HYPONYMS loop
    so text channels can keep taking max over variants."""
    from .sig2 import atoms_of
    cache_p = Path(store.dir) / "_vocab.json"
    cache = (json.loads(cache_p.read_text()) if cache_p.exists()
             else {})
    tl = text.lower()
    for atom in atoms_of(tl):
        w = atom.split()[-1]
        if w not in cache:
            cache[w] = _attested(store, w, cap)
            cache_p.write_text(json.dumps(cache, indent=1))
        if cache[w]:
            return [text] + [tl.replace(w, c) for c in cache[w]]
    return [text]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./myenv/bin/python tests/test_vocab.py`
Expected: `3 passed`. If `test_vessel_covers_cookware` fails, change the test call to `hyponym_lemmas("vessel", depth=3)` AND make `corpus_variants`'s call chain use depth 3 (`_attested` gains a `depth` pass-through) — never hand-add lemmas.

- [ ] **Step 6: Replace the hand table at both call sites**

In `python/elidedb/scenario.py`, delete the dict and its comment block:

```python
# CATEGORY-WORD EXPANSION: "vessel" starves SigLIP/PE (ledger: sup
# 247, prec 0.25) and SAM 3 cannot ground it at all (UNGROUNDABLE,
# measured). Generic English hyponyms — no metadata, any corpus. Text
# channels take max over variants; the binding audit tries variants
# until one grounds.
_HYPONYMS = {"vessel": ("pot", "pan", "bowl"),
             "container": ("box", "drawer", "bin"),
             "utensil": ("spoon", "fork", "knife")}
```

and replace the variants block inside `search_set`:

```python
    variants = [text]
    for w, syns in _HYPONYMS.items():
        if w in tl:
            variants = [text] + [tl.replace(w, s) for s in syns]
            break
```

with:

```python
    # CATEGORY-WORD EXPANSION, corpus-attested: WordNet supplies the
    # candidate hyponyms (dictionary), the store's SigLIP2 frame
    # space decides which exist HERE (data) — "vessel" starves the
    # text channels (measured: sup 247, prec 0.25) and SAM cannot
    # ground it; the specific attested terms can.
    try:
        from .vocab import corpus_variants
        variants = corpus_variants(store, text)
    except Exception:
        variants = [text]
```

Check for other `_HYPONYMS` references before deleting: `grep -n "_HYPONYMS" python/elidedb/*.py scripts/*.py`. Known second user: `_binding_audit`'s variant grounding in scenario.py and `capture` in fit_set_weights.py — point BOTH at `corpus_variants` the same way (in `capture`, replace:

```python
    from elidedb.scenario import _HYPONYMS
    variants = [text]
    for w_, syns in _HYPONYMS.items():
        if w_ in text.lower():
            variants = [text] + [text.lower().replace(w_, s_)
                                 for s_ in syns]
            break
```

with:

```python
    from elidedb.vocab import corpus_variants
    variants = corpus_variants(db, text)
```

). If `_binding_audit` builds variants differently (per-phrase, not per-query), adapt with `_attested`-backed lemmas for the phrase's head noun via `corpus_variants(store, phrase)` — same function, phrase in place of query.

- [ ] **Step 7: Refit + bench — q03 is the canary**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/fit_set_weights.py
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py
```
Acceptance: q03 (the vessel query, hyponym-dependent, baseline 5/10) stays ≥ 4/10 and mean prec ≥ Task 3's row. First run pays the attestation encodes (~2 s per new noun, cached in `lake/bench/_vocab.json`). If q03 regresses: print `lake/bench/_vocab.json` — if pot/pan/bowl are not attested, the corpus-preference threshold (`sc > base`) is too strict; relax to `sc > 0.9 * base` as a fitted-constant candidate and rerun ONCE, reporting both rows.

- [ ] **Step 8: Commit**

```bash
git add python/elidedb/vocab.py tests/test_vocab.py python/elidedb/scenario.py scripts/fit_set_weights.py lake/bench/_vocab.json lake/bench/_set_weights.json BENCHMARKS.md
git commit -m "Corpus-attested vocabulary (WordNet x SigLIP2 frames) replaces _HYPONYMS — last hand dictionary deleted; kitchen attests pot/pan, a street corpus would attest boat"
```

---

### Task 7: InternVideo2-Stage2 1B video-native text channel (cost-gated)

Thesis T5. The only new-encoder bet; explicitly allowed to fail. **Bailout criteria — stop and record a negative result in BENCHMARKS.md notes if:** (a) the model cannot load on MPS/CPU after the fallbacks below (e.g. its remote code hard-imports `flash_attn`, which does not build on macOS), or (b) ingest ETA on 1,122 episodes exceeds 60 min, or (c) total task time exceeds 90 min.

**Files:**
- Create: `scripts/iv2_probe.py`
- Create: `python/elidedb/iv2.py`
- Create: `scripts/iv2_ingest.py`
- Modify: `python/elidedb/scenario.py` (new channel try-block after the sig2 block)
- Modify: `scripts/fit_set_weights.py` (`CH` list + `capture`)

- [ ] **Step 1: Probe the model API (do not guess it)**

Create `scripts/iv2_probe.py`:

```python
"""Load-probe for InternVideo2-Stage2 1B on this machine. Prints the
feature-method names its remote code actually exposes; the channel
adapts to whichever exists. Bailout evidence if it cannot load."""
import sys

import torch
from transformers import AutoModel, AutoTokenizer

MID = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"

try:
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MID, trust_remote_code=True, torch_dtype=torch.float16)
except Exception as e:
    print(f"LOAD FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
cands = [n for n in dir(model) if any(
    k in n.lower() for k in ("vid", "video", "vision", "txt", "text"))
    and not n.startswith("_")]
print("feature-method candidates:", cands)
print("config frames:", getattr(model.config, "num_frames", "?"))
```

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/iv2_probe.py`
Expected: either `feature-method candidates: [... 'get_vid_feat', 'get_txt_feat' ...]` (or `encode_vision`/`encode_text` names), or `LOAD FAILED: …`. On LOAD FAILED → execute the bailout: append the exact error under a `### InternVideo2 negative result (2026-07-27)` heading in BENCHMARKS.md, commit that, and SKIP the rest of this task.

- [ ] **Step 2: Channel module with method-name adaptation**

Create `python/elidedb/iv2.py`:

```python
"""InternVideo2-Stage2 1B (arXiv 2403.15377) — the only true
VIDEO-native text channel: stage-2 trains video-text contrastively
with temporal modeling, which frame-pooled image encoders (PE, SigLIP,
X-CLIP pooling) structurally lack. Distilled small variants are
reported weak on retrieval; it is 1B or nothing. One vector per
episode (4-frame clip), table iv2_vectors."""
from __future__ import annotations

import numpy as np

_S = {}

MID = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"

_VID_NAMES = ("get_vid_feat", "encode_vision", "encode_video")
_TXT_NAMES = ("get_txt_feat", "encode_text", "get_text_features")


def _call_first(model, names, *args):
    for n in names:
        if hasattr(model, n):
            out = getattr(model, n)(*args)
            return out[0] if isinstance(out, tuple) else out
    raise AttributeError(f"none of {names} on {type(model).__name__}")


def _load():
    if "model" not in _S:
        import torch
        from transformers import AutoModel, AutoTokenizer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _S["tok"] = AutoTokenizer.from_pretrained(
            MID, trust_remote_code=True)
        _S["model"] = AutoModel.from_pretrained(
            MID, trust_remote_code=True,
            torch_dtype=torch.float16).to(dev).eval()
        _S["dev"] = dev
    return _S


def text_vec(text):
    import torch
    s = _load()
    with torch.no_grad():
        tok = s["tok"](text, padding="max_length", truncation=True,
                       max_length=40, return_tensors="pt").to(s["dev"])
        t = _call_first(s["model"], _TXT_NAMES, tok)
    t = t.float().cpu().numpy().reshape(-1)
    return t / (np.linalg.norm(t) + 1e-8)


def iv2_lookup(store, text):
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "iv2_vectors")
    key = {}
    for r, (s, a) in enumerate(zip(
            tbl.column("stream").to_pylist(),
            (int(v) for v in tbl.column("ts").to_pylist()))):
        key[(str(s), a)] = r
    sc = np.asarray(vecs) @ text_vec(text)

    def lookup(s, a, b):
        r = key.get((str(s), a))
        return float(sc[r]) if r is not None else float("nan")
    return lookup, None
```

If the probe showed the text method takes raw strings or `input_ids` rather than the tokenizer dict, adapt the ONE call `_call_first(s["model"], _TXT_NAMES, tok)` accordingly (e.g. `(tok.input_ids)`) — that call is the single adaptation point, same for the video call in the ingest below.

- [ ] **Step 3: Ingest script**

Create `scripts/iv2_ingest.py`:

```python
"""Ingest InternVideo2-Stage2 1B episode vectors: 4 frames per
episode -> one video-native vector -> iv2_vectors. Mirrors
sig2_ingest's episode walk; ~1,122 episodes, expect 15-40 min on MPS.
Prints ETA every 100 episodes; ABORT (ctrl-c safe, appends nothing)
if projected total exceeds 60 min — that is the cost gate."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.iv2 import MID, _call_first, _load, _VID_NAMES  # noqa: E402


def main():
    import torch
    from elidedb.video import FrameSet
    s = _load()
    store = Store.open(sys.argv[1] if len(sys.argv) > 1
                       else "lake/bench")
    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = store.table("frames").scan()
    rows_s, rows_a, rows_b, vecs = [], [], [], []
    t0 = time.time()
    for ri, (st, a, b) in enumerate(recs):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), st),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 4:
            continue
        pick = np.linspace(0, len(sel) - 1, 4).round().astype(int)
        try:
            dec = FrameSet(store, "frames",
                           sel.take(pick)).decode(width=224)
        except Exception:
            continue        # stream-boundary episode in filtered store
        if len(dec) < 4:
            continue
        import torchvision.transforms.functional as TF
        from PIL import Image
        imgs = [Image.fromarray(d[1]).resize((224, 224))
                for d in sorted(dec)]
        px = torch.stack([TF.to_tensor(i) for i in imgs])
        px = TF.normalize(px, [0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
        # (1, C, T, H, W) fp16 — InternVideo2 clip layout
        px = px.permute(1, 0, 2, 3).unsqueeze(0).to(
            s["dev"], torch.float16)
        with torch.no_grad():
            f = _call_first(s["model"], _VID_NAMES, px)
        f = f.float().cpu().numpy().reshape(-1)
        f = f / (np.linalg.norm(f) + 1e-8)
        rows_s.append(st); rows_a.append(a); rows_b.append(b)
        vecs.append(f.astype(np.float32))
        if (ri + 1) % 100 == 0:
            el = time.time() - t0
            eta = el / (ri + 1) * len(recs)
            print(f"  {ri + 1}/{len(recs)} {el:.0f}s "
                  f"ETA {eta / 60:.0f}min", flush=True)
            if eta > 3600:
                print("COST GATE: ETA > 60min, aborting")
                return
    V = np.stack(vecs)
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V).reshape(-1)), V.shape[1]),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("iv2_vectors").append(
        tbl, kind="embeddings",
        meta={"model": MID, "dim": int(V.shape[1]),
              "frames_per_episode": 4})
    print(json.dumps({"rows": len(tbl), "dim": int(V.shape[1]),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
```

Run in the FOREGROUND with a generous timeout:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/iv2_ingest.py lake/bench
```
Expected: progress lines with ETA, final JSON `{"rows": ~1100, "dim": …}`. If the video call errors on tensor layout, try `(B, T, C, H, W)` (drop the permute, use `px.unsqueeze(0)` directly) — the two layouts are the only candidates; check the remote code's `encode_vision` signature if both fail, then bail out per the criteria if unresolvable.

- [ ] **Step 4: Wire the channel into live + fit**

In `python/elidedb/scenario.py`, insert AFTER the sig2 try-block (anchor: the line `        pass` that closes the sig2 `except Exception:` — insert immediately before `    try:` of the vid block):

```python
    try:
        # video-NATIVE text alignment (InternVideo2-Stage2, temporal
        # modeling the frame-pooled channels lack) — content channel
        from .iv2 import iv2_lookup
        vs = []
        for vtext in variants:
            look, _ = iv2_lookup(store, vtext)
            vs.append(np.array([look(*k) for k in keys]))
        ch["iv2"] = variant_max(vs)
    except Exception:
        pass
```

In `scripts/fit_set_weights.py`: change

```python
CH = ["pe", "act", "vid", "obj", "mot", "prf", "sig2", "conj"]
```
→
```python
CH = ["pe", "act", "vid", "obj", "mot", "prf", "sig2", "conj", "iv2"]
```

and in `capture()`, after the sig2/conj block, add:

```python
    try:
        from elidedb.iv2 import iv2_lookup
        vs = []
        for vt in variants:
            look, _ = iv2_lookup(db, vt)
            vs.append(np.array([look(*k) for k in keys]))
        out["iv2"] = variant_max(vs)
    except Exception:
        out["iv2"] = np.full(len(keys), np.nan)
    return out, sq is not None
```

(replacing the bare `    return out, sq is not None` at the end).

- [ ] **Step 5: Refit + bench + ledger**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/fit_set_weights.py
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python scripts/bench_truth.py
```
Expected: fit prints iv2 weights per type; ledger row appended. Success = mean prec or binding queries improve with LOQO not degrading; fitted iv2 weight 0 = honest negative result, keep the vectors (they cost nothing at query time when weight is 0) and record it.

- [ ] **Step 6: Commit**

```bash
git add scripts/iv2_probe.py scripts/iv2_ingest.py python/elidedb/iv2.py python/elidedb/scenario.py scripts/fit_set_weights.py lake/bench/_set_weights.json BENCHMARKS.md
git commit -m "InternVideo2-Stage2 1B channel: video-native text alignment (temporal modeling frame-pooled channels lack); role fitted, outcome ledger-recorded"
```

---

### Task 8: Desk serves `search_set` (the product surface)

The stable-benchmark directive exists because the Desk was serving the OLD captioned `search_context` while all numbers measured `search_set`. Close the contract: the surface the user clicks must be the surface the ledger measures.

**Files:**
- Modify: `python/elidedb/desk.py:604-621` (the `kind == "context"` branch)

- [ ] **Step 1: Replace the context branch**

In `python/elidedb/desk.py`, replace:

```python
    if kind == "context":
        hits, stats = db.search_context(
            body["text"], k=int(body.get("k", 8)),
            pool=int(body.get("pool", 48)),
            deep=6 if body.get("rerank") else 0,
            t0=body.get("t0"), t1=body.get("t1"),
            streams=body.get("streams") or None)
        # a result you can check: attach the clip's caption when the store
        # has captions
        try:
            for h in hits:
                ex = db.explain(h["t0"], h["t1"], h["stream"])
                if ex:
                    h["caption"] = ex[0]["caption"]
        except Exception:
            pass
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
```

with:

```python
    if kind == "context":
        # PRODUCT SURFACE = MEASURED SURFACE: this is the exact
        # search_set the ledger benchmarks (stable-benchmark
        # directive — the old captioned search_context served here
        # while acceptance was measured elsewhere; never again).
        from elidedb.scenario import search_set
        r = search_set(db, body["text"], purity="fast",
                       k_max=int(body.get("k", 10)))
        hits = [{"stream": c["stream"], "t0": c["t0"], "t1": c["t1"],
                 "score": c["score"]} for c in r["clips"]]
        lo, hi = body.get("t0"), body.get("t1")
        if lo is not None:
            hits = [h for h in hits if h["t1"] >= int(lo)]
        if hi is not None:
            hits = [h for h in hits if h["t0"] <= int(hi)]
        stats = {"channels": r.get("channels", []),
                 "scored": r.get("scored", 0),
                 "direction_filtered": r.get("direction_filtered", 0),
                 "no_match": bool(r.get("no_match")),
                 "set_ms": r.get("ms")}
        return {"hits": hits, "stats": stats,
                "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
```

- [ ] **Step 2: Smoke it exactly as the UI calls it**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./myenv/bin/python -c "
import sys; sys.path.insert(0, 'python')
from elidedb import Store
from elidedb import desk
desk.STORES['bench'] = Store.open('lake/bench')
r = desk.api_query('bench', {'type': 'context', 'text': 'close the drawer', 'k': 10})
assert r['hits'], 'empty result for a supported query'
assert 'caption' not in r['hits'][0], 'captioned path still alive'
print(len(r['hits']), 'hits', r['stats'])"
```
Expected: `10 hits {'channels': [...], 'scored': 1122, ...}`. If `STORES` is not the dict name at module level, read the top of desk.py for the store registry and adapt the smoke only (not the handler).

- [ ] **Step 3: Commit**

```bash
git add python/elidedb/desk.py
git commit -m "Desk context search now serves search_set — product surface equals measured surface (stable-benchmark contract closed); captioned search_context path removed from UI"
```

---

### Task 9: Sprint record — BENCHMARKS.md narrative + memory

**Files:**
- Modify: `BENCHMARKS.md` (narrative section above the ledger table)
- Modify: `/Users/sudharshanramesh/.claude/projects/-Users-sudharshanramesh-Studies-MyProjects-StreetDex/memory/elidedb-no-hardwire-rule.md` (NEXT section)
- Modify: `/Users/sudharshanramesh/.claude/projects/-Users-sudharshanramesh-Studies-MyProjects-StreetDex/memory/MEMORY.md` (if a new memory file is warranted)

- [ ] **Step 1: Write the sprint narrative**

Add to BENCHMARKS.md, above the ledger table, a `## Binding-filter sprint (2026-07-27)` section of at most 15 lines: baseline row → final row, per-task ledger deltas (numbers only), the T1/T2 thesis in two sentences, and any negative results (IV2, fitted-membership rejections) stated plainly.

- [ ] **Step 2: Update memory**

In `elidedb-no-hardwire-rule.md`, replace the `NEXT:` list with the post-sprint state: which of (gate-in-fit, fit-live unification, corpus vocabulary) are DONE, current ledger, and what remains. Keep the description line's date fresh.

- [ ] **Step 3: Commit (repo files only — memory lives outside the repo)**

```bash
git add BENCHMARKS.md
git commit -m "Binding-filter sprint record: ledger trajectory + thesis + negative results, numbers only"
```

---

## Execution order & dependency notes

Task 0 → 1 → 2 → 3 are strictly sequential (each edits code the next anchors on). Task 4 depends on 3 (diagnoses its outcome). Task 5 depends on 3 (same `score_query`). Task 6 depends on 1 (uses `variant_max`-era capture) and should follow 5 to keep fit churn linear. Task 7 is independent after 3 but ordered late because it is the expensive bet. Task 8 only needs Task 0. Task 9 is last.

**Decision points the executor resolves from data, not by asking:**
- Task 3 outcome (1)/(2)/(3) → Task 4 tells you which lever is next; proceed with the plan either way and record findings.
- Task 6 canary regression → the single `0.9 * base` relaxation rerun, both rows reported.
- Task 7 bailout criteria are exact; a negative result is a completed task.

**Definition of done for the sprint:** every task committed with its ledger row; final BENCHMARKS.md narrative; fit-live divergence eliminated (Tasks 2+5); zero hand dictionaries remaining (`grep -rn "_HYPONYMS" python scripts` returns nothing); Desk serves `search_set`. Target — not a promise — is binding queries off 0.00 and mean precision > 0.30 while yield holds ≥ 0.26; if the measured result falls short, the ledger says so and the honest gap is the deliverable.
