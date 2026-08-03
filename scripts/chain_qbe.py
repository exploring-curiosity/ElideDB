"""CHAIN QbE: episode similarity from ALIGNED EVENT SEQUENCES.

The measured gap this attacks: appearance-era channels score chain-
shape queries at chance (template-identity prec 0.07-0.13 for three of
four templates; only the visibly-different precarious clears it),
because every sim episode shares one scene and differs only in event
ORDER. The store's own element tables hold that order - this scores it.

An episode becomes its event sequence, each event a token of
    (kind, object SLOT, duration, motion vector)
where the slot abstracts instance identity to a repetition pattern:
object ids are renamed A, B, C by order of first appearance, so
"pick A, stack A, pick B, stack B" matches across episodes without
knowing which blocks played the parts. This leans exactly on what the
binding work bought (same-block consistency 8% -> ~45-50%): slots are
only as real as binding coherence, and this benchmark measures whether
that is already enough.

Similarity = Needleman-Wunsch global alignment; a token match scores
kind equality + slot-pattern agreement + motion cosine, gaps cost.
QbE protocol identical to the appearance smoke (same seeds, same
k = ceil(1.5*support) ceiling) so the numbers are directly comparable.

EVAL-side script: truth is read only to pick seeds and grade - the
similarity itself reads nothing but store tables.

    python scripts/chain_qbe.py [--store lake/sim_chains]
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402

# Defaults are the MEASURED-BEST configuration (ablation 2026-08-02):
# kind-only scores 0.25-0.55 yield while adding slots (binding at 45%
# coherence) and motion (chance for this task) DILUTES it to 0.20 flat.
# The oracle - the same alignment over truth primitives + slots -
# scores 1.00 yield on every template, so the mechanism is validated
# and every missing point is element quality. Re-raise W_SLOT / W_MOT
# as binding and motion earn it.
W_KIND = 1.0
W_SLOT = 0.0
W_MOT = 0.0
GAP = -0.6
MISMATCH_FLOOR = -0.8


def sequences(db):
    """{episode -> [token, ...]} from events (+ motion vectors), sorted
    by event time. Token: (kind, slot, dur_s, mot_unit_vec|None)."""
    ep = db.table("episodes").scan().to_pydict()
    ep_t0 = {int(t): int(e) for e, t in zip(ep["episode_index"], ep["ts"])}
    mv = db.table("motion_vectors").scan().to_pydict()
    mot = {}
    for s, a, v in zip(mv["stream"], mv["ts"], mv["vector"]):
        v = np.asarray(v, np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            mot[(str(s), int(a))] = v / n
    ev = db.table("events").scan().to_pydict()
    rows = defaultdict(list)
    for j in range(len(ev["ts"])):
        e = ep_t0.get(int(ev["ts"][j]))
        if e is None:
            continue
        rows[e].append((int(ev["ev_t0"][j]), int(ev["ev_t1"][j]),
                        str(ev["kind"][j]), int(ev["object_id"][j]),
                        mot.get((str(ev["stream"][j]), int(ev["ts"][j])))))
    seqs = {}
    for e, rr in rows.items():
        rr.sort()
        slot_of, seq = {}, []
        for a, b, kind, oid, m in rr:
            if oid >= 0 and oid not in slot_of:
                slot_of[oid] = len(slot_of)
            seq.append((kind, slot_of.get(oid, -1), (b - a) / 1e9, m))
        seqs[e] = seq
    return seqs


def match(t1, t2):
    s = W_KIND * (1.0 if t1[0] == t2[0] else -0.5)
    # slot agreement: both refer to the same POSITION in their episode's
    # cast of objects (the repetition pattern, not the instance)
    if t1[1] >= 0 and t2[1] >= 0:
        s += W_SLOT * (1.0 if t1[1] == t2[1] else -0.5)
    if t1[3] is not None and t2[3] is not None:
        s += W_MOT * float(t1[3] @ t2[3])
    return max(s, MISMATCH_FLOOR)


def align(a, b):
    """Needleman-Wunsch, normalised by the longer sequence so long
    episodes are not rewarded for length alone."""
    n, m = len(a), len(b)
    if not n or not m:
        return 0.0
    D = np.full((n + 1, m + 1), -1e9, np.float32)
    D[0, :] = np.arange(m + 1, dtype=np.float32) * GAP
    D[:, 0] = np.arange(n + 1, dtype=np.float32) * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = max(D[i - 1, j - 1] + match(a[i - 1], b[j - 1]),
                          D[i - 1, j] + GAP,
                          D[i, j - 1] + GAP)
    return float(D[n, m]) / max(n, m)


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    seqs = sequences(db)
    eps = sorted(seqs)
    print(f"{len(eps)} episodes, mean seq len "
          f"{np.mean([len(seqs[e]) for e in eps]):.1f}")

    # pairwise chain similarity (150x150 alignments)
    S = np.zeros((len(eps), len(eps)), np.float32)
    from tqdm import tqdm
    for i in tqdm(range(len(eps)), desc="align", unit="ep"):
        for j in range(i + 1, len(eps)):
            S[i, j] = S[j, i] = align(seqs[eps[i]], seqs[eps[j]])

    # same protocol + seeds as the appearance smoke, graded on truth
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    pos = {e: i for i, e in enumerate(eps)}
    rs = np.random.RandomState(0)
    print(f"{'query':<20} {'sup':>4} {'ret':>4} {'true':>4} "
          f"{'yield':>6} {'prec':>6}   (appearance baseline y/p)")
    base = {"swap": (0.20, 0.13), "precarious": (0.55, 0.37),
            "push_then_build": (0.10, 0.07),
            "build_unstack_move": (0.15, 0.10)}
    for target in ("swap", "precarious", "push_then_build",
                   "build_unstack_move"):
        pool = sorted(e for e, tm in tmpl.items() if tm == target)
        seeds = sorted(int(x) for x in rs.choice(pool, 5, replace=False))
        support = len(pool) - len(seeds)
        k = math.ceil(1.5 * support)
        si = [pos[e] for e in seeds]
        score = S[si].max(0)
        for x in si:
            score[x] = -1e9
        order = np.argsort(-score)[:k]
        got = [eps[i] for i in order]
        true = sum(1 for e in got if tmpl.get(e) == target)
        b = base[target]
        print(f"{target:<20} {support:>4} {len(got):>4} {true:>4} "
              f"{true/support:>6.2f} {true/len(got):>6.2f}   "
              f"({b[0]:.2f}/{b[1]:.2f})")


if __name__ == "__main__":
    main()
