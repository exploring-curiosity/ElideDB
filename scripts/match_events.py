"""V2/V3: match queries against the answers table. Frame unification.

A query is parsed into the same frame the extractor writes - verb
class, participant noun, destination noun - and matching is per-field
(Marengo-style aspect fusion), never one pooled score:

  verb    closed-class uniform-English map (open/close/put/take...) ->
          compatible topology verbs. Generic dictionary knowledge,
          applied identically to every query (L9-allowed).
  name    cosine(query noun vec, participant name_vec) in SigLIP text
          space - the namer's words and the query's words meet in the
          embedding space, never as strings, so "green block" matches
          "a green object" without either being special-cased.
  sign    open-vs-close needs an articulation SIGN convention, which
          is camera geometry, not knowledge we may hardwire. It is
          CALIBRATED per store: the sign whose demos correlate best
          with the act-channel's own open-posterior wins. Corpus-
          derived, self-recognized, no assumption survives a camera
          flip.

Prints the two product metrics per query (event channel ALONE) plus
verb-only and name-only ablations (V3), against the 0.29/0.23 fused
baseline. Read-only; no artifacts written.

  python scripts/match_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402

# closed-class, uniform English: query verb words -> topology verbs.
# "move" appears everywhere because the extractor's move covers any
# relocation; direction verbs map to the articulation channel.
VERB_MAP = {
    "open": {"open"},
    "close": {"close"}, "closes": {"close"}, "shut": {"close"},
    "put": {"move", "bring", "put_away"},
    "place": {"move", "bring", "put_away"},
    "pick": {"move", "put_away"},
    "take": {"move", "bring"},
    "move": {"move", "bring", "put_away"},
    "holds": {"close", "open", "move"},
}


def parse_query(text, atoms_of):
    tl = text.lower()
    verbs = set()
    for w, tv in VERB_MAP.items():
        if w in tl.split() or any(t.startswith(w) for t in tl.split()):
            verbs |= tv
    atoms = atoms_of(tl)[:2]
    return verbs, atoms


def main():
    from elidedb.sig2 import _text_vec, atoms_of

    db = Store.open("lake/bench")
    ans = db.table("answers").scan().to_pydict()
    n = len(ans["ts"])
    print(f"answers table: {n} rows")
    from elidedb.embeddings import _vec_table
    _, NV = _vec_table(db, "answers", column="name_vec")
    NV = np.asarray(NV, np.float32)
    nrm = np.linalg.norm(NV, axis=1, keepdims=True)
    NV = NV / np.maximum(nrm, 1e-8)
    has_name = (nrm[:, 0] > 1e-6)

    # group answer rows per demo
    keys = sorted({(s, int(a), int(b)) for s, a, b in
                   zip(ans["stream"], ans["ts"], ans["t1"])})
    kidx = {k: i for i, k in enumerate(keys)}
    rows_of = {i: [] for i in range(len(keys))}
    for r in range(n):
        rows_of[kidx[(ans["stream"][r], int(ans["ts"][r]),
                      int(ans["t1"][r]))]].append(r)
    verb_of = {i: ans["verb"][rows_of[i][0]] for i in rows_of}
    art_of = {i: float(ans["articulation"][rows_of[i][0]])
              for i in rows_of}

    # ---- articulation sign calibration (corpus-derived, no geometry
    # assumption): among demos the extractor called open/close, which
    # art sign correlates with the act-channel's open posterior?
    sign = 1.0
    try:
        from elidedb.action_channel import act_lookup
        look, _ = act_lookup(db, "robot arm opens the drawer")
        a_open = np.array([look(*keys[i]) for i in range(len(keys))])
        arts = np.array([art_of[i] for i in range(len(keys))])
        m = np.isfinite(a_open) & (np.abs(arts) > 1.5)
        if m.sum() >= 20:
            c = float(np.corrcoef(np.sign(arts[m]), a_open[m])[0, 1])
            sign = 1.0 if c >= 0 else -1.0
            print(f"articulation sign calibrated: {sign:+.0f} "
                  f"(corr {c:+.2f} on {int(m.sum())} demos)")
    except Exception as e:
        print(f"sign calibration unavailable ({type(e).__name__}); "
              f"using +1")

    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    modes = ("full", "verb-only", "name-only")
    print(f"\n{'q':>4} {'sup':>4} {'K':>4} | " +
          " | ".join(f"{m:>16}" for m in modes) +
          "   (yield/prec per mode)")
    agg = {m: ([], []) for m in modes}
    for qi in sorted(sup):
        if sup[qi] == 0:
            continue
        text = QUERIES[qi]
        verbs, atoms = parse_query(text, atoms_of)
        qv = []
        for a in atoms:
            v = np.asarray(_text_vec(a), np.float32)
            qv.append(v / (np.linalg.norm(v) + 1e-8))
        lab = np.array([1 if truth.get((qi, k[0], k[1])) == 1 else 0
                        for k in keys])
        K = int(np.ceil(sup[qi] * 1.5))

        # per-demo field scores
        vs = np.zeros(len(keys))
        ns = np.zeros(len(keys))
        for i in range(len(keys)):
            v = verb_of[i]
            ok = v in verbs if verbs else True
            # open/close ride the calibrated articulation sign
            if verbs & {"open", "close"}:
                a = sign * art_of[i]
                want_open = "open" in verbs
                ok = (a > 1.5) if want_open else (a < -1.5)
                vs[i] = abs(art_of[i]) if ok else 0.0
            else:
                vs[i] = 1.0 if ok else 0.0
            if qv:
                rr = [r for r in rows_of[i] if has_name[r]]
                if rr:
                    sims = NV[rr] @ np.stack(qv).T      # (rows, atoms)
                    ns[i] = float(sims.max())
        cells = []
        for m in modes:
            if m == "full":
                sc = vs * (1.0 + ns)
            elif m == "verb-only":
                sc = vs
            else:
                sc = ns
            order = np.argsort(-sc)
            tru = int(lab[order[:K]].sum())
            y, p = tru / sup[qi], tru / K
            agg[m][0].append(y); agg[m][1].append(p)
            cells.append(f"{y:.2f}/{p:.2f}      ")
        print(f"q{qi:02d} {sup[qi]:>4} {K:>4} | " + " | ".join(cells))
    print("-" * 100)
    print(f"{'mean':>13} | " + " | ".join(
        f"{np.mean(agg[m][0]):.2f}/{np.mean(agg[m][1]):.2f}      "
        for m in modes) + f"   (fused baseline 0.29/0.23)")


if __name__ == "__main__":
    main()
