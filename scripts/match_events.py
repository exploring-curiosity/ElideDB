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

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402

# closed-class, uniform English: query verb words -> topology verbs.
# "move" appears everywhere because the extractor's move covers any
# relocation; direction verbs map to the articulation channel.
# v2 verbs are a geometric partition (largest class 28%, was 80%
# when every named track answered "move"), so the map targets the
# TRANSITION the query describes. The PREPOSITION is the stronger
# signal and wins when present: "into the drawer" is put_into
# whatever the sentence's main verb happens to be.
VERB_MAP = {
    "open": {"open"},
    "close": {"close"}, "closes": {"close"}, "shut": {"close"},
    "put": {"put_into", "put_on", "put_away"},
    "place": {"put_into", "put_on", "put_away"},
    "pick": {"put_into", "put_on", "take_out"},
    "take": {"take_out"},
    "move": {"put_into", "put_on", "put_away"},
    # "holds the handle and closes" - the sentence's OTHER verb is
    # the event. Mapping "holds" to three classes matched 837 of 1,122
    # demos and buried the 62%-correct `close` label under everything
    # else, so a grasp word contributes nothing on its own.
    "holds": set(),
}
REL_VERBS = {"into": {"put_into"}, "on": {"put_on"},
             "out": {"take_out"}}


# closed-class English prepositions -> geometric relation classes
PREP_MAP = (("on top", "on"), ("onto", "on"), ("into", "into"),
            ("in ", "into"), ("out of", "out"), ("from", "out"),
            ("on ", "on"))


def parse_query(text, atoms_of):
    tl = text.lower()
    verbs = set()
    for w, tv in VERB_MAP.items():
        if w in tl.split() or any(t.startswith(w) for t in tl.split()):
            verbs |= tv
    rel = ""
    for pat, r in PREP_MAP:
        if pat in tl:
            rel = r
            break
    if rel in REL_VERBS and not (verbs & {"open", "close"}):
        verbs = REL_VERBS[rel]
    atoms = atoms_of(tl)[:2]
    return verbs, atoms, rel


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

    # NAME SPECIFICITY, corpus-derived: "black object" passes detector
    # verification in any crop, so vague names flooded v2. A name that
    # appears on half the corpus carries no information about any one
    # demo; classic IDF downweights it without a single hand rule.
    from collections import Counter
    cnt = Counter(nm for nm in ans["name"] if nm)
    n_named = max(sum(cnt.values()), 1)
    import math
    idf = {nm: math.log(n_named / c) / math.log(n_named)
           for nm, c in cnt.items()}
    row_idf = np.array([idf.get(nm, 0.0) for nm in ans["name"]],
                       np.float32)

    # group answer rows per demo
    keys = sorted({(s, int(a), int(b)) for s, a, b in
                   zip(ans["stream"], ans["ts"], ans["t1"])})
    kidx = {k: i for i, k in enumerate(keys)}
    rows_of = {i: [] for i in range(len(keys))}
    for r in range(n):
        rows_of[kidx[(ans["stream"][r], int(ans["ts"][r]),
                      int(ans["t1"][r]))]].append(r)
    verb_of = {i: ans["verb"][rows_of[i][0]] for i in rows_of}
    # v2 verbs from scripts/verb_recompute.py when present
    import json as _json
    vp = ROOT / "artifacts/verbs_v2.json"
    if vp.exists():
        v2 = _json.loads(vp.read_text())
        hit = 0
        for i in rows_of:
            k = f"{keys[i][0]}|{keys[i][1]}"
            if k in v2:
                verb_of[i] = v2[k]; hit += 1
        print(f"verbs v2 applied to {hit}/{len(rows_of)} demos")
    art_of = {i: float(ans["articulation"][rows_of[i][0]])
              for i in rows_of}
    # geometry per demo: dest/origin boxes + the articulated region.
    # box columns are FixedSizeList; flatten once.
    BX = np.array(ans["box"], np.int32).reshape(-1, 4) \
        if "box" in ans else None
    AB = np.array(ans["art_box"], np.int32).reshape(-1, 4) \
        if "art_box" in ans else None

    def rel_score(i, qrel):
        """Geometric relation between the participant's END state and
        the articulated region. Closed-class geometry, no camera
        assumptions beyond box containment/overlap:
          into  the moved thing's last box sits INSIDE the articulated
                region (it went in), or it has an origin but no dest
                (it disappeared)
          on    a dest box exists (visible at end) and OVERLAPS the
                articulated region
          out   an origin box sits inside the region (it came out)"""
        if not qrel or BX is None:
            return 0.5
        ab = AB[rows_of[i][0]]
        has_art = (ab[2] - ab[0]) * (ab[3] - ab[1]) > 0
        dests = [BX[r] for r in rows_of[i] if ans["kind"][r] == "dest"]
        origs = [BX[r] for r in rows_of[i] if ans["kind"][r] == "origin"]

        def center_in(b, rgn):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return rgn[0] <= cx <= rgn[2] and rgn[1] <= cy <= rgn[3]

        def overlap(b, rgn):
            return not (b[2] < rgn[0] or b[0] > rgn[2]
                        or b[3] < rgn[1] or b[1] > rgn[3])

        if qrel == "into":
            if origs and not dests:
                return 1.0                    # vanished: went inside
            if has_art and any(center_in(d, ab) for d in dests):
                return 1.0
            return 0.2
        if qrel == "on":
            if dests and (not has_art
                          or any(overlap(d, ab) for d in dests)):
                return 1.0
            return 0.3
        if qrel == "out":
            if has_art and any(center_in(o, ab) for o in origs):
                return 1.0
            return 0.3
        return 0.5

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
        verbs, atoms, qrel = parse_query(text, atoms_of)
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
            # ONE verb path for every query. open/close used to be
            # routed around the verb field into the raw articulation
            # sign, which is exactly the signal measured useless
            # (corr +0.04); the cavity-based verb label is 2.2-2.5x
            # over prior on the two drawer queries and the matcher
            # has to actually read it.
            v = verb_of[i]
            vs[i] = 1.0 if (not verbs or v in verbs) else 0.0
            if qv:
                rr = [r for r in rows_of[i] if has_name[r]]
                if rr:
                    sims = NV[rr] @ np.stack(qv).T      # (rows, atoms)
                    w = row_idf[rr][:, None]
                    ns[i] = float((sims * (0.3 + 0.7 * w)).max())
        rel = np.array([rel_score(i, qrel) for i in range(len(keys))])
        cells = []
        for m in modes:
            if m == "full":
                sc = vs * (1.0 + ns) * (0.5 + rel)
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
