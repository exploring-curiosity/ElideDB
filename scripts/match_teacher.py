"""Match queries against the five-element teacher. Events are a SET.

A compositional query is a set of required events, not one label:
"pick up a green object from table and put it into the drawer" asks
for a put_into whose participant is greenish, inside a demo that also
opens the drawer. The single-verb model could not state that, which is
why q00/q01/q02 measured exactly zero however good the verb was.

Scoring, per field, with per-field confidences (the Marengo shape):

    verb   does the demo contain the required event kind at all
    name   cosine(query noun, that EVENT's participant name_vec) -
           the name is attached to the event, so "green" has to be
           the thing that went in, not merely present somewhere
    scene  cosine(query, demo gist) - a weak prior, never a verdict
    agent  demos with no self-moving thing cannot be manipulations

Prints the two product metrics plus per-field ablations.

  python scripts/match_teacher.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402

# closed-class English -> required event kinds. Uniform across queries.
PREP = (("on top", "put_on"), ("onto", "put_on"), ("into", "put_into"),
        ("out of", "take_out"), ("in ", "put_into"), ("on ", "put_on"))
VERBW = {"open": "open", "opens": "open", "close": "close",
         "closes": "close", "shut": "close"}


def parse(text, atoms_of):
    tl = text.lower()
    need = set()
    for w in tl.split():
        if w in VERBW:
            need.add(VERBW[w])
    if not need:
        for pat, k in PREP:
            if pat in tl:
                need.add(k)
                break
    # "from the drawer ... on the table" is a take_out even though the
    # sentence's preposition of record is "on"
    if "from the drawer" in tl or "out of" in tl:
        need = {"take_out"} | (need - {"put_on", "put_into"})
    return need, atoms_of(tl)[:2]


def main():
    from elidedb.embeddings import _vec_table
    from elidedb.sig2 import _text_vec, atoms_of

    db = Store.open("lake/bench")
    ev = db.table("events").scan().to_pydict()
    an = db.table("answers2").scan().to_pydict()
    print(f"events {len(ev['ts'])} rows, answers2 {len(an['ts'])} demos")
    _, EV = _vec_table(db, "events", column="name_vec")
    EV = np.asarray(EV, np.float32)
    n = np.linalg.norm(EV, axis=1, keepdims=True)
    EV = EV / np.maximum(n, 1e-8)
    _, SV = _vec_table(db, "answers2", column="scene_vec")
    SV = np.asarray(SV, np.float32)

    keys = [(an["stream"][i], int(an["ts"][i])) for i in range(len(an["ts"]))]
    kidx = {k: i for i, k in enumerate(keys)}
    by = defaultdict(list)
    for r in range(len(ev["ts"])):
        k = (ev["stream"][r], int(ev["ts"][r]))
        if k in kidx:
            by[kidx[k]].append(r)
    span = np.array(an["agent_span"], np.float32)

    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    modes = ("full", "event-only", "name@event", "scene-only")
    print(f"\n{'q':>4} {'sup':>4} {'K':>4} {'need':>22} | " +
          " | ".join(f"{m:>11}" for m in modes))
    agg = {m: ([], []) for m in modes}
    for qi in sorted(sup):
        if sup[qi] == 0:
            continue
        need, atoms = parse(QUERIES[qi], atoms_of)
        qv = np.stack([_text_vec(a) / (np.linalg.norm(_text_vec(a)) + 1e-8)
                       for a in atoms]) if atoms else None
        # the scene gist is pooled from pe_vectors, so it must be
        # compared with PE's OWN text encoder - names live in SigLIP2
        # space, scenes in PE space, and mixing them is a dimension
        # error that would have been a silent wrong answer if the two
        # happened to match in width
        from elidedb.pe import _text_vec as _pe_text
        sq = np.asarray(_pe_text(QUERIES[qi]), np.float32)
        sq = sq / (np.linalg.norm(sq) + 1e-8)
        lab = np.array([1 if truth.get((qi, k[0], k[1])) == 1 else 0
                        for k in keys])
        K = int(np.ceil(sup[qi] * 1.5))

        e_sc = np.zeros(len(keys))
        n_sc = np.zeros(len(keys))
        for i in range(len(keys)):
            rr = by.get(i, [])
            hits = [r for r in rr if ev["kind"][r] in need] if need else rr
            e_sc[i] = 1.0 if hits else 0.0
            if qv is not None and hits:
                m = [r for r in hits if ev["name"][r]]
                if m:
                    n_sc[i] = float((EV[m] @ qv.T).max())
        s_sc = SV @ sq
        gate = (span >= 0.25).astype(float)
        cells = []
        for mo in modes:
            if mo == "full":
                sc = gate * e_sc * (1.0 + n_sc) + 0.05 * s_sc
            elif mo == "event-only":
                sc = e_sc
            elif mo == "name@event":
                sc = n_sc
            else:
                sc = s_sc
            tru = int(lab[np.argsort(-sc)[:K]].sum())
            y, p = tru / sup[qi], tru / K
            agg[mo][0].append(y); agg[mo][1].append(p)
            cells.append(f"{y:.2f}/{p:.2f}".rjust(11))
        print(f"q{qi:02d} {sup[qi]:>4} {K:>4} {str(sorted(need)):>22} | "
              + " | ".join(cells))
    print("-" * 96)
    print(f"{'mean':>16} {'':>10} | " + " | ".join(
        f"{np.mean(agg[m][0]):.2f}/{np.mean(agg[m][1]):.2f}".rjust(11)
        for m in modes) + "   (fused floor 0.29/0.23)")


if __name__ == "__main__":
    main()
