"""Score every arm on the transfer ladder. Arm selection happens HERE, on
ood_val - never on test, which is read once at the end.

THE LADDER, weakest shift to strongest:

  val         held-out SCENES, same tasks. Already used to pick each
              checkpoint, so it is a selection set and not a result.
  test        held-out scenes, touched once.
  ood_val     rcasa_eval - ArrangeTea and OpenFridge, two task families that
              appear NOWHERE in the training corpus. Fridge is an unseen
              object that moves like the cabinets and microwaves the model did
              train on, so an OpenFridge query is graded correct when it
              returns Open/hinged: the cross-object case stated as the goal.
              ArrangeTea is a wholly novel multi-step task and only matches
              itself.
  ood_test    bridge - real robot video, real kitchens, a different embodiment.
              Read once.

The ood_val pool deliberately MIXES the OOD queries into the in-domain pool.
Scoring rcasa_eval against itself would ask a much easier question - 27 clips
of two visually distinct tasks separate on almost anything. Making an unseen
fridge compete against 156 in-domain distractors is the question that matters.

    python -m relmo.vjood --tags s_rec0.1,s_sig,s_d256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrank, vjz  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402
from relmo.vjzeval import evaluate, report  # noqa: E402


def recs_for(dataset, ids=None, suffix="", root="vjrec6", phase=None,
              arc=0.0):
    rec, sig = vjz.dirs(root, dataset, suffix=suffix)
    have = sorted(p.stem for p in rec.glob("*.npz")
                  if not p.name.startswith("."))
    if ids is not None:
        have = [i for i in have if i in ids]
    d = vjz.gather(have, dataset, want_y=False, rec_dir=rec, sig_dir=sig,
                   phase=phase, arc=arc)
    # a record without its aligned SigLIP companion cannot be encoded; drop it
    # here rather than letting it fail deep inside a batch
    return {i: v for i, v in d.items()
            if v["sig"] is not None and len(v["sig"]) == len(v["a"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--rec-root", default="vjrec6")
    ap.add_argument("--phase", default="rcasa_train",
                    help="vjphase model to subtract; '' disables it")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--test", action="store_true",
                    help="also read the untouched test split - do this ONCE")
    a = ap.parse_args()

    sp = load_split()
    ph = None
    if a.phase:
        from relmo.vjphase import load as load_phase
        ph = load_phase(a.phase)
        print(f"window-phase correction: {a.phase} (fitted on TRAIN only, "
              f"travels to ood unchanged)")
    Rin = recs_for("rcasa", root=a.rec_root, phase=ph)
    Rood = recs_for("rcasa_eval", root=a.rec_root, phase=ph)
    print(f"records: rcasa {len(Rin)} | rcasa_eval {len(Rood)}")

    base = {i: Rin[i]["a"] for i in Rin}
    base.update({i: Rood[i]["a"] for i in Rood})
    inpool = (sp["val"] | sp["test"]) & set(Rin)
    ood_q = set(Rood)
    ood_pool = inpool | ood_q

    def run(name, desc):
        rows = {}
        rows["ood_val"] = evaluate(desc, ood_q, ood_pool, boot=a.boot)
        rows["val"] = evaluate(desc, sp["val"] & set(Rin),
                               sp["val"] & set(Rin), boot=a.boot)
        if a.test:
            rows["test_primary"] = evaluate(desc, sp["test"] & set(Rin),
                                            inpool, boot=a.boot)
            rows["test_deploy"] = evaluate(desc, sp["test"] & set(Rin),
                                           set(Rin), boot=a.boot)
        return rows

    results = {"frozen a_t": run("frozen a_t", base)}
    for tag in [t for t in a.tags.split(",") if t.strip()]:
        tag = tag.strip()
        try:
            model, _ = vjrank.load_ckpt(tag)
        except FileNotFoundError:
            print(f"  no checkpoint {tag} - skipped")
            continue
        # the TRAINED representation through the UNTRAINED matcher. Held next
        # to the frozen row this separates the two things training could have
        # bought: a better descriptor, or a better comparison of descriptors.
        d = vjrank.encode_all(model, Rin)
        d.update(vjrank.encode_all(model, Rood))
        if len(d) < len(base) * 0.9:
            print(f"  {tag}: only {len(d)}/{len(base)} encodable "
                  f"(missing siglip records?)")
        results[f"{tag} z+dtw"] = run(tag, d)

    cols = ["val", "ood_val"] + (["test_primary", "test_deploy"] if a.test
                                 else [])
    print(f"\n{'arm':16s} " + " ".join(f"{c:>22s}" for c in cols))
    print("-" * (16 + 23 * len(cols)))
    for name, rows in results.items():
        cells = []
        for c in cols:
            r = rows[c]
            ci = r.get("ci95", (float("nan"),) * 2)
            cells.append(f"{r['overall']:.3f} [{ci[0]:.2f},{ci[1]:.2f}]")
        print(f"{name:16s} " + " ".join(f"{c:>22s}" for c in cells))
    print(f"\nchance: " + "  ".join(
        f"{c} {results['frozen a_t'][c]['chance']:.3f}" for c in cols))

    best = max((k for k in results if k != "frozen a_t"),
               key=lambda k: results[k]["ood_val"]["overall"], default=None)
    if best:
        print(f"\nSELECTED ON ood_val: {best}")
        report(f"{best} - ood_val (unseen tasks vs in-domain distractors)",
               results[best]["ood_val"])
        report("frozen a_t - ood_val", results["frozen a_t"]["ood_val"])
        if a.test:
            report(f"{best} - test PRIMARY (train-disjoint pool)",
                   results[best]["test_primary"])
            report(f"{best} - test DEPLOYMENT (full index)",
                   results[best]["test_deploy"])
            report("frozen a_t - test PRIMARY",
                   results["frozen a_t"]["test_primary"])
    R.log("vjood", tags=a.tags, touched_test=bool(a.test),
          **{f"{k}_{c}": round(v[c]["overall"], 4)
             for k, v in results.items() for c in cols})


if __name__ == "__main__":
    main()
