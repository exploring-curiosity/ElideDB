"""Retire channels IN THE STORE, not in code.

spaces() enrolls any *_vectors table whose meta declares a `model`; a
channel therefore leaves the fusion by a log commit that REMOVES that
declaration and states why. The store carries its own roster - a code-
level drop list would be a prior the next corpus inherits blind.

Measured basis (bench_qbe_weighted --drop, coh^4 mean yield q03/04/05):

    full fusion (8 ch)          0.69
    - pe, xclip                 0.74   <- the drop IMPROVES fusion
    - iv2                       0.63   <- iv2 is NOT redundant: q03
                                          0.59 -> 0.44. The roster's
                                          "iv2 does sig2's noun job"
                                          guess was WRONG for QbE and
                                          is withdrawn on this number.
    - pe, xclip, act            0.73   <- act's exit costs 0.01 (noise);
                                          it leaves per the user's
                                          roster call, and cheaply

Rows are rewritten unchanged (pe 34 MB, xclip 6 MB, act 1 MB - the cost
of an honest ledger); only the meta changes. Rebuilding the channel
later is one commit that re-declares `model`.

    python scripts/retire_channels.py [--store lake/fresh_bench]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402

# what the first run blanked, from the tables' own earlier commits
RESTORE = {
    "pe_vectors": "PE-Core-L-14-336",
    "xclip_vectors": "microsoft/xclip-large-patch14",
    "action_probs": "vjepa2-vitl + ssv2 attentive probe",
}

RETIRE = {
    "pe_vectors": "0.89 rank-dup of sig2; fusion 0.69 -> 0.74 without "
                  "pe+xclip (measured 2026-08-01)",
    "xclip_vectors": "0.70 dup of pe/sig2, cross-frame attention was "
                     "discarded at write; dropped with pe, same measure",
    "action_probs": "user roster call 2026-08-01 (no earned purpose); "
                    "exit measured at -0.01 fusion yield (noise)",
    "vjepa_vectors": "spec says never whole-frame; vjepa_part (tubelets "
                     "seeded from trajectories) replaces it at zero "
                     "measured cost (coh^4 0.72 -> 0.73, 2026-08-02)",
}


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    for name, why in RETIRE.items():
        if name not in db.tables():
            print(f"{name}: absent, nothing to retire")
            continue
        t = db.table(name)
        meta = dict(t.state().meta or {})
        # model="" is the tombstone, and the key stays in the fold -
        # test truthiness, or a re-run "re-retires" a retired table and
        # blanks its retired_model record (which is exactly what the
        # first run of this script did; restored below)
        if not meta.get("model"):
            if not meta.get("retired_model") and name in RESTORE:
                meta["retired_model"] = RESTORE[name]
                t.replace(t.scan(), kind=t.state().kind or "vectors",
                          meta=meta)
                print(f"{name}: retired_model restored")
            else:
                print(f"{name}: already retired")
            continue
        # the log FOLDS meta with dict.update, so a key can never be
        # removed by omission - only overwritten. Empty string is the
        # tombstone: spaces() tests truthiness, and the history keeps
        # what the model was in retired_model.
        meta["retired_model"] = meta["model"]
        meta["model"] = ""
        meta["retired"] = why
        t.replace(t.scan(), kind=t.state().kind or "vectors", meta=meta)
        assert not (t.state().meta or {}).get("model")
        print(f"{name}: retired ({meta['retired_model']})")


if __name__ == "__main__":
    main()
