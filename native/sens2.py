"""How much are boundaries worth NOW? sensitivity.py, re-run honestly.

native/sensitivity.py measured retrieval against degraded boundaries and
concluded segmentation was not the bottleneck: yield 0.267 at perfect
boundaries, 0.267 at boundary F1 0.5, flat. That conclusion steered
several decisions in this build and it is STALE.

It was measured with encode.py's V-JEPA2, which mean-pools and is
therefore order-blind - action AUC 0.521 against a chance of 0.500. An
encoder that cannot represent the action is the binding constraint, so
boundary quality had nothing to show. u6retr.py then found:

    siglip2_rank   oracle spans 0.595  ->  real spans 0.262
    vjepa2         oracle spans 0.167  ->  real spans 0.214

so with a working encoder the boundaries bind hard, and the slope has
to be re-measured before anyone quotes it again.

Same degradation ladder as the original so the two are comparable; only
the encoder changes. The output is the answer to "what boundary F1 does
step 5 need?", which the original answered with "any".

    python native/sens2.py --eps 40      # ~15 min
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg, f1_at                                # noqa: E402
from sensitivity import degrade, units_from                    # noqa: E402
from u6retr import score, units                                # noqa: E402
from unitenc import FPS                                        # noqa: E402


def main():
    import encode as E
    import pyarrow.parquet as pq
    from tqdm import tqdm

    NEPS = arg("--eps", 40, int)
    ENCS = arg("--enc", "siglip2_rank,vjepa2").split(",")
    LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl, bnd = {}, {}
    for e, tm, a, b in zip(t["episode"], t["template"], t["t0"],
                           t["t1"]):
        tmpl[int(e)] = tm
        bnd.setdefault(int(e), set()).update(
            [round(float(a), 2), round(float(b), 2)])

    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))[:NEPS]
    print(f"sensitivity v2 — {len(dirs)} episodes, encoders {ENCS}, "
          f"{len(LEVELS)} degradation levels", flush=True)

    rng = np.random.default_rng(0)
    media, spans, qual = {}, {lv: {} for lv in LEVELS}, \
        {lv: [] for lv in LEVELS}
    for d in tqdm(dirs, desc="decode", unit="ep"):
        ei = int(d.name[2:])
        if ei not in bnd:
            continue
        cam = sorted(d.glob("cam*.mp4"))[0]
        dur = E.probe_duration(cam)
        media[ei] = E.decode(cam, fps=FPS, w=256)
        truth = sorted(bnd[ei])
        for lv in LEVELS:
            bs = degrade(truth, dur, lv, rng)
            qual[lv].append(f1_at(bs, truth, 0.05 * dur)[0])
            spans[lv][ei] = units_from(bs, dur)

    print(f"\n{'encoder':<15}{'boundary F1':<14}{'yield':<9}"
          f"{'prec':<9}{'units/ep':<10}{'support'}")
    for enc in ENCS:
        for lv in LEVELS:
            seqs = {}
            for ei, F in tqdm(media.items(),
                              desc=f"{enc} lv{lv}", unit="ep",
                              leave=False):
                seqs[ei] = units(enc, f"sens{lv}", ei, F, spans[lv][ei])
            y, p, su, rt = score(seqs, tmpl)
            nu = np.mean([len(v) for v in seqs.values()])
            print(f"{enc:<15}{np.mean(qual[lv]):<14.3f}{y:<9.3f}"
                  f"{p:<9.3f}{nu:<10.1f}{su:<9.1f}{rt:.1f}", flush=True)
        print()


if __name__ == "__main__":
    main()
