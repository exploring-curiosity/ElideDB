"""Gate d: the disguise battery scored on PREDICTOR STATES.

Same instrument as vdyn (4 real clips x EVAL disguises + 60 undisguised
distractors, content-free), same fixed-mean scoring - the only change
is the representation: the predictor's state trajectory over the clip's
frozen latents, mean-pooled. The bars to beat, measured earlier:

    raw-pixel floor (32x32 thumbs)   sev-rho 0.553   struct-rho 0.390
    frozen vits fixed-mean (global)  AUC 0.987 P@1 1.000  0.499/0.423

A sim-trained predictor scored on real corpora - transfer is part of
what this measures; the number is reported either way.

    SDX_ENC=vits SDX_RES=320 python native/vwm_battery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import vaug                                               # noqa: E402
import vcore                                              # noqa: E402
from vdyn import latent, _l2                              # noqa: E402
from vtrans import CLIPS, distractors, spearman           # noqa: E402


def main():
    import vwm
    model = vwm.load()
    B = vaug.battery(prims=vaug.EVAL_PRIMS)
    pmap = dict(B)
    sev = {n: vaug.severity(p) for n, p in B}
    vcore.feature_dim()
    D = distractors()
    from tqdm import tqdm

    items, R = [], {}
    bar = tqdm(total=len(CLIPS) * len(B) + len(D), unit="clip",
               desc="states")
    for corpus, mid, t0, t1 in CLIPS:
        for n, p in B:
            k = (corpus, n)
            items.append(k)
            g, c = latent(corpus, mid, t0, t1, n, p)
            h, _ = model.states(vwm.build_input(g, c))
            R[k] = _l2(h.mean(0))
            bar.update(1)
    for j, (corpus, mid, a, b) in enumerate(D):
        k = (f"dist{j}", "identity")
        items.append(k)
        g, c = latent(corpus, mid, a, b, "identity", vaug.identity_params())
        h, _ = model.states(vwm.build_input(g, c))
        R[k] = _l2(h.mean(0))
        bar.update(1)
    bar.close()

    M = np.stack([R[k] for k in items])
    S = M @ M.T
    same = np.array([[a[0] == b[0] for b in items] for a in items])
    off = ~np.eye(len(items), dtype=bool)
    qi = [i for i, k in enumerate(items) if not k[0].startswith("dist")]
    pos = S[np.ix_(qi, range(len(items)))][
        same[np.ix_(qi, range(len(items)))] & off[np.ix_(qi, range(len(items)))]]
    neg = S[np.ix_(qi, range(len(items)))][
        (~same[np.ix_(qi, range(len(items)))]) & off[np.ix_(qi, range(len(items)))]]
    auc = float((pos[:, None] > neg[None, :]).mean()
                + 0.5 * (pos[:, None] == neg[None, :]).mean())
    p1 = float(np.mean([
        items[int(np.argmax(np.where(off[i], S[i], -np.inf)))][0]
        == items[i][0] for i in qi]))
    sr, st = [], []
    for corpus, _, _, _ in CLIPS:
        ii = [i for i, k in enumerate(items) if k[0] == corpus]
        i0 = [i for i in ii if items[i][1] == "identity"][0]
        sr.append(spearman([S[i0, i] for i in ii if i != i0],
                           [-sev[items[i][1]] for i in ii if i != i0]))
        fs, td = [], []
        for a2 in ii:
            for b2 in ii:
                if a2 < b2:
                    fs.append(S[a2, b2])
                    td.append(-vaug.tdist(pmap[items[a2][1]],
                                          pmap[items[b2][1]]))
        st.append(spearman(fs, td))
    print(f"\npredictor-state fixed-mean:  AUC {auc:.3f}  P@1 {p1:.3f}  "
          f"sev-rho {np.mean(sr):.3f}  struct-rho {np.mean(st):.3f}")
    print("bars: pixel floor 0.553/0.390; frozen fixed-mean "
          "0.987/1.000/0.499/0.423")


if __name__ == "__main__":
    main()
