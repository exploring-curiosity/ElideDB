"""Full-span ITM: 4 frames across the WHOLE episode, logit margin.

The first corpus pass sampled this way and put q09's eggplant at rank
4 and q00's trues at 1/5/8 - then the artifact was overwritten by the
3-window experiment, whose thirds sample PARTIAL actions. That hurts
for a principled reason: the ITM model was trained on 4-frame clips
spanning whole actions, so the full-span sample is in-distribution
and a third of an action is not. This recomputes the full-span signal
as its own artifact. Artifacts are cheap; only overwriting them is
expensive.

  python scripts/itm_fullspan.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402
from elidedb.iv2 import V_MEAN, V_STD                        # noqa: E402
from itm_probe import _itm_head                              # noqa: E402

NF = 4
QIDS = list(range(11))


def main():
    import cv2
    import torch

    from elidedb.video import FrameSet

    db = Store.open("lake/bench")
    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = db.table("frames").scan()
    head, m, dev = _itm_head()
    toks = [m.tokenizer(QUERIES[qi], padding="max_length", truncation=True,
                        max_length=m._config.max_txt_l,
                        return_tensors="pt").to(dev) for qi in QIDS]

    S = np.full((len(keys), len(QIDS)), np.nan, np.float32)
    t0, done = time.time(), 0
    for i, (s, a, b) in enumerate(keys):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            continue
        pi = np.linspace(0, len(sel) - 1,
                         min(NF, len(sel))).round().astype(int)
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode(width=224)
        except Exception:
            continue
        fr = [f for _, f in sorted(dec)]
        if len(fr) < 2:
            continue
        fs = [cv2.resize(f, (224, 224)) for f in fr]
        x = (np.stack(fs).astype(np.float32) / 255.0 - V_MEAN) / V_STD
        px = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(dev, m.dtype)
        with torch.no_grad():
            vis, _ = m.encode_vision(px, test=True)
            vam = torch.ones(vis.shape[:2], dtype=torch.long, device=dev)
            for j, tok in enumerate(toks):
                out = m.get_text_encoder()(
                    tok.input_ids, attention_mask=tok.attention_mask,
                    encoder_hidden_states=vis, encoder_attention_mask=vam,
                    return_dict=True, mode="multi_modal")
                lg = head(out.last_hidden_state[:, 0]).float()[0]
                S[i, j] = float(lg[1] - lg[0])
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(keys)}  {el:.0f}s  "
                  f"ETA {el / done * len(keys) / 60:.0f}min", flush=True)

    out = ROOT / "artifacts/itm_fullspan.npz"
    np.savez(out, S=S, qids=np.array(QIDS),
             streams=np.array([k[0] for k in keys]),
             ts=np.array([k[1] for k in keys], np.int64),
             t1=np.array([k[2] for k in keys], np.int64))
    print(json.dumps({"episodes": int(done),
                      "seconds": round(time.time() - t0, 1),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
