"""ITM over the FULL corpus x all graded queries. No pools, no caps.

The probe said the discarded itm_head out-separates the 7B VLM judge
on all three calibration queries (AUC 0.963/0.897/0.968 vs
0.887/0.751/0.845) at half the cost. Pools lie (the crop channel's
1.00-on-80 was 0.00-on-1,121), so the claim is only real if it holds
over all 1,121 episodes - which is affordable because the expensive
half of ITM is query-INDEPENDENT: encode each episode's video tokens
once, then every query is one cheap fused BERT forward over the cached
tokens. 11 queries cost barely more than one.

Two lessons from the first full pass are baked in here:
  logits, not softmax   P(match) saturates - dozens of negatives at
                        0.99+ and q10's true episodes at ranks 61/135
                        DESPITE scores of 0.94/0.99. Softmax destroyed
                        resolution exactly at the top of the list; the
                        raw logit margin keeps it.
  3 windows, PER-WINDOW one 4-frame window can miss the object
                        entirely (q09's second true episode scored
                        0.765 at rank 190). But max-pooling windows
                        was measured WORSE (mean yield 0.40 -> 0.36,
                        q00 0.38 -> 0.12): a negative gets three
                        chances to fluke a high margin - the same
                        extreme-value trap that killed patch MaxSim.
                        So the matrix stores every window's score and
                        the pooling is chosen by measurement, offline.

Writes the raw (episodes x queries) logit-margin matrix to
ml/itm_scores3.npz. Evaluation is a separate script so re-scoring is
never needed to re-analyze.

  python scripts/itm_full.py [--limit N]
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
QIDS = list(range(11))          # q00..q10: every graded query + the gate


def main():
    import cv2
    import torch

    from elidedb.video import FrameSet

    argv = sys.argv
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else None

    db = Store.open("lake/bench")
    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    if limit:
        keys = keys[:limit]
    frames_tbl = db.table("frames").scan()
    head, m, dev = _itm_head()

    # tokenize every query once
    toks = []
    for qi in QIDS:
        toks.append(m.tokenizer(QUERIES[qi], padding="max_length",
                                truncation=True,
                                max_length=m._config.max_txt_l,
                                return_tensors="pt").to(dev))

    S3 = np.full((len(keys), len(QIDS), 3), np.nan, np.float32)
    t0 = time.time()
    done = 0
    for i, (s, a, b) in enumerate(keys):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            continue
        # decode enough frames that the 3 windows actually cover the
        # episode (3 windows x NF frames), not 3 slices of one sample
        pi = np.unique(np.linspace(0, len(sel) - 1,
                                   min(3 * NF, len(sel))).round()
                       .astype(int))
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode(width=224)
        except Exception:
            continue
        fr = [f for _, f in sorted(dec)]
        if len(fr) < 2:
            continue
        for w0 in range(3):                              # early/mid/late
            lo = int(w0 * len(fr) / 3)
            hi = max(int((w0 + 1) * len(fr) / 3), lo + 1)
            win = fr[lo:hi]
            wi = np.linspace(0, len(win) - 1,
                             min(NF, len(win))).round().astype(int)
            fs = [cv2.resize(win[k], (224, 224)) for k in wi]
            x = (np.stack(fs).astype(np.float32) / 255.0 - V_MEAN) / V_STD
            px = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(
                dev, m.dtype)
            with torch.no_grad():
                vis, _ = m.encode_vision(px, test=True)  # per window
                vam = torch.ones(vis.shape[:2], dtype=torch.long,
                                 device=dev)
                for j, tok in enumerate(toks):           # cheap per query
                    out = m.get_text_encoder()(
                        tok.input_ids, attention_mask=tok.attention_mask,
                        encoder_hidden_states=vis,
                        encoder_attention_mask=vam,
                        return_dict=True, mode="multi_modal")
                    lg = head(out.last_hidden_state[:, 0]).float()[0]
                    S3[i, j, w0] = float(lg[1] - lg[0])
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(keys)}  {el:.0f}s  "
                  f"ETA {el / done * len(keys) / 60:.0f}min", flush=True)

    out = ROOT / "ml/itm_scores3.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, S=np.nanmax(S3, axis=2), S3=S3, qids=np.array(QIDS),
             streams=np.array([k[0] for k in keys]),
             ts=np.array([k[1] for k in keys], np.int64),
             t1=np.array([k[2] for k in keys], np.int64))
    print(json.dumps({"episodes": int(done), "queries": len(QIDS),
                      "seconds": round(time.time() - t0, 1),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
