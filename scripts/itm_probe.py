"""The reranker we already owned: InternVideo2's ITM head, measured.

Every load of iv2_stage2_1b has printed the same warning since the
channel landed: "Some weights ... were not used: ['itm_head.bias',
'itm_head.weight', 'temp']". That is not noise - it is the CROSS-
ENCODER half of the retrieval recipe the model was trained with. The
BLIP/UMT/InternVideo2 papers all retrieve the same way: cosine over
the contrastive (ITC) embeddings for recall, then rerank the top
candidates with the ITM head, where text tokens CROSS-ATTEND to the
video tokens and a binary classifier reads the fused CLS. The channel
shipped with only the cheap half; the fine-grained matcher sat unused
on disk.

Cost shape: ITM is quadratic-ish (one fused forward per candidate per
query), so it cannot replace the cosine scan - but the teacher is
allowed to be expensive, and unlike a VLM judge the ITM head was
TRAINED on exactly this yes/no question over exactly this
architecture's features.

Protocol is identical to teacher_probe.py (same queries, same seed,
same sample construction), so the numbers are directly comparable to
the 7B VLM's AUC 0.887/0.751/0.845 on q09/q08/q04.

  python scripts/itm_probe.py [--q 9,8,4] [--neg 40]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402
from elidedb.iv2 import MDIR, V_MEAN, V_STD, load_model      # noqa: E402

NF = 4


def _itm_head():
    """The checkpoint's own 2-class matcher, loaded from the weights
    AutoModel discards (the class never declares the attribute)."""
    import torch
    sd = torch.load(f"{MDIR}/pytorch_model.bin", map_location="cpu",
                    weights_only=True, mmap=True)
    w, b = sd["itm_head.weight"], sd["itm_head.bias"]
    m, dev = load_model()
    head = torch.nn.Linear(w.shape[1], 2)
    head.load_state_dict({"weight": w, "bias": b})
    return head.to(dev, m.dtype).eval(), m, dev


def itm_score(head, m, dev, frames_hwc, text):
    """P(match) from the fused cross-encoder, one clip one query."""
    import cv2
    import torch
    fs = [cv2.resize(f, (224, 224)) for f in frames_hwc]
    x = (np.stack(fs).astype(np.float32) / 255.0 - V_MEAN) / V_STD
    px = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(dev, m.dtype)
    with torch.no_grad():
        vis, _ = m.encode_vision(px, test=True)          # [1, N, C]
        tok = m.tokenizer(text, padding="max_length", truncation=True,
                          max_length=m._config.max_txt_l,
                          return_tensors="pt").to(dev)
        out = m.get_text_encoder()(
            tok.input_ids, attention_mask=tok.attention_mask,
            encoder_hidden_states=vis,
            encoder_attention_mask=torch.ones(vis.shape[:2],
                                              dtype=torch.long,
                                              device=dev),
            return_dict=True, mode="multi_modal")
        cls = out.last_hidden_state[:, 0]
        p = torch.softmax(head(cls).float(), dim=-1)[0, 1]
    return float(p)


def main():
    from elidedb.video import FrameSet

    argv = sys.argv
    qs = [int(x) for x in (argv[argv.index("--q") + 1].split(",")
                           if "--q" in argv else ["9", "8", "4"])]
    nneg = int(argv[argv.index("--neg") + 1]) if "--neg" in argv else 40

    db = Store.open("lake/bench")
    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = db.table("frames").scan()
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    def frames_of(s, a, b):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            return []
        pi = np.linspace(0, len(sel) - 1,
                         min(NF, len(sel))).round().astype(int)
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode(width=224)
        except Exception:
            return []
        return [f for _, f in sorted(dec)]

    head, m, dev = _itm_head()
    rng = np.random.default_rng(0)
    for qi in qs:
        text = QUERIES[qi]
        pos = [k for k in keys if truth.get((qi, k[0], k[1])) == 1][:20]
        rest = [k for k in keys if truth.get((qi, k[0], k[1])) != 1]
        neg = [rest[i] for i in rng.choice(len(rest),
                                           min(nneg, len(rest)),
                                           replace=False)]
        rows, t0 = [], time.time()
        for k, lab in [(k, 1) for k in pos] + [(k, 0) for k in neg]:
            fr = frames_of(*k)
            if len(fr) < 2:
                continue
            rows.append((lab, itm_score(head, m, dev, fr, text)))
        A = np.array(rows, float)
        y, s = A[:, 0].astype(bool), A[:, 1]
        r = np.argsort(np.argsort(s)) + 1
        auc = ((r[y].sum() - y.sum() * (y.sum() + 1) / 2)
               / (y.sum() * (~y).sum()))
        n_pos = int(y.sum())
        K = max(1, int(np.ceil(n_pos * 1.5)))
        top = np.argsort(-s)[:K]
        tru = int(y[top].sum())
        print(f"q{qi:02d} sup {sup[qi]:3d}  sample {len(A):3d} "
              f"({n_pos} true)  AUC {auc:.3f}  "
              f"yield {tru / n_pos:.2f}  prec {tru / len(top):.2f}  "
              f"{(time.time() - t0) / len(A):.1f}s/clip", flush=True)
        print(f"      P(match)  true {s[y].mean():.3f}  "
              f"false {s[~y].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
