"""Are SigLIP2's PATCH tokens text-aligned? Decide before ingesting.

Pooled image embeddings answer "what is this scene", which is why
every channel scores near zero on queries that name a small object:
one 1152-d vector for a 640x480 frame cannot both describe the scene
and assert that an eggplant is in it. Cutting the object out and
embedding it separately was measured dead at corpus scale (crops lose
the context that identifies the thing).

Late interaction takes the third option: keep the PATCH grid from the
same single full-frame forward pass and score MaxSim per query token
(ColBERT's operator; Video-ColBERT arXiv 2503.19009 applies it to
text-to-video). Nothing is cut, nothing is detected, context is intact.

The premise it rests on: a patch token must live in the TEXT space.
SigLIP pools with an attention (MAP) head, so raw patch tokens do not
- they are pre-head features. This projects each patch through the
head's own value/output path individually, which is what the head
would compute if that patch were the only one attended (the MaskCLIP
construction). Whether that lands in text space is an empirical
question, so it is asked here, on one frame, before anything is built:

  does the best-scoring patch for "an eggplant" sit somewhere
  DIFFERENT from the best patch for "a banana", and does the frame's
  patch-max separate a frame that has the object from one that does
  not?

  python scripts/probe_patch_align.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402


def patch_vectors(model, proc, dev, images):
    """(n, patches, dim) patch tokens pushed through the MAP head's
    value path, so each is a stand-alone image embedding."""
    import torch
    with torch.no_grad():
        px = proc(images=images, return_tensors="pt").to(dev)
        vm = model.vision_model
        out = vm(**px, output_hidden_states=False)
        H = out.last_hidden_state                    # (n, P, D)
        head = vm.head                               # MAP head
        # value/output projection of the pooling attention, applied
        # per patch instead of to the attention-weighted sum
        attn = head.attention
        D = H.shape[-1]
        Wv, bv = attn.in_proj_weight[2 * D:], attn.in_proj_bias[2 * D:]
        V = H @ Wv.T + bv
        V = attn.out_proj(V)
        # the head's residual MLP, same as the pooled path
        V = V + head.mlp(head.layernorm(V))
        V = V / V.norm(dim=-1, keepdim=True)
    return V.float().cpu().numpy()


def main():
    import torch
    from PIL import Image

    from elidedb.device import pick
    from elidedb.sig2 import MID, _text_vec
    from elidedb.video import FrameSet
    from transformers import AutoModel, AutoProcessor

    dev, dtype = pick()
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=dtype,
                                      low_cpu_mem_usage=True).to(dev).eval()

    db = Store.open("lake/bench")
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    true_of = {}
    for q, s, a, v in zip(t["query_id"], t["stream"], t["t0"], t["true"]):
        if int(v) == 1:
            true_of.setdefault(int(q), []).append((s, int(a)))

    frames_tbl = db.table("frames").scan()

    def frames_for(s, a, n=4):
        ep = db.table("episodes").scan()
        m = ep.filter(pc.and_(pc.equal(ep.column("stream"), s),
                              pc.equal(ep.column("ts"), a)))
        b = int(m.column("t1")[0].as_py())
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        pick_i = np.linspace(0, len(sel) - 1, n).round().astype(int)
        dec = FrameSet(db, "frames", sel.take(pick_i)).decode()
        return [Image.fromarray(f) for _, f in sorted(dec)]

    egg = true_of[9][0]        # an episode that contains an eggplant
    ban = true_of[10][0]       # one that contains a banana
    qs = ["an eggplant", "a banana", "a drawer"]
    QT = np.stack([_text_vec(q) for q in qs])

    print(f"{'frame source':>18} " + " ".join(f"{q:>14}" for q in qs))
    for name, key in (("eggplant ep", egg), ("banana ep", ban)):
        ims = frames_for(*key)
        P = patch_vectors(model, proc, dev, ims)        # (n, P, D)
        S = P @ QT.T                                    # (n, P, q)
        pooled_max = S.max(axis=1).max(axis=0)          # best patch, any frame
        g = int(np.sqrt(P.shape[1]))
        loc = [np.unravel_index(int(S[:, :, j].max(0).argmax()), (g, g))
               for j in range(len(qs))]
        print(f"{name + ' patchmax':>18} " +
              " ".join(f"{v:>14.3f}" for v in pooled_max))
        print(f"{'  best patch (r,c)':>18} " +
              " ".join(f"{str(l):>14}" for l in loc))
        # the pooled embedding, for reference: this is what ships today
        with torch.no_grad():
            px = proc(images=ims, return_tensors="pt").to(dev)
            F = model.get_image_features(**px)
            F = (F / F.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        print(f"{name + ' POOLED':>18} " +
              " ".join(f"{v:>14.3f}" for v in (F @ QT.T).max(0)))


if __name__ == "__main__":
    main()
