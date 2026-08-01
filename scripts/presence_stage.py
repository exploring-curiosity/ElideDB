"""FORCED-CHOICE object identification at the boundary - the VLM stage.

Three question forms were measured on the same boundary before this
one was chosen, worst to best:
  pairwise A/B action     826 comparisons, zero movement - at the
                          boundary everything is a pick-place at the
                          same table; the action does not discriminate
  yes/no presence, 640px  separates on average (true +2.72 / false
                          +0.08) but saturates with 10 negatives above
                          the best true - yes-bias survives polarity
  FORCED CHOICE, 640px    "which of these is picked up or moved" over
                          distractors drawn from the other registered
                          queries' noun atoms (query-log derived, not
                          dataset metadata): true +4.16 / false -3.31,
                          q09's eggplants at ranks 3 and 4 of 60 by
                          this margin ALONE
The forced choice kills the yes-bias because saying yes to the target
means saying no to everything else. Boundary is ordered by the choice
margin alone (fusing base back in was measured worse, 3/12 vs 3/4);
outside the boundary the base order stands.

  python scripts/presence_stage.py [--q 0,1,2,7,8,9,10] [--depth 60]
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import queries                                  # noqa: E402

QUERIES = queries()
from elidedb import Store                                    # noqa: E402

JUDGE = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
NF = 4
WIDTH = 640


def main():
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    from elidedb.rerank import _load
    from elidedb.sig2 import atoms_of
    from elidedb.video import FrameSet

    argv = sys.argv
    qsel = [int(x) for x in (argv[argv.index("--q") + 1].split(",")
                             if "--q" in argv
                             else ["0", "1", "2", "7", "8", "9", "10"])]
    depth = int(argv[argv.index("--depth") + 1]) if "--depth" in argv \
        else 60

    d = np.load(ROOT / "ml/teacher_base.npz", allow_pickle=True)
    B, qids = d["B"], [int(q) for q in d["qids"]]
    keys = list(zip([str(s) for s in d["streams"]],
                    [int(v) for v in d["ts"]], [int(v) for v in d["t1"]]))
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    sup = {}
    for (q, _s, _a), v in truth.items():
        sup[q] = sup.get(q, 0) + v

    db = Store.open("lake/bench")
    frames_tbl = db.table("frames").scan()
    model, processor, cfg, yes_ids, no_ids = _load(JUDGE)

    def frames_of(i):
        s, a, b = keys[i]
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        pi = np.linspace(0, len(sel) - 1,
                         min(NF, len(sel))).round().astype(int)
        try:
            dec = FrameSet(db, "frames", sel.take(pi)).decode(width=WIDTH)
        except Exception:
            return []
        return [Image.fromarray(f) for _, f in sorted(dec)]

    # distractors: first noun atom of every registered query - the
    # query log is product input, available in any deployment, and is
    # NOT dataset metadata. CONCRETE atoms only: "a yellow object" and
    # "the banana" can both be correct for one clip, so a generic atom
    # in the menu splits the probability mass with its concrete twin
    # and the margin collapses (measured: q01 0.47 -> 0.35, q02 0.43
    # -> 0.36 with generics in the pool).
    allatoms = []
    for qq in QUERIES[:11]:
        a = atoms_of(qq.lower())[:1]
        if a and a[0] not in allatoms and "object" not in a[0]:
            allatoms.append(a[0])
    tok = processor.tokenizer

    def choice_margin(ims, target, seed=0):
        # GENERIC target ("a yellow object"): identity is the wrong
        # question - the model rightly answers "the banana" and the
        # target loses to its own referent. Ask the PROPERTY instead,
        # with the modifier lifted from the atom itself.
        if "object" in target:
            prop = target.replace("object", "").strip()
            for art in ("a ", "an ", "the "):
                if prop.startswith(art):
                    prop = prop[len(art):]
            prop = prop.strip()
            opts = [f"{prop}", "some other color", "nothing is moved"]
        else:
            # menu ENSEMBLE: near-tied margins flip when the menu
            # changes (q09's trues went ranks 3/4 -> out of top-3 on a
            # distractor swap). Different seeds draw different
            # distractor subsets; the caller averages.
            rng = np.random.default_rng(seed)
            pool = [a for a in allatoms if a != target]
            pick = [pool[i] for i in rng.permutation(len(pool))[:5]]
            opts = [target] + pick + ["none of these"]
        letters = "ABCDEFGHIJ"[:len(opts)]
        menu = "  ".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        lids = {letters[i]: sorted({tok.encode(x)[0] for x in
                                    (letters[i], " " + letters[i])})
                for i in range(len(opts))}
        if "object" in target:
            q = (f"These are {len(ims)} frames from one robot "
                 f"manipulation clip. What color is the object the "
                 f"robot picks up or moves? {menu}. "
                 f"Answer with the letter only.")
        else:
            q = (f"These are {len(ims)} frames from one robot "
                 f"manipulation clip. Which of these is picked up or "
                 f"moved by the robot in this clip? {menu}. "
                 f"Answer with the letter only.")
        prompt = apply_chat_template(processor, cfg, q, num_images=len(ims))
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for k, im in enumerate(ims):
                p = f"{td}/f{k}.jpg"
                im.save(p, "JPEG", quality=92)
                paths.append(p)
            r = generate(model, processor, prompt, image=paths,
                         max_tokens=1, verbose=False)
        if r.logprobs is None:
            return 0.0
        lg = mx.array(r.logprobs).reshape(-1)
        sc = {L: max(float(lg[i]) for i in ids) for L, ids in lids.items()}
        return sc["A"] - max(v for L, v in sc.items() if L != "A")

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    results = {}
    margins_out = {}
    for j, qi in enumerate(qids):
        if qi not in qsel or sup.get(qi, 0) == 0:
            continue
        text = QUERIES[qi]
        atoms = atoms_of(text.lower())[:2] or [text]
        lab = np.array([1 if truth.get((qi, s, a)) == 1 else 0
                        for s, a, _ in keys])
        base = B[:, j].astype(float)
        order = list(np.argsort(-base))
        K = int(np.ceil(sup[qi] * 1.5))
        band = order[:max(depth, K)]

        t0 = time.time()
        target = atoms[0]
        marg = np.zeros(len(band))
        for bi, i in enumerate(band):
            ims = frames_of(i)
            if not ims:
                marg[bi] = -10.0
                continue
            marg[bi] = float(np.mean(
                [choice_margin(ims, target, seed=sd) for sd in (0, 1, 2)]
            )) if "object" not in target else choice_margin(ims, target)
        # INFORMATIVENESS GATE, self-recognized: if (almost) the whole
        # band answers the same way, the question measured nothing
        # here - the base is already saturated on this attribute (q01's
        # band is all yellow movers because the base put them there),
        # and reordering by an uninformative margin is pure noise
        # (measured: q01 0.47 -> 0.35). Only reorder when the band
        # actually splits.
        frac = float((marg > 0).mean())
        if 0.15 <= frac <= 0.85:
            band_sorted = [band[i] for i in np.argsort(-marg)]
        else:
            band_sorted = list(band)
        new_order = band_sorted + order[len(band):]

        def yld(o):
            return int(lab[o[:K]].sum()) / sup[qi]

        y0, y1 = yld(order), yld(new_order)
        results[qi] = (y0, y1, K, int(lab[new_order[:K]].sum()))
        margins_out[str(qi)] = {"band": [int(i) for i in band],
                                "marg": [float(v) for v in marg]}
        print(f"q{qi:02d} sup {sup[qi]:3d} K {K:3d} atoms {atoms}  "
              f"yield {y0:.2f} -> {y1:.2f}  "
              f"({(time.time() - t0) / len(band):.1f}s/clip)", flush=True)

    (ROOT / "ml/presence_margins.json").write_text(json.dumps(margins_out))
    if results:
        print(f"\nmean yield on staged queries: "
              f"{np.mean([v[0] for v in results.values()]):.2f} -> "
              f"{np.mean([v[1] for v in results.values()]):.2f}")


if __name__ == "__main__":
    main()
