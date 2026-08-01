"""Two pilots on ground-truth close/open episode clips.

A. NumPro (arXiv 2411.10332): stamp frame numbers on the images — the VLM
   reads them by OCR and gains temporal order. Our measured deficit is
   exactly order-blindness (reversal-contrast AUC 0.14). Does numbering
   fix the absolute direction question?
B. ViLL-E-style (arXiv 2604.12148): use the VLM as an EMBEDDING model —
   the next-token DISTRIBUTION under an action prompt, one forward pass,
   no generation loop. Clip side sees before/after frames; query side is
   text-only through the same model. Retrieval = cosine. Does this space
   separate close from open where SigLIP (cos 0.951) cannot?
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.rerank import DEEP_VLM, _load, as_change_question  # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402


def load_clips(db, n_per=8):
    from PIL import Image
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = {"close": [], "open": []}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        k = (k or "").lower()
        if "drawer" not in k:
            continue
        w = ("close" if ("close" in k and "put" not in k) else
             "open" if ("open" in k and "put" not in k) else None)
        if w and len(eps[w]) < n_per:
            eps[w].append((stream_of.get(int(i)), int(a), int(b)))
    frames_tbl = db.table("frames").scan()
    clips, labels = [], []
    for kind, lst in eps.items():
        for s, a, b in lst:
            sel = frames_tbl.filter(pc.and_(
                pc.equal(frames_tbl.column("stream"), s),
                pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                        pc.less_equal(frames_tbl.column("ts"), b))))
            if len(sel) < 2:
                continue
            dec = FrameSet(db, "frames", sel.take(
                np.array([0, len(sel) - 1]))).decode(width=448)
            if len(dec) < 2:
                continue
            clips.append([Image.fromarray(d[1]) for d in sorted(dec)])
            labels.append(kind)
    return clips, np.array(labels)


def stamp(im, num):
    """NumPro overlay: large red number, bottom-right (the paper's best
    position/color/size after their design search)."""
    from PIL import ImageDraw, ImageFont
    im = im.copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
    except Exception:
        font = ImageFont.load_default()
    d.text((im.width - 70, im.height - 75), str(num), fill=(255, 0, 0),
           font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return im


def score(clips, question, model_id):
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg, yes_ids, no_ids = _load(model_id)
    tmp = Path(tempfile.mkdtemp(prefix="pilot_nv_"))
    out = []
    for ci, frames in enumerate(clips):
        prompt = apply_chat_template(processor, cfg, question,
                                     num_images=len(frames))
        paths = []
        for j, im in enumerate(frames):
            fp = tmp / f"c{ci}_{j}.jpg"
            im.save(fp, "JPEG", quality=85)
            paths.append(str(fp))
        r = generate(model, processor, prompt, image=paths, max_tokens=1,
                     verbose=False)
        a = mx.array(r.logprobs).reshape(-1)
        y = max(float(a[i]) for i in yes_ids)
        n = max(float(a[i]) for i in no_ids)
        out.append(y - n)
    return np.array(out)


def vocab_vec(question, paths, model_id):
    """Next-token log-distribution — the one-forward 'caption in
    distribution space'. Returns softmax probs (sparse-ish, cosine-able)."""
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg, _, _ = _load(model_id)
    prompt = apply_chat_template(processor, cfg, question,
                                 num_images=len(paths))
    r = generate(model, processor, prompt, image=paths or None,
                 max_tokens=1, verbose=False)
    a = mx.array(r.logprobs).reshape(-1)
    p = mx.softmax(a)
    v = np.array(p, np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def auc(pos, neg):
    return float(np.mean([[p > n for n in neg] for p in pos]))


def main():
    db = Store.open("lake/bridge4h")
    clips, labels = load_clips(db)
    print(f"{(labels == 'close').sum()} close + "
          f"{(labels == 'open').sum()} open clips")
    Qc = as_change_question("closing the drawer")
    Qo = as_change_question("opening the drawer")

    # ---- A. NumPro ---------------------------------------------------------
    numbered = [[stamp(c[0], 1), stamp(c[1], 2)] for c in clips]
    for model_id, tag in [(None, "2B"), (DEEP_VLM, "7B")]:
        mid = model_id or "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        mc = score(numbered, Qc, mid)
        mo = score(numbered, Qo, mid)
        a_abs = auc(mc[labels == "close"], mc[labels == "open"])
        a_con = auc((mc - mo)[labels == "close"], (mc - mo)[labels == "open"])
        print(f"NumPro {tag}: absolute AUC {a_abs:.2f} (was 0.36) | "
              f"swap-contrast AUC {a_con:.2f} (was "
              f"{'0.86' if tag == '2B' else '0.91'})", flush=True)

    # ---- B. ViLL-E-style vocabulary embedding ------------------------------
    CLIP_Q = ("The first image is the start of a short robot clip and the "
              "second is the end. In one word, the action that happened "
              "between them is:")
    TEXT_Q = ("A short robot clip is described as: {}. In one word, the "
              "action in that clip is:")
    tmp = Path(tempfile.mkdtemp(prefix="pilot_ve_"))
    for model_id, tag in [("mlx-community/Qwen2-VL-2B-Instruct-4bit", "2B"),
                          (DEEP_VLM, "7B")]:
        cv = []
        for ci, c in enumerate(clips):
            paths = []
            for j, im in enumerate(c):
                fp = tmp / f"{tag}_{ci}_{j}.jpg"
                im.save(fp, "JPEG", quality=85)
                paths.append(str(fp))
            cv.append(vocab_vec(CLIP_Q, paths, model_id))
        cv = np.stack(cv)
        qc = vocab_vec(TEXT_Q.format("closing the drawer"), [], model_id)
        qo = vocab_vec(TEXT_Q.format("opening the drawer"), [], model_id)
        sc, so = cv @ qc, cv @ qo
        print(f"ViLL-E-style {tag}: "
              f"close-query AUC {auc(sc[labels == 'close'], sc[labels == 'open']):.2f} | "
              f"open-query AUC {auc(so[labels == 'open'], so[labels == 'close']):.2f} | "
              f"contrast close-query AUC "
              f"{auc((sc - so)[labels == 'close'], (sc - so)[labels == 'open']):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
