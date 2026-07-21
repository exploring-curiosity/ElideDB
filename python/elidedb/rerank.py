"""Relational reranking with a vision-language model.

WHY THIS EXISTS
---------------
SigLIP-style embeddings encode *what is present* in a frame, not *how things
relate*. "two people working together on a laptop" and "two people standing
near a laptop" land in almost the same place in embedding space, which is why
appearance-only search returns every clip containing two people.

A VLM actually reads the pixels and answers a grounded question about the
relation. It is ~1000x more expensive per window, so it is used exactly as a
database uses an expensive operator: LAST, on a small candidate set that
cheap operators already pruned.

    ts/stream predicates  ->  ANN/IVF shortlist  ->  exact cosine  ->  VLM
    (log zone maps)           (~40-80 vectors)      (rank)            (top N)

SCORING
-------
Generation is not used for scoring: small instruct VLMs are heavily
yes-biased and answer "Yes" to nearly any yes/no question (measured — both a
true and a false frame generated "Yes"). Instead we take ONE forward pass and
read the next-token distribution:

    score = max logP("Yes"|image,question) - max logP("No"|image,question)

That is a calibrated, continuous margin: positive leans yes, negative leans
no, and the magnitude is comparable across candidates. Measured on a known
true/false pair from the lab capture: TRUE +1.09 vs FALSE +0.53.

The final ordering fuses the retrieval score with the relational margin, so
a candidate must be both visually similar AND relationally correct.
"""
from __future__ import annotations

import re

import numpy as np

DEFAULT_VLM = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
_VLM_CACHE: dict = {}


def _load(model_id: str):
    if model_id not in _VLM_CACHE:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, processor = load(model_id)
        cfg = load_config(model_id)
        tok = processor.tokenizer
        yes = sorted({tok.encode(s)[0] for s in ("Yes", "yes", " Yes")})
        no = sorted({tok.encode(s)[0] for s in ("No", "no", " No")})
        _VLM_CACHE[model_id] = (model, processor, cfg, yes, no)
    return _VLM_CACHE[model_id]


def as_question(query: str) -> str:
    """Turn a retrieval phrase into a grounded yes/no question.

    The compositional operators are prose here: the VLM reasons over the
    whole sentence, which is precisely the capability embeddings lack."""
    q = query.strip()
    q = re.sub(r"\s+AND\s+", " and ", q, flags=re.I)
    q = re.sub(r"\s+NOT\s+", " but no ", q, flags=re.I)
    q = re.sub(r"(^|\s)-(\w)", r"\1no \2", q)
    if not q:
        return "Is anything notable happening? Answer yes or no."
    return (f"Does this image show: {q}? "
            "Answer only yes or no.")


def score_images(images, question: str, model_id: str = DEFAULT_VLM):
    """P(yes) - P(no) margin per image, in one forward pass each."""
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg, yes_ids, no_ids = _load(model_id)
    prompt = apply_chat_template(processor, cfg, question, num_images=1)
    out = []
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.gettempdir()) / "_elidedb_rerank.jpg"
    for im in images:
        im.save(tmp, "JPEG", quality=88)
        r = generate(model, processor, prompt, image=[str(tmp)],
                     max_tokens=1, verbose=False)
        if r.logprobs is None:
            out.append(0.0)
            continue
        a = mx.array(r.logprobs).reshape(-1)
        y = max(float(a[i]) for i in yes_ids)
        n = max(float(a[i]) for i in no_ids)
        out.append(y - n)
    return out


def rerank_hits(store, hits, query, top_n: int = 12, alpha: float = 0.7,
                model_id: str = DEFAULT_VLM, width: int = 512):
    """Re-order retrieval hits by relational correctness.

    Only the first `top_n` hits are examined (the expensive operator runs
    last, on a pruned set). `alpha` weights the VLM margin against the
    retrieval score; both are rank-normalised so they are commensurable.
    Returns (hits, info) with `vlm` and `fused` attached to each reranked hit.
    """
    from PIL import Image
    if not hits:
        return hits, {"reranked": 0}
    head, tail = hits[:top_n], hits[top_n:]
    rot = store.meta.get("display", {}).get("rotate", 0)

    frames, keep = [], []
    for h in head:
        mid = (h["t0"] + h["t1"]) // 2
        w, _ = store.window(mid - 500_000_000, mid + 500_000_000,
                            tables=["frames"])
        fs = w.get("frames")
        dec = fs.decode(stream=h["stream"], width=width, limit=1) if fs else []
        if not dec:
            continue
        im = Image.fromarray(dec[0][1])
        if rot:
            im = im.rotate(rot, expand=True)
        frames.append(im)
        keep.append(h)
    if not frames:
        return hits, {"reranked": 0, "note": "no frames decodable"}

    question = as_question(query)
    margins = score_images(frames, question, model_id)

    # rank-normalise both signals to [0,1] so alpha is meaningful
    def ranknorm(v):
        v = np.asarray(v, dtype=float)
        if len(v) < 2:
            return np.ones_like(v)
        r = v.argsort().argsort().astype(float)
        return r / (len(v) - 1)
    rn_vlm = ranknorm(margins)
    rn_ret = ranknorm([h["score"] for h in keep])
    for h, m, fv, fr in zip(keep, margins, rn_vlm, rn_ret):
        h["vlm"] = round(float(m), 3)
        h["fused"] = round(float(alpha * fv + (1 - alpha) * fr), 4)
    keep.sort(key=lambda h: -h["fused"])
    return keep + tail, {
        "reranked": len(keep), "question": question, "model": model_id,
        "vlm_min": round(float(min(margins)), 3),
        "vlm_max": round(float(max(margins)), 3)}
