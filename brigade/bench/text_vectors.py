#!/usr/bin/env python3
"""Cache SigLIP2 text vectors for the requests the head trains on.

    myenv/bin/python brigade/bench/text_vectors.py

Runs in RelMo's interpreter because SigLIP2's text tower needs transformers
4.57 while pi0.5 pins 4.53.2. Nothing here reaches the database — these are
inputs to a training run, and at serve the same tower embeds whatever the human
says, live, and throws it away.
"""
import json, os, sys
import numpy as np

TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "eval_logs", "reasoner")

def main() -> int:
    import torch
    from transformers import AutoModel, AutoProcessor

    rows = json.load(open(os.path.join(TRAIN, "segments.json")))
    texts = sorted({r["instruction"] for r in rows})
    # Paraphrases the demo will actually be asked, embedded now so the head can
    # be evaluated on them without the tower being needed at train time.
    texts += ["put the bowl away", "put the bottle away", "tidy up the bowl",
              "make the stove ready", "put the cream cheese away",
              "put it back", "and the bottle too", "now get the stove going"]
    texts = sorted(set(texts))

    mid = "google/siglip2-base-patch16-224"
    m = AutoModel.from_pretrained(mid, dtype=torch.float32).eval()
    pr = AutoProcessor.from_pretrained(mid)
    with torch.no_grad():
        t = pr(text=texts, padding="max_length", max_length=64, return_tensors="pt")
        e = m.get_text_features(**t).float().numpy()
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    np.savez(os.path.join(TRAIN, "text_vectors.npz"),
             texts=np.array(texts, dtype=object), vecs=e.astype(np.float32))
    print(f"cached {len(texts)} text vectors, dim {e.shape[1]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
