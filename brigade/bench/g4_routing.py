#!/usr/bin/env python3
"""G4 — can a NOUN find the right clips, with no text stored anywhere?

    myenv/bin/python brigade/bench/g4_routing.py

The one text path this project's own research sanctions
(`native/relmo/RELMO.md:1405`): SigLIP 2's text tower already shares a space
with the `sig` channel, and `sig` is one of the three inputs to RelMo's
embedding. So a noun can be projected into the store's geometry with no new
corpus and, critically, **without a single word being written to the database**.
The word exists only for the microseconds it takes to embed it.

What is NOT expected to work, measured previously and recorded there: verbs and
direction. "Open the cabinet doors" and "Close the cabinet doors" sit at cosine
0.9974 in that tower — the tower erases the very distinction the robot cares
about. So this gate asks only what the research says is answerable: does the
noun route to clips containing that object?

Scoring uses the recording ids, which encode which episode is which. The system
never sees them; they are the harness's answer key, like the labels in G1.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "native")))

import numpy as np

G1 = "../eval_logs/g1"
BASIS = "rcasa"

# Which clip is about which object. Harness-side ground truth; never shown.
SUBJECT_OF = {"g4": "bowl", "g1": "bowl", "g8": "bowl",
              "g9": "wine bottle", "g6": "cream cheese", "g2": "wine bottle"}
QUERIES = ["a black bowl", "a wine bottle", "a box of cream cheese"]
# which clips SHOULD come back for each query, by subject
WANT = {"a black bowl": "bowl", "a wine bottle": "wine bottle",
        "a box of cream cheese": "cream cheese"}


def main() -> int:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    import imageio.v3 as iio

    clips = sorted(p for p in glob.glob(os.path.join(G1, "g*_0.mp4"))
                   if ".frontview." not in p and ".sideview." not in p)
    if not clips:
        print(f"no clips in {G1}; run g1_readers.py --record first")
        return 1

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mid = "google/siglip2-base-patch16-224"
    print(f"loading {mid} on {dev} ...", flush=True)
    model = AutoModel.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
    proc = AutoProcessor.from_pretrained(mid)

    # ---- the clips, through the IMAGE tower (this is the `sig` channel) ----
    vecs, names = [], []
    for c in clips:
        v = iio.imread(c, plugin="pyav")
        idx = np.linspace(0, len(v) - 1, 8).astype(int)
        imgs = [Image.fromarray(np.asarray(v[i])) for i in idx]
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt").to(dev)
            e = model.get_image_features(**x).float().cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        vecs.append(e.mean(0))
        names.append(os.path.basename(c).split("_")[0])
    V = np.stack(vecs)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    # ---- the nouns, through the TEXT tower ----
    print(f"\n{len(clips)} clips, {len(QUERIES)} noun queries, nothing stored\n"
          + "=" * 70)
    hits = 0
    rows = []
    for q in QUERIES:
        with torch.no_grad():
            t = proc(text=[q], padding="max_length", max_length=64,
                     return_tensors="pt").to(dev)
            e = model.get_text_features(**t).float().cpu().numpy()[0]
        e = e / (np.linalg.norm(e) + 1e-9)
        sims = V @ e
        order = np.argsort(-sims)
        want = WANT[q]
        top = [names[i] for i in order]
        top_subj = [SUBJECT_OF[n] for n in top]
        n_want = sum(1 for s in SUBJECT_OF.values() if s == want)
        k = n_want
        got = sum(1 for s in top_subj[:k] if s == want)
        hits += got == n_want
        print(f'\n  "{q}"   (should return {n_want} clip(s) of the {want})')
        for i in order:
            mark = "<-- want" if SUBJECT_OF[names[i]] == want else ""
            print(f"     {sims[i]:+.4f}  {names[i]}  {SUBJECT_OF[names[i]]:<14}{mark}")
        print(f"     precision@{k}: {got}/{n_want}")
        rows.append(dict(query=q, want=want, support=n_want, got=got,
                         ranking=[[names[i], float(sims[i])] for i in order]))

    print("\n" + "=" * 70)
    print(f"G4: {hits}/{len(QUERIES)} nouns routed to the right clips at "
          f"precision@support = 1.0")
    print("(chance for a 2-of-6 query is 0.07; for 1-of-6 it is 0.17)")
    json.dump(rows, open(os.path.join(G1, "g4.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
