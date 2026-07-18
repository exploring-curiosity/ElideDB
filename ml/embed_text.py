#!/usr/bin/env python3
"""Embed one text query with SigLIP and write raw float32 to stdout.

This is the C++ <-> sidecar text-query contract (see meta.json's
embed_text_cmd): argv[-1] is the query text, stdout is exactly `dim` little-
endian float32 values, L2-normalized. Nothing else may be printed to stdout.
"""
import argparse
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/siglip-so400m-patch14-384")
    ap.add_argument("text")
    args = ap.parse_args()

    from mlx_embeddings.utils import load
    import mlx.core as mx  # noqa: F401

    model, processor = load(args.model)
    # SigLIP was trained with fixed-length padded text; mirror that here.
    inputs = processor(text=[args.text], padding="max_length",
                       max_length=64, truncation=True, return_tensors="np")
    out = model.get_text_features(mx.array(inputs["input_ids"]))
    emb = np.array(out, dtype=np.float32)[0]
    emb /= np.linalg.norm(emb)
    sys.stdout.buffer.write(emb.tobytes())


if __name__ == "__main__":
    main()
