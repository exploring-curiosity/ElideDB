"""Deterministic train / val / test splits.

Fixing a real defect: the first trainer computed a holdout and then
discarded it (`tr, _ = sh.split()`), so every episode went into
training and nothing was ever validated in-domain. With no in-domain
validation there is no way to tell learning from memorisation, which
is exactly the question that matters here.

Splits are by EPISODE ID HASH, not by order, so they stay stable as
the generator appends - an episode never migrates between splits, and
a resumed run sees the same partition it started with.

  train  80%   gradient updates
  val    10%   model selection, early stopping, probe fitting
  test   10%   touched once, at the end, never for selection

The robot-arm corpus is not in any of these. It is a separate,
out-of-domain evaluation and it never trains anything.
"""
from __future__ import annotations

import hashlib

TRAIN, VAL, TEST = "train", "val", "test"


def split_of(episode_id: str, seed: str = "relmo-v2") -> str:
    h = hashlib.sha256(f"{seed}/{episode_id}".encode()).digest()
    v = int.from_bytes(h[:4], "big") % 100
    if v < 80:
        return TRAIN
    if v < 90:
        return VAL
    return TEST


def partition(files, dataset=None):
    """files -> {split: [files]}.

    Hashing the episode id is right for corpora WE generate. It is
    wrong for an imported corpus that ships its own splits: Kubric
    MOVi-C/D/E hold out entire OBJECTS AND BACKGROUNDS in their test
    split, which is a stronger generalization test than any id hash
    can express - and re-hashing it would silently mix training
    objects into the held-out number. So when the manifest declares
    native_splits, the manifest wins."""
    out = {TRAIN: [], VAL: [], TEST: []}
    native = {}
    if dataset:
        from relmo import registry as R
        man = R.read_manifest(dataset)
        if man.get("config", {}).get("native_splits"):
            native = {e["id"]: e["split"].replace("validation", VAL)
                      for e in man.get("episodes", [])}
    for f in files:
        out[native.get(f.stem) or split_of(f.stem)].append(f)
    return out
