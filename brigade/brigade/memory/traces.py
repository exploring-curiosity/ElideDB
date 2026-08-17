"""The head's view of a clip: RelMo's trace, summarised without losing time.

The 512-d column in the database is a MEAN over the descriptor trace. It has to
be — it is the stage-1 prefilter, and a prefilter is one vector per row by
definition. But a mean over time cannot tell apart two behaviours that visit the
same pixels in a different order, and in a kitchen where every behaviour shares
one room and one camera that is most of what there is to tell apart.

MEASURED (bench/relmo_probe.py and bench/trace_feature.py; 59 spans, 10
behaviours, chance 0.100, nearest-neighbour with every overlapping span barred
so the question is "is a DIFFERENT performance of this behaviour found"):

    pooled mean, RoboCasa basis                     0.300
    pooled mean, no whitening                       0.475
    full trace, RelMo's DTW  (stage 2)              0.576
    per-channel temporal STD, both channels         0.661
    per-channel temporal STD, SigLIP2 only          0.712   <- what is used
    net direction (last - first)                    0.085
    all six blocks concatenated                     0.136

Three things fall out of that table and all three shaped this file.

THE SPREAD, NOT THE MEAN. The mean says what the kitchen looks like, and the
answer is the same for all ten behaviours because it is the same kitchen. The
per-channel standard deviation says how much each feature moved while the clip
ran, which is what actually differs. This is the same shape of result as the
delta-appearance channel elsewhere in this codebase.

FUSION DESTROYS IT. Cosine over concatenated unit blocks is the MEAN of the
per-block cosines, so five uninformative blocks drown one good one: 0.712 alone,
0.136 with everything attached. Selecting the best single view beats fusing.

DIRECTION IS NOT AVAILABLE HERE. `last - first` scores below chance. A 15 s span
holds several seconds of approach and retreat, and the net displacement of a
descriptor over that is nearly zero plus noise. It is kept in the code, unused,
because "we tried it and it was 0.085" is worth more than silence.

Nothing here is fitted. Three reductions of an array and an L2 — no per-corpus
parameter, no label, nothing that has to be re-learned for a different kitchen.
"""

from __future__ import annotations

import os

import numpy as np

from ..config import CFG

TRACES = os.path.join(CFG.artifacts, "traces")
BASIS = os.path.join(CFG.artifacts, "basis.npz")

# Which reductions the head reads, chosen by the table above. `("std_sig",)` is
# the measured best; the others exist so the choice can be re-measured rather
# than trusted, and `bench/trace_feature.py` re-runs that comparison.
BLOCKS = ("std_sig",)
_WIDTH = dict(mean_fix=1024, std_fix=1024, dir_fix=1024,
              mean_sig=768, std_sig=768, dir_sig=768)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def blocks_of(fix: np.ndarray, sig: np.ndarray) -> dict[str, np.ndarray]:
    """The trace, reduced. Each block L2'd on its own.

    Normalising per block is not tidiness: concatenated raw, the mean block's
    norm dwarfs the rest and the similarity becomes the mean's similarity with
    noise attached. Same discipline RelMo applies to its own two channels.
    """
    out = {}
    for name, x in (("fix", np.atleast_2d(np.asarray(fix, np.float32))),
                    ("sig", np.atleast_2d(np.asarray(sig, np.float32)))):
        out[f"mean_{name}"] = _unit(x.mean(0))
        out[f"std_{name}"] = _unit(x.std(0))
        out[f"dir_{name}"] = _unit(x[-1] - x[0])
    return out


def feature(clip_id: str, root: str | None = None,
            blocks: tuple[str, ...] = BLOCKS) -> np.ndarray | None:
    """clip_id -> the head's input vector, or None if the trace is missing."""
    p = os.path.join(root or TRACES, f"{clip_id}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    b = blocks_of(z["fix"], z["sig"])
    v = np.concatenate([b[k] for k in blocks]) / np.sqrt(len(blocks))
    return v.astype(np.float32)


def features(clip_ids: list[str], root: str | None = None,
             blocks: tuple[str, ...] = BLOCKS) -> tuple[np.ndarray, list[str]]:
    """-> (N, D) and the ids that actually had a trace, in the same order."""
    out, kept = [], []
    for c in clip_ids:
        f = feature(c, root, blocks)
        if f is not None:
            out.append(f)
            kept.append(c)
    if not out:
        return np.zeros((0, dim(blocks)), np.float32), []
    return np.stack(out), kept


def dim(blocks: tuple[str, ...] = BLOCKS) -> int:
    return sum(_WIDTH[b] for b in blocks)
