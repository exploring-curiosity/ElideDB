"""Text embedding for episodic recall.

MiniLM (all-MiniLM-L6-v2, 384-d, ~90 MB) over the robot's own event text. This
is deliberately the most boring retrieval available: text-over-text with a
sentence encoder is the part of this system least likely to embarrass anyone on
stage, which is exactly why the agent's decisions hang off it rather than off
video similarity.

Loaded once, lazily, and announced — a 90 MB lazy load inside the first tick
would look like a hang.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import CFG

log = logging.getLogger("brigade.embed")

_model = None
_lock = threading.Lock()


def load(warm: bool = True):
    """Load the sentence encoder. Safe to call repeatedly."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer

        name = CFG.memory.text_model
        log.info("loading text encoder %s (first run downloads ~90MB)", name)
        model = SentenceTransformer(name)
        if warm:
            model.encode(["warmup"], normalize_embeddings=True)
        _model = model
        log.info("text encoder ready, dim=%d", model.get_sentence_embedding_dimension())
        return _model


def is_loaded() -> bool:
    return _model is not None


def encode(texts: str | list[str]) -> np.ndarray:
    """Embed text(s) to unit-norm float32. Returns (n, 384).

    Normalised at the source so every downstream comparison is a cosine and
    nobody has to remember to normalise again.
    """
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    if not batch:
        return np.zeros((0, CFG.memory.text_dim), dtype=np.float32)
    model = load()
    # No progress bar: this runs once per event write and once per recall, and a
    # tqdm bar per single-item encode is pure terminal noise.
    vecs = model.encode(
        batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.shape[1] != CFG.memory.text_dim:
        raise ValueError(
            f"{CFG.memory.text_model} returned dim {vecs.shape[1]}, "
            f"but the schema declares VECTOR({CFG.memory.text_dim})"
        )
    return vecs
