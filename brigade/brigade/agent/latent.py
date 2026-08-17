"""The reasoning layer: retrieved clips in, a command for pi0.5 out. No words.

This is the answer to the constraint that shaped the whole architecture. Nothing
textual may be stored, and the reasoning layer may not be generative — but pi0.5
acts only on a language prefix. The way through is that pi0.5's prefix is a
TENSOR, not a string:

    lang_emb = paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)                       # [images..., language]

Nothing requires that tensor to have come from a tokenizer. So the memory layer
emits it directly and no vocabulary exists anywhere at serve time.

WHY THIS IS BUILDABLE, measured before it was written (bench/a_latent.py,
scratchpad interpolation run, three episodes per arm on libero_goal/4):

    baseline                     3/3
    lang_emb + noise @ 0.05 std  3/3
    lang_emb + noise @ 0.50 std  3/3      <- half a std costs nothing
    WRONG instruction embedded   0/3      <- the control: the prefix does steer

    blend 1.0 / 0.9 / 0.7 / 0.5  3/3      <- follows the majority component
    blend 0.3                    0/3

Two facts fall out, and together they specify the head. The policy tolerates
large *unstructured* error, and it follows whichever real instruction holds more
than half the weight — while still distinguishing two instructions whose
embeddings sit at cosine 0.9727. The direction that matters is small and
specific; everything else is slack. So the head does not need to regress
409,600 numbers accurately. It needs to put **more than half its weight on the
right prototype**, which is an argmax problem with a wide margin.

WHAT IS AND IS NOT STORED. The prototype bank holds instruction embeddings
computed once during training and frozen into a model asset. Instructions are
therefore used at TRAINING time and never at serve, and the database holds clips
and RelMo vectors only — no text, no labels, no state. That is the same status
as any pretrained checkpoint: a model that has seen language is not a memory
that stores it.

MEMORY IS LOAD-BEARING BY CONSTRUCTION. The head's only view of what to do comes
from the RelMo vectors of retrieved clips. With memory off, retrieval returns
nothing, the head has no input, and no command exists — the robot cannot act at
all rather than acting badly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("brigade.latent")


@dataclass
class Command:
    """What the memory layer hands the robot."""
    lang_emb: torch.Tensor        # (1, T, D) — the prefix pi0.5 consumes
    weights: np.ndarray           # mixing weights over prototypes, for the UI
    margin: float                 # top weight minus runner-up
    evidence: list                # recording ids the command was built from

    @property
    def confident(self) -> bool:
        """Above the measured cliff. Blends at 0.5 succeed and 0.3 fail, so a
        top weight under half is not a command — it is a question, and the
        agent should ask rather than guess."""
        return bool(self.weights.max() > 0.5)


class PrototypeBank(nn.Module):
    """Frozen instruction embeddings. Built once at training, never at serve.

    Stored as a model asset rather than in the database, which is the whole
    point: the memory holds video, the model holds whatever it learned.
    """

    def __init__(self, embs: torch.Tensor):
        super().__init__()
        # (K, T, D) — registered as a buffer so it moves with the module and is
        # saved with it, but is never a gradient target.
        self.register_buffer("bank", embs)

    @property
    def k(self) -> int:
        return int(self.bank.shape[0])

    def blend(self, weights: torch.Tensor) -> torch.Tensor:
        """(K,) weights -> (1, T, D). The command."""
        w = weights.reshape(-1, 1, 1).to(self.bank.dtype).to(self.bank.device)
        return (self.bank * w).sum(0, keepdim=True)


class InstructionHead(nn.Module):
    """RelMo vectors of retrieved clips (+ the live request) -> prototype weights.

    Deliberately tiny and deliberately not generative. It emits K numbers. The
    command is a weighted sum of frozen prototypes, so the output space is
    exactly the span of instructions the robot has been trained on and cannot
    drift into nonsense the way a decoder can.
    """

    def __init__(self, k: int, clip_dim: int = 512, req_dim: int = 768,
                 hidden: int = 256, use_request: bool = True):
        super().__init__()
        self.use_request = use_request
        d_in = clip_dim + (req_dim if use_request else 0)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, k),
        )

    def forward(self, clip_vec: torch.Tensor, req_vec: torch.Tensor | None = None
                ) -> torch.Tensor:
        x = clip_vec
        if self.use_request:
            if req_vec is None:
                raise ValueError("head was built with use_request=True")
            x = torch.cat([clip_vec, req_vec], dim=-1)
        return self.net(x)                     # logits over prototypes


class Reasoner(nn.Module):
    """Bank + head. The whole reasoning layer.

    Pooling across retrieved clips is a mean of the per-clip logits, not of the
    vectors: RelMo returns several spans of varying relevance and averaging
    their *opinions* lets a single strong match outvote two weak ones, which
    averaging the inputs would not.
    """

    def __init__(self, bank: PrototypeBank, head: InstructionHead):
        super().__init__()
        self.bank, self.head = bank, head

    def command(self, clip_vecs: np.ndarray, req_vec: np.ndarray | None = None,
                evidence: list | None = None, temperature: float = 1.0) -> Command | None:
        """Retrieved clips -> a Command, or None when memory returned nothing.

        Returning None is the ablation working, not an error path: with no
        clips there is no command, so the robot has nothing to execute.
        """
        if clip_vecs is None or len(clip_vecs) == 0:
            return None
        dev = self.bank.bank.device
        c = torch.as_tensor(np.atleast_2d(clip_vecs), dtype=torch.float32, device=dev)
        r = None
        if self.head.use_request:
            r = torch.as_tensor(np.atleast_2d(req_vec), dtype=torch.float32, device=dev)
            r = r.expand(c.shape[0], -1)
        with torch.no_grad():
            logits = self.head(c, r).mean(0) / max(temperature, 1e-6)
            w = torch.softmax(logits, dim=-1)
        wn = w.detach().cpu().numpy()
        order = np.argsort(-wn)
        margin = float(wn[order[0]] - (wn[order[1]] if len(order) > 1 else 0.0))
        return Command(lang_emb=self.bank.blend(w), weights=wn, margin=margin,
                       evidence=list(evidence or []))


# ---------------------------------------------------------------- serve-side

class LatentPrefix:
    """Installs a Command as pi0.5's language prefix.

    Patches `embed_language_tokens`, which is where a string would otherwise
    have become a tensor. Used as a context manager so the policy is always
    returned to normal — a leaked patch would silently drive every later
    episode from the same stale command.
    """

    def __init__(self, policy, command: Command):
        self.pwe = policy.model.paligemma_with_expert
        self.command = command
        self._orig = None

    def __enter__(self):
        self._orig = self.pwe.embed_language_tokens
        emb = self.command.lang_emb

        def emit(tokens):
            # Width is whatever the pipeline tokenized to; the bank was built
            # through the same path, so these agree. Guarded anyway because a
            # silent shape mismatch would broadcast into garbage.
            e = emb.to(tokens.device)
            if e.shape[1] != tokens.shape[1]:
                raise ValueError(f"prefix width {e.shape[1]} != expected {tokens.shape[1]}")
            return e.expand(tokens.shape[0], -1, -1)

        self.pwe.embed_language_tokens = emit
        return self

    def __exit__(self, *exc):
        self.pwe.embed_language_tokens = self._orig
        return False


def build_bank(policy, instructions: list[str], tokenize) -> PrototypeBank:
    """Embed each instruction ONCE, through the policy's own path.

    `tokenize(text) -> tokens` must be the pipeline's real tokenizer, so the
    prototypes live exactly where genuine prefixes live rather than near them.
    """
    pwe = policy.model.paligemma_with_expert
    embs = []
    with torch.no_grad():
        for text in instructions:
            embs.append(pwe.embed_language_tokens(tokenize(text)).squeeze(0))
    return PrototypeBank(torch.stack(embs))
