#!/usr/bin/env python3
"""Test A — can pi0.5 be commanded by a TENSOR instead of a string?

    .venv-libero/bin/python brigade/bench/a_latent.py --episodes 3

This decides the architecture. Under the owner's ruling nothing textual may be
stored, so a command must either be generated (ruled out) or never be words at
all. pi0.5's prefix is assembled as

    lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)                      # [images..., language]

so the language contribution is a (B, T, D) tensor and nothing requires it to
have come from a tokenizer. If a small head can emit that tensor, the memory
layer commands the robot with no vocabulary anywhere.

The risk is not whether the tensor *fits* — it obviously does — but whether the
action expert tolerates a prefix that is NEAR the manifold of real token
embeddings without being exactly on it. A trained head will never reproduce the
embedding exactly, so this measures how much error the policy forgives.

Arms:

  baseline   untouched, the reference success rate
  noise@s    lang_emb + Gaussian noise at s x the embedding's own std
  wrong      the embedding of a DIFFERENT task's instruction

`wrong` is the control and it matters more than the rest. If the robot succeeds
just as often with the wrong instruction embedded, then the prefix is not
steering anything on this scene, every other number here is meaningless, and
the premise of a latent command is dead for a different reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def patch(policy, mode: str, scale: float = 0.0, other_tokens=None):
    """Wrap embed_language_tokens. Returns a restore().

    Patching at the EMBEDDING boundary, not the tokenizer, is the point: it is
    exactly where a head would plug in, so a pass here is evidence about the
    real design and not about a mock of it.
    """
    pwe = policy.model.paligemma_with_expert
    original = pwe.embed_language_tokens

    def wrapped(tokens):
        emb = original(tokens)
        if mode == "baseline":
            return emb
        if mode == "wrong":
            alt = other_tokens.to(tokens.device)
            if alt.shape[1] != tokens.shape[1]:          # pad/crop to the same width
                if alt.shape[1] < tokens.shape[1]:
                    pad = tokens[:, alt.shape[1]:] * 0
                    alt = torch.cat([alt, pad], dim=1)
                else:
                    alt = alt[:, :tokens.shape[1]]
            return original(alt)
        if mode == "noise":
            return emb + torch.randn_like(emb) * (emb.std() * scale)
        raise ValueError(mode)

    pwe.embed_language_tokens = wrapped
    return lambda: setattr(pwe, "embed_language_tokens", original)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--goal", type=int, default=4)
    ap.add_argument("--out", default="../eval_logs/a_latent.json")
    a = ap.parse_args()

    from brigade.agent.pilot import Pilot

    p = Pilot(device="mps")
    p.load()
    p.open_scene("libero_goal", a.goal)
    instruction = p.default_instruction
    print(f"\nscene libero_goal/{a.goal}: {instruction!r}\n" + "=" * 70, flush=True)

    # Capture the token ids of a DIFFERENT instruction for the control arm, by
    # letting the real pipeline tokenize it once.
    grabbed = {}
    pwe = p.policy.model.paligemma_with_expert
    orig = pwe.embed_language_tokens

    def grab(tokens):
        grabbed["t"] = tokens.detach().clone()
        return orig(tokens)

    pwe.embed_language_tokens = grab
    p.env.envs[0].task_description = "put the wine bottle on the rack"
    p.run("put the wine bottle on the rack", keep_frames=False, max_steps=2)
    pwe.embed_language_tokens = orig
    other = grabbed.get("t")
    print(f"control instruction tokenized: {tuple(other.shape)}", flush=True)

    ARMS = [("baseline", "baseline", 0.0),
            ("noise 0.05", "noise", 0.05),
            ("noise 0.10", "noise", 0.10),
            ("noise 0.25", "noise", 0.25),
            ("noise 0.50", "noise", 0.50),
            ("wrong instruction", "wrong", 0.0)]

    rows = []
    for label, mode, scale in ARMS:
        restore = patch(p.policy, mode, scale, other)
        ok = 0
        secs = []
        for i in range(a.episodes):
            ep = p.run(instruction, keep_frames=False, seed=10000 + i)
            ok += bool(ep.success)
            secs.append(ep.seconds)
        restore()
        rows.append(dict(arm=label, ok=ok, n=a.episodes, mean_s=float(np.mean(secs))))
        print(f"  {label:<20} {ok}/{a.episodes}   mean {np.mean(secs):.0f}s", flush=True)

    print("\n" + "=" * 70)
    base = rows[0]["ok"]
    wrong = rows[-1]["ok"]
    print(f"baseline {base}/{a.episodes}   wrong-instruction {wrong}/{a.episodes}")
    if wrong >= base and base > 0:
        print("CONTROL FAILED: the prefix is not steering this scene. Every other\n"
              "row is uninterpretable — a latent command cannot be evaluated here.")
    else:
        tol = [r for r in rows[1:-1] if r["ok"] >= base]
        print(f"tolerated perturbation up to: "
              f"{tol[-1]['arm'] if tol else 'none — even 0.05 hurts'}")
        print("A is live if the policy holds at 0.10+; a trained head lands far\n"
              "closer than that to the true embedding.")
    json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
