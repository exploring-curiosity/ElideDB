"""Does pi0.5 follow a BLEND of two instruction embeddings?

If a head is going to emit the language tensor, the practical form is a mixture
over learned prototypes: predict weights, sum. That only works if the space
between two real instruction embeddings is navigable rather than a cliff.

alpha=1.0 is the right instruction, alpha=0.0 the wrong one. Where success
falls off tells us how confident the head's weights have to be.
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, "/Users/sudharshanramesh/Studies/MyProjects/StreetDex/brigade")
import numpy as np, torch
from brigade.agent.pilot import Pilot

p = Pilot(device="mps"); p.load(); p.open_scene("libero_goal", 4)
RIGHT = p.default_instruction
WRONG = "put the wine bottle on the rack"
print(f"right={RIGHT!r}  wrong={WRONG!r}", flush=True)

pwe = p.policy.model.paligemma_with_expert
orig = pwe.embed_language_tokens
grab = {}


def capture(tokens):
    grab["t"] = tokens.detach().clone()
    return orig(tokens)


# capture token ids for both instructions through the real pipeline
for name, instr in (("right", RIGHT), ("wrong", WRONG)):
    pwe.embed_language_tokens = capture
    p.env.envs[0].task_description = instr
    p.run(instr, keep_frames=False, max_steps=2)
    grab[name] = grab["t"]
pwe.embed_language_tokens = orig

with torch.no_grad():
    E_R = orig(grab["right"])
    W = grab["wrong"]
    if W.shape[1] != grab["right"].shape[1]:
        W = W[:, :grab["right"].shape[1]] if W.shape[1] > grab["right"].shape[1] \
            else torch.cat([W, grab["right"][:, W.shape[1]:] * 0], 1)
    E_W = orig(W)
print(f"embeddings {tuple(E_R.shape)}  cos(right,wrong)="
      f"{torch.nn.functional.cosine_similarity(E_R.flatten(), E_W.flatten(), 0).item():.4f}",
      flush=True)

rows = []
for alpha in (1.0, 0.9, 0.7, 0.5, 0.3):
    def mixed(tokens, a=alpha):
        return a * E_R.to(tokens.device) + (1 - a) * E_W.to(tokens.device)
    pwe.embed_language_tokens = mixed
    ok = sum(bool(p.run(RIGHT, keep_frames=False, seed=10000 + i).success) for i in range(3))
    pwe.embed_language_tokens = orig
    rows.append((alpha, ok))
    print(f"  alpha={alpha:<5} toward the RIGHT instruction   {ok}/3", flush=True)

json.dump(rows, open("/Users/sudharshanramesh/Studies/MyProjects/StreetDex/eval_logs/interp.json", "w"))
print("\nA prototype-mixing head needs weights confident enough to stay above the "
      "alpha where this breaks.")
