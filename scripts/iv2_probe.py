"""Load-probe for InternVideo2-Stage2 1B on this machine. Prints the
feature-method names its remote code actually exposes; the channel
adapts to whichever exists. Bailout evidence if it cannot load."""
import sys

import torch
from transformers import AutoModel, AutoTokenizer

MID = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"

try:
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MID, trust_remote_code=True, torch_dtype=torch.float16)
except Exception as e:
    print(f"LOAD FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
cands = [n for n in dir(model) if any(
    k in n.lower() for k in ("vid", "video", "vision", "txt", "text"))
    and not n.startswith("_")]
print("feature-method candidates:", cands)
print("config frames:", getattr(model.config, "num_frames", "?"))
