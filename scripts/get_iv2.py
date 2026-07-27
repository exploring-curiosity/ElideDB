"""Assemble models/iv2_stage2_1b: InternVideo2-Stage2 1B, MPS-loadable.

The official OpenGVLab repo ships a raw 8.6GB training checkpoint and
no code. This puts together the pieces that DO work here:
  - modeling code: TIGER-AI-Lab/VLM2Vec's single-file HF port
    (config already runs use_flash_attn=false)
  - weights: ziyjiang/InternVideo2-1B — fp16 towers-only
    re-serialization of the same official checkpoint (2.8GB)
  - three local patches (idempotent, marked "MPS port"):
    1. guard the two hard flash_attn imports (macOS has no flash_attn;
       the non-flash path never calls them)
    2. LayerScale parameter named `gamma` to match the checkpoint's
       ls{1,2}.gamma keys — with the port's default `weight` naming,
       80 layer scales silently stay random and the tower NaNs
    3. bert config resolved next to the modeling file instead of the
       VLM2Vec repo's hardcoded relative path

Re-runnable; refreshes the HF dynamic-module cache copy as well.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

RAW = ("https://raw.githubusercontent.com/TIGER-AI-Lab/VLM2Vec/main/"
       "src/model/baseline_backbone/internvideo2/")
FILES = ["modeling_internvideo2.py", "config_bert_large.json",
         "config.json", "__init__.py"]
DIR = Path("models/iv2_stage2_1b")


def patch(src):
    if "MPS port" in src:
        return src          # already patched
    src = src.replace(
        "from flash_attn.flash_attn_interface import "
        "flash_attn_varlen_qkvpacked_func\n"
        "from flash_attn.bert_padding import unpad_input, pad_input",
        "# MPS port: flash_attn does not build on macOS; the config\n"
        "# runs use_flash_attn=false so these symbols are never\n"
        "# called — guard the import like the FusedMLP ones above.\n"
        "try:\n"
        "    from flash_attn.flash_attn_interface import "
        "flash_attn_varlen_qkvpacked_func\n"
        "    from flash_attn.bert_padding import unpad_input, "
        "pad_input\n"
        "except Exception:\n"
        "    flash_attn_varlen_qkvpacked_func = None\n"
        "    unpad_input = pad_input = None")
    src = src.replace(
        "        self.weight = nn.Parameter(init_values * "
        "torch.ones(dim))",
        "        # MPS port: checkpoint names this ls{1,2}.gamma\n"
        "        self.gamma = nn.Parameter(init_values * "
        "torch.ones(dim))")
    src = src.replace("self.weight.float()", "self.gamma.float()")
    src = src.replace("x.mul_(self.weight)", "x.mul_(self.gamma)")
    src = src.replace("x * self.weight", "x * self.gamma")
    src = src.replace(
        '    bert_config = BertConfig.from_json_file("./src/model/'
        'vlm_backbone/internvideo2/config_bert_large.json")',
        "    # MPS port: resolve the bert config next to this file\n"
        "    _bc = os.path.join(os.path.dirname(os.path.abspath("
        "__file__)),\n"
        '                       "config_bert_large.json")\n'
        "    bert_config = BertConfig.from_json_file(_bc)")
    return src


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        p = DIR / f
        if not p.exists():
            urllib.request.urlretrieve(RAW + f, p)
    mp = DIR / "modeling_internvideo2.py"
    mp.write_text(patch(mp.read_text()))
    from huggingface_hub import hf_hub_download
    w = hf_hub_download("ziyjiang/InternVideo2-1B",
                        "pytorch_model.bin")
    dst = DIR / "pytorch_model.bin"
    if not dst.exists():
        dst.symlink_to(w)
    # refresh HF's dynamic-module cache copy if it exists
    cache = (Path.home() / ".cache/huggingface/modules/"
             "transformers_modules/iv2_stage2_1b")
    if cache.exists():
        shutil.copy(mp, cache / mp.name)
        shutil.copy(DIR / "config_bert_large.json", cache)
    print(f"ready: {DIR}")


if __name__ == "__main__":
    main()
