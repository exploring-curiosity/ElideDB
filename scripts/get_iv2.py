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


MARKER = "_tok_helpers_vendored"  # current patch level (patches 4+5)


def patch(src):
    if MARKER in src:
        return src          # already at the current patch level
    # patch 4: transformers>=5 removed the bert pruning/chunking helpers
    # from pytorch_utils; extend the import chain with vendored reference
    # implementations (see models/iv2_stage2_1b for the applied form)
    src = src.replace(
        "except ImportError:\n"
        "    # transformers>=4.57\n"
        "    from transformers.pytorch_utils import (\n"
        "        apply_chunking_to_forward,\n"
        "        find_pruneable_heads_and_indices,\n"
        "        prune_linear_layer,\n"
        "    )",
        "except ImportError:\n"
        "    try:\n"
        "        # transformers 4.57..4.x\n"
        "        from transformers.pytorch_utils import (\n"
        "            apply_chunking_to_forward,\n"
        "            find_pruneable_heads_and_indices,\n"
        "            prune_linear_layer,\n"
        "        )\n"
        "    except ImportError:\n"
        f"        # MPS port ({MARKER}): transformers>=5 removed\n"
        "        # these; vendored reference implementations.\n"
        "        def find_pruneable_heads_and_indices(heads, n_heads,\n"
        "                                             head_size,\n"
        "                                             already_pruned_heads):\n"
        "            mask = torch.ones(n_heads, head_size)\n"
        "            heads = set(heads) - already_pruned_heads\n"
        "            for head in heads:\n"
        "                head = head - sum(1 if h < head else 0\n"
        "                                  for h in already_pruned_heads)\n"
        "                mask[head] = 0\n"
        "            mask = mask.view(-1).contiguous().eq(1)\n"
        "            index = torch.arange(len(mask))[mask].long()\n"
        "            return heads, index\n"
        "\n"
        "        def prune_linear_layer(layer, index, dim=0):\n"
        "            index = index.to(layer.weight.device)\n"
        "            W = layer.weight.index_select(dim, index)"
        ".clone().detach()\n"
        "            b = None\n"
        "            if layer.bias is not None:\n"
        "                b = (layer.bias if dim == 1\n"
        "                     else layer.bias[index]).clone().detach()\n"
        "            new_size = list(layer.weight.size())\n"
        "            new_size[dim] = len(index)\n"
        "            new_layer = nn.Linear(new_size[1], new_size[0],\n"
        "                                  bias=layer.bias is not None)\n"
        "            new_layer = new_layer.to(layer.weight.device)\n"
        "            with torch.no_grad():\n"
        "                new_layer.weight.copy_(W.contiguous())\n"
        "                if b is not None:\n"
        "                    new_layer.bias.copy_(b.contiguous())\n"
        "            return new_layer\n"
        "\n"
        "        def apply_chunking_to_forward(forward_fn, chunk_size,\n"
        "                                      chunk_dim, *input_tensors):\n"
        "            if chunk_size > 0:\n"
        "                num_chunks = (input_tensors[0].shape[chunk_dim]\n"
        "                              // chunk_size)\n"
        "                chunks = tuple(t.chunk(num_chunks, dim=chunk_dim)\n"
        "                               for t in input_tensors)\n"
        "                out = tuple(forward_fn(*c) for c in zip(*chunks))\n"
        "                return torch.cat(out, dim=chunk_dim)\n"
        "            return forward_fn(*input_tensors)")
    # patch 5: transformers>=5 removed the private tokenizer
    # character-class helpers as well
    src = src.replace(
        "from transformers.tokenization_utils import PreTrainedTokenizer, "
        "_is_control, _is_punctuation, _is_whitespace",
        "try:\n"
        "    from transformers.tokenization_utils import (\n"
        "        PreTrainedTokenizer, _is_control, _is_punctuation, "
        "_is_whitespace)\n"
        "except ImportError:\n"
        f"    # MPS port ({MARKER}): transformers>=5 removed the\n"
        "    # private character-class helpers; vendored versions.\n"
        "    from transformers import PreTrainedTokenizer\n"
        "\n"
        "    def _is_whitespace(char):\n"
        "        if char in (\" \", \"\\t\", \"\\n\", \"\\r\"):\n"
        "            return True\n"
        "        return unicodedata.category(char) == \"Zs\"\n"
        "\n"
        "    def _is_control(char):\n"
        "        if char in (\"\\t\", \"\\n\", \"\\r\"):\n"
        "            return False\n"
        "        return unicodedata.category(char).startswith(\"C\")\n"
        "\n"
        "    def _is_punctuation(char):\n"
        "        cp = ord(char)\n"
        "        if (33 <= cp <= 47 or 58 <= cp <= 64\n"
        "                or 91 <= cp <= 96 or 123 <= cp <= 126):\n"
        "            return True\n"
        "        return unicodedata.category(char).startswith(\"P\")")
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
        "    # MPS port: resolve the bert config next to this file,\n"
        "    # falling back to the assembled model dir (transformers\n"
        "    # caches this .py WITHOUT sibling json files)\n"
        "    _bc_cands = [\n"
        "        os.path.join(os.path.dirname(os.path.abspath("
        "__file__)),\n"
        '                     "config_bert_large.json"),\n'
        "        os.path.join(os.getcwd(),\n"
        '                     "models/iv2_stage2_1b/'
        'config_bert_large.json"),\n'
        "    ]\n"
        "    _bc = next(p for p in _bc_cands if os.path.exists(p))\n"
        "    bert_config = BertConfig.from_json_file(_bc)")
    return src


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    mp = DIR / "modeling_internvideo2.py"
    if mp.exists() and MARKER not in mp.read_text():
        src = mp.read_text()
        if "_bc_cands" not in src:
            mp.unlink()      # pre-v2: refetch and repatch from scratch
        # v2 files just need the new patch applied in place below
    for f in FILES:
        p = DIR / f
        if not p.exists():
            urllib.request.urlretrieve(RAW + f, p)
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
