# Third-party notices

ElideDB itself is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
This file records what the software depends on and under which terms, so that
anyone evaluating it, and anyone later negotiating a commercial license, can see
the full picture at a glance. License tags below were verified against the
upstream model cards and project pages on 2026-08-23.

## The product pipeline

These are the only models the shipped read and write path uses. Both are
downloaded from Hugging Face on first use and are never modified or fine-tuned.

| component | source | license | commercial use |
|---|---|---|---|
| V-JEPA 2 (ViT-L) | [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | MIT | permitted |
| SigLIP 2 (base) | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | Apache-2.0 | permitted |

Supporting libraries: PyTorch (BSD-3), Hugging Face Transformers (Apache-2.0),
NumPy (BSD-3), tqdm (MIT/MPL-2.0). All permissive.

**FFmpeg** is invoked as an external program (a subprocess), never linked or
redistributed. Users install it themselves (`brew install ffmpeg` / `apt
install ffmpeg`). FFmpeg's own LGPL/GPL terms therefore apply to the FFmpeg
binary, not to ElideDB.

**There are no restrictions against commercial use anywhere in the product
pipeline.** The noncommercial restriction on this repository is ElideDB's own
license choice, not something inherited from a dependency.

## Demo-only models

The two public web demos (`deploy/`) use a wider set of text towers. None of
these are on the product path.

| component | source | license | note |
|---|---|---|---|
| SigLIP so400m | `google/siglip-so400m-patch14-384` | Apache-2.0 | |
| SigLIP 2 so400m | `google/siglip2-so400m-patch14-384` | Apache-2.0 | |
| X-CLIP large | `microsoft/xclip-large-patch14` | MIT | |
| Perception Encoder | `facebook/PE-Core-L14-336` | Apache-2.0 | |
| InternVideo2-Stage2 1B | `ziyjiang/InternVideo2-1B` | MIT (tag) | community re-serialization of the OpenGVLab checkpoint; verify against the upstream release before any commercial deployment |
| DINOv3 ConvNeXt-Tiny | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | DINOv3 License (Meta) | commercial use permitted, but carries obligations: redistribute the agreement, display "Built with DINOv3", acceptable-use policy. Demo only |

## Datasets

**No dataset ships in the product.** A customer's store contains only the
customer's own video and traces derived from it. Datasets appear in this
project solely for evaluation and for the public demo store.

| dataset | license | where it is used |
|---|---|---|
| BridgeData V2 | CC BY 4.0 | source of the public demo store; attribution required, commercial use permitted |
| RoboCasa | MIT (tooling) | local evaluation only |
| KITTI | CC BY-NC-SA 3.0 | local evaluation only; noncommercial |
| Oxford RobotCar | CC BY-NC-SA 4.0 | local evaluation only; noncommercial |
| Something-Something v2 | research license | one offline probe; not in any store or result that ships |

The KITTI and Oxford RobotCar terms are noncommercial. They are used here to
test generalization across domains and never leave this machine; any future
commercial evaluation would substitute cleared footage.

**Attribution for the demo store:** the public demo store on Hugging Face is
derived from BridgeData V2 (Walke et al., CoRL 2023), used under CC BY 4.0.
