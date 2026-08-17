# Running LIBERO + learned VLA policies on Apple Silicon

**The widely-repeated claim that LIBERO requires Linux is false.** LeRobot's
`[libero]` extra is pinned `sys_platform == 'linux'`, and that pin exists for a
single dependency — `hf-egl-probe`, a headless-EGL *device prober*. macOS renders
MuJoCo through CGL and never needs it. Everything else works natively on arm64.

Verified on this machine (M-series, macOS 15.5, Python 3.11.6) on 2026-08-16.

---

## What works, measured

| Thing | Result |
|---|---|
| LIBERO task suites enumerated | **130 tasks** across spatial/object/goal/10/90 |
| `OffScreenRenderEnv` | make 2.8 s, reset 0.8 s |
| Sim throughput | **26.9 Hz** with two 256×256 cameras (37 ms/step) |
| LeRobot `LiberoEnv` vec-env | resets + steps; `Box(-1,1,(1,7))` actions |
| π0.5 checkpoint load | **3.62 B params on MPS**, 40 s load |

The tasks are genuine household work, not block-stacking. Read straight from the
benchmark:

```
libero_10:  'put the black bowl in the bottom drawer of the cabinet and close it'
            'turn on the stove and put the moka pot on it'
            'put the yellow and white mug in the microwave and close it'
libero_90:  'open the top drawer of the cabinet and put the bowl in it'
libero_goal:'turn on the stove'  'put the cream cheese in the bowl'
```

---

## Install

Use a **separate venv**. LIBERO pins `robosuite==1.4.0`, which will fight any
newer robosuite already in your project.

```bash
python3 -m venv .venv-libero
V=.venv-libero/bin

# LIBERO's own stack. mujoco MUST be <3.9 — robosuite 1.4.0 breaks on 3.11.
$V/pip install "robosuite==1.4.0" "mujoco<3.9.0,>=3.0.0" "bddl==1.0.1" \
               gymnasium termcolor matplotlib imageio "numpy<2" torch torchvision

# The benchmark itself. --no-deps is the whole trick: it skips hf-egl-probe,
# which cannot build on macOS and is the only reason the Linux pin exists.
git clone --depth 1 https://github.com/huggingface/lerobot-libero.git
$V/pip install --no-deps -e ./lerobot-libero

# LIBERO prompts interactively on first import. Answer it once, non-interactively.
echo "N" | $V/python -c "import libero.libero"   # writes ~/.libero/config.yaml

$V/pip install lerobot num2words
```

### Version constraints that actually bite

* `mujoco < 3.9` — robosuite 1.4.0 fails on 3.11.
* `transformers` — lerobot wants `>=4.57.1,<5.0.0`; **π0.5 wants 4.53**. They
  conflict. Pick per policy (see below).
* `huggingface-hub < 0.36` for lerobot; transformers 5.x demands `>=1.5`, so do
  not let transformers upgrade to 5.x.
* `num2words` — undeclared runtime dep of the SmolVLM processor.

---

## Policies

### SmolVLA — ungated, small, recommended to start

```bash
$V/python -m lerobot.scripts.lerobot_eval \
  --output_dir=./eval_logs/smolvla_goal0 \
  --env.type=libero --env.task=libero_goal --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=2 \
  --policy.path=HuggingFaceVLA/smolvla_libero --policy.device=mps \
  --env.max_parallel_tasks=1
```

`HuggingFaceVLA/smolvla_libero` — 1.22 GB, SmolVLM2-500M backbone, no gate.

### π0.5 — highest published scores, but two extra hurdles

Published 97.5% avg (LeRobot's reproduction) / 96.85% (Physical Intelligence).

**Hurdle 1 — patched transformers.** π0.5 requires openpi's `transformers_replace`
overlay; without it you get *"An incorrect transformer version is used"*, because
`transformers.models.siglip.check` does not exist upstream.

```bash
$V/pip install "transformers==4.53.2"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Physical-Intelligence/openpi.git
cd openpi && git sparse-checkout set src/openpi/models_pytorch/transformers_replace
cp -r src/openpi/models_pytorch/transformers_replace/* \
      ../.venv-libero/lib/python3.11/site-packages/transformers/

# verify
$V/python -c "from transformers.models.siglip import check; \
  print(check.check_whether_transformers_replace_is_installed_correctly())"   # True
```

**Hurdle 2 — a gated repo.** The tokenizer pulls `google/paligemma-3b-pt-224`,
which is `gated=manual`. You must accept the license at
<https://huggingface.co/google/paligemma-3b-pt-224> and wait for approval.
Until then the eval dies at `make_pre_post_processors` with a 403. Nothing in
the pipeline can route around this — it is a licence gate, not a bug.

A harmless warning to ignore on load:

```
Warning: Could not remap state dict keys: Missing key(s) ... embed_tokens.weight
```

The weights ARE loaded — inspected directly: `std 0.186, absmax 11.6`, i.e.
trained Gemma embeddings, not random init (~0.02 std). It is a key-naming
notice from weight tying.

---

## Why this matters

The control is **fully learned end-to-end**: two camera images plus an 8-dim
proprioceptive state, conditioned on a natural-language instruction, mapped
directly to a 7-dim end-effector delta + gripper, every step. Opening drawers,
grasping, multi-step sequencing all emerge from the policy. There are no
scripted primitives, no waypoints, and no state writes anywhere in the loop.
