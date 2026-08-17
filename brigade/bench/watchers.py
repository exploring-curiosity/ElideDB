#!/usr/bin/env python3
"""Which reasoning layer can sit inside the robot's memory loop?

    myenv/bin/python brigade/bench/watchers.py

Two numbers decide it, and accuracy alone is not enough: the watcher runs on
every recall, so its latency is part of the robot's cadence. The control loop
acts every 57 ms and thinks every ~573 ms, and an episode is ~10 s. Anything
that costs seconds per recall is not a memory, it is a batch job.

Scored on the same blind clips as `watcher_gate.py`: episodes with known
outcomes, labels never shown to the model.

The candidates are deliberately unequal in size — the question is how much
capability is actually needed to read "where did the bowl end up" off five
frames, not which model is best in general.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = "../eval_logs/watcher_gate"
PLACES = ["the table", "the top of the cabinet", "the stove", "the plate",
          "the wine rack", "inside the black bowl"]


def frames(path, n=5, size=384):
    import imageio.v3 as iio
    from PIL import Image

    v = iio.imread(path, plugin="pyav")
    last = len(v) - 1
    idx = sorted(set((last * (np.linspace(0, 1, n) ** 2)).astype(int).tolist() + [last]))
    return [Image.fromarray(np.asarray(v[i])).resize((size, size)) for i in idx]


def question(subject: str, n: int) -> str:
    return (f"These {n} frames are consecutive moments from one robot episode in "
            f"a kitchen, in order. Watch what the robot does. Where does the "
            f"{subject} END UP? Choose exactly one of: " + "; ".join(PLACES)
            + ". Reply with the choice only.")


# ---------------------------------------------------------------- candidates

def run_smolvlm(rows, model_id="HuggingFaceTB/SmolVLM2-500M-Instruct"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float32).to("mps").eval()
    load_s = time.time() - t0

    out = []
    for r in rows:
        imgs = frames(r["clip"])
        msg = [{"role": "user", "content":
                [{"type": "image", "image": im} for im in imgs]
                + [{"type": "text", "text": question(r["subject"], len(imgs))}]}]
        inputs = proc.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("mps", dtype=torch.float32)
        t = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=24)
        said = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip().lower()
        out.append((said, time.time() - t))
    return out, load_s


def run_qwen(rows, model_id="mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    t0 = time.time()
    model, processor = load(model_id)
    load_s = time.time() - t0

    out = []
    for r in rows:
        imgs = frames(r["clip"])
        prompt = apply_chat_template(processor, model.config,
                                     question(r["subject"], len(imgs)),
                                     num_images=len(imgs))
        t = time.time()
        g = generate(model, processor, prompt, imgs, max_tokens=24, verbose=False)
        said = (g if isinstance(g, str) else getattr(g, "text", str(g))).strip().lower()
        out.append((said, time.time() - t))
    return out, load_s


CANDIDATES = [
    ("SmolVLM2-500M   (video, 0.5B)", run_smolvlm),
    ("Qwen3-VL-30B-A3B (general, 4bit)", run_qwen),
]


def main() -> int:
    rows = json.load(open(os.path.join(OUT, "labels.json")))
    rows = [r for r in rows if r["truth"].lower() in " ".join(PLACES).lower()]
    print(f"\n{len(rows)} blind clips with known outcomes\n" + "=" * 72)

    table = []
    for name, fn in CANDIDATES:
        print(f"\n--- {name} ---", flush=True)
        try:
            said, load_s = fn(rows)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        ok, lat = 0, []
        for r, (txt, dt) in zip(rows, said):
            hit = r["truth"].lower() in txt
            ok += hit
            lat.append(dt)
            print(f"  {'HIT ' if hit else 'miss'}  want {r['truth']:<12} "
                  f"got {txt[:46]!r}  ({dt * 1e3:.0f} ms)", flush=True)
        table.append((name, ok, len(rows), float(np.median(lat)) * 1e3, load_s))
        print(f"  -> {ok}/{len(rows)}   median {np.median(lat) * 1e3:.0f} ms "
              f"(load {load_s:.0f}s)", flush=True)

    print("\n" + "=" * 72)
    print(f"{'watcher':<34}{'correct':>9}{'median':>12}{'load':>9}")
    for name, ok, n, ms, load_s in table:
        print(f"{name:<34}{f'{ok}/{n}':>9}{f'{ms:.0f} ms':>12}{f'{load_s:.0f}s':>9}")
    print(f"\nchance is ~{1 / len(PLACES):.2f} ({len(PLACES)} options)")
    json.dump([dict(watcher=n, correct=o, n=t, median_ms=m, load_s=l)
               for n, o, t, m, l in table],
              open(os.path.join(OUT, "watchers.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
