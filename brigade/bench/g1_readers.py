#!/usr/bin/env python3
"""G1/G2 — can a fast, local reader say what happened, with NO options given?

    .venv-libero/bin/python brigade/bench/g1_readers.py --record
    myenv/bin/python        brigade/bench/g1_readers.py --read

Two changes from the first gate, and both matter.

**Open vocabulary.** The earlier version handed the model the six places in the
kitchen and asked it to pick one. That is a scene vocabulary written into the
prompt — the exact hardwiring this project forbids. The question here names no
places at all. The truth strings exist only inside this harness, for scoring,
and are never shown to any model. A reader that answers "on top of the wooden
cabinet" has genuinely read it; one given a menu has not.

**512-pixel memory camera.** The clips are no longer whatever the policy ate.
See Pilot.memory_frame.

Scored by substring against a small set of surface forms for the same place
("cabinet" / "cupboard"). That is scoring vocabulary, not system vocabulary: it
lives here, it never reaches the model, and its only job is to avoid marking a
correct answer wrong because it said cupboard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = "../eval_logs/g1"

# (libero_goal id, what the question is about, accepted surface forms of the truth)
CASES = [
    (4, "black bowl", ["cabinet", "cupboard"]),
    (1, "black bowl", ["stove", "burner", "hob", "cooktop"]),
    (8, "black bowl", ["plate", "dish"]),
    (9, "wine bottle", ["rack", "wine rack", "holder"]),
    (6, "cream cheese", ["bowl"]),
    (2, "wine bottle", ["cabinet", "cupboard"]),
]

QUESTION = ("The images are frames from one video, in order, showing a robot "
            "arm working at a kitchen counter. Look at the LAST frame. "
            "Where is the {subject} resting at the end? "
            "Answer with just the place, in a few words.")


# ------------------------------------------------------------------ phase one

def record(n_each: int) -> int:
    from brigade.agent.pilot import Pilot
    from brigade.memory.relmo import write_clip

    os.makedirs(OUT, exist_ok=True)
    p = Pilot(device="mps")
    p.load()
    rows = []
    for goal, subject, truth in CASES:
        p.open_scene("libero_goal", goal)
        for k in range(n_each):
            extra = {c: [] for c in ("frontview", "sideview")}
            def grab(_buf, _step, _p=p, _e=extra):
                if _step % 2 == 0:
                    for c in _e:
                        _e[c].append(_p.memory_frame(c))
            ep = p.run(p.default_instruction, keep_frames=True, on_frame=grab)
            path = os.path.join(OUT, f"g{goal}_{k}.mp4")
            write_clip(ep.frames, path, fps=10)
            for c, buf in extra.items():
                if buf:
                    write_clip(buf, path.replace(".mp4", f".{c}.mp4"), fps=10)
            ended = {o.label: o.place for o in ep.final_obs}
            rows.append(dict(clip=path, goal=goal, subject=subject, truth=truth,
                             instruction=p.default_instruction,
                             success=bool(ep.success), ended=ended,
                             res=int(p.MEM_RES)))
            print(f"  {'ok  ' if ep.success else 'FAIL'} g{goal} {p.default_instruction}"
                  f"\n        ended {ended}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "labels.json"), "w"), indent=1)
    print(f"\nwrote {len(rows)} clips at {p.MEM_RES}px to {OUT}")
    return 0


# ------------------------------------------------------------------ phase two

def frames(path, n=4, size=512, cam="agentview"):
    """Few frames, high resolution. The reduced task is 'read the END frame',
    which RelMo makes possible by returning a located span rather than a whole
    recording — so the budget goes on pixels, not on time."""
    import imageio.v3 as iio
    from PIL import Image

    if cam != "agentview":
        alt = path.replace(".mp4", f".{cam}.mp4")
        if os.path.exists(alt):
            path = alt
    v = iio.imread(path, plugin="pyav")
    last = len(v) - 1
    if n <= 1:
        # EXACTLY one. FastVLM allows a single image token per prompt and the
        # dedup below silently produced two, which it rejects outright.
        idx = [last]
    else:
        idx = sorted(set((last * (np.linspace(0, 1, n) ** 2)).astype(int).tolist() + [last]))
    return [Image.fromarray(np.asarray(v[i])).resize((size, size)) for i in idx]


def read_mlx(rows, model_id, n_frames=4, cam="agentview"):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    t0 = time.time()
    model, processor = load(model_id)
    load_s = time.time() - t0
    out = []
    for r in rows:
        imgs = frames(r["clip"], n=n_frames, cam=cam)
        prompt = apply_chat_template(processor, model.config,
                                     QUESTION.format(subject=r["subject"]),
                                     num_images=len(imgs))
        t = time.time()
        g = generate(model, processor, prompt, imgs, max_tokens=32, verbose=False)
        said = (g if isinstance(g, str) else getattr(g, "text", str(g))).strip()
        out.append((said, time.time() - t))
    return out, load_s


def read_moondream(rows, model_id="vikhyatk/moondream2", n_frames=1, cam="agentview"):
    """Moondream answers about ONE image, which is the reduced task exactly."""
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision="2025-06-21",
        dtype=torch.float32, device_map={"": "mps"})
    load_s = time.time() - t0
    out = []
    for r in rows:
        img = frames(r["clip"], n=1, cam=cam)[-1]          # the END frame
        q = (f"Where is the {r['subject']} resting? "
             f"Answer with just the place, in a few words.")
        t = time.time()
        said = str(model.query(img, q)["answer"]).strip()
        out.append((said, time.time() - t))
    return out, load_s


def read_moondream_crop(rows, model_id="vikhyatk/moondream2", cam="agentview"):
    """Point at the subject FIRST, then read a crop around it.

    Moondream's native skills are point/detect — it returns (x, y) for an
    object you name — and free-form relational VQA is not what it is best at.
    Asking "where is the bowl resting" over a whole 512px kitchen makes the
    relevant evidence a handful of pixels; localising first and then reading a
    tight crop spends the model's resolution where the answer actually is.

    Nothing here is a scene prior: the noun comes from the human's question,
    the crop is a fixed fraction of the frame, and the follow-up question names
    no places.
    """
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision="2025-06-21",
        dtype=torch.float32, device_map={"": "mps"})
    load_s = time.time() - t0
    out = []
    for r in rows:
        img = frames(r["clip"], n=1, cam=cam)[-1]
        W, H = img.size
        t = time.time()
        try:
            pts = model.point(img, r["subject"]).get("points", [])
        except Exception:
            pts = []
        if pts:
            cx, cy = float(pts[0]["x"]) * W, float(pts[0]["y"]) * H
            half = W * 0.30
            box = (max(0, cx - half), max(0, cy - half),
                   min(W, cx + half), min(H, cy + half))
            view = img.crop(box).resize((W, H))
        else:
            view = img
        q = (f"What is the {r['subject']} sitting on or in? "
             f"Answer with just the thing it rests on.")
        said = str(model.query(view, q)["answer"]).strip()
        out.append((said + ("" if pts else "  [not located]"), time.time() - t))
    return out, load_s


# (label, fn). FastVLM accepts exactly ONE image per prompt — it raised
# "Expected up to 1 image tokens per prompt, got 4" — so it reads the end frame,
# which is the reduced task anyway.
CAMS = os.environ.get("G1_CAMS", "agentview,frontview,sideview").split(",")
READERS = []
for _cam in CAMS:
    READERS += [
        (f"Moondream2      {_cam:<10}", lambda rows, c=_cam: read_moondream(rows, cam=c)),
        (f"Moondream2-crop {_cam:<10}", lambda rows, c=_cam: read_moondream_crop(rows, cam=c)),
        (f"FastVLM-0.5B    {_cam:<10}", lambda rows, c=_cam: read_mlx(
            rows, "mlx-community/FastVLM-0.5B-bf16", n_frames=1, cam=c)),
    ]


def scored(said: str, truth: list[str]) -> bool:
    s = said.lower()
    return any(t in s for t in truth)


def read() -> int:
    rows = json.load(open(os.path.join(OUT, "labels.json")))
    print(f"\nG1/G2 — {len(rows)} blind clips at {rows[0].get('res', '?')}px, "
          f"NO options offered\n" + "=" * 74)
    table = []
    for name, fn in READERS:
        print(f"\n--- {name} ---", flush=True)
        try:
            said, load_s = fn(rows)
        except Exception as exc:  # noqa: BLE001
            print(f"  UNAVAILABLE: {type(exc).__name__}: {str(exc)[:130]}")
            continue
        ok, lat, answers = 0, [], []
        for r, (txt, dt) in zip(rows, said):
            hit = scored(txt, r["truth"])
            ok += hit
            lat.append(dt)
            answers.append(txt)
            print(f"  {'HIT ' if hit else 'miss'} want {r['truth'][0]:<10} "
                  f"got {txt[:56]!r}  ({dt * 1e3:.0f} ms)", flush=True)
        n_uniq = len(set(a.strip().lower() for a in answers))
        table.append((name, ok, len(rows), float(np.median(lat)) * 1e3, load_s, n_uniq))
        print(f"  -> {ok}/{len(rows)}  median {np.median(lat) * 1e3:.0f} ms  "
              f"distinct answers {n_uniq}/{len(rows)}", flush=True)

    print("\n" + "=" * 74)
    print(f"{'reader':<26}{'G1 correct':>12}{'G2 median':>12}{'distinct':>10}{'load':>8}")
    for name, ok, n, ms, load_s, uq in table:
        print(f"{name:<26}{f'{ok}/{n}':>12}{f'{ms:.0f} ms':>12}"
              f"{f'{uq}/{n}':>10}{f'{load_s:.0f}s':>8}")
    print("\nG1 passes at >=4/6.  G2 passes at <=1000 ms.")
    print("`distinct` is the constant-predictor check: a reader answering the "
          "same thing every time\nis not reading, whatever it scores.")
    json.dump([dict(reader=n, correct=o, n=t, median_ms=m, load_s=l, distinct=u)
               for n, o, t, m, l, u in table],
              open(os.path.join(OUT, "g1.json"), "w"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    if a.record:
        return record(a.n)
    if a.read:
        return read()
    ap.error("--record (libero venv) or --read (myenv)")


if __name__ == "__main__":
    sys.exit(main())
