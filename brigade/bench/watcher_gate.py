#!/usr/bin/env python3
"""Can a VLM read an object's fate OFF a clip? The gate for a clips-only memory.

    .venv-libero/bin/python brigade/bench/watcher_gate.py --record
    myenv/bin/python        brigade/bench/watcher_gate.py --watch

If memory stores clips and nothing else — no locations, no labels, no norms —
then every semantic answer has to be read out of the clips at query time. The
reasoning layer is not a nicety in that design, it is the only thing that turns
retrieval into an answer. So its accuracy is the ceiling on the whole system,
and it is measured before anything is built on it.

Two phases because they need different interpreters: the simulator runs under
transformers 4.53.2 for pi0.5, the watcher under 4.57 for Qwen3-VL. Phase one
records clips with known outcomes; phase two watches them blind and is scored.
The labels are written to disk in phase one and NEVER shown to the watcher.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = "../eval_logs/watcher_gate"
MODEL = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"

# (libero_goal id, the object the question is about, the place it should end up)
CASES = [
    (4, "black bowl", "cabinet"),
    (1, "black bowl", "stove"),
    (8, "black bowl", "plate"),
    (9, "wine bottle", "wine rack"),
    (6, "cream cheese", "bowl"),
    (0, "drawer", "open"),
]
PLACES = ["the table", "the top of the cabinet", "the stove", "the plate",
          "the wine rack", "inside the black bowl"]


def record(n_each: int = 1) -> int:
    from brigade.agent.pilot import Pilot
    from brigade.memory.relmo import write_clip

    os.makedirs(OUT, exist_ok=True)
    p = Pilot(device="mps")
    p.load()
    rows = []
    for goal, subject, truth in CASES:
        p.open_scene("libero_goal", goal)
        for k in range(n_each):
            ep = p.run(p.default_instruction, keep_frames=True)
            path = os.path.join(OUT, f"g{goal}_{k}.mp4")
            write_clip(ep.frames, path, fps=10)
            end = {o.label: o.place for o in ep.final_obs}
            rows.append(dict(clip=path, goal=goal, subject=subject,
                             instruction=p.default_instruction,
                             success=bool(ep.success), truth=truth, ended=end))
            print(f"  {'ok  ' if ep.success else 'FAIL'} g{goal}  {p.default_instruction}"
                  f"  -> {end}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "labels.json"), "w"), indent=1)
    print(f"\nwrote {len(rows)} clips + labels to {OUT}")
    return 0


def frames(path, n=5, bias="end", size=384):
    """Sample n frames. `bias='end'` clusters them near the finish.

    The question is where something ENDS UP, and a uniform sample spends most of
    its budget on the approach: with eight uniform frames the last placement gets
    one or two, and the watcher answers with wherever the object spent the most
    time — which is the table it started on. Weighting toward the end shows the
    outcome instead of the journey.
    """
    import imageio.v3 as iio
    from PIL import Image

    v = iio.imread(path, plugin="pyav")
    last = len(v) - 1
    if bias == "end":
        # quadratic spacing: sparse early, dense late, final frame always included
        idx = (last * (np.linspace(0, 1, n) ** 2)).astype(int)
    else:
        idx = np.linspace(0, last, n).astype(int)
    idx = sorted(set(idx.tolist() + [last]))
    return [Image.fromarray(np.asarray(v[i])).resize((size, size)) for i in idx]


def watch() -> int:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    rows = json.load(open(os.path.join(OUT, "labels.json")))
    print(f"loading {MODEL} ...", flush=True)
    model, processor = load(MODEL)

    # Only well-formed cases: the drawer episode FAILED to record (the robot
    # never opened it) and "open" is not one of the offered places, so scoring
    # it would be scoring my own bug rather than the watcher.
    rows = [r for r in rows if r["truth"].lower() in " ".join(PLACES).lower()]
    results = {}
    for bias in ("uniform", "end"):
      ok = 0
      print(f"\n--- frame sampling: {bias} ---", flush=True)
      for r in rows:
        imgs = frames(r["clip"], bias=bias)
        # The question names the object but NOT the answer, and the options are
        # the same for every case, so the watcher cannot pick up the target from
        # the phrasing.
        q = (f"These {len(imgs)} frames are consecutive moments from one robot "
             f"episode in a kitchen, in order. Watch what the robot does. "
             f"Where does the {r['subject']} END UP? Choose exactly one of: "
             + "; ".join(PLACES) + ". Reply with the choice only.")
        prompt = apply_chat_template(processor, model.config, q, num_images=len(imgs))
        t0 = time.time()
        out = generate(model, processor, prompt, imgs, max_tokens=32, verbose=False)
        said = (out if isinstance(out, str) else getattr(out, "text", str(out))).strip().lower()
        hit = r["truth"].lower() in said
        ok += hit
        r[f"watched_{bias}"] = said
        r[f"correct_{bias}"] = bool(hit)
        print(f"  {'HIT ' if hit else 'miss'}  want {r['truth']:<12} got {said[:52]!r}"
              f"  ({time.time() - t0:.1f}s)", flush=True)
      results[bias] = ok
      print(f"  -> {ok}/{len(rows)}", flush=True)

    print(f"\nWATCHER  uniform {results['uniform']}/{len(rows)}   "
          f"end-weighted {results['end']}/{len(rows)}")
    json.dump(rows, open(os.path.join(OUT, "watched.json"), "w"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    if a.record:
        return record(a.n)
    if a.watch:
        return watch()
    ap.error("pass --record (libero venv) or --watch (myenv)")


if __name__ == "__main__":
    sys.exit(main())
