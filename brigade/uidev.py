#!/usr/bin/env python3
"""The dashboard with the kitchen but no policy — for looking at the view.

    python uidev.py            # http://127.0.0.1:8099

Loading pi0.5 costs 40 s and RelMo's basis another 85, which is the wrong
feedback loop for adjusting a light or a material. This opens the same LIBERO
scene, streams the same transforms from the same exporter, and drives the arm
with a slow sweep so there is motion to look at. Everything the browser receives
is byte-identical to the real thing; only the thing choosing the actions differs.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from brigade.api.server import LIVE, create_app, serve_in_thread
from brigade.memory.store import Memory
from brigade.world.scene import SceneExporter


def main() -> int:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(4)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    mem = Memory()
    app = create_app(LIVE, lambda: mem)

    @app.get("/api/prompts")
    def prompts():
        return dict(prompts=["put the bowl away", "put the bottle away",
                             "make the stove ready"])

    @app.get("/api/videomem")
    def videomem():
        return dict(ready=False, basis=None, n=0, hits=[], note="ui dev mode")

    serve_in_thread(app, port=int(os.environ.get("BRIGADE_PORT", 8099)))
    print(f"\n  ui dev  ->  http://127.0.0.1:{os.environ.get('BRIGADE_PORT', 8099)}\n",
          flush=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.reset()
    ex = SceneExporter(env.env.sim)
    LIVE.publish_scene(*ex.scene(task=task.language))
    LIVE.set_status("ui-dev", task.language, instruction=task.language,
                    request="(no policy loaded)", memory_used=True)
    LIVE.say("act", "ui dev mode — the arm sweeps, nothing is being decided")
    print(f"scene: {ex.scene(task.language)[0]['n_geoms']} visual geoms", flush=True)

    t = 0.0
    while True:
        # A slow lissajous on the end-effector delta: enough motion to judge
        # lighting, shadows and material without pretending to be a policy.
        a = np.array([0.35 * np.sin(t), 0.35 * np.sin(1.3 * t), 0.25 * np.sin(0.7 * t),
                      0.0, 0.0, 0.3 * np.sin(0.5 * t), np.sign(np.sin(0.25 * t))])
        # LIBERO raises on a step past the horizon rather than auto-resetting,
        # so the sweep has to restart the episode itself or the dev server dies
        # a couple of hundred steps in with "executing action in terminated
        # episode" — which looks exactly like the page having crashed.
        try:
            _obs, _r, done, _info = env.step(a)
        except ValueError:
            done = True
        if done:
            env.reset()
            t = 0.0
        LIVE.publish_frame(ex.frame_bytes())
        t += 0.06
        time.sleep(1 / 30)


if __name__ == "__main__":
    sys.exit(main())
