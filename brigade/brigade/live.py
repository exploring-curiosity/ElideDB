"""Run the kitchen live, with the dashboard attached.

    .venv-libero/bin/python -m brigade.live            # then open :8099

This is the v2 driver. `run.py` is the v1 one and is built on the fact store —
beliefs, norms, a task queue with a goal string in it — every part of which the
owner's ruling removed. Nothing here writes a fact. The loop is:

    the cameras stream into the store, always
    a human types one thing
    the memory layer retrieves, reasons, and emits a command
    pi0.5 executes the command as a latent prefix
    the kitchen keeps what changed

The main thread owns the simulator, because MuJoCo's GL context does; uvicorn
runs beside it and the two meet through `Live`, one lock and a queue.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")

from .api.server import LIVE, create_app, serve_in_thread
from .config import CFG

log = logging.getLogger("brigade.live")

REASONER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "eval_logs", "reasoner", "reasoner.pt")


class LiveKitchen:
    """The serve loop with a viewport bolted on."""

    def __init__(self, live=LIVE, scene: int = 0, device: str = "mps"):
        from .agent.serve import Kitchen

        self.live = live
        self.k = Kitchen(REASONER, scene=scene, device=device)

    # ---- boot ---------------------------------------------------------------

    def boot(self) -> None:
        self.live.set_status("booting", "starting RelMo and the policy")
        info = self.k.open()
        self.live.publish_scene(*self.k.pilot.scene_payload)
        self.live.publish_frame(self.k.pilot.scene.frame_bytes())
        s = info["store"]
        self.live.say("memory",
                      f"video memory: {s['clips']} spans, {s['indexed']} indexed, "
                      f"{s['seconds']:.0f}s of video, basis {info['basis']}")
        self.live.say("memory",
                      "the store holds video, vectors and timestamps. No label, "
                      "no caption, no object name, no location.")
        self.live.set_status("idle", "watching")

    # ---- the viewport -------------------------------------------------------

    def _publish(self, payload, _step) -> None:
        self.live.publish_frame(payload if isinstance(payload, (bytes, bytearray))
                                else self.k.pilot.scene.frame_bytes())

    def _watch(self, seconds: float) -> None:
        """Idle, but with the viewport updating so the room stays live."""
        def fn(frame, _i):
            self.k.store.write(frame)
            self.live.publish_frame(self.k.pilot.scene.frame_bytes())

        self.k.pilot.idle(seconds, on_frame=fn)

    # ---- the loop -----------------------------------------------------------

    def loop(self) -> None:
        while True:
            try:
                cmd = self.live.commands.get(timeout=0.5)
            except queue.Empty:
                # The cameras do not stop because nobody is talking. This is
                # where most of the store's video comes from.
                self._watch(2.0)
                continue
            self.serve(cmd)

    def serve(self, cmd: dict) -> None:
        heard = (cmd.get("request") or "").strip()
        use_memory = bool(cmd.get("use_memory", True))
        if not heard:
            return
        self.live.say("human", heard)
        self.live.set_status("recalling", heard, request=heard,
                             instruction="", memory_used=use_memory)

        t = self.k.hear(heard, on_frame=self._publish, memory=use_memory)

        if t.clips:
            stage = "stage 1 + DTW" if t.clips[0].stage == "dtw" else "stage 1"
            self.live.say("memory",
                          f"{len(t.clips)} spans in {t.retrieval_ms:.1f} ms ({stage}) · "
                          + ", ".join(f"{c.clip_id[2:8]} {c.score:.2f}"
                                      for c in t.clips[:4]),
                          clips=[c.clip_id for c in t.clips])
        if t.chose:
            self.live.say("reason",
                          f"command: {t.chose!r} — top weight "
                          f"{t.weights.max():.2f}, margin {t.margin:.2f}, "
                          f"{t.reason_ms:.1f} ms")
        if t.note:
            self.live.say("note", t.note)
        if t.acted:
            self.live.say("act", f"{'goal satisfied' if t.succeeded else 'ran to the step limit'}"
                                 f" in {t.act_s:.0f}s")
        self.live.record(t.to_json())
        self.live.set_status("idle", "watching", request=heard,
                             instruction=t.chose, memory_used=use_memory,
                             resolution=t.to_json())

    def close(self) -> None:
        self.k.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    lk = LiveKitchen(scene=a.scene, device=a.device)
    app = create_app(LIVE, lambda: lk.k.store)
    serve_in_thread(app, port=a.port)
    print(f"dashboard: http://localhost:{a.port}", flush=True)
    try:
        lk.boot()
        lk.loop()
    except KeyboardInterrupt:
        return 0
    finally:
        lk.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
