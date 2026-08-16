"""Bring the kitchen up with its console attached.

    python -m brigade.run

The ordering here is the whole point and is not rearrangeable: the simulation
boots and then runs on the MAIN thread, because MuJoCo's macOS render context is
main-thread-only and violating that kills the process without a traceback. The
HTTP console is the worker.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import CFG
from .world.sim import SimRunner


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="brigade.run", description="Brigade — The Pass")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-ui", action="store_true", help="run the kitchen headless")
    p.add_argument("--no-agent", action="store_true", help="no autonomy; manual control only")
    p.add_argument("--forget", action="store_true", help="wipe memory before starting")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("brigade")

    runner = SimRunner()
    log.info("booting kitchen (layout %d, style %d) — takes a few seconds…",
             CFG.world.layout_id, CFG.world.style_id)
    t0 = time.time()
    runner.boot()
    log.info("kitchen up in %.1fs", time.time() - t0)

    objects = runner.call(lambda env: sorted(env.objects.keys()))
    fixtures = runner.call(lambda env: env.get_ep_meta()["brigade_fixtures"])
    log.info("objects: %s", ", ".join(objects))
    log.info("home=%s stage=%s counter=%s",
             fixtures["home_cabinet"], fixtures["stage_cabinet"], fixtures["counter"])

    # Memory before agent: an agent whose memory is offline should not start at
    # all, rather than start and improvise.
    memory = None
    agent = None
    try:
        from .memory.store import Memory

        memory = Memory()
        info = memory.setup()
        log.info("memory: %s %s | native vectors: %s | indexes: %s",
                 info["flavor"], info["version"], info["native_vectors"],
                 ", ".join(info["vector_indexes"]) or "none")
        if args.forget:
            memory.wipe()
            log.info("memory wiped — the robot starts knowing nothing")
        log.info("memory holds: %s", memory.stats())
    except Exception as exc:
        log.error("MEMORY UNAVAILABLE: %s", exc)
        log.error("the agent will not start; the kitchen runs under manual control only")

    if memory is not None and not args.no_agent:
        from .agent.loop import Agent

        # Load the text encoder up front and say so. A lazy 90 MB load inside the
        # first tick hides behind an idle-looking agent and reads as a hang.
        from .memory import embed

        log.info("loading text encoder…")
        embed.load()
        agent = Agent(runner, memory)
        agent.start()
        log.info("agent online — it decides for itself what needs doing")

    if not args.no_ui:
        from .api.server import serve_in_background

        serve_in_background(runner, host=args.host, port=args.port,
                            memory=memory, agent=agent)
        log.info("console: http://%s:%d", args.host, args.port)

    log.info("running at %d Hz — ctrl-c to stop", CFG.world.control_freq)
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        if agent is not None:
            agent.stop()
        runner.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
