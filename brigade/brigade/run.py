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

    if not args.no_ui:
        from .api.server import serve_in_background

        serve_in_background(runner, host=args.host, port=args.port)
        log.info("console: http://%s:%d", args.host, args.port)

    log.info("running at %d Hz — ctrl-c to stop", CFG.world.control_freq)
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        log.info("stopping")
        runner.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
