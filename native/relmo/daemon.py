"""Run a relmo module DETACHED, so a job outlives the terminal that
started it.

The autonomous loop is supposed to keep producing and consuming while
nobody is watching; a generator that dies because a session closed is
not autonomous. Every long job goes through here.

Double-fork + setsid: the worker ends up in its own session with no
controlling terminal, reparented to init, immune to SIGHUP and to the
parent's process group being killed. stdout/stderr go to a log under
data/relmo/logs/, and the pid is written next to it so status.py can
tell "running" from "died".

    python -m relmo.daemon tracks2 --name physgen_v3
    python -m relmo.daemon train_wm --dataset physgen_v3 --head fdnn
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

LOGS = R.ROOT / "data" / "relmo" / "logs"


def spawn(module: str, args: list[str], tag: str | None = None):
    LOGS.mkdir(parents=True, exist_ok=True)
    tag = tag or module
    log = LOGS / f"{tag}.log"
    pidf = LOGS / f"{tag}.pid"
    if os.fork() != 0:                       # parent returns to caller
        for _ in range(50):                  # wait for the pid file
            if pidf.exists():
                break
            time.sleep(0.1)
        return int(pidf.read_text()) if pidf.exists() else None
    os.setsid()                              # new session, no tty
    if os.fork() != 0:                       # never a session leader
        os._exit(0)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(os.open(os.devnull, os.O_RDONLY))
    pidf.write_text(str(os.getpid()))
    os.chdir(str(Path(__file__).resolve().parents[1]))
    R.log("daemon_start", module=module, args=args, pid=os.getpid(),
          log=str(log))
    os.execv(sys.executable,
             [sys.executable, "-m", f"relmo.{module}"] + args)


def alive(tag: str):
    pidf = LOGS / f"{tag}.pid"
    if not pidf.exists():
        return None
    pid = int(pidf.read_text())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m relmo.daemon <module> [args]")
    p = spawn(sys.argv[1], sys.argv[2:])
    print(f"detached pid {p}  log {LOGS / sys.argv[1]}.log")
