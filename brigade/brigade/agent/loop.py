"""The autonomous loop. This is the agent.

Nobody drives Brigade. Every few seconds it looks at the kitchen, compares what
it sees against what it remembers, writes its own tasks from the difference, and
does them. A human can add a task, but the robot does not need one.

    OBSERVE   what is in the kitchen now
    NOTICE    diff against memory  ->  self-authored tasks
    CLAIM     take the highest-priority task (FOR UPDATE SKIP LOCKED)
    RECALL    norms + beliefs + episodic recall for this goal
    DECIDE    pick a plan and a strategy, and log WHICH memories drove it
    ACT       run skills
    LEARN     write the outcome; promote a repeated placement to a norm

The loop is a background thread — the simulation owns the main thread. It only
touches the world through SimRunner.

What makes this an agent rather than a script is the NOTICE step. Everything
downstream of it exists in ordinary automation; the robot writing its own work
list from the difference between the world and its memory is the part that does
not.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import CFG
from ..memory.store import Memory, Recalled
from ..world import skills
from ..world.sim import SimRunner, SkillResult

log = logging.getLogger("brigade.agent")

PICK_STRATEGIES = ("top", "side")


@dataclass
class Beat:
    """One thing the agent did, for the console's activity feed."""

    t: float
    phase: str          # notice | decide | act | learn | idle
    text: str
    detail: str = ""
    ok: bool | None = None
    recalled: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(t=self.t, phase=self.phase, text=self.text, detail=self.detail,
                    ok=self.ok, recalled=self.recalled)


class Agent:
    def __init__(self, runner: SimRunner, memory: Memory | None = None):
        self.runner = runner
        self.mem = memory or Memory()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.beats: list[Beat] = []
        self._beats_lock = threading.Lock()
        self.tick = 0
        self.current: str | None = None
        self.paused = False
        # Objects the robot has already looked at and filed; used so NOTICE
        # reports genuinely new things rather than re-firing every tick.
        self._known_locations: dict[str, str] = {}
        self.stats = dict(noticed=0, tasks_done=0, tasks_failed=0, norms_learned=0)

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="brigade-agent", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return dict(running=self.running, tick=self.tick, current=self.current,
                    paused=self.paused, **self.stats)

    def feed(self, limit: int = 40) -> list[dict]:
        with self._beats_lock:
            return [b.as_dict() for b in self.beats[-limit:]][::-1]

    def _beat(self, phase: str, text: str, detail: str = "", ok: bool | None = None,
              recalled: list[str] | None = None) -> None:
        b = Beat(time.time(), phase, text, detail, ok, recalled or [])
        with self._beats_lock:
            self.beats.append(b)
            if len(self.beats) > 400:
                del self.beats[:-200]
        log.info("[%s] %s%s", phase, text, f" — {detail}" if detail else "")

    # ---- the loop -----------------------------------------------------------

    def _census(self) -> None:
        """First look: learn what the kitchen normally looks like.

        Without this the agent treats the kitchen's resting state as a mess and
        tries to put away the kettle that lives on the counter. A kettle on the
        counter is not out of place — it is where kettles are. What is out of
        place is what CHANGED.

        So the first observation is a baseline: everything visible is recorded as
        a belief and accepted as normal. Only movement after this point is a
        surprise, and only surprises become work. (Beliefs, not norms: the robot
        knows where things ARE, and still has to discover where they BELONG.)
        """
        snap = self._snapshot()
        for iid, o in snap["objects"].items():
            if "error" in o:
                continue
            loc = o.get("location")
            if not loc or loc == "unknown":
                continue
            self._known_locations[iid] = loc
            self.mem.see(iid, o["label"], loc, o["pos"],
                         float(o.get("location_confidence", 1.0)))
        counted = len(self._known_locations)
        self.mem.remember(
            f"first look at the kitchen: {counted} things, all where they normally are",
            kind="observation", payload=dict(baseline=self._known_locations.copy()),
        )
        self._beat("notice", f"took stock of the kitchen — {counted} things",
                   "this is normal; from here I only act on what changes")

    def _run(self) -> None:
        # Wait for the kitchen; the agent is useless before the world exists.
        while not self._stop.is_set() and not self.runner.running:
            time.sleep(0.5)
        self._beat("idle", "agent online", "watching the kitchen")
        try:
            self._census()
        except Exception as exc:
            log.exception("census failed")
            self._beat("idle", "could not take stock", str(exc)[:120], ok=False)

        while not self._stop.is_set():
            try:
                if self.paused:
                    time.sleep(0.5)
                    continue
                self.tick += 1
                self._observe_and_notice()
                task = self.mem.claim()
                if task is None:
                    self.current = None
                    time.sleep(CFG.agent.tick_s)
                    continue
                self.current = task["goal"]
                try:
                    self._do_task(task)
                except Exception as exc:
                    # A task that dies mid-flight must be released, not left in
                    # 'running' forever. Without this one exception wedges the
                    # queue permanently: nothing is pending, so nothing is ever
                    # claimed again, and the agent looks alive while doing
                    # nothing at all.
                    log.exception("task failed")
                    self.mem.finish(task["id"], False, f"error: {exc}")
                    self.stats["tasks_failed"] += 1
                    self._beat("act", f"task failed: {task['goal']}", str(exc)[:160], ok=False)
                self.current = None
            except Exception as exc:  # a bad tick must not end the agent
                log.exception("tick failed")
                self._beat("idle", "tick failed", str(exc)[:160], ok=False)
                time.sleep(1.0)

    # ---- observe + notice ---------------------------------------------------

    def _snapshot(self) -> dict:
        return self.runner.call(lambda env: env.world_snapshot())

    def _observe_and_notice(self) -> None:
        """Look at the kitchen; turn surprises into work.

        Two kinds of surprise are actionable:
          * an object is somewhere it does not belong (out of place)
          * an object is somewhere new since we last looked (it moved)
        Both become tasks the robot wrote for itself.
        """
        snap = self._snapshot()
        objects = snap["objects"]

        for iid, o in objects.items():
            if "error" in o or o.get("grasped"):
                continue
            loc, label = o.get("location"), o.get("label")
            if not loc or loc == "unknown":
                continue

            known = self._known_locations.get(iid)
            self._known_locations[iid] = loc

            # Record what we can see. This is the spatial memory being written.
            self.mem.see(iid, label, loc, o["pos"], float(o.get("location_confidence", 1.0)))

            if known is not None and known == loc:
                continue  # nothing changed for this object

            self._beat("notice", f"{label} {iid} moved",
                       f"{known} → {_short(loc)}" if known else f"appeared in {_short(loc)}")

            # Is it out of place? Only surfaces count as "out"; a thing inside a
            # cupboard is put away by definition.
            on_surface = "counter" in loc or "island" in loc
            if not on_surface:
                continue
            if self.mem.has_open_task(iid):
                continue

            self.stats["noticed"] += 1
            self._beat("notice", f"{label} left out on {_short(loc)}",
                       "no instruction given — deciding for myself")
            self.mem.remember(
                f"noticed a {label} ({iid}) left out on {loc}",
                kind="observation", subject=iid,
                payload=dict(location=loc, label=label),
            )
            self.mem.enqueue(f"put the {label} away", origin="self", priority=5,
                             subject=iid, payload=dict(instance_id=iid, label=label))

    # ---- act ----------------------------------------------------------------

    def _do_task(self, task: dict) -> None:
        goal = task["goal"]
        payload = task.get("payload") or {}

        # A door job from a human: "open the cab_2". Done by hand like everything.
        if payload.get("door"):
            name, action = payload["door"], payload.get("action", "open")
            self._beat("act", f"{action} {_short(name)} by hand", "asked by the user")
            skill = skills.open_fixture if action == "open" else skills.close_fixture
            res = self._run_skill(skill(name, "right"), action, name, timeout_s=240.0)
            self.mem.remember(
                f"{action}ed {name} by hand — {'success' if res.ok else 'failure'} — {res.detail}",
                kind="outcome", subject=name, outcome="success" if res.ok else "failure",
                task_id=str(task["id"]),
            )
            self.mem.finish(task["id"], res.ok, res.detail)
            self.stats["tasks_done" if res.ok else "tasks_failed"] += 1
            return

        iid = payload.get("instance_id") or task.get("subject")
        label = payload.get("label")
        t0 = time.time()

        if iid is None:
            self.mem.finish(task["id"], False, "no subject")
            return
        if label is None:
            label = self.runner.call(lambda env: env.object_label(iid))

        # ---- RECALL -------------------------------------------------------
        r0 = time.time()
        norm = self.mem.home_for(label)
        recalls: list[Recalled] = self.mem.recall(f"where does a {label} go", k=3)
        recall_ms = (time.time() - r0) * 1000.0

        if norm:
            home = norm["home_location"]
            why = (f"memory: {label}s belong in {_short(home)} "
                   f"({norm['source']}, seen {norm['n_episodes']}x)")
            # Logged BEFORE acting, so the audit trail records the decision even
            # if the acting then fails. A decisions table that only contains
            # successes is not an audit trail.
            self.mem.decide(f"put {iid} in {home}", why, recalls, recall_ms,
                            str(task["id"]))
            self._beat("decide", f"I know where {label}s go", why,
                       recalled=[r.text for r in recalls])
        else:
            # ---- no memory -> EXPLORE ------------------------------------
            self.mem.decide(
                f"explore for {label}",
                f"nothing in memory about where {label}s go; searching the kitchen",
                recalls, recall_ms, str(task["id"]),
            )
            self._beat("decide", f"no memory of where {label}s go",
                       "searching the cabinets", recalled=[r.text for r in recalls])
            home = self._explore_for(label, task)
            if home is None:
                self.mem.remember(
                    f"searched the kitchen and found nowhere that {label}s are kept",
                    kind="reflection", subject=label, outcome="failure",
                )
                self.mem.finish(task["id"], False, "found nowhere to put it")
                self.stats["tasks_failed"] += 1
                return
            why = f"found {label}s already in {_short(home)} while searching"
            self.mem.decide(f"put {iid} in {home}", why, recalls, recall_ms,
                            str(task["id"]))

        # ---- ACT ----------------------------------------------------------
        # Which grasps are even geometrically possible is read from the object
        # (its own horizontal_radius vs the gripper aperture); WHICH of those to
        # try first is read from memory. Physics constrains the menu, experience
        # orders it.
        options = self.runner.call(lambda env: skills.strategies_for(env, iid))
        strategy, strat_why = self.mem.best_strategy("pick", label, options)
        self._beat("act", f"fetching the {label}", strat_why)

        # Drive to the OBJECT, not to the fixture it sits on — a counter is metres
        # long and "at the counter" does not mean "able to touch the bowl".
        self._run_skill(skills.drive_to_object(iid), "drive_to", iid)
        got = self._run_skill(skills.pick(iid, strategy), "pick", iid, strategy)
        self.mem.record_attempt("pick", label, strategy, got.ok)

        if not got.ok:
            # ---- learn from failure and retry differently -----------------
            self.mem.remember(
                f"pick of {label} ({iid}) failed with the {strategy} approach: {got.detail}",
                kind="outcome", subject=iid, outcome="failure",
                payload=dict(skill="pick", strategy=strategy),
            )
            alt, alt_why = self.mem.best_strategy("pick", label, options)
            if alt == strategy:
                others = [s for s in options if s != strategy]
                if not others:
                    self.mem.finish(task["id"], False, got.detail)
                    self.stats["tasks_failed"] += 1
                    self._beat("learn", f"no other way to grip a {label}", got.detail, ok=False)
                    return
                alt = others[0]
                alt_why = f"{strategy} just failed; trying {alt}"
            self._beat("learn", f"{strategy} grasp failed", alt_why, ok=False)
            got = self._run_skill(skills.pick(iid, alt), "pick", iid, alt)
            self.mem.record_attempt("pick", label, alt, got.ok)
            strategy = alt

        if not got.ok:
            self.mem.remember(
                f"could not pick up the {label} ({iid}) at all",
                kind="outcome", subject=iid, outcome="failure",
            )
            self.mem.finish(task["id"], False, got.detail)
            self.stats["tasks_failed"] += 1
            self._beat("learn", f"gave up on the {label}", got.detail, ok=False)
            return

        # If home is a cabinet whose doors are shut, the robot opens one BY HAND
        # before placing. If one door's opening is not enough for the object to
        # land inside, it opens the second and tries again — adaptation, not
        # teleportation.
        self._run_skill(skills.drive_to(home), "drive_to", home)
        from ..world import doors as D

        needs_door = self.runner.call(lambda env: D.openable(env, home))
        if needs_door:
            norms = self.runner.call(lambda env: D.door_norms(env, home))
            if not norms or max(norms.values()) < D.SEE_INSIDE:
                self._beat("act", f"opening {_short(home)} by hand", "crack the handle, sweep the door")
                self._run_skill(skills.open_fixture(home, "right"), "open", home, timeout_s=240.0)
        put = self._run_skill(skills.place(iid, home), "place", iid, timeout_s=120.0)
        if not put.ok and needs_door and (put.payload or {}).get("door_shut") is not True:
            self._beat("learn", "did not land inside; opening the other door", "", ok=False)
            self._run_skill(skills.open_fixture(home, "left"), "open", home, timeout_s=240.0)
            got2 = self._run_skill(skills.pick(iid, strategy), "pick", iid, strategy)
            if got2.ok:
                put = self._run_skill(skills.place(iid, home), "place", iid, timeout_s=120.0)

        # ---- LEARN ---------------------------------------------------------
        seconds = time.time() - t0
        final = self.runner.call(lambda env: env.locate_detail(iid))
        ok = put.ok or final["location"] == home
        self.mem.remember(
            f"put the {label} ({iid}) in {final['location']} — "
            f"{'success' if ok else 'failure'} — {seconds:.1f}s — strategy={strategy}",
            kind="outcome", subject=iid, outcome="success" if ok else "failure",
            task_id=str(task["id"]),
            payload=dict(label=label, location=final["location"], strategy=strategy,
                         seconds=round(seconds, 1)),
        )
        self.mem.see(iid, label, final["location"], _pos(self.runner, iid),
                     float(final["confidence"]))
        self._known_locations[iid] = final["location"]
        self.mem.record_attempt("place", label, "overhead", ok)

        if ok:
            before = self.mem.home_for(label)
            after = self.mem.learn_norm(label, home)
            if after and (before is None or before["n_episodes"] != after["n_episodes"]
                          or before["home_location"] != after["home_location"]):
                self.stats["norms_learned"] += 1
                self._beat(
                    "learn", f"{label}s go in {_short(home)}",
                    f"confidence {after['confidence']:.2f} after {after['n_episodes']} time(s)",
                    ok=True,
                )
            self.stats["tasks_done"] += 1
            self._beat("act", f"put the {label} away", f"{_short(home)} · {seconds:.0f}s", ok=True)
        else:
            self.stats["tasks_failed"] += 1
            self._beat("act", f"failed to put the {label} away", put.detail, ok=False)

        self.mem.finish(task["id"], ok, put.detail)

    # ---- exploration --------------------------------------------------------

    def _explore_for(self, label: str, task: dict) -> str | None:
        """Open cabinets until we find where things of this kind already live.

        This is the cold-memory path, and the cost of it is the point: it is what
        the robot has to do exactly once, and never again once the norm exists.
        """
        # Nearest first. Searching in dictionary order makes the robot criss-cross
        # the kitchen, which is both slow and reads as aimless on camera; a person
        # looking for bowls checks the cupboard they are standing next to.
        # Only fixtures the HAND can open count: exploration means physically
        # pulling doors now, so a fridge this gripper cannot work is not a place
        # the robot can look, and pretending to see inside it would be fake.
        def _by_distance(env):
            import numpy as np

            from ..world import doors as D

            here = np.array(env._get_observations()["robot0_base_pos"])[:2]
            cands = [
                t["name"] for t in env.placement_targets()
                if t["kind"] == "container" and D.openable(env, t["name"])
            ]
            return sorted(
                cands,
                key=lambda n: float(np.linalg.norm(np.array(env.fixtures[n].pos)[:2] - here)),
            )

        targets = self.runner.call(_by_distance)
        opened = 0
        for name in targets:
            if self._stop.is_set():
                return None
            self._beat("act", f"looking in {_short(name)}",
                       "driving over and pulling the door open")
            res = self._run_skill(skills.survey(name), "survey", name, timeout_s=240.0)
            if not res.ok:
                # Could not get the door open enough to see. Real answer; move on.
                self.mem.remember(
                    f"could not see inside {name}: {res.detail}",
                    kind="observation", subject=name, outcome="failure",
                )
                continue
            opened += 1
            labels = (res.payload or {}).get("labels", {})
            match = [i for i, lab in labels.items() if lab == label]
            self.mem.remember(
                f"looked in {name}: {', '.join(labels.values()) if labels else 'empty'}",
                kind="observation", subject=name,
                payload=dict(fixture=name, found=list(labels)),
            )
            for i, lab in labels.items():
                self.mem.see(i, lab, name, _pos(self.runner, i), 1.0)
            if match:
                self._beat("learn", f"found {len(match)} {label}(s) in {_short(name)}",
                           f"after opening {opened} cabinet(s)", ok=True)
                self.mem.remember(
                    f"{label}s are kept in {name} — found {len(match)} there while searching",
                    kind="reflection", subject=label,
                    payload=dict(label=label, location=name, opened=opened),
                )
                return name
        self._beat("learn", f"no {label}s found anywhere", f"opened {opened} cabinets", ok=False)
        return None

    # ---- helpers ------------------------------------------------------------

    def _run_skill(self, controller, name: str, subject: str = "",
                   strategy: str = "default", timeout_s: float | None = None) -> SkillResult:
        res = self.runner.run_skill(name, controller, subject=subject, strategy=strategy,
                                    timeout_s=timeout_s)
        self._beat("act", f"{name} {_short(subject)}".strip(), res.detail, ok=res.ok)
        return res


def _pos(runner: SimRunner, iid: str) -> list[float]:
    try:
        return list(map(float, runner.call(lambda env: env.object_pose(iid))))
    except Exception:
        return [0.0, 0.0, 0.0]


def _short(name: str) -> str:
    return (name or "").replace("_main_group", "")
