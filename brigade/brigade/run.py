#!/usr/bin/env python3
"""Brigade: run the robot, the memory and the dashboard as one process.

    python -m brigade.run                      # dashboard on :8099
    python -m brigade.run --seed               # teach it a history first
    python -m brigade.run --ab "put the bowl away" --trials 3

The whole system in one loop:

    human types something vague
        -> Resolver reads memory (pgvector) and picks a full instruction
        -> Pilot hands that instruction to pi0.5 and steps the simulator
        -> the scene's geom transforms stream to the browser as it moves
        -> the outcome is written back as an event, a norm and a video vector

Why one process and not a service per box: MuJoCo's GL context on macOS belongs
to the main thread, so the simulator cannot be moved off it. Everything else
therefore comes to the simulator — uvicorn runs in a daemon thread, RelMo runs
in a subprocess because its `transformers` pin is incompatible with pi0.5's, and
Postgres is where it always was.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import time
import uuid

os.environ.setdefault("MUJOCO_GL", "cgl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .agent.pilot import Pilot
from .agent.resolver import Resolver
from .api.server import LIVE, create_app, serve_in_thread
from .config import CFG
from .memory import relmo as relmo_mod
from .memory.db import DB, MemoryUnavailable
from .memory.store import Memory

log = logging.getLogger("brigade")

# One kitchen, ten goals: LIBERO's `libero_goal` suite is the same scene with
# the same four objects (bowl, plate, wine bottle, cream cheese) throughout —
# verified by comparing the BDDL object sets, all ten identical, and the initial
# placements, which differ by at most 0.08 across all 79 state dimensions. That
# is what makes an underspecified request interesting: "put the bowl away" has
# four physically legal answers in this one room and the sentence picks none.
SCENE_SUITE = "libero_goal"

# What a human types to try it. Not a task list — these are underspecified on
# purpose, and none of them is a string pi0.5 has ever been trained on.
PROMPTS = ["put the bowl on the stove", "where is the bowl?", "put it back",
           "and the bottle too", "now get the stove going", "feed the cat"]

# The history the robot is given in --seed: (goal id in libero_goal, repeats).
# The instruction is read from the benchmark, never typed here, so the words the
# robot learns are the words the goal actually carries. A household that keeps
# bowls on top of the cabinet and wine on the rack is just a household; the
# single bowl-on-plate episode is there because real habits have exceptions,
# and a norm that cannot survive one is not a norm.
SEED_HISTORY = [(4, 3), (9, 2), (7, 1), (6, 1), (8, 1)]


class Brigade:
    """The agent: memory, policy and world, wired together."""

    def __init__(self, live=LIVE, device: str = "mps", continuous: bool = True):
        self.live = live
        # Persistence is a CHOICE with a measured price, not a free upgrade:
        # pi0.5 is an episodic policy and a carried-over kitchen is out of its
        # training distribution. See RESULTS.md for the two numbers.
        self.continuous = continuous
        self.mem = Memory()
        self.resolver = Resolver(self.mem)
        self.pilot = Pilot(device=device)
        self.relmo = relmo_mod.RelMoSidecar()
        self.clips = os.path.join(CFG.artifacts, "clips")
        # ONE continuous kitchen. LIBERO resets between episodes by design, so
        # without this the robot's spatial memory describes a world that has
        # already been rewound — accurate about the past, useless about the
        # present, and depended on by nothing. Carrying the state forward is
        # what makes a belief worth holding.
        self.world_state = None if continuous else False   # False = never carry

    # ---- boot ---------------------------------------------------------------

    def goals(self) -> dict[int, str]:
        """The goal id -> instruction map, read from LIBERO, cached."""
        if not hasattr(self, "_goals"):
            from libero.libero import benchmark

            s = benchmark.get_benchmark_dict()[SCENE_SUITE]()
            self._goals = {i: s.get_task(i).language for i in range(s.n_tasks)}
        return self._goals

    def enter(self, task_id: int) -> None:
        """Stand in the kitchen with a particular goal in force.

        Switching goal switches the BDDL the env is built from, because that is
        what LIBERO's success checker reads. Physically it is the same room —
        the ten goal scenes carry identical objects and near-identical initial
        placements — so this changes what counts as done, not what is there.
        """
        if (self.pilot.suite, self.pilot.task_id) == (SCENE_SUITE, task_id):
            return
        self.pilot.open_scene(SCENE_SUITE, task_id)
        # A rebuilt env starts from the benchmark's placement, so the persistent
        # kitchen is put back before anything looks at it.
        if self.continuous:
            self.pilot.restore(self.world_state)
        self.live.publish_scene(*self.pilot.scene_payload)
        self.live.publish_frame(self.pilot.scene.frame_bytes())

    def boot(self, with_relmo: bool = True) -> None:
        self.live.set_status("booting", "connecting to memory")
        info = self.mem.setup()
        self.live.say("memory", f"{info['flavor']} {info['version']} · "
                                f"native vectors: {info['native_vectors']} · "
                                f"indexes: {', '.join(info['vector_indexes']) or 'none'}")
        # Warm the sentence encoder now: the first recall otherwise pays a 4.5 s
        # lazy model load and the dashboard's headline latency reads as seconds.
        from .memory import embed

        embed.load()

        self.live.set_status("booting", "loading pi0.5 (3.6B, ~40s)")
        self.pilot.load()
        self.live.say("act", "pi0.5 loaded — 98.0% over 400 LIBERO episodes, measured")

        self.live.set_status("booting", "building the kitchen")
        self.enter(SEED_HISTORY[0][0])

        if with_relmo:
            self.live.set_status("booting", "starting RelMo video memory (~85s)")
            if self.relmo.start():
                self.live.say("memory", f"RelMo ready — basis {self.relmo.basis!r} "
                                        f"over {self.relmo.n_basis} recordings")
            else:
                self.live.say("memory", f"RelMo unavailable: {self.relmo.error}")

        self.live.set_status("idle", "waiting for an instruction")

    # ---- one request --------------------------------------------------------

    def handle(self, request: str, use_memory: bool = True,
               goal: int | None = None) -> dict:
        """Resolve, act, remember. The three steps of the whole system.

        `goal` fixes which BDDL predicate judges the episode. The A/B passes the
        goal that the memory-on arm resolved to, so both arms are scored against
        the same definition of having done the thing — otherwise "memory off
        failed" would only mean "it was marked against a different target".
        """
        # A question is answered from memory and nothing is executed. Routing it
        # here rather than into the policy is not a shortcut: "where is the
        # bowl" has no motor answer, and pretending it does would be the exact
        # kind of fake the rest of this avoids.
        if self.resolver.is_question(request):
            return dict(ok=self.where_is(request, use_memory).get("correct", False),
                        question=True)

        self.live.set_status("thinking", "reading memory", request=request,
                             memory_used=use_memory, instruction=None)

        res = self.resolver.resolve(request, use_memory=use_memory)
        self.live.set_status("thinking", res.rationale[:80], request=request,
                             memory_used=use_memory, instruction=res.instruction,
                             resolution=res.to_json())
        if use_memory:
            self.live.say("memory", f'recall "{request}" → '
                                    f'{len(res.recalled)} rows in {res.latency_ms:.1f} ms')

        task_id = self.mem.enqueue(request, origin="user")
        self.mem.decide(chose=res.instruction or "ABSTAIN", rationale=res.rationale,
                        recalled=res.recalled, latency_ms=res.latency_ms, task_id=task_id)

        if res.abstained:
            # Honest failure: the robot has no experience to draw on and says so
            # rather than picking something and calling it a decision.
            self.live.say("fail", f'abstained: {res.rationale}')
            self.mem.finish(task_id, False, "abstained")
            self.live.record(dict(request=request, memory=use_memory, instruction=None,
                                  success=False, seconds=0.0, abstained=True))
            self.live.set_status("idle", "abstained — nothing in memory matches")
            return dict(ok=False, abstained=True, resolution=res.to_json())

        instruction = res.instruction
        goal_id = goal if goal is not None else res.task_id
        if goal_id is None:
            goal_id = self.pilot.task_id
        self.enter(goal_id)

        self.live.say("act", f'executing "{instruction}"')
        self.live.set_status("acting", instruction, request=request,
                             memory_used=use_memory, instruction=instruction,
                             resolution=res.to_json())

        ep = self.pilot.run(instruction,
                            from_state=self.world_state if self.continuous else None,
                            on_frame=lambda buf, step: self.live.publish_frame(buf))
        # The world keeps what the episode did to it — including the mess. In one
        # measured run the robot knocked the bowl off the cabinet while reaching
        # for the bottle, and that stayed knocked off, which is the whole point.
        if self.continuous:
            self.world_state = ep.end_state
            self.pilot.restore(self.world_state)
            self.live.publish_frame(self.pilot.scene.frame_bytes())
        seen = self.observe_after(ep, use_memory)

        if ep.already_satisfied:
            # Only possible in a persistent world, and worth surfacing: the goal
            # held before the robot moved, so the success is the world's, not
            # the policy's, and counting it would inflate every number here.
            self.live.say("memory", "already done before I started — "
                                    "not counting that as work")
        # A goal that already held before the robot moved is not an achievement.
        # Counting it as one is how a persistent world quietly inflates every
        # number: three memory-off beats "succeeded" in 1 second flat, having
        # inherited a kitchen the memory-on arm had already tidied.
        did_work = bool(ep.success and not ep.already_satisfied)
        self.live.say("ok" if ep.success else "fail",
                      f'{"succeeded" if ep.success else "failed"} in {ep.seconds:.0f}s '
                      f'({ep.steps} steps)'
                      + ("  [goal already held at the start]" if ep.already_satisfied else ""))
        # The memory-off arm does not write to memory. Two reasons, and the
        # second one is a bug that would otherwise be invisible: an ablated run
        # has no memory to write to, so writing would not be ablated; and the
        # row it wrote would record the raw phrase ("put the bowl away") as a
        # successful *instruction*, which the next resolution would then recall
        # and rank — the control arm quietly poisoning the treatment arm.
        event_id = None
        if use_memory:
            event_id = self.resolver.learn(request, instruction, ep.success, ep.seconds,
                                           task_id, suite=SCENE_SUITE, goal_id=goal_id)
        self.mem.finish(task_id, ep.success, instruction)
        self.live.record(dict(request=request, memory=use_memory, instruction=instruction,
                              success=did_work, seconds=ep.seconds, abstained=False,
                              already=bool(ep.already_satisfied),
                              goal=self.goals().get(goal_id, "")))

        self.remember_video(ep, event_id, instruction)
        self.live.set_status("idle", "waiting for an instruction",
                             request=request, instruction=instruction,
                             memory_used=use_memory, resolution=res.to_json())
        return dict(ok=did_work, already=bool(ep.already_satisfied),
                    resolution=res.to_json(), episode=ep.to_json())

    # ---- spatial memory -----------------------------------------------------

    def observe_after(self, ep, use_memory: bool = True) -> list:
        """Look at what moved during an episode, and write down where it is now.

        Both states come from the episode itself: `start_obs` is captured after
        the env reset, `final_obs` at termination but before the vector env's
        NEXT_STEP autoreset undoes it. Comparing anything captured outside
        `run()` records "nothing moved" for every successful episode.

        Only what actually moved is recorded, so `last_seen` stays a signal of
        what the robot has been doing rather than a heartbeat.

        With memory off nothing is written — the ablation has to cover *making*
        memories as well as reading them, or the off arm quietly accumulates the
        beliefs that the on arm is supposed to be the only one to have.
        """
        from .world.observe import moved

        changed = moved(ep.start_obs, ep.final_obs)
        if not use_memory:
            return changed
        for o in changed:
            self.mem.see(o.instance, o.label, o.place, o.pos,
                         confidence=1.0 if o.distance < 0.25 else 0.6)
        if changed:
            self.live.say("memory", "observed: " + ", ".join(
                f"{o.label} → {o.place}" for o in changed))
        return changed

    def where_is(self, request: str, use_memory: bool = True) -> dict:
        """Answer a question about the world. No episode runs.

        Scored against the LIVE kitchen, which is only a fair test because the
        kitchen persists: the answer is checked against where the object
        actually is right now, not against a record kept on the side. While
        episodes still reset, this comparison was impossible — the sim after an
        action showed the bowl back at its starting place, and a correct answer
        was marked WRONG.

        What is tested is retention plus revision: the belief was written
        several episodes ago, has to survive them, and has to be *updated* by
        the one that moved the object again.
        """
        if not use_memory:
            ans = dict(answered=False, latency_ms=0.0, correct=False,
                       text="memory is off — the robot has no record of where anything was put")
        else:
            ans = self.resolver.answer_where(request)
            if ans.get("answered"):
                # Checked against the LIVE kitchen, which is meaningful only
                # because the kitchen persists. While episodes reset, the sim
                # after an action showed the bowl back where it started, so the
                # only available answer key was a record kept on the side.
                live = {o.label: o.place for o in self.pilot.look()}
                actual = live.get(ans["label"])
                ans["actual_place"] = actual
                ans["correct"] = bool(actual and actual == ans["place"])
        self.live.say("memory" if ans.get("answered") else "fail", "Q: " + request)
        self.live.say("ok" if ans.get("correct") else "fail", "A: " + ans["text"])
        self.live.record(dict(request=request, memory=use_memory, question=True,
                              instruction=ans.get("text"), success=bool(ans.get("correct")),
                              seconds=0.0, abstained=not ans.get("answered")))
        self.live.set_status("idle", ans["text"], request=request,
                             memory_used=use_memory, instruction=ans["text"])
        return ans

    # ---- video memory -------------------------------------------------------

    def remember_video(self, ep, event_id: str, instruction: str) -> None:
        """Encode what the episode LOOKED like and index it beside the text."""
        if not (self.relmo.ready and ep.frames):
            return
        rec_id = f"brigade-{uuid.uuid4().hex[:10]}"
        path = os.path.join(self.clips, f"{rec_id}.mp4")
        try:
            self.live.set_status("encoding", "RelMo is encoding the episode")
            relmo_mod.write_clip(ep.frames, path, fps=10)
            vec, meta = self.relmo.encode(path, rec_id)
            if vec is None:
                self.live.say("memory", f"RelMo encode failed: {meta.get('error')}")
                return
            relmo_mod.index_recording(
                self.mem.db, rec_id, vec, store="brigade", basis_id=self.relmo.basis,
                kitchen_id=self.mem.kitchen, robot_id=self.mem.robot,
                duration_s=ep.seconds, n_steps=ep.steps, lang=instruction,
                outcome="success" if ep.success else "failure",
                event_id=event_id, clip_path=path,
            )
            hits = relmo_mod.similar_recordings(self.mem.db, rec_id, k=4)
            self.live.say("memory", f"video memory +1 ({meta.get('ms', 0):.0f} ms encode)"
                                    + (f" · closest past clip cos {hits[0]['score']:.3f}"
                                       if hits else ""))
        except Exception as exc:  # noqa: BLE001
            log.warning("video memory failed: %s", exc)
            self.live.say("memory", f"video memory failed: {exc}")

    # ---- the loop -----------------------------------------------------------

    def loop(self) -> None:
        """Main thread forever: drain the command queue, run the robot."""
        while True:
            try:
                cmd = self.live.commands.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.handle(cmd["request"], cmd.get("use_memory", True))
            except Exception as exc:  # noqa: BLE001 — a bad request must not end the shift
                log.exception("request failed")
                self.live.say("fail", f"error: {exc}")
                self.live.set_status("idle", f"error: {exc}")


# ---------------------------------------------------------------- entry points

def seed(bg: Brigade, repeats_scale: int = 1) -> None:
    """Give the robot a history by making it live one.

    Every row this writes is the record of an episode that actually ran. The
    demo would be a lie if the memory were INSERTed directly — the point being
    demonstrated is that experience accumulates, so it has to be experience.
    """
    total = sum(n for _, n in SEED_HISTORY) * repeats_scale
    done = 0
    print(f"\nseeding memory with {total} real episodes "
          f"(~{total * 14 / 60:.0f} min of robot time)\n", flush=True)
    for goal_id, n in SEED_HISTORY:
        instruction = bg.goals()[goal_id]
        bg.enter(goal_id)
        for _ in range(n * repeats_scale):
            done += 1
            print(f"  [{done}/{total}] {instruction} ... ", end="", flush=True)
            bg.live.set_status("acting", f"seeding {done}/{total}", instruction=instruction)
            ep = bg.pilot.run(instruction,
                              on_frame=lambda b, s: bg.live.publish_frame(b))
            seen = bg.observe_after(ep)
            eid = bg.resolver.learn(instruction, instruction, ep.success, ep.seconds,
                                    suite=SCENE_SUITE, goal_id=goal_id)
            bg.remember_video(ep, eid, instruction)
            # Seeding is deliberately NOT continuous: it stands for the robot's
            # past days, so each episode starts fresh and it genuinely performs
            # the task three times. Carrying state here would make repeats
            # trivially satisfied and no norm would ever be learned. The shift
            # that follows continues from wherever seeding left the kitchen.
            if bg.continuous:
                bg.world_state = ep.end_state
            print(f"{'ok' if ep.success else 'FAILED'} in {ep.seconds:.0f}s"
                  + (f"  [saw {', '.join(f'{o.label}→{o.place}' for o in seen)}]" if seen else ""),
                  flush=True)
    print("\nlearned norms:")
    for n in bg.mem.norms():
        tally = ", ".join(f"{e['location']}x{e['n_episodes']}"
                          for e in bg.mem.norm_evidence(n["label"]))
        print(f"  {n['label']:<14} -> {n['home_location']:<20} "
              f"conf {n['confidence']:.2f}   [{tally}]")


def ab(bg: Brigade, request: str, trials: int = 3) -> list:
    """The measurement: the same request, with memory and without.

    Controlled: same kitchen, same policy, same seeds, and — critically — the
    same goal predicate deciding what counts as success, taken from what the
    memory arm resolved to. The only thing that varies is whether the sentence
    handed to pi0.5 came out of the database or straight from the human.
    """
    probe = bg.resolver.resolve(request)
    if probe.abstained:
        print(f'\ncannot run the A/B for "{request}": memory has nothing to '
              f"resolve it to, so there is no goal to score either arm against.")
        return []
    goal_id = probe.task_id
    goal_text = bg.goals().get(goal_id, "?")

    print(f"\n{'=' * 78}\nA/B — \"{request}\"  x{trials} per arm")
    print(f"  goal in force : {goal_text}  (libero_goal/{goal_id})")
    print(f"  memory ON  says: {probe.instruction!r}")
    print(f"  memory OFF says: {request!r}   (the raw words, unchanged)")
    print("=" * 78, flush=True)

    rows = []
    for use_memory in (True, False):
        for i in range(trials):
            r = bg.handle(request, use_memory=use_memory, goal=goal_id)
            instr = (r.get("resolution") or {}).get("instruction")
            secs = (r.get("episode") or {}).get("seconds", 0.0)
            rows.append(dict(memory=use_memory, ok=r["ok"], instruction=instr,
                             seconds=secs, goal=goal_text, request=request))
            print(f"  memory={'ON ' if use_memory else 'OFF'} #{i + 1}  "
                  f"{'SUCCESS' if r['ok'] else 'fail   '}  {secs:>5.0f}s  "
                  f"-> {instr or 'ABSTAINED'}", flush=True)

    print(f"\n{'-' * 78}")
    for tag, flag in (("MEMORY ON ", True), ("MEMORY OFF", False)):
        sel = [r for r in rows if r["memory"] is flag]
        ok = sum(1 for r in sel if r["ok"])
        print(f"  {tag}  {ok}/{len(sel)}   "
              f"{100 * ok / len(sel) if sel else 0:>5.0f}%")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", action="store_true", help="run a history, then serve")
    ap.add_argument("--wipe", action="store_true", help="forget everything first")
    ap.add_argument("--ab", metavar="REQUEST", action="append", default=[],
                    help="run the memory on/off measurement; repeatable")
    ap.add_argument("--results", default="../eval_logs/brigade_results.json")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--no-relmo", action="store_true")
    ap.add_argument("--episodic", action="store_true",
                    help="reset the kitchen between actions (LIBERO's default)")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-16s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    bg = Brigade(device=args.device, continuous=not args.episodic)
    try:
        if not DB.healthy():
            raise MemoryUnavailable(f"cannot reach {DB._safe_dsn()}")
    except MemoryUnavailable as exc:
        print(f"\nmemory is unreachable: {exc}\n"
              f"Brigade will not start without it — that is the design.\n", file=sys.stderr)
        return 2

    if not args.no_serve:
        app = create_app(LIVE, lambda: bg.mem)

        @app.get("/api/prompts")
        def prompts():
            return dict(prompts=PROMPTS)

        @app.get("/api/videomem")
        def videomem():
            n = bg.mem.db.query(
                "SELECT count(*) AS c FROM relmo_recordings WHERE store = 'brigade'")[0]["c"]
            rows = bg.mem.db.query(
                """SELECT recording_id FROM relmo_recordings
                   WHERE store='brigade' ORDER BY ts DESC LIMIT 1""")
            hits = relmo_mod.similar_recordings(bg.mem.db, rows[0]["recording_id"], 4) \
                if rows else []
            return dict(ready=bg.relmo.ready, basis=bg.relmo.basis, n=int(n), hits=hits,
                        note=None if bg.relmo.ready else (bg.relmo.error or "not started"))

        serve_in_thread(app, port=args.port)
        print(f"\n  dashboard  ->  http://127.0.0.1:{args.port}\n", flush=True)

    if args.wipe:
        bg.mem.wipe()
        bg.mem.db.execute("DELETE FROM relmo_recordings")
        print("memory wiped")

    bg.boot(with_relmo=not args.no_relmo)

    if args.seed:
        seed(bg)

    if args.ab:
        import json

        all_rows = []
        for request in args.ab:
            all_rows += ab(bg, request, args.trials)
        if all_rows:
            on = [r for r in all_rows if r["memory"]]
            off = [r for r in all_rows if not r["memory"]]
            print(f"\n{'=' * 78}\nOVERALL   memory ON  "
                  f"{sum(r['ok'] for r in on)}/{len(on)}   |   memory OFF  "
                  f"{sum(r['ok'] for r in off)}/{len(off)}\n{'=' * 78}")
            os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
            json.dump(dict(
                rows=all_rows,
                norms=[dict(n) for n in bg.mem.norms()],
                stats=bg.mem.stats(),
            ), open(args.results, "w"), indent=1, default=str)
            print(f"wrote {args.results}")

    if args.no_serve:
        return 0
    print("\nready — type into the dashboard, or Ctrl-C to stop\n", flush=True)
    try:
        bg.loop()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        bg.relmo.stop()
        bg.pilot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
