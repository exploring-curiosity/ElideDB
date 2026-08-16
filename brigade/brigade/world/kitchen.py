"""BrigadeKitchen — one persistent kitchen the agent lives in.

Every RoboCasa task ships as its own env class that builds a scene, runs one
episode, and is thrown away. Brigade needs the opposite: a single kitchen that
stays up for hours while many different tasks get done in it. So we subclass the
base `Kitchen` env (which is registered and instantiable on its own) and give it
a fixed pool of objects instead of a task.

Two consequences worth stating, because they are the whole point:

  * The kitchen is PINNED (layout, style, seed). The same kitchen every boot is
    what makes "it remembers where the bowls are" mean anything — a randomised
    scene would make memory worthless by construction.
  * There is no `_check_success`. Brigade has no episode to succeed at. Outcomes
    are judged per-skill, against the world, and written to memory.

The object pool is declarative (POOL below) so the demo's cast is one edit away.
Two bowls start inside a cabinet: that is the "existing stack" the agent has to
discover for itself in act 1 before it can learn where bowls belong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType
from robocasa.utils import object_utils as OU


@dataclass(frozen=True)
class PoolItem:
    """One object that exists in the kitchen for the whole run.

    `home` is only the *initial* placement. Where the object actually is at any
    moment is a question for perception and memory, never for this table.
    """

    instance_id: str
    category: str
    where: str  # "cab" | "stage" | "counter"
    # RoboCasa tags assets as graspable or not, and asking for a graspable
    # variant of a category that has none (plate, for one) filters the candidate
    # set to empty and the scene fails to build. So this is per-item: demand it
    # only for objects the robot is actually expected to pick up.
    graspable: bool = True


# The cast. Bowls dominate because the demo's through-line is "where do bowls
# belong" — the agent must learn it, be corrected on it, and act on it.
#
# Sizing note: layout 1 offers ~1.5 m² of counter across two fixtures and ~0.3 m²
# per cabinet shelf. Placement is rejection-sampled and gives up after 50 tries,
# so the cast is spread deliberately across four fixtures rather than piled onto
# the one counter the robot happens to face.
POOL: tuple[PoolItem, ...] = (
    # Pre-existing stack, inside the home cabinet, unseen until the agent opens
    # it. Act 1 depends on these being discoverable but not visible.
    PoolItem("bowl_a", "bowl", "cab"),
    PoolItem("bowl_b", "bowl", "cab"),
    # The bowls that get "dropped" on the counter during the demo. They wait in
    # a different, closed cabinet so their arrival is genuinely an arrival.
    PoolItem("bowl_c", "bowl", "stage"),
    PoolItem("bowl_d", "bowl", "stage"),
    # Tea cast, on the counter where a human would leave them.
    PoolItem("mug_a", "mug", "counter"),
    PoolItem("kettle_a", "kettle_non_electric", "counter"),
    # A distractor, so perception has to discriminate rather than "the only
    # thing on the counter is the answer".
    PoolItem("can_a", "can", "counter"),
)


class BrigadeKitchen(Kitchen):
    """A kitchen with a cast of objects and no task."""

    # ---- scene construction -------------------------------------------------

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # The cabinet the agent will learn to use as the bowls' home. Registering
        # it as a ref (rather than looking it up ad hoc) keeps it stable across
        # resets and puts it in ep_meta for provenance.
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        # A second cabinet, distinct from the home cabinet, to hold the objects
        # that get dropped into play later. Chosen by capacity rather than name
        # so the pool still places if the layout changes.
        self.stage_cab = self._pick_stage_cabinet()
        # Where the robot starts. The counter, so the opening shot has the
        # workspace in frame.
        self.init_robot_base_ref = self.counter

    def _pick_stage_cabinet(self):
        """The roomiest cabinet that is not the bowls' home."""
        best, best_area = None, 0.0
        for name, fxtr in self.fixtures.items():
            if not name.startswith("cab_") or name == self.cab.name:
                continue
            try:
                regions = fxtr.get_reset_regions(env=self)
            except Exception:
                continue
            area = sum(r["size"][0] * r["size"][1] for r in regions.values())
            if area > best_area:
                best, best_area = fxtr, area
        # If the layout has only one usable cabinet, staging shares it. The demo
        # is weaker (the dropped bowls start beside the stack) but nothing breaks.
        return best if best is not None else self.cab

    def _get_obj_cfgs(self):
        # Window sizes are tuned against the measured extents of this layout
        # (cabinet shelves ~0.94 x 0.34, counter regions ~0.54-1.04 x 0.65) and
        # carry no offset — an offset pushes the sampling window off the surface
        # and then every attempt fails.
        #
        # Size these generously. Placement is rejection-sampled and each failure
        # rebuilds the entire model, so a window that is merely *sufficient*
        # (two bowls in 0.50 x 0.30 needed ~42 retries) turns boot into a minute
        # of silent thrashing. Roomy windows are the difference between a 3 s
        # boot and a 40 s one.
        placements = {
            "cab": lambda: dict(fixture=self.cab, size=(0.85, 0.30), pos=(None, 1.0)),
            "stage": lambda: dict(fixture=self.stage_cab, size=(0.85, 0.30), pos=(None, 1.0)),
            "counter": lambda: dict(
                fixture=self.counter,
                sample_region_kwargs=dict(ref=self.cab),
                size=(0.50, 0.40),
                pos=("ref", -1.0),
            ),
        }
        return [
            dict(
                name=item.instance_id,
                obj_groups=item.category,
                graspable=item.graspable,
                placement=placements[item.where](),
            )
            for item in POOL
        ]

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "Keep the kitchen in order."
        meta["brigade_pool"] = [
            dict(instance_id=i.instance_id, category=i.category) for i in POOL
        ]
        meta["brigade_fixtures"] = dict(
            home_cabinet=self.cab.name,
            stage_cabinet=self.stage_cab.name,
            counter=self.counter.name,
            sink=self.sink.name,
            stove=self.stove.name,
        )
        return meta

    def _check_success(self):
        # Brigade has no episode and therefore nothing to succeed at. Outcomes
        # are per-skill and are judged by world.skills against the world state.
        return False

    # ---- read the world -----------------------------------------------------
    #
    # Everything below is how perception and memory ground themselves. Note that
    # none of it is used to DECIDE anything — the agent decides from camera
    # images and from memory. These are used to (a) turn "grasp that bowl" into a
    # gripper pose, which is what a real robot's depth sensor does, and (b) score
    # whether a skill actually worked.

    def object_pose(self, instance_id: str) -> np.ndarray:
        """World-frame xyz of an object in the pool."""
        return np.array(self.sim.data.body_xpos[self.obj_body_id[instance_id]])

    def object_label(self, instance_id: str) -> str:
        """Human-readable category, e.g. 'bowl'. Comes from RoboCasa's own
        asset metadata, not from a hand-written map."""
        return self.get_obj_lang(instance_id)

    def storage_fixtures(self) -> dict[str, object]:
        """Fixtures an object can be *inside* — the candidates for `locate`."""
        out = {}
        for name, fxtr in self.fixtures.items():
            if any(k in name for k in ("cab_", "drawer", "fridge", "microwave", "oven")):
                out[name] = fxtr
        return out

    def locate(self, instance_id: str) -> str:
        """Which fixture is this object in or on?"""
        return self.locate_detail(instance_id)["location"]

    def locate_detail(self, instance_id: str) -> dict:
        """Locate an object, and say how the answer was reached.

        Four layers, most specific first. The layering exists because a single
        test is not enough: requiring physical *contact* with a counter geom
        reported 'unknown' for roughly a third of drops that were visibly sitting
        on the counter — the object had come to rest on a sink lip or a raised
        edge, which is a counter to any human and not a counter geom to MuJoCo.
        Spatial memory is written from this, so a brittle answer here becomes a
        robot that cannot find a bowl it is looking straight at.

        `method` is returned so the memory layer can set confidence from it, and
        so 'unknown' is distinguishable from 'guessed'.
        """
        pos = self.object_pose(instance_id)

        # 1. Inside a container. Wins outright: a bowl in an open cabinet is in
        #    the cabinet, not on the shelf it happens to touch.
        for name, fxtr in self.storage_fixtures().items():
            try:
                if OU.obj_inside_of(self, instance_id, fxtr):
                    return dict(location=name, method="containment", confidence=1.0)
            except Exception:
                continue

        surfaces = {n: f for n, f in self.fixtures.items() if "counter" in n or "island" in n}

        # 2. Physically resting on a counter. Most precise when it fires.
        try:
            if OU.check_obj_any_counter_contact(self, instance_id) and surfaces:
                nearest = min(
                    surfaces,
                    key=lambda n: float(np.linalg.norm(np.array(surfaces[n].pos)[:2] - pos[:2])),
                )
                return dict(location=nearest, method="contact", confidence=1.0)
        except Exception:
            pass

        # 3. Standing within a surface's footprint, whatever it is resting on.
        for name, fxtr in surfaces.items():
            try:
                if OU.point_in_fixture(pos, fxtr, only_2d=True):
                    return dict(location=name, method="footprint", confidence=0.8)
            except Exception:
                continue

        # 4. Near something, at counter height. Reported at low confidence, which
        #    is the memory layer's cue to re-verify by looking before trusting it.
        if surfaces:
            nearest = min(
                surfaces,
                key=lambda n: float(np.linalg.norm(np.array(surfaces[n].pos)[:2] - pos[:2])),
            )
            gap = float(np.linalg.norm(np.array(surfaces[nearest].pos)[:2] - pos[:2]))
            if gap < 1.5:
                return dict(location=nearest, method="proximity", confidence=0.5)

        return dict(location="unknown", method="none", confidence=0.0)

    def placement_targets(self) -> list[dict]:
        """Fixtures an object can be put on or in.

        Used by the console's drop/move controls and, later, by the agent when it
        has to choose somewhere to put a thing down. Reports whether each target
        is a container (has a door) so the caller knows an open is required.
        """
        out = []
        for name, fxtr in self.fixtures.items():
            is_store = any(
                k in name for k in ("cab_", "drawer", "fridge", "microwave", "oven")
            )
            is_surface = "counter" in name or "island" in name
            if not (is_store or is_surface):
                continue
            try:
                regions = fxtr.get_reset_regions(env=self)
            except TypeError:
                try:
                    regions = fxtr.get_reset_regions(self)
                except Exception:
                    regions = {}
            except Exception:
                regions = {}
            if not regions:
                continue  # nowhere to actually put anything
            out.append(
                dict(
                    name=name,
                    kind="container" if is_store else "surface",
                    openable=hasattr(fxtr, "open_door"),
                    is_open=self.fixture_is_open(name),
                )
            )
        return sorted(out, key=lambda d: (d["kind"], d["name"]))

    def fixture_is_open(self, fixture_name: str) -> bool | None:
        """True/False for openable fixtures, None for ones with no door."""
        fxtr = self.fixtures.get(fixture_name)
        if fxtr is None or not hasattr(fxtr, "is_open"):
            return None
        try:
            return bool(fxtr.is_open(self))
        except Exception:
            return None

    def world_snapshot(self) -> dict:
        """Everything true about the world right now.

        Used for skill verification and for the dashboard's ground-truth column
        (which exists precisely so a viewer can see where memory and reality
        disagree). The agent does not read this to make decisions.
        """
        objects = {}
        for item in POOL:
            try:
                where = self.locate_detail(item.instance_id)
                objects[item.instance_id] = dict(
                    label=self.object_label(item.instance_id),
                    pos=self.object_pose(item.instance_id).tolist(),
                    location=where["location"],
                    located_by=where["method"],
                    location_confidence=where["confidence"],
                    grasped=bool(OU.check_obj_grasped(self, item.instance_id)),
                )
            except Exception as exc:  # an object can be mid-teleport
                objects[item.instance_id] = dict(error=str(exc))
        doors = {
            n: self.fixture_is_open(n)
            for n in self.storage_fixtures()
            if self.fixture_is_open(n) is not None
        }
        return dict(objects=objects, doors=doors)
