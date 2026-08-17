"""Looking at the kitchen: what is here, and where did it end up.

This is what turns an episode into a *belief*. The robot finishes an episode and
asks the world where things are now; the answer is written to `object_beliefs`,
and that row is the only reason it can answer "where did you put the bowl?" a
minute later — or five minutes and three episodes later, when nothing about the
current camera frame would tell you.

Two ideas, both deliberately dumb:

**A thing is something that can move.** Movable objects are the bodies carrying
a free joint. That is MuJoCo's own definition of "not bolted down", so the robot
does not need a list of what a kitchen contains — a house with a samovar in it
gets a samovar in this list the moment one is added to the scene.

**A place is the nearest thing that can be named.** Bodies MuJoCo calls `_main`
are the scene's top-level objects and fixtures; an object's location is
whichever of those it is closest to, excluding itself. So "on the cabinet" is
measured, not asserted, and the same code says "on the table" when the bowl is
back on the table.

The honest limit, stated because it matters for what the demo can claim: this is
read out of mjData rather than out of pixels. It is the position a perfect
perception stack would report. Brigade is a memory system, not a perception one,
and nothing downstream would change if these numbers arrived from a detector —
but they do not, and that is worth knowing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .scene import _raw_data, _raw_model


@dataclass
class Observation:
    instance: str          # 'akita_black_bowl_1'
    label: str             # 'bowl' — what a human would call it
    pos: list[float]
    place: str             # 'wooden cabinet'
    distance: float        # metres to that place; large = "nowhere in particular"

    def to_json(self) -> dict:
        return dict(instance=self.instance, label=self.label, place=self.place,
                    pos=[round(v, 4) for v in self.pos], distance=round(self.distance, 4))


def _name(m, kind, i: int) -> str:
    from mujoco import mj_id2name

    return mj_id2name(m, kind, i) or f"{kind}{i}"


def pretty(body_name: str) -> str:
    """'wooden_cabinet_1_main' -> 'wooden cabinet'. Strips MuJoCo's bookkeeping
    suffixes and instance numbers so what reaches the database and the screen is
    something a person would say."""
    n = re.sub(r"_(main|base|body|root)$", "", body_name)
    n = re.sub(r"_\d+$", "", n)
    return n.replace("_", " ").strip()


def head_label(body_name: str) -> str:
    """'akita_black_bowl_1_main' -> 'bowl'.

    The last real word of the name. Same rule the resolver uses on a human's
    sentence, so a belief written about `akita_black_bowl_1` is found by
    somebody who says "bowl".
    """
    words = [w for w in pretty(body_name).split() if not w.isdigit()]
    return words[-1].lower() if words else pretty(body_name)


def movables(sim) -> list[tuple[int, str]]:
    """Bodies with a free joint — the things that can be picked up or pushed."""
    from mujoco import mjtJoint, mjtObj

    m = _raw_model(sim)
    out = []
    for b in range(m.nbody):
        n = int(m.body_jntnum[b])
        if not n:
            continue
        adr = int(m.body_jntadr[b])
        if any(int(m.jnt_type[adr + i]) == int(mjtJoint.mjJNT_FREE) for i in range(n)):
            out.append((b, _name(m, mjtObj.mjOBJ_BODY, b)))
    return out


def places(sim) -> list[tuple[int, str]]:
    """Nameable locations: the scene's top-level bodies, plus the table.

    Sub-bodies are excluded on purpose. A cabinet publishes `cabinet_top`,
    `cabinet_middle` and `cabinet_bottom` at *identical* positions, so keeping
    them would make the nearest-place answer a coin toss between three names for
    the same spot.
    """
    from mujoco import mjtObj

    m = _raw_model(sim)
    out = []
    for b in range(m.nbody):
        nm = _name(m, mjtObj.mjOBJ_BODY, b)
        if nm.endswith("_main") or nm == "table":
            out.append((b, nm))
    return out


def _half_extent(m, g: int) -> np.ndarray | None:
    """Local half-size of one geom, per axis.

    Deliberately NOT `geom_rbound`: that is a bounding *sphere* radius, so a
    1.7 m-wide tabletop gets a 0.85 m half-height and the table's box reaches
    up to z=1.66. Every support test then fails, because nothing in the kitchen
    is above the table. Measured, and it is why this function exists.
    """
    from .scene import BOX, CAPSULE, CYLINDER, ELLIPSOID, MESH, PLANE, SPHERE

    t, s = int(m.geom_type[g]), np.asarray(m.geom_size[g], dtype=float)
    if t == BOX or t == ELLIPSOID:
        return s[:3].copy()
    if t == SPHERE:
        return np.array([s[0], s[0], s[0]])
    if t in (CAPSULE, CYLINDER):
        return np.array([s[0], s[0], s[1] + (s[0] if t == CAPSULE else 0.0)])
    if t == PLANE:
        return np.array([s[0] if s[0] > 0 else 10.0, s[1] if s[1] > 0 else 10.0, 0.002])
    if t == MESH:
        mid = int(m.geom_dataid[g])
        if mid < 0:
            return None
        v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.asarray(m.mesh_vert[v0:v0 + nv], dtype=float)
        return np.maximum(np.abs(v.max(0)), np.abs(v.min(0)))
    return None


def _extent(sim, body: int) -> tuple[np.ndarray, np.ndarray] | None:
    """World axis-aligned box around every geom of a body's whole subtree.

    A body's *origin* is not its shape: the cabinet's origin sits at its base
    while its usable surface is 20 cm higher, and the stove's origin is its
    control panel rather than the burner. Answering "what is this resting on"
    from origins alone put the bowl on the table when it was on the cabinet.

    The subtree matters too — a cabinet's drawers and top are separate child
    bodies, and the parent on its own bounds almost nothing.

    An oriented box becomes an axis-aligned one the standard way: |R| applied
    to the local half-extent.
    """
    m, d = _raw_model(sim), _raw_data(sim)
    lo, hi = None, None
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        anc, ok = b, (b == body)
        while not ok and anc > 0:
            anc = int(m.body_parentid[anc])
            ok = anc == body
        if not ok:
            continue
        h = _half_extent(m, g)
        if h is None:
            continue
        R = np.abs(np.asarray(d.geom_xmat[g], dtype=float).reshape(3, 3))
        world_h = R @ h
        c = np.asarray(d.geom_xpos[g], dtype=float)
        a, z = c - world_h, c + world_h
        lo = a if lo is None else np.minimum(lo, a)
        hi = z if hi is None else np.maximum(hi, z)
    return None if lo is None else (lo, hi)


def observe(sim) -> list[Observation]:
    """Where is everything, right now.

    "Where" means what the object is resting on or in, which is a support
    question, not a nearest-neighbour one. Nearest-body answers "the bottle is
    at the bowl" whenever four objects share a tabletop — technically the
    closest thing, useless as a place. So a candidate has to be BELOW the object
    and to contain it in plan view, and among those the highest one wins: the
    bowl on the cabinet is over the table too, but the cabinet is what holds it.
    """
    d = _raw_data(sim)
    objs, locs = movables(sim), places(sim)
    xpos = np.asarray(d.xpos)
    boxes = {b: _extent(sim, b) for b, _ in locs}

    out = []
    for b, nm in objs:
        p = np.asarray(xpos[b], dtype=float)
        best, best_area, best_gap = None, float("inf"), float("inf")
        for lb, lname in locs:
            if lb == b or boxes.get(lb) is None:
                continue
            lo, hi = boxes[lb]
            # how far outside this thing's footprint the object is, in plan view
            dx = max(lo[0] - p[0], 0.0, p[0] - hi[0])
            dy = max(lo[1] - p[1], 0.0, p[1] - hi[1])
            gap = float(np.hypot(dx, dy))
            if gap > 0.02:
                continue                       # not over it at all
            # Either resting on top of it, or down inside it. The second case is
            # not a detail: a bottle in a wine rack sits BELOW the rack's top, so
            # a pure "is it above" test calls the rack no place at all and the
            # answer falls through to the table.
            on_top = -0.02 <= p[2] - hi[2] <= 0.15
            inside = lo[2] - 0.02 <= p[2] <= hi[2] + 0.02
            if not (on_top or inside):
                continue
            # Prefer the most SPECIFIC place. Everything in the room is over the
            # table, so without this the answer is always "the table"; the
            # smallest footprint containing the object is the one worth naming.
            area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
            if area < best_area:
                best, best_area, best_gap = lname, area, gap
        if best is None:
            # nothing holds it — in the gripper, or somewhere with no word for it
            best, best_gap = "nowhere in particular", float("inf")
        out.append(Observation(
            instance=re.sub(r"_main$", "", nm), label=head_label(nm),
            pos=[float(v) for v in p],
            place=pretty(best) if best != "nowhere in particular" else best,
            distance=best_gap,
        ))
    return out


def moved(before: list[Observation], after: list[Observation],
          tol: float = 0.02) -> list[Observation]:
    """Which objects actually changed position during the episode.

    Only these are worth writing: re-asserting that the untouched plate is still
    where it was costs a row and buys nothing, and it would make `last_seen`
    meaningless as a signal of what the robot has been doing.
    """
    prev = {o.instance: np.asarray(o.pos) for o in before}
    return [o for o in after
            if o.instance in prev and float(np.linalg.norm(np.asarray(o.pos) - prev[o.instance])) > tol]
