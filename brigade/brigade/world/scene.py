"""The MuJoCo scene, as something a browser can draw.

The view in the dashboard is not a video of the simulator. It is the simulator's
own geometry, drawn by the client, from the same arrays MuJoCo integrates. That
distinction is the whole reason this module exists:

  * a rendered stream would prove only that a renderer ran;
  * shipping `geom_xpos`/`geom_xmat` proves the physics moved, because those
    arrays are the output of `mj_step` and nothing else can write them.

It also happens to be far cheaper. Measured on LIBERO's kitchen scene: 239 geoms
of which 86 are visual, a 0.5 MB one-time mesh payload, and **7 KB per frame**.
No OpenGL context is involved at any point here, which matters on macOS where a
GL context is pinned to the main thread — the transform stream can therefore be
read from the web thread while the sim renders policy observations on the main
one.

The one trap, and it is not guessable: in robosuite/LIBERO **group 1 is the
visual mesh and group 0 is the collision hull** — the opposite of the MuJoCo
convention most references describe. Drawing group 0 gets you a scene of grey
capsules covering the furniture. The names give it away (`table_collision`,
`robot0_link0_collision`), which is how it was caught.
"""

from __future__ import annotations

import numpy as np

# mjtGeom. Only the ones LIBERO scenes actually contain are handled; anything
# else is dropped loudly rather than drawn wrong.
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = range(8)
TYPE_NAME = {PLANE: "plane", SPHERE: "sphere", CAPSULE: "capsule",
             ELLIPSOID: "ellipsoid", CYLINDER: "cylinder", BOX: "box", MESH: "mesh"}

VISUAL_GROUP = 1


def _raw_model(sim):
    """robosuite wraps MjModel; reach the binding object underneath."""
    return getattr(sim.model, "_model", sim.model)


def _raw_data(sim):
    """As above for MjData.

    Deliberately not one shared helper keyed on "whichever attribute exists":
    robosuite's data wrapper carries a `_model` attribute *as well*, so a
    first-match-wins lookup silently hands back the model and every transform
    read fails later with a confusing AttributeError on `geom_xpos`.
    """
    return getattr(sim.data, "_data", sim.data)


class SceneExporter:
    """Turns one MuJoCo model into a static scene + a per-step transform feed.

    Model and data are looked up through the sim on EVERY access, never cached.
    robosuite rebuilds the simulation on reset — `env.reset()` can construct a
    fresh MjSim, and with it fresh MjModel and MjData objects — so a reference
    grabbed in `__init__` goes stale the first time an episode starts. It does
    not raise: the old MjData is still a perfectly valid object frozen at the
    moment it was orphaned, so `frame_bytes()` keeps returning well-formed
    frames of a world that stopped existing.

    That failure is invisible from every direction. The socket is up, frames
    arrive at 14/s, the geometry is right, the numbers are plausible — and
    nothing on screen ever moves. It was caught by comparing one geom's local
    matrix across two frames and finding them byte-identical while the robot was
    demonstrably mid-episode.
    """

    def __init__(self, sim_or_getter):
        # A callable, because the sim OBJECT is replaced too, not just its data:
        # robosuite's reset path is _destroy_sim -> _load_model -> _initialize_sim,
        # which constructs a new MjSim. Holding the sim itself would go stale in
        # exactly the same silent way as holding its MjData.
        self._get = sim_or_getter if callable(sim_or_getter) else (lambda: sim_or_getter)
        self.geoms = self._visual_geoms()

    @property
    def sim(self):
        return self._get()

    @property
    def m(self):
        return _raw_model(self.sim)

    @property
    def d(self):
        return _raw_data(self.sim)

    # ---- static ------------------------------------------------------------

    def _visual_geoms(self) -> np.ndarray:
        """Visual geoms, in draw order. Falls back to 'everything' if a scene
        has no group-1 geoms at all, so an unusual model degrades to an ugly
        picture rather than an empty one."""
        vis = np.where(self.m.geom_group == VISUAL_GROUP)[0]
        return vis if len(vis) else np.arange(self.m.ngeom)

    def scene(self, task: str = "") -> tuple[dict, bytes]:
        """The one-time payload: a small JSON header plus one binary blob.

        Returns (header, blob). Meshes are NOT in the header — serialising
        LIBERO's kitchen meshes as JSON numbers measured **134 MB**, because a
        float32 costs four bytes and its decimal spelling costs about twenty.
        The same geometry packed as raw float32/uint32 is ~5 MB, so vertices
        travel as bytes and the header carries only offsets into them.

        Layout of `blob`, mesh by mesh in header order:
            verts    float32[nv*3]
            normals  float32[nv*3]
            faces    uint32 [nf*3]
        """
        m = self.m
        mesh_ids = sorted({int(m.geom_dataid[i]) for i in self.geoms
                           if m.geom_type[i] == MESH and m.geom_dataid[i] >= 0})
        remap = {mid: k for k, mid in enumerate(mesh_ids)}

        chunks: list[bytes] = []
        meshes: list[dict] = []
        off = 0
        for mid in mesh_ids:
            v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            f0, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            verts = np.ascontiguousarray(m.mesh_vert[v0:v0 + nv], dtype=np.float32)
            # mesh_face indices are LOCAL to each mesh (0-based within it), so
            # they are shipped unmodified and the client offsets nothing.
            faces = np.ascontiguousarray(m.mesh_face[f0:f0 + nf], dtype=np.uint32)
            normals = np.ascontiguousarray(m.mesh_normal[v0:v0 + nv], dtype=np.float32)

            rec = dict(nv=nv, nf=nf, verts=off)
            chunks.append(verts.tobytes()); off += verts.nbytes
            rec["normals"] = off
            chunks.append(normals.tobytes()); off += normals.nbytes
            rec["faces"] = off
            chunks.append(faces.tobytes()); off += faces.nbytes
            meshes.append(rec)

        geoms = []
        for i in self.geoms:
            i = int(i)
            gt = int(m.geom_type[i])
            geoms.append(dict(
                name=self._name(i),
                type=TYPE_NAME.get(gt, "box"),
                size=[round(float(x), 5) for x in m.geom_size[i]],
                rgba=[round(float(x), 4) for x in self._rgba(i)],
                mesh=remap.get(int(m.geom_dataid[i]), -1) if gt == MESH else -1,
                body=self._body_name(int(m.geom_bodyid[i])),
            ))

        header = dict(task=task, meshes=meshes, geoms=geoms,
                      n_geoms=len(geoms), n_meshes=len(meshes), blob_bytes=off)
        return header, b"".join(chunks)

    def _rgba(self, i: int) -> list[float]:
        """Colour, preferring the material over the geom's own rgba.

        MuJoCo's own precedence: a geom with a material takes the material's
        colour, and its `geom_rgba` is usually left at the default white that
        would make the whole kitchen look like polystyrene.

        The twist that actually decides how this looks: **28 of the 86 visual
        geoms carry a textured material whose `mat_rgba` is (1,1,1,1)** — the
        colour is in the texture, not the material, so honouring mat_rgba alone
        renders the floor, every wall and most of the furniture bone white. The
        textures themselves total 142 MB in this one scene and are not going
        over a websocket, so each is collapsed to its mean colour. That is
        frankly an approximation — a chequerboard becomes grey — but it puts
        the wood, tile and steel back in roughly the right hue for a fraction
        of a megabyte.
        """
        mid = int(self.m.geom_matid[i])
        if mid < 0:
            return np.asarray(self.m.geom_rgba[i], dtype=float).tolist()
        rgba = np.asarray(self.m.mat_rgba[mid], dtype=float).copy()
        tint = self._texture_tint(mid)
        if tint is not None:
            rgba[:3] *= tint
        return rgba.tolist()

    def _texture_tint(self, matid: int) -> np.ndarray | None:
        """Mean RGB of a material's texture, in [0,1]. None if untextured."""
        if not hasattr(self, "_tint_cache"):
            self._tint_cache: dict[int, np.ndarray | None] = {}
        if matid in self._tint_cache:
            return self._tint_cache[matid]

        m = self.m
        # mat_texid is (nmat, mjNTEXROLE) in current MuJoCo — a texture per
        # role, not a single id. The first non-negative role is the one that
        # carries the surface colour.
        roles = np.asarray(m.mat_texid[matid]).ravel()
        cand = roles[roles >= 0]
        if not len(cand):
            self._tint_cache[matid] = None
            return None

        tid = int(cand[0])
        adr = int(m.tex_adr[tid])
        n = int(m.tex_width[tid]) * int(m.tex_height[tid]) * int(m.tex_nchannel[tid])
        chan = int(m.tex_nchannel[tid])
        data = np.asarray(m.tex_data[adr:adr + n], dtype=np.uint8)
        if data.size < chan:
            self._tint_cache[matid] = None
            return None
        # Stride-sample: a 3072x512 texture is 4.7 M bytes and the mean of every
        # 64th texel is the same colour for this purpose.
        px = data[: (data.size // chan) * chan].reshape(-1, chan)[::64]
        tint = (px[:, :3].mean(axis=0) / 255.0).astype(float)
        self._tint_cache[matid] = tint
        return tint

    def _name(self, i: int) -> str:
        from mujoco import mj_id2name, mjtObj
        return mj_id2name(self.m, mjtObj.mjOBJ_GEOM, i) or f"geom{i}"

    def _body_name(self, b: int) -> str:
        from mujoco import mj_id2name, mjtObj
        return mj_id2name(self.m, mjtObj.mjOBJ_BODY, b) or f"body{b}"

    # ---- per step ----------------------------------------------------------

    def transforms(self) -> np.ndarray:
        """(n_visual, 12) float32: xyz followed by a row-major 3x3.

        Read straight out of mjData with no copy of anything else. This is the
        entire per-frame message — ~7 KB for LIBERO's kitchen.
        """
        pos = np.asarray(self.d.geom_xpos[self.geoms], dtype=np.float32)
        mat = np.asarray(self.d.geom_xmat[self.geoms], dtype=np.float32).reshape(-1, 9)
        return np.hstack([pos, mat])

    def frame_bytes(self) -> bytes:
        return self.transforms().tobytes()
