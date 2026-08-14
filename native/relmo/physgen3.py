"""Physics generator v3 - v2 dynamics, TEXTURED surfaces.

One variable changed, because trackval measured exactly one failure:
CoTracker3 on v2 renders scores 0.4px EPE on the CHECKERED floor and
5.0px median / 75% jump-rate on the flat-colored objects. A point
tracker needs local appearance to lock onto; v2's uniform-rgba
primitives give it nothing (the aperture problem, literally staged).
Real objects have texture, so v2 was not just hard - it was HARDER
than reality, which is the wrong side of the sim-to-real gap.

v3 gives every object and every driver a randomized speckle/checker
texture (MuJoCo builtin textures with random marks). Same dynamics,
same alphabet, same camera contract - pixelgt reconstructs v3 scenes
by gen_version dispatch.

Measured before any tracker finetuning: if texture alone fixes the
tracker on sim, finetuning starts from a sane baseline instead of
compensating for an artificially untrackable world.
"""
from __future__ import annotations

import numpy as np

from relmo import physgen2 as P2

GEN_VERSION = 3
FPS = P2.FPS
SEG_DS = P2.SEG_DS
SHAPES = P2.SHAPES
DRIVERS = P2.DRIVERS


def _tex(name, rng):
    """A randomized trackable texture + material pair.

    Always checker (the only builtin with dense local structure -
    'flat' and 'gradient' render as uniform color, verified on a
    frame), contrast enforced, plus max-contrast random marks. The
    checkered FLOOR is where CoTracker scored 0.4px; this gives the
    objects the same property with per-object randomized appearance."""
    c1 = rng.uniform(0.15, 0.85, 3)
    sgn = np.where(c1 > 0.5, -1.0, 1.0)
    c2 = np.clip(c1 + sgn * rng.uniform(0.25, 0.5, 3), 0.02, 0.98)
    mk = np.clip(1.0 - c1, 0.0, 1.0)
    # texuniform repeats PER METER and objects are ~5-9cm, so the
    # repeat must be large to put several cells across an object;
    # random="0.12" densifies the speckle (default 0.01 is invisible)
    rep = int(rng.integers(25, 80))
    tex = (f'<texture name="t_{name}" type="2d" builtin="checker" '
           f'rgb1="{c1[0]:.2f} {c1[1]:.2f} {c1[2]:.2f}" '
           f'rgb2="{c2[0]:.2f} {c2[1]:.2f} {c2[2]:.2f}" '
           f'mark="random" random="0.12" '
           f'markrgb="{mk[0]:.2f} {mk[1]:.2f} {mk[2]:.2f}" '
           f'width="128" height="128"/>')
    mat = (f'<material name="m_{name}" texture="t_{name}" '
           f'texrepeat="{rep} {rep}" texuniform="true" '
           f'specular="{rng.uniform(0.05, 0.3):.2f}" '
           f'shininess="{rng.uniform(0.05, 0.4):.2f}"/>')
    return tex, mat


def scene(rng):
    """v2 scene, then swap flat rgba for randomized textures.

    IMPORTANT: consume the texture draws from a rng FORKED off the
    main stream, so the main stream's draw count stays identical to
    v2 - scene() must leave rng exactly where v2's scene() would, or
    the camera reconstruction contract in pixelgt breaks."""
    xml, plan, launch, driver = P2.scene(rng)
    trng = np.random.default_rng(rng.integers(2 ** 31))
    # after the fork the main stream has advanced by exactly ONE draw
    # regardless of how many objects exist - pixelgt accounts for it
    texs, mats = [], []
    import re
    names = re.findall(r'<geom name="g_(o\d+)"', xml)
    for nm in names + ["drv"]:
        t, m_ = _tex(nm, trng)
        texs.append(t)
        mats.append(m_)
    xml = xml.replace('<material name="gridm"',
                      "".join(texs) + "".join(mats)
                      + '<material name="gridm"')
    for nm in names:
        xml = re.sub(
            (rf'(<geom name="g_{nm}"[^>]*?) '
             rf'rgba="[\d.]+ [\d.]+ [\d.]+ 1"'),
            rf'\1 material="m_{nm}"', xml)
    # drivers: their geom has no name; retexture by rgba signature
    xml = re.sub(r'rgba="0\.75 0\.25 0\.2 1"', 'material="m_drv"', xml)
    xml = re.sub(r'rgba="0\.2 0\.55 0\.75 1"', 'material="m_drv"', xml)
    xml = re.sub(r'rgba="0\.8 0\.6 0\.2 1"', 'material="m_drv"', xml)
    xml = re.sub(r'rgba="0\.6 0\.3 0\.7 1"', 'material="m_drv"', xml)
    return xml, plan, launch, driver


def run_episode(seed, out_dir, seconds=3.6):
    return P2.run_episode(seed, out_dir, seconds, scene_fn=scene,
                          gen_version=GEN_VERSION)
