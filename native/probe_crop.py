"""Does spatially-local pooling actually help `crop`? Answer before building.

`crop` is the weakest transform in the battery and it degrades fastest
with corpus size (0.734 -> 0.638 between A/B and full scale). The stated
reason is that GeM pools the WHOLE frame into one vector, so removing a
third of the field of view shifts every frame vector, and rank pooling
faithfully preserves that shift. R-MAC is the standard answer.

Running that as a store A/B costs ~4.5 GPU-hours. This costs ten
minutes, and it measures the quantity that A/B would measure a proxy of:
in the c2 space that actually ranks results, how much closer does a
cropped clip stay to ITSELF than to other clips?

    margin = cos(crop(a), a) - mean_b!=a cos(crop(a), b)

Stability alone is not the goal - a constant vector is perfectly stable
and useless - so the cross term is what makes this a retrieval question.

Selection honesty: the clips come from HELD-OUT media (SDX_SKIP past
everything the A/Bs used), so whichever pooling wins here was not chosen
on the media the final grade reports.

Four candidates:
    gem       incumbent, whole-frame GeM
    rmac2     uniform 2x2 grid, pooled -> L2 -> summed
    rmac3     uniform 3x3 grid
    trueR3    R-MAC as published (Tolias et al. 2016): square regions at
              scales L=1..3 with ~40% overlap, which is the part the
              uniform grid drops. A non-overlapping grid lets content
              slide across a cell boundary under a crop, which is the
              exact failure the overlap exists to absorb.
"""
import sys
import numpy as np

sys.path.insert(0, "native")
import vcore                                              # noqa: E402
import vnuis                                              # noqa: E402
import vsrc                                               # noqa: E402

CLIP_S = 8.0            # same span the benchmark hands the system
PER_CORPUS = 2
SKIP = {"sim": 24, "bridge": 15, "car": 0, "drone": 12}   # past the A/B


def true_rmac(P, torch, L):
    """R-MAC as published: square regions at L scales, ~40% overlap."""
    B, N, D = P.shape
    side = int(round(float(N) ** 0.5))
    if side * side != N:
        return None
    G = P.reshape(B, side, side, D)
    out = None
    for l in range(1, L + 1):
        rs = int(round(2.0 * side / (l + 1)))       # region side
        if rs < 1:
            continue
        n = l + 1                                    # regions per axis
        if n > 1:
            step = max((side - rs) / (n - 1), 0.0)
        else:
            step = 0.0
        ys = sorted({min(int(round(i * step)), side - rs) for i in range(n)})
        for y0 in ys:
            for x0 in ys:
                R = G[:, y0:y0 + rs, x0:x0 + rs, :].reshape(B, -1, D)
                g = (R.float().clamp(min=1e-6).pow(vcore.GEM_P)
                     .mean(1).pow(1.0 / vcore.GEM_P))
                g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                out = g if out is None else out + g
    return out


def clips():
    """(id, frames) from media no A/B ever saw."""
    out = []
    for name in ("sim", "bridge", "car", "drone"):
        got = 0
        for s in vsrc.sources(name, skip=SKIP[name]):
            if s.dur < CLIP_S + 6.0:
                continue
            t0 = round((s.dur - CLIP_S) / 2.0, 1)
            F = s.cut(t0, t0 + CLIP_S)
            if len(F) < 8:
                continue
            out.append((s.id, F))
            got += 1
            if got >= PER_CORPUS:
                break
    return out


def c2_of(F):
    """The channel that actually ranks: rank-pooled, scene-basis removed."""
    wins, e = vcore.encode_clip(F)
    if wins is None or not e["valid"].any():
        return None
    return e["c2"][e["valid"]]


def setcos(A, B):
    """Set-to-set similarity, exactly how vqbe.score reads a query: max
    over the query's sub-windows, averaged over the candidate's."""
    if A is None or B is None or not len(A) or not len(B):
        return np.nan
    return float((A @ B.T).max(1).mean())


def run(mode, cl, orig_rmac):
    if mode == "trueR3":
        vcore.POOL, vcore._rmac = "rmac3", (
            lambda P, t, g: true_rmac(P, t, 3))
    else:
        vcore.POOL, vcore._rmac = mode, orig_rmac
    vcore.POOL_FALLBACK.clear()

    ident, crop = [], []
    for _mid, F in cl:
        ident.append(c2_of(F))
        crop.append(c2_of(vnuis.crop(F, np.random.RandomState(7))))

    n = len(cl)
    self_id = [setcos(ident[i], ident[i]) for i in range(n)]
    self_cr = [setcos(crop[i], ident[i]) for i in range(n)]
    cross = [setcos(crop[i], ident[j])
             for i in range(n) for j in range(n) if i != j]
    # rank-1: is the cropped clip closest to its OWN original?
    hit = sum(1 for i in range(n)
              if setcos(crop[i], ident[i])
              > max(setcos(crop[i], ident[j]) for j in range(n) if j != i))
    d = vcore.feature_dim()
    fb = "  FELL BACK TO GEM" if vcore.POOL_FALLBACK else ""
    print(f"  {mode:<8}{d:<7}{np.nanmean(self_id):<9.3f}"
          f"{np.nanmean(self_cr):<9.3f}{np.nanmean(cross):<9.3f}"
          f"{np.nanmean(self_cr) - np.nanmean(cross):<9.3f}{hit}/{n}{fb}")
    return float(np.nanmean(self_cr) - np.nanmean(cross)), hit


def main():
    orig_rmac = vcore._rmac
    cl = clips()
    print(f"{len(cl)} held-out clips of {CLIP_S:.0f}s "
          f"({', '.join(sorted({c.split('/')[0] for c, _ in cl}))})")
    print(f"encoder {vcore.ENCODER.split('/')[-1]}  res {vcore.RES}\n")
    print(f"  {'pool':<8}{'dim':<7}{'self':<9}{'crop':<9}{'cross':<9}"
          f"{'margin':<9}rank1")
    res = {}
    for mode in ("gem", "rmac2", "rmac3", "trueR3"):
        # dim is cached from the first model call; R-MAC keeps the same
        # width (regions are summed, not concatenated) so this is safe,
        # but assert it rather than trust it.
        res[mode] = run(mode, cl, orig_rmac)
    base = res["gem"][0]
    print(f"\n  margin vs gem ({base:.3f}):")
    for m, (v, h) in res.items():
        if m == "gem":
            continue
        print(f"    {m:<8}{v - base:+.3f}"
              f"   {'BETTER' if v > base else 'worse'}")


if __name__ == "__main__":
    main()
