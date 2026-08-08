"""gem vs true R-MAC across the WHOLE battery, on held-out media.

The first probe asked one question - crop - on 8 clips, and answered it:
the uniform-grid R-MAC that was queued for a 4.5-hour store A/B is WORSE
than the incumbent (margin 0.477/0.478 vs 0.494), while R-MAC as
published is better (0.532). The difference between those two is the
region OVERLAP, which is the part the uniform grid dropped.

Before spending GPU-hours on a store build, two things have to hold that
the first probe could not show:

  1. the crop win survives more clips and more than one crop seed
  2. it does not COST anything on the other five transforms. A pooling
     that fixes crop and wrecks warp or tempo is not an improvement, and
     crop-only evidence cannot see that.

Same margin as before, per transform:

    margin = cos(T(a), a) - mean_b!=a cos(T(a), b)

Held-out media throughout, so the pooling that wins here was not chosen
on the media the final grade reports.
"""
import sys
import time
import numpy as np

sys.path.insert(0, "native")
sys.path.insert(0, "/private/tmp/claude-501")
import vcore                                              # noqa: E402
import vnuis                                              # noqa: E402
import vsrc                                               # noqa: E402
from probe_crop import SKIP, c2_of, setcos, true_rmac     # noqa: E402

CLIP_S = 8.0
PER_CORPUS = 4
SEEDS = (7, 11, 23)


def clips():
    out = []
    for name in ("sim", "bridge", "car", "drone"):
        got = 0
        for s in vsrc.sources(name, skip=SKIP[name]):
            if s.dur < CLIP_S + 6.0:
                continue
            # spread the sample through the media instead of always the
            # middle: one fixed offset would sample one kind of moment.
            for frac in (0.25, 0.6):
                t0 = round((s.dur - CLIP_S) * frac, 1)
                F = s.cut(t0, t0 + CLIP_S)
                if len(F) >= 8:
                    out.append((f"{s.id}@{t0:.0f}", F))
                    got += 1
                if got >= PER_CORPUS:
                    break
            if got >= PER_CORPUS:
                break
    return out


def measure(mode, cl, orig):
    if mode == "trueR3":
        vcore.POOL, vcore._rmac = "rmac3", lambda P, t, g: true_rmac(P, t, 3)
    else:
        vcore.POOL, vcore._rmac = mode, orig
    vcore.POOL_FALLBACK.clear()

    from tqdm import tqdm
    ident = [c2_of(F) for _m, F in
             tqdm(cl, desc=f"{mode} identity", unit="clip", leave=False)]
    out = {}
    n = len(cl)
    todo = [(t, s) for t in vnuis.BATTERY for s in SEEDS]
    bar = tqdm(todo, desc=f"{mode} battery", unit="pass", leave=False)
    acc = {t: [] for t in vnuis.BATTERY}
    for tname, seed in bar:
        tf = vnuis.BATTERY[tname]
        alt = [c2_of(tf(F, np.random.RandomState(seed))) for _m, F in cl]
        self_s = np.nanmean([setcos(alt[i], ident[i]) for i in range(n)])
        cross = np.nanmean([setcos(alt[i], ident[j])
                            for i in range(n) for j in range(n) if i != j])
        acc[tname].append(self_s - cross)
    for t, v in acc.items():
        out[t] = float(np.mean(v))
    if vcore.POOL_FALLBACK:
        print(f"  !! {mode} FELL BACK TO GEM - result is not this mode")
    return out


def main():
    orig = vcore._rmac
    cl = clips()
    t0 = time.time()
    print(f"{len(cl)} held-out clips of {CLIP_S:.0f}s, "
          f"{len(SEEDS)} seeds per transform")
    print(f"encoder {vcore.ENCODER.split('/')[-1]}  res {vcore.RES}\n")
    res = {m: measure(m, cl, orig) for m in ("gem", "trueR3")}
    names = list(vnuis.BATTERY)
    print(f"  {'pool':<9}" + "".join(f"{t:<12}" for t in names) + "mean")
    for m, d in res.items():
        mu = float(np.mean([d[t] for t in names]))
        print(f"  {m:<9}" + "".join(f"{d[t]:<12.3f}" for t in names)
              + f"{mu:.3f}")
    print(f"\n  {'delta':<9}" + "".join(
        f"{res['trueR3'][t] - res['gem'][t]:<+12.3f}" for t in names)
        + f"{np.mean([res['trueR3'][t] - res['gem'][t] for t in names]):+.3f}")
    win = sum(1 for t in names if res["trueR3"][t] > res["gem"][t])
    print(f"\n  trueR3 wins {win}/{len(names)} transforms;"
          f" mean margin {np.mean([res['trueR3'][t] for t in names]):.3f}"
          f" vs {np.mean([res['gem'][t] for t in names]):.3f}")
    print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
