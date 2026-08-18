"""PRIMITIVE ACTION dataset: one primitive, one manipulated object,
three arms, and NO FAILED EPISODES.

Owner directive (2026-08-10): "a failed episode creates more noise" -
the corpus audit showed vx300s chains succeeding 19% and even panda
chains only 65%, so everything benchmarked so far sat on partly
broken data. This generator emits the SIMPLEST possible episodes and
RETRIES until the generator's own verification passes in full
(every event ok + every state claim ok + end state ok). An episode
that cannot pass within the attempt budget is discarded, never
written.

One object at a time: each template manipulates exactly ONE block.
stack/unstack need a second block to exist (the relation's other
end), but it is never touched.

The arm is RECORDED IN META (the grow corpora never recorded it -
that provenance hole ends here), along with the target primitive and
the attempt count.

    SDX_ARM=panda  python scripts/sim_prims.py --per-prim 12 --start 0
    SDX_ARM=xarm7  python scripts/sim_prims.py --per-prim 12 --start 1000
    SDX_ARM=vx300s python scripts/sim_prims.py --per-prim 12 --start 2000

Output: data/prim_actions/ep<id>/ - same layout as every sim corpus.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sim_chains as SC  # noqa: E402  (arm chosen via SDX_ARM at import)

# minimal templates: ONE manipulated block (index 0); block 1, where
# present, is the relation's other end and is never touched
PRIM_TEMPLATES = {
    "prim_pick": dict(n=1, steps=[("pick", 0)]),
    "prim_place": dict(n=1, steps=[("pick", 0), ("place", 0, "Z0")]),
    "prim_stack": dict(n=2, steps=[("pick", 0), ("stack", 0, 1)]),
    "prim_unstack": dict(n=2, steps=[("pick", 0), ("stack", 0, 1),
                                     ("unstack", 0, "Z1")]),
    "prim_push": dict(n=1, steps=[("push", 0, "Z0")]),
}
TARGET = {"prim_pick": "pick", "prim_place": "place",
          "prim_stack": "stack", "prim_unstack": "unstack",
          "prim_push": "push"}
SC.TEMPLATES.update(PRIM_TEMPLATES)

MAX_ATTEMPTS = 20     # clean-gated; attempts are ~1.5s, discards are worse


def main():
    argv = sys.argv
    per = int(argv[argv.index("--per-prim") + 1]) \
        if "--per-prim" in argv else 12
    start = int(argv[argv.index("--start") + 1]) \
        if "--start" in argv else 0
    out = ROOT / (argv[argv.index("--out") + 1] if "--out" in argv
                  else "data/prim_actions")
    out.mkdir(parents=True, exist_ok=True)
    arm = SC.ARM_NAME
    log, discarded = [], 0
    t0 = time.time()
    from tqdm import tqdm
    jobs = [(tn, k) for tn in PRIM_TEMPLATES for k in range(per)]
    bar = tqdm(jobs, unit="ep", desc=f"prims/{arm} (~{len(jobs)*3}s)")
    ep_id = start
    for tname, k in bar:
        ok_rec = None
        for att in range(MAX_ATTEMPTS):
            rng = np.random.default_rng(
                77000 + start * 13 + ep_id * 31 + att)
            # bind directly (sample_binding round-robins templates;
            # we need THIS one): same shape/colour/zone draws as the
            # chain generator, randomised per attempt
            T = SC.TEMPLATES[tname]
            shapes = [SC.SHAPES[rng.integers(2)]
                      for _ in range(T["n"])]
            colors = list(rng.choice(list(SC.PALETTE), size=T["n"],
                                     replace=False))
            spec = list(zip(shapes, colors))
            zperm = list(rng.permutation(list(SC.ZONES)))
            zone_bind = {f"Z{i}": zperm[i] for i in range(3)}
            ep_dir = out / f"ep{ep_id:04d}"
            if ep_dir.exists():
                shutil.rmtree(ep_dir)
            rec = SC.run_episode(ep_id, tname, spec, zone_bind, rng,
                                 out, [])
            clean = all(a == 1 for a in rec.get("grasp_attempts", []))
            if rec["success"] and rec["state_success"] and clean:
                ok_rec = rec
                # provenance + label INTO the meta (the grow corpora
                # never recorded the arm; that hole ends here)
                mf = ep_dir / "meta.json"
                meta = json.loads(mf.read_text())
                meta["arm"] = arm
                meta["target_prim"] = TARGET[tname]
                meta["attempts"] = att + 1
                meta["clean"] = True
                mf.write_text(json.dumps(meta, indent=1))
                break
            shutil.rmtree(ep_dir, ignore_errors=True)
        if ok_rec is None:
            discarded += 1
            bar.write(f"  DISCARDED {tname} after {MAX_ATTEMPTS} "
                      f"attempts ({arm})")
        else:
            log.append(ok_rec)
            ep_id += 1
    print(json.dumps({
        "arm": arm, "kept": len(log), "discarded": discarded,
        "mean_attempts": round(float(np.mean(
            [json.loads((out / f'ep{r["episode"]:04d}' /
                         'meta.json').read_text())["attempts"]
             for r in log])), 2) if log else None,
        "wall_min": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
