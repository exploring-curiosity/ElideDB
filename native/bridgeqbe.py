"""REAL robot data. Everything so far was measured on the synthetic corpus.

sim is 150 MuJoCo episodes whose classes differ chiefly by how many
times a motif repeats - which turned out to be the whole story there,
and is an artifact of how that corpus was generated. Bridge is real:
50,415 teleoperated episodes, 22,199 task strings, 175 tasks with >= 50
episodes, and episodes are SHORT (mean 7.1 s) single actions rather
than chains. So the count bottleneck does not apply here at all.

What bridge tests instead is the harder and more useful thing:

    open the drawer   484 episodes      close the drawer   418
    open microwave    330               close microwave    330

Same kitchen, same objects, same gripper - OPPOSITE MOTION. Appearance
alone cannot separate those pairs; a mean over frames is identical
under time reversal. This is exactly the case rank pooling was adopted
for (it flips sign when the clip reverses), so it is the honest test of
today's main change on data nobody generated for us.

Truth (task strings) is EVAL ONLY - it selects which episodes belong
together and never enters the encoder or the matcher.

    python native/bridgeqbe.py --tasks 8 --per 40
"""
from __future__ import annotations

import collections
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402
from unitenc import _norm                                      # noqa: E402

BR = ROOT / "data/bridge"
STREAM = "observation.images.image_0"
FPS = 5.0
NF = 8                    # frames per unit
CACHE = Path("/private/tmp/claude-501/bridgeqbe")

# The discriminations that matter: reversible pairs in the same scene.
PAIRS = [("open the drawer", "close the drawer"),
         ("open microwave", "close microwave"),
         ("put carrot on plate", "take carrot off plate")]


def episodes():
    """episode_index -> (task, video file, start seconds, n frames)."""
    import pyarrow.parquet as pq
    t = pq.read_table(BR / "meta/episodes/chunk-000/file-000.parquet")
    d = t.to_pydict()
    fi = d[f"videos/{STREAM}/file_index"]
    ts = d[f"videos/{STREAM}/from_timestamp"]
    out = {}
    for i, ep in enumerate(d["episode_index"]):
        tk = d["tasks"][i]
        tk = tk[0] if isinstance(tk, list) and tk else str(tk)
        out[int(ep)] = (str(tk), int(fi[i]), float(ts[i]),
                        int(d["length"][i]))
    return out


def decode_clip(file_idx, t0, dur, w=256):
    """Byte-range-ish read: seek to the episode, decode only its frames.

    This is the database's own discipline - never decode a whole file to
    reach seven seconds of it.
    """
    path = BR / f"videos/{STREAM}/chunk-000/file-{file_idx:03d}.mp4"
    if not path.exists():
        return []
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(path),
           "-t", f"{dur:.3f}", "-vf", f"scale={w}:-2", "-pix_fmt",
           "rgb24", "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return []
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True)
    try:
        W0, H0 = [int(x) for x in probe.stdout.strip().split(",")[:2]]
    except Exception:                                    # noqa: BLE001
        return []
    h = int(round(H0 * w / W0 / 2)) * 2
    n = len(r.stdout) // (w * h * 3)
    if n < 2:
        return []
    return np.frombuffer(r.stdout[:n * w * h * 3],
                         np.uint8).reshape(n, h, w, 3)


def encode_episode(F, mode="rank"):
    """One vector per episode. Episodes are ~7 s single actions, so the
    unit IS the episode - no segmentation question arises."""
    from gebdback import BACKBONES, KIND
    fn, mid = KIND[BACKBONES["siglip2"][0]], BACKBONES["siglip2"][1]
    idx = np.linspace(0, len(F) - 1, NF).round().astype(int)
    V = _norm(fn(mid, [np.ascontiguousarray(F[i]) for i in idx]))
    m = V.mean(0)
    if mode == "mean":
        return m / max(np.linalg.norm(m), 1e-8)
    T = len(V)
    al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
    r = (al[:, None] * V).sum(0)
    r = r / max(np.linalg.norm(r), 1e-8)
    v = np.concatenate([m, r])
    return v / max(np.linalg.norm(v), 1e-8)


def score(vecs, lab, seed_n=5):
    groups = collections.defaultdict(list)
    for e in vecs:
        groups[lab[e]].append(e)
    rs = np.random.RandomState(0)
    ys, ps, sups, rets = [], [], [], []
    per = {}
    for tk, pool in sorted(groups.items()):
        if len(pool) < seed_n + 1:
            continue
        sd = sorted(rs.choice(pool, seed_n, replace=False).tolist())
        support = len(pool) - len(sd)
        k = math.ceil(1.5 * support)
        cand = [e for e in vecs if e not in sd]
        S = {e: max(float(vecs[e] @ vecs[s]) for s in sd)
             for e in cand}
        loo = [max(float(vecs[a] @ vecs[b]) for b in sd if b != a)
               for a in sd]
        cut = min(loo) if loo else -np.inf
        ranked = sorted(S, key=lambda x: -S[x])[:k]
        got = [e for e in ranked if S[e] >= cut]
        tr = sum(1 for e in got if lab[e] == tk)
        ys.append(tr / support)
        ps.append(tr / len(got) if got else 0.0)
        sups.append(support)
        rets.append(len(got))
        per[tk] = (tr / support, tr / len(got) if got else 0.0, support)
    return ys, ps, sups, rets, per


def main():
    from tqdm import tqdm
    per_task = arg("--per", 40, int)
    ntask = arg("--tasks", 8, int)
    mode = arg("--mode", "rank")

    meta = episodes()
    bytask = collections.defaultdict(list)
    for ep, (tk, fi, t0, ln) in meta.items():
        if tk.strip():
            bytask[tk].append(ep)
    # the reversible pairs first, then fill with other high-support tasks
    want = []
    for a, b in PAIRS:
        if len(bytask.get(a, [])) >= 10 and len(bytask.get(b, [])) >= 10:
            want += [a, b]
    for tk, eps in sorted(bytask.items(), key=lambda kv: -len(kv[1])):
        if len(want) >= ntask:
            break
        if tk not in want and len(eps) >= 30:
            want.append(tk)
    print(f"bridge QbE — {len(want)} tasks, <= {per_task} episodes each, "
          f"encoder siglip2_{mode}")
    for tk in want:
        print(f"   {len(bytask[tk]):5d} avail  {tk[:60]}")

    CACHE.mkdir(parents=True, exist_ok=True)
    vecs, lab = {}, {}
    rs = np.random.RandomState(0)
    for tk in want:
        eps = sorted(bytask[tk])
        if len(eps) > per_task:
            eps = sorted(rs.choice(eps, per_task,
                                   replace=False).tolist())
        for ep in tqdm(eps, desc=tk[:22], unit="ep", leave=False):
            fp = CACHE / f"{mode}_{ep}.npy"
            if fp.exists():
                vecs[ep] = np.load(fp)
                lab[ep] = tk
                continue
            tkk, fi, t0, ln = meta[ep]
            F = decode_clip(fi, t0, max(ln / FPS, 1.0))
            if len(F) < 3:
                continue
            v = encode_episode(F, mode)
            np.save(fp, v)
            vecs[ep] = v
            lab[ep] = tk

    print(f"\nencoded {len(vecs)} episodes")
    ys, ps, sups, rets, per = score(vecs, lab)
    if not ys:
        print("no task had enough episodes")
        return
    print(f"\n{'task':<34}{'yield':<9}{'prec':<9}{'support'}")
    for tk, (y, p, s) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"  {tk[:32]:<32}{y:<9.3f}{p:<9.3f}{s}")
    print(f"\nOVERALL yield {np.mean(ys):.3f}  prec {np.mean(ps):.3f}  "
          f"support {np.mean(sups):.1f}  returned {np.mean(rets):.1f}")


if __name__ == "__main__":
    main()
