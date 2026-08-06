"""Is 0.721 a PHYSICAL-AI ceiling, or an artifact of the robot arm?

Everything in this build was measured on one embodiment. The encoder
ceiling (action AUC 0.721 -> yield ~0.49) is either a general fact
about frozen encoders and event retrieval, or a quirk of a MuJoCo arm
corpus. That cannot be settled with more arm data.

Four embodiments, one encoder, one metric:

    arm     sim_chains       prim labels (pick/place/stack/...)
    car     KITTI 22 drives  OXTS GPS/IMU  -> kinematic events
    drone   AGZ 45 min       onboard GPS   -> kinematic events
    drone   UZH-FPV          laser-tracker pose -> kinematic events

Truth is vision-independent everywhere, so no domain's labels can be
recovered from the pixels the encoder sees. That is what makes the
comparison honest rather than circular.

PAIRING RULE. unitenc.py takes pairs across EPISODES so shared
background cannot win the AUC. AGZ is one continuous flight with no
episodes, so the same protection here is temporal: a pair is only
scored if the two units are from different media OR more than
SEPARATION seconds apart. Without that, two units 3 s apart look alike
because the drone had not moved, and the AUC would measure
autocorrelation rather than event discrimination.

    python native/crossdom.py --domains agz,kitti,uzhfpv --enc r50_rank
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402
from unitenc import NF, _norm, auc                             # noqa: E402

SEPARATION = 60.0        # seconds; see PAIRING RULE above
W = 256                  # decode width, matching encode.py


def _load(paths):
    from PIL import Image
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:                                # noqa: BLE001
            continue
        h = max(int(im.height * W / im.width), 1)
        out.append(np.asarray(im.resize((W, h))))
    return out


def units_for(spans, frame_at, fps, enc, batch=24):
    """Encode each labelled span with the SAME pooling used on the arm."""
    from gebdback import BACKBONES, KIND
    # accept both "resnet50:rank" and the unitenc names ("r50_rank")
    ALIAS = {"r50": "resnet50", "siglip2": "siglip2",
             "dino": "dinov3_ct"}
    if ":" in enc:
        back, mode = enc.split(":")
    else:
        stem, mode = enc.rsplit("_", 1)
        back = ALIAS.get(stem, stem)
    if back not in BACKBONES:
        raise ValueError(f"unknown backbone {back} from {enc}")
    fn, mid = KIND[BACKBONES[back][0]], BACKBONES[back][1]
    vecs, keep = [], []
    for i0 in range(0, len(spans), batch):
        chunk = spans[i0:i0 + batch]
        flat, idx = [], []
        for (a, b, _lab) in chunk:
            ids = np.linspace(a * fps, b * fps, NF).round().astype(int)
            fr = _load([frame_at(int(j)) for j in ids])
            if len(fr) < 3:
                idx.append(None)
                continue
            idx.append((len(flat), len(flat) + len(fr)))
            flat += fr
        if not flat:
            continue
        V = _norm(fn(mid, flat))
        for s, ix in zip(chunk, idx):
            if ix is None:
                continue
            seg = V[ix[0]:ix[1]]
            m = seg.mean(0)
            T = len(seg)
            al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
            r = (al[:, None] * seg).sum(0)
            r = r / max(np.linalg.norm(r), 1e-8)
            vecs.append(np.concatenate([m, r]) if mode == "rank" else m)
            keep.append(s)
    return (_norm(np.stack(vecs)) if vecs else np.zeros((0, 2))), keep


def domain_auc(V, spans, media):
    """Same-label vs different-label cosine, with the separation rule."""
    S = V @ V.T
    pos, neg = [], []
    n = len(V)
    for i in range(n):
        for j in range(i + 1, n):
            far = media[i] != media[j] or \
                abs(spans[i][0] - spans[j][0]) > SEPARATION
            if not far:
                continue
            (pos if spans[i][2] == spans[j][2] else neg).append(S[i, j])
    return auc(np.array(pos), np.array(neg)), len(pos), len(neg)


def main():
    import domains as D
    from collections import Counter
    from tqdm import tqdm
    enc = arg("--enc", "r50_rank")
    want = arg("--domains", "agz,kitti,uzhfpv").split(",")
    cap = arg("--cap", 400, int)

    print(f"CROSS-DOMAIN action AUC — encoder {enc}, "
          f"separation {SEPARATION:.0f}s\n", flush=True)
    print(f"{'domain':<16}{'units':<8}{'classes':<10}{'AUC':<9}"
          f"{'pos/neg pairs'}")

    for name in want:
        try:
            if name == "agz":
                d = D.agz()
                sp = d["spans"][:cap]
                fps = d["fps"]

                def fa(i, d=d):
                    return d["frame_path"](max(int(i) + 1, 1))
                V, kept = units_for(sp, fa, fps, enc)
                media = ["agz"] * len(kept)
            elif name == "kitti":
                drives = D.kitti_drives()
                V_all, kept_all, med = [], [], []
                for dr in tqdm(drives, desc="kitti", unit="drive",
                               leave=False):
                    dd = D.kitti(dr)
                    if not dd["spans"]:
                        continue
                    v, k = units_for(dd["spans"], dd["frame_path"],
                                     dd["fps"], enc)
                    if len(v):
                        V_all.append(v)
                        kept_all += k
                        med += [dr.name] * len(k)
                V = _norm(np.concatenate(V_all)) if V_all else np.zeros((0, 2))
                kept, media = kept_all, med
            else:
                continue
            if len(V) < 10:
                print(f"{name:<16}too few units ({len(V)})")
                continue
            a, np_, nn = domain_auc(V, kept, media)
            cl = Counter(s[2] for s in kept)
            print(f"{name:<16}{len(V):<8}{len(cl):<10}{a:<9.3f}"
                  f"{np_}/{nn}   {dict(cl)}", flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"{name:<16}FAILED {type(e).__name__}: {e}")

    print(f"\nreference — robot arm (sim), same encoder family: "
          f"r50_rank 0.579, siglip2_rank 0.691 (held out, 60 eps)")
    print("chance = 0.500")


if __name__ == "__main__":
    main()
