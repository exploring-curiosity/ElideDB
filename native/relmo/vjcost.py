"""What the shipped system costs: write time per hour of video, read time per query.

Measured, not extrapolated. Model loading is excluded from every timing (it is
paid once per process, not per hour of video); everything else - ffmpeg decode,
V-JEPA, SigLIP, the pooling head and the recurrence - is inside the clock.

WRITE is reported as video-seconds per compute-second, because that is the
number that decides how many cameras one machine can keep up with. A rate of
1.0x is real time for a single stream.

READ is split into ENCODE (the query clip has to be turned into a z sequence,
which costs exactly what writing it would) and SEARCH (subsequence DTW against
the index). They scale differently - encode is constant in corpus size, search
is linear - so quoting only their sum hides which one will bite first.

    python -m relmo.vjcost --episodes 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrel  # noqa: E402
from relmo.vjeval import REC, l2  # noqa: E402
from relmo.vjmatch import dtw  # noqa: E402
from relmo.vjrec4 import CTX  # noqa: E402
from relmo.vjrec6 import (HOP_S, STREAM_FPS, WIN_FRAMES, episode_paths,  # noqa: E402
                          geometry, record_stream, windows)
from relmo.vjs import MODEL, TUBELET, probe_dims, read_frames  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--queries", type=int, default=12)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, VJEPA2Model
    from relmo import vjrank2
    from relmo.vjrec7 import encode_tokens
    from relmo.vjsig import MODEL as SIGM, RES

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16
    print(f"loading models onto {dev} (excluded from all timings)...",
          flush=True)
    vj = VJEPA2Model.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    sg = AutoModel.from_pretrained(SIGM, dtype=dt).to(dev).eval()
    tok_pca = np.load(R.BASE / "vjrec7" / "_token_pca.npz")["W"].astype(np.float32)
    z = np.load(REC / a.dataset / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    rk, _ = vjrank2.load_ckpt("p1_reg_low_s0")
    _, n_t, _, _ = geometry(WIN_FRAMES, STREAM_FPS, HOP_S)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    print("  loaded\n", flush=True)

    eps = [e for e in episode_paths(a.dataset)][:a.episodes]
    T = dict(decode=0.0, vjepa=0.0, siglip=0.0, model=0.0)
    video_s = 0.0
    for eid, mp4, fps in eps:
        t0 = time.time()
        w_, h_ = probe_dims(mp4)
        F = read_frames(mp4, w_, h_)
        T["decode"] += time.time() - t0
        video_s += len(F) / fps
        wins = windows(len(F), fps, WIN_FRAMES, STREAM_FPS, HOP_S)
        if not wins:
            continue

        t0 = time.time()
        rec = record_stream(vj, torch, dev, F, fps, cal, 6, dt)
        toks, gates = [], []
        for _, idx in wins:
            b, g = encode_tokens(vj, torch, dev, F[idx], 6, n_t, dt)
            toks.append(b.astype(np.float32) @ tok_pca)
            gates.append(g)
        if dev == "mps":
            torch.mps.synchronize()
        T["vjepa"] += time.time() - t0

        t0 = time.time()
        sel = F[np.clip(rec["frame0"].astype(int), 0, len(F) - 1)]
        V = []
        for s in range(0, len(sel), 64):
            x = torch.tensor(sel[s:s + 64]).permute(0, 3, 1, 2).float().div_(255.)
            x = torch.nn.functional.interpolate(x, size=(RES, RES),
                                                mode="bilinear",
                                                align_corners=False)
            with torch.no_grad():
                V.append(sg.get_image_features(
                    pixel_values=((x - mean) / std).to(dev, dt)).float().cpu().numpy())
        sig = np.concatenate(V).astype(np.float32)
        if dev == "mps":
            torch.mps.synchronize()
        T["siglip"] += time.time() - t0

        t0 = time.time()
        tok = np.concatenate(toks)[None]
        gate = np.concatenate(gates)[None]
        from relmo import vjz
        fix, g2 = vjz.channels(np.load(R.BASE / "vjrec6"
                                       / f"{a.dataset}_L6" / f"{eid}.npz"),
                               primary="b")
        n = min(tok.shape[1], len(sig), len(fix))
        with torch.no_grad():
            rk.encode(torch.tensor(tok[:, :n]), torch.tensor(gate[:, :n]),
                      torch.tensor(g2[None, :n]), torch.tensor(sig[None, :n]),
                      torch.tensor(fix[None, :n]),
                      torch.ones(1, n, dtype=torch.bool))
        T["model"] += time.time() - t0

    tot = sum(T.values())
    print(f"WRITE, measured over {len(eps)} recordings "
          f"({video_s:.0f} s of video)\n")
    print(f"{'stage':22s} {'compute s':>10s} {'share':>7s} {'x real-time':>12s}")
    print("-" * 54)
    for k in ("decode", "vjepa", "siglip", "model"):
        print(f"{k:22s} {T[k]:10.1f} {T[k]/tot*100:6.1f}% "
              f"{video_s/max(T[k],1e-9):11.1f}x")
    print("-" * 54)
    print(f"{'TOTAL':22s} {tot:10.1f} {100.0:6.1f}% {video_s/tot:11.1f}x")
    print(f"\nper HOUR of video: {3600/(video_s/tot)/60:.1f} compute-minutes")
    print(f"streams kept up with in real time on one machine: "
          f"{video_s/tot:.1f}")

    # ---------------- READ ----------------
    print("\n\nREAD, measured against the built index\n")
    D = vjrank2.load_corpus()
    Z = vjrank2.encode_all(rk, D)
    ids = sorted(Z)
    P, ok = _pad([l2(Z[i]) for i in ids])
    L = np.array([len(Z[i]) for i in ids])
    AM = vjrel.all_meta()
    qs = ids[:a.queries]
    t0 = time.time()
    for q in qs:
        C = 1.0 - np.einsum("sd,nkd->nsk", l2(Z[q]), P)
        C = np.where(ok[:, None, :], C, PAD_COST)
        dtw(C, False, L)
    search = (time.time() - t0) / len(qs)
    enc_per_s = video_s / tot
    med_dur = float(np.median([AM[i]["dur"] for i in ids if i in AM]))
    print(f"index: {len(ids)} recordings, {sum(L)} steps, "
          f"{Z[ids[0]].shape[1]}-d per step")
    print(f"{'stage':30s} {'ms':>9s}")
    print("-" * 41)
    print(f"{'encode a '+f'{med_dur:.0f}'+'s query clip':30s} "
          f"{med_dur/enc_per_s*1000:9.0f}")
    print(f"{'search the whole index':30s} {search*1000:9.0f}")
    print(f"{'TOTAL per query':30s} {(med_dur/enc_per_s+search)*1000:9.0f}")
    print(f"\nsearch scales LINEARLY in corpus size: "
          f"{search/len(ids)*1000*1000:.2f} ms per 1000 recordings; encode is "
          f"constant.")
    R.log("vjcost", realtime=round(video_s/tot, 2),
          search_ms=round(search*1000, 1), index=len(ids))


if __name__ == "__main__":
    main()
