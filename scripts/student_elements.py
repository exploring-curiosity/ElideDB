"""The student produces all five elements itself, fast.

Same methodology as the teacher, not the teacher's outputs. The teacher
spends 3.4s per demo and almost all of it in one place: naming a
participant costs a 7B VLM generation plus a Grounding-DINO
verification per crop. Everything else it does - agent from flow,
contact from motion onset, cavity transitions, the answer rollup - is
geometry and costs ~0.5s with no model at all.

So the student keeps the geometry verbatim and replaces ONLY the
expensive step: a distilled head maps a crop's SigLIP embedding
directly to the name VECTOR the teacher's generator+verifier would
have produced. Names were never compared as strings anywhere in this
system - matching is cosine in name space - so predicting the vector
is the whole job, and it turns 3.4s of autoregressive generation into
one matmul.

    stage E1  agent, participants, events, answer   flow geometry
    stage E2  participant name vectors              distilled head
    stage E3  scene                                 pooled, free

  python scripts/student_elements.py --train     fit the name head
  python scripts/student_elements.py --all       produce + time
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_teacher import (articulated_box, cavity_series,     # noqa: E402
                           CAV_MIN, REL_MIN, NFRAMES)
from elidedb import Store                                      # noqa: E402
from extract_events import (_crop, agent_track,                # noqa: E402
                            causal_participants)

HEAD = ROOT / "models/student_v1/namer.pt"


def crop_embed(crops, model, proc, dev):
    """SigLIP image embedding for a batch of crops."""
    import torch
    from PIL import Image
    if not crops:
        return np.zeros((0, 1152), np.float32)
    with torch.no_grad():
        px = proc(images=[Image.fromarray(c) for c in crops],
                  return_tensors="pt").to(dev)
        F = model.get_image_features(**px)
        F = F / F.norm(dim=-1, keepdim=True)
    return F.float().cpu().numpy()


def structure(db, frames_tbl, key):
    """E1: everything the teacher derives from geometry, verbatim."""
    from elidedb.video import FrameSet
    s, a, b = key
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), s),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                pc.less_equal(frames_tbl.column("ts"), b))))
    n = len(sel)
    if n < 4:
        return None
    pick = np.unique(np.linspace(0, n - 1, min(NFRAMES, n))
                     .round().astype(int))
    try:
        dec = sorted(FrameSet(db, "frames", sel.take(pick)).decode())
    except Exception:
        return None
    times = [int(t) for t, _ in dec]
    frames = [f for _, f in dec]
    if len(frames) < 4:
        return None
    H, W = frames[0].shape[:2]
    # ONE motion pass on the GPU, shared by both consumers
    from extract_events import motion_mags
    mg = motion_mags(frames)
    tr, span, masks, all_tracks = agent_track(frames, mg)
    abox = articulated_box(frames, masks.any(0), mg)
    ev = []

    def stamp(i):
        return times[min(max(i, 0), len(times) - 1)]

    if tr is not None:
        ev.append({"kind": "agent", "role": "agent", "t0": stamp(tr[0][0]),
                   "t1": stamp(tr[-1][0] + 1), "box": list(tr[-1][3]),
                   "crop": None, "conf": float(span)})
    cav = cavity_series(frames, abox)
    if cav is not None and len(cav) >= 4:
        base = float(np.median(cav[:2]))
        run = None
        for i in range(1, len(cav)):
            d = float(cav[i]) - base
            k = "open" if d >= CAV_MIN else ("close" if d <= -CAV_MIN
                                             else None)
            if k and run is None:
                run = (k, i)
            elif run and k != run[0]:
                ev.append({"kind": run[0], "role": "container",
                           "t0": stamp(run[1] - 1), "t1": stamp(i),
                           "box": list(abox), "crop": None,
                           "conf": abs(d)})
                run = (k, i) if k else None
        if run:
            ev.append({"kind": run[0], "role": "container",
                       "t0": stamp(run[1] - 1), "t1": stamp(len(cav) - 1),
                       "box": list(abox), "crop": None,
                       "conf": abs(float(cav[-1]) - base)})
    for origin, dest, onset, life, area in causal_participants(
            all_tracks, tr, len(frames) - 1):
        c0 = np.array([(origin[0] + origin[2]) / 2,
                       (origin[1] + origin[3]) / 2])
        c1 = np.array([(dest[0] + dest[2]) / 2, (dest[1] + dest[3]) / 2])
        # object-relative, exactly as the teacher: see build_teacher's
        # REL_MIN note - a frame-relative threshold was measuring track
        # truncation, not motion, and typed 82% of everything "adjust".
        odiag = float(np.hypot(origin[2] - origin[0],
                               origin[3] - origin[1]))
        disp = float(np.linalg.norm(c1 - c0)) / max(odiag, 1.0)

        def inside(c, rgn):
            return (rgn is not None and rgn[0] <= c[0] <= rgn[2]
                    and rgn[1] <= c[1] <= rgn[3])
        k = ("adjust" if disp < REL_MIN else
             "take_out" if (inside(c0, abox) and not inside(c1, abox))
             else "put_into" if inside(c1, abox) else "put_on")
        ev.append({"kind": "contact", "role": "participant",
                   "t0": stamp(onset - 1), "t1": stamp(onset),
                   "box": list(origin), "crop": _crop(frames[0], origin),
                   "conf": 1.0})
        ev.append({"kind": "release", "role": "participant",
                   "t0": stamp(onset + life - 1),
                   "t1": stamp(onset + life), "box": list(dest),
                   "crop": _crop(frames[-1], dest), "conf": 1.0})
        ev.append({"kind": k, "role": "participant", "t0": stamp(onset),
                   "t1": stamp(onset + life), "box": list(dest),
                   "crop": _crop(frames[0], origin), "conf": 1.0})
    ev.sort(key=lambda e: e["t0"])
    return {"span": float(span), "events": ev}


def main():
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoProcessor

    from elidedb.device import pick
    from elidedb.embeddings import _vec_table
    from elidedb.scenario import _episodes
    from elidedb.sig2 import MID
    argv = sys.argv
    do_train = "--train" in argv
    do_all = "--all" in argv
    dev, dtype = pick()
    proc = AutoProcessor.from_pretrained(MID)
    sig = AutoModel.from_pretrained(MID, dtype=dtype,
                                    low_cpu_mem_usage=True).to(dev).eval()
    # --store lets the write path run against any store, not just
    # the benchmark one; the elements are corpus-independent.
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    keys = _episodes(db)
    frames_tbl = db.table("frames").scan()

    class Namer(nn.Module):
        """crop image embedding -> the teacher's name VECTOR. Names are
        never compared as strings in this system, so the vector is the
        entire product of 3.4s of VLM generation + detector check."""
        def __init__(self, d=1152):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, 1024), nn.GELU(),
                                   nn.Linear(1024, d))

        def forward(self, x):
            y = x + self.f(x)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    if do_train:
        # teacher targets: the events table's own name vectors
        ev = db.table("events").scan().to_pydict()
        _, NV = _vec_table(db, "events", column="name_vec")
        NV = np.asarray(NV, np.float32)
        BX = np.array(ev["box"], np.int32).reshape(-1, 4)
        kidx = {(k[0], k[1]): i for i, k in enumerate(keys)}
        want = [r for r in range(len(ev["ts"]))
                if ev["name"][r] and np.linalg.norm(NV[r]) > 1e-6]
        print(f"{len(want)} named teacher events to imitate")
        by = {}
        for r in want:
            by.setdefault(kidx[(str(ev["stream"][r]), int(ev["ts"][r]))],
                          []).append(r)
        X, Y, t0 = [], [], time.time()
        from elidedb.video import FrameSet
        for n, (i, rows) in enumerate(by.items()):
            s, a, b = keys[i]
            sel = frames_tbl.filter(pc.and_(
                pc.equal(frames_tbl.column("stream"), s),
                pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                        pc.less_equal(frames_tbl.column("ts"), b))))
            try:
                dec = sorted(FrameSet(db, "frames",
                                      sel.take(np.array([0, len(sel) - 1]))
                                      ).decode())
            except Exception:
                continue
            crops, tgt = [], []
            for r in rows:
                im = dec[0][1] if ev["kind"][r] != "release" else dec[-1][1]
                c = _crop(im, BX[r])
                if c.shape[0] >= 12 and c.shape[1] >= 12:
                    crops.append(c); tgt.append(NV[r])
            if crops:
                X.append(crop_embed(crops, sig, proc, dev))
                Y.append(np.stack(tgt))
            if (n + 1) % 200 == 0:
                print(f"  {n+1}/{len(by)} {time.time()-t0:.0f}s", flush=True)
        X = np.concatenate(X); Y = np.concatenate(Y)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
        print(f"pairs {X.shape} in {time.time()-t0:.0f}s")
        net = Namer().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4,
                                weight_decay=1e-2)
        Xt = torch.tensor(X, device=dev, dtype=torch.float32)
        Yt = torch.tensor(Y, device=dev, dtype=torch.float32)
        t1 = time.time()
        for e in range(300):
            perm = torch.randperm(len(Xt), device=dev)
            for j in range(0, len(perm), 256):
                b_ = perm[j:j + 256]
                loss = (1 - (net(Xt[b_]) * Yt[b_]).sum(-1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if (e + 1) % 100 == 0:
                print(f"  epoch {e+1} cosine loss {float(loss):.4f}")
        HEAD.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), HEAD)
        print(f"saved {HEAD}, train {time.time()-t1:.0f}s")
        return

    if not do_all:
        return
    net = Namer().to(dev)
    net.load_state_dict(torch.load(HEAD, map_location=dev,
                                   weights_only=True))
    net.eval()
    rows = {k: [] for k in ("ts", "t1", "stream", "kind", "role",
                            "ev_t0", "ev_t1", "conf")}
    vecs, spans = [], []
    t0 = time.time()
    # PARALLEL BY DEMO, BATCHED ON THE GPU.
    #
    # This loop used to run one demo at a time, and it was the whole
    # write cost: profiled at 537 ms/demo of geometry, of which 87% is
    # dense Farneback optical flow (agent_track 243 ms, articulated_box
    # 227 ms), plus ~310 ms of per-demo SigLIP crop embedding. Meanwhile
    # the INGEST of the same corpus ran at 1,109 frames/s on 13.6 cores.
    # The element pass was using one.
    #
    # Two changes, neither of which alters a single output value:
    #   - structure() runs across demos in a thread pool. It is OpenCV
    #     and ffmpeg underneath, both of which drop the GIL, so threads
    #     get real parallelism (the store's own loader already reads
    #     concurrently for the same reason).
    #   - crops are embedded ONE CHUNK AT A TIME instead of one demo at
    #     a time, so SigLIP sees batches of hundreds rather than fours.
    # Demos are independent and the rows are ts-sorted before the
    # commit, so results are identical to the sequential path.
    # NO THREADS. The motion pass runs on the GPU instead (see
    # motion_mags): the work that made this slow was per-pixel and
    # belongs on device, not spread across cores. Crops are still
    # embedded a CHUNK at a time so SigLIP sees batches of hundreds
    # rather than fours, which is also GPU work.
    CHUNK = 32
    print(f"  {len(keys)} demos, GPU motion, chunk {CHUNK}", flush=True)
    for base in range(0, len(keys), CHUNK):
        batch = keys[base:base + CHUNK]
        sts = [structure(db, frames_tbl, kk) for kk in batch]
        allcrops = []
        for st in sts:
            if st is None:
                continue
            allcrops += [e["crop"] for e in st["events"]
                         if e["crop"] is not None]
        NV = np.zeros((0, 1152), np.float32)
        if allcrops:
            emb = crop_embed(allcrops, sig, proc, dev)
            with torch.no_grad():
                NV = net(torch.tensor(emb, device=dev,
                                      dtype=torch.float32)).cpu().numpy()
        ptr = 0
        for k, st in zip(batch, sts):
            if st is None:
                continue
            spans.append((k, st["span"]))
            for e in st["events"]:
                rows["ts"].append(k[1]); rows["t1"].append(k[2])
                rows["stream"].append(k[0]); rows["kind"].append(e["kind"])
                rows["role"].append(e["role"]); rows["ev_t0"].append(e["t0"])
                rows["ev_t1"].append(e["t1"]); rows["conf"].append(e["conf"])
                if e["crop"] is not None:
                    vecs.append(NV[ptr]); ptr += 1
                else:
                    vecs.append(np.zeros(1152, np.float32))
        n = min(base + CHUNK, len(keys))
        el = time.time() - t0
        print(f"  {n}/{len(keys)} {el:.0f}s ({el/n:.3f}s/demo)", flush=True)
    wall = time.time() - t0
    V = np.stack(vecs)
    tbl = pa.table({
        "ts": pa.array(rows["ts"], pa.int64()),
        "t1": pa.array(rows["t1"], pa.int64()),
        "stream": pa.array(rows["stream"]),
        "kind": pa.array(rows["kind"]),
        "role": pa.array(rows["role"]),
        "ev_t0": pa.array(rows["ev_t0"], pa.int64()),
        "ev_t1": pa.array(rows["ev_t1"], pa.int64()),
        "name": pa.array([""] * len(rows["ts"])),
        "conf": pa.array(rows["conf"], pa.float32()),
        "name_vec": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V.astype(np.float16)).reshape(-1),
                     pa.float16()), 1152),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    db.table("events_s").replace(tbl, kind="events",
                                 meta={"producer": "student-v1"})
    print(json.dumps({"demos": len(spans), "events": tbl.num_rows,
                      "seconds": round(wall, 1),
                      "s_per_demo": round(wall / max(len(spans), 1), 3),
                      "teacher_s_per_demo": 3.4}))


if __name__ == "__main__":
    main()
