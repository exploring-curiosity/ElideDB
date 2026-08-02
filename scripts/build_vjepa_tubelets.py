"""V-JEPA per participant - the last structural gap in the spec.

The spec (user, 2026-08-01): vjepa understands the PHYSICS AND DYNAMICS
of the agent and every participant. Per participant, NEVER whole-frame
("whole frame says nothing"). Seeded from the ELEMENTS, not re-detected.
Only participants that moved significantly, plus the agent.

Every clause maps to something the store now has:

    seeded from elements    the tubelet's boxes come straight from the
                            trajectories table - no detector runs here
    moved significantly     the event join already computed it: a
                            participant bound to an event IS the most-
                            moving thing in that span. Corpus-derived,
                            no threshold invented for this script.
    plus the agent          trajectories.is_agent
    never whole-frame       each clip is the track's own crops

One row per (interval, object): 16 frames sampled across the track's
life, cropped to its box at each sampled instant, through V-JEPA2 ViT-L
(fpc64-256, mean-pooled tokens - the recipe the episode channel
measured at ~291 ms/clip). The table declares `model`, so spaces()
enrolls it (pooled per episode by span containment); the per-row
granularity is what the joins consume.

The episode-level vjepa_vectors stays until this channel beats it on
the bench - retirement is a measurement here, not a hunch.

    python scripts/build_vjepa_tubelets.py [--store lake/fresh_bench]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
from build_dinov3 import segments                              # noqa: E402

MID = "facebook/vjepa2-vitl-fpc64-256"
NFRAMES = 16
CKPT_EVERY = 300


def tubelet_set(db):
    """{(stream, track_ts, t1, object_id) -> [(ts, box)...]} for every
    agent track and every event-bound participant track, PER EPISODE.

    The first cut of this keyed `wanted` by (stream, object_id) alone,
    and identity's recurrence turned that against it: an object bound to
    one event in one episode pulled in its tracks from EVERY episode it
    recurs in - 50,075 tubelets, a 4-hour projection, for a spec that
    says "participants that moved significantly" in THAT episode. The
    membership is (stream, event-span, object): the binding is to the
    episode where the movement happened, so a track qualifies only when
    an event inside its own span names its object.
    """
    tr = db.table("trajectories").scan().to_pydict()
    # every track once, keyed; and the tracks of each (stream, object)
    tracks_of = defaultdict(list)
    keys = {}
    for i in range(len(tr["ts"])):
        s, oid = str(tr["stream"][i]), int(tr["object_id"][i])
        k = (s, int(tr["track_ts"][i]), int(tr["t1"][i]), oid)
        if k not in keys:
            keys[k] = bool(tr["is_agent"][i])
            tracks_of[(s, oid)].append(k)
    # an event marks exactly the tracks of ITS object that overlap ITS
    # span - which, with per-segment tracks on a gapped timeline, is
    # that one episode's track and nothing recurring elsewhere
    ev = db.table("events").scan().to_pydict()
    wanted = set()
    for s, a, b, oid in zip(ev["stream"], ev["ts"], ev["t1"],
                            ev["object_id"]):
        if int(oid) < 0:
            continue
        for k in tracks_of.get((str(s), int(oid)), ()):
            if k[1] <= int(b) and int(a) <= k[2]:
                wanted.add(k)
    wanted |= {k for k, ag in keys.items() if ag}
    n_agent = sum(1 for k in wanted if keys[k])
    out = {k: [] for k in wanted}
    for i in range(len(tr["ts"])):
        k = (str(tr["stream"][i]), int(tr["track_ts"][i]),
             int(tr["t1"][i]), int(tr["object_id"][i]))
        if k in out:
            out[k].append(
                (int(tr["ts"][i]),
                 (tr["x0"][i], tr["y0"][i], tr["x1"][i], tr["y1"][i])))
    for v in out.values():
        v.sort()
    print(f"{len(out):,} tubelets ({n_agent:,} agent tracks in scope)")
    return out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    tubes = tubelet_set(db)

    # group tubelets by the segment that holds their frames
    ft, segs = segments(db)
    seg_span = []
    ts_col = ft.column("ts").to_pylist()
    for sname, i, n_b in segs:
        seg_span.append((sname, int(ts_col[i]), int(ts_col[i + n_b - 1]),
                         i, n_b))
    by_seg = defaultdict(list)
    for k in tubes:
        s, a, b, oid = k
        for j, (sn, t0, t1, i, n_b) in enumerate(seg_span):
            if sn == s and a >= t0 and a <= t1:
                by_seg[j].append(k)
                break
    n_clips = sum(len(v) for v in by_seg.values())
    print(f"{n_clips:,} clips over {len(by_seg):,} segments; projection "
          f"~{n_clips * 0.29 / 60:.0f} min encode + ~6 min decode",
          flush=True)

    import torch
    from transformers import AutoModel, AutoVideoProcessor
    print("loading V-JEPA2 ViT-L ...", flush=True)
    proc = AutoVideoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(
        MID, dtype=torch.float16, low_cpu_mem_usage=True).to("mps").eval()

    ck = db.dir / "_cache" / "build_tubelets.npz"
    rows, done_segs = [], set()
    if ck.exists():
        d = np.load(ck, allow_pickle=True)
        rows = list(d["rows"])
        done_segs = set(int(x) for x in d["done"])
        print(f"resuming: {len(rows):,} rows, {len(done_segs)} segments done")

    cost = defaultdict(float)
    todo = sorted(by_seg)
    for n_done, seg_no in enumerate(tqdm(todo, desc="tubelets",
                                         unit="seg")):
        if seg_no in done_segs:
            continue
        sname, t0, t1, i, n_b = seg_span[seg_no]
        a = time.time()
        chunk = FrameSet(db, "frames", ft.slice(i, n_b)).decode()
        cost["decode"] += time.time() - a
        if not chunk:
            done_segs.add(seg_no)
            continue
        frame_at = {int(ts): im for ts, im in chunk}
        clips, metas = [], []
        for k in by_seg[seg_no]:
            s, ta, tb, oid = k
            pts = tubes[k]
            pick = np.linspace(0, len(pts) - 1, NFRAMES).round().astype(int)
            arr = []
            for j in pick:
                ts_j, (x0, y0, x1, y1) = pts[j]
                im = frame_at.get(ts_j)
                if im is None:
                    continue
                x0, y0 = max(int(x0), 0), max(int(y0), 0)
                x1, y1 = min(int(x1), im.shape[1]), min(int(y1), im.shape[0])
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    # one shape per clip: the processor np.stacks the
                    # frames, and a track's box breathes as the object
                    # moves - resize here, to the model's own input
                    # size, so no second resize happens downstream
                    arr.append(cv2.resize(im[y0:y1, x0:x1], (256, 256),
                                          interpolation=cv2.INTER_AREA))
            if len(arr) < 4:
                continue
            while len(arr) < NFRAMES:
                arr.append(arr[-1])
            clips.append(arr[:NFRAMES])
            metas.append(k)
        a = time.time()
        for c, k in zip(clips, metas):
            inp = proc(c, return_tensors="pt")
            pv = inp["pixel_values_videos"].to("mps", torch.float16)
            with torch.no_grad():
                out = model(pixel_values_videos=pv)
            v = out.last_hidden_state[0].mean(0).float().cpu().numpy()
            n = np.linalg.norm(v)
            if not np.isfinite(v).all():
                raise FloatingPointError(f"NaN tubelet at {k}")
            rows.append((*k, (v / (n + 1e-8)).astype(np.float16)))
        cost["encode"] += time.time() - a
        done_segs.add(seg_no)
        if (n_done + 1) % CKPT_EVERY == 0:
            np.savez(ck, rows=np.array(rows, object),
                     done=np.array(sorted(done_segs)))
    print(f"cost: {json.dumps({k: round(v/60,1) for k, v in cost.items()})}"
          f" min", flush=True)

    dim = len(rows[0][4])
    tbl = pa.table({
        "ts": pa.array([a for _, a, _, _, _ in rows], pa.int64()),
        "t1": pa.array([b for _, _, b, _, _ in rows], pa.int64()),
        "stream": pa.array([s for s, *_ in rows]),
        "object_id": pa.array([o for _, _, _, o, _ in rows], pa.int32()),
        "vector": pa.array([v.astype(np.float32) for *_, v in rows],
                           pa.list_(pa.float32(), dim)),
    })
    tbl = tbl.take(pc.sort_indices(tbl, sort_keys=[
        ("object_id", "ascending"), ("ts", "ascending")]))
    db.table("vjepa_part_vectors").set_layout(
        "object_id", sort_by=["object_id", "ts"], min_group_rows=256)
    db.table("vjepa_part_vectors").replace(
        tbl, kind="vectors",
        meta={"model": MID, "dim": dim, "unit": "participant_track",
              "seeded_from": "trajectories (elements), never re-detected",
              "scope": "agent + event-bound participants"})
    got = len(db.table("vjepa_part_vectors").scan())
    assert got == len(tbl), (got, len(tbl))
    ck.unlink(missing_ok=True)
    print(f"vjepa_part_vectors: {got:,} rows "
          f"({len(set(r[3] for r in rows)):,} objects)")


if __name__ == "__main__":
    main()
