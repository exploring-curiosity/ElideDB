"""v1 (custom SDX/SFI binaries) → v2 (Parquet lake) migration.

Reads the v1 store's own files — SDX columnar chunks, SFI frame tables,
embedding runs — and rewrites each as Parquet tables under a transaction
log. Raw media files stay exactly where they are; only indexes and sensor
rows move. This is the "no custom datatypes" pivot: the v1 formats keep
working for the C++ engine, but the go-forward store is open-format.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

from .store import Store

_SDX_TYPES = {0: ("i8", 8), 1: ("f8", 8), 2: ("f4", 4), 3: ("i8", 8), 4: ("i2", 2)}


def read_sdx(path) -> pa.Table:
    """Minimal SDX v1 reader (layout: FORMAT.md §1)."""
    data = Path(path).read_bytes()
    assert data[:4] == b"SDX1" and data[-4:] == b"SDX1", f"bad SDX: {path}"
    (flen,) = struct.unpack_from("<I", data, len(data) - 8)
    off = len(data) - 8 - flen
    def u16():
        nonlocal off; (v,) = struct.unpack_from("<H", data, off); off += 2; return v
    def u32():
        nonlocal off; (v,) = struct.unpack_from("<I", data, off); off += 4; return v
    def i64():
        nonlocal off; (v,) = struct.unpack_from("<q", data, off); off += 8; return v
    def s():
        nonlocal off
        n = u16()
        v = data[off:off + n].decode(); off += n; return v
    _ver, _flags = u16(), u16()
    _sid, _units = s(), s()
    _clock_off = i64()
    _chunk_target = u32()
    ncol = u16()
    cols = []
    for _ in range(ncol):
        name = s(); ty = data[off]; off += 2
        cols.append((name, ty))
    nchunk = u32()
    spans = [[] for _ in cols]
    for _ in range(nchunk):
        _rows = u32(); off_pad = u32()  # noqa: F841
        for c in range(ncol):
            (doff, dlen, _mn, _mx) = struct.unpack_from("<QQQQ", data, off)
            off += 32
            spans[c].append((doff, dlen))
    arrays = {}
    for c, (name, ty) in enumerate(cols):
        np_ty, _w = _SDX_TYPES[ty]
        parts = [np.frombuffer(data, dtype=np_ty,
                               count=dlen // int(np_ty[1]),
                               offset=doff) for (doff, dlen) in spans[c]]
        arrays["ts" if c == 0 else name] = np.concatenate(parts) if parts else \
            np.array([], np_ty)
    return pa.table({k: pa.array(v) for k, v in arrays.items()})


def _sfi_frame_table(sfi_path, source_path, stream, width, height, codec):
    sys.path.insert(0, str(Path(__file__).parents[2] / "ml"))
    from sfi_reader import read_sfi
    sfi = read_sfi(str(sfi_path))
    fr = sfi.frames
    n = len(fr)
    return pa.table({
        "ts": pa.array(fr["pts_ns"].astype("int64")),
        "byte_offset": pa.array(fr["byte_offset"].astype("int64")),
        "packet_size": pa.array(fr["packet_size"].astype("int32")),
        "keyframe": pa.array((fr["flags"] & 1).astype(bool)),
        "width": pa.array(np.full(n, width, "int32")),
        "height": pa.array(np.full(n, height, "int32")),
        "codec": pa.array([codec] * n),
        "source": pa.array([str(source_path)] * n),
        "stream": pa.array([stream] * n),
    })


def migrate_v1(v1_store: str, out_path: str, name: str,
               run_id: str | None = None, verbose=True) -> Store:
    v1 = Path(v1_store)
    cur = int((v1 / "CURRENT").read_text().strip())
    man = json.loads((v1 / "manifests" / f"manifest-{cur}.json").read_text())
    db = Store.create(out_path, name)

    # video: every SFI → rows of ONE frame_index table ("frames")
    for vs in man.get("video_streams", []):
        for seg in vs["segments"]:
            t = _sfi_frame_table(v1 / seg["sfi_path"], seg["source_path"],
                                 vs["stream_id"], vs.get("width", 0),
                                 vs.get("height", 0), "mjpeg")
            db.table("frames").append(t, kind="frame_index",
                                      meta={"migrated_from": seg["sfi_path"]})
            if verbose:
                print(f"  frames << {vs['stream_id']} ({len(t)} rows)")

    # sensors: every SDX → its own timeseries table
    for ss in man.get("sensor_streams", []):
        t = read_sdx(v1 / ss["sdx_path"])
        tname = ss["stream_id"].replace("/", "_").replace(" ", "_").lower()
        db.table(tname).append(t, meta={"migrated_from": ss["sdx_path"]})
        if verbose:
            print(f"  {tname} << {len(t):,} rows")

    # embeddings run → embeddings + centroids tables
    if run_id is None and man.get("semantic_run"):
        run_id = man["semantic_run"]["run_id"]
    if run_id:
        run = v1 / "ml" / run_id
        win = json.loads((run / "windows.json").read_text())
        dim = win["dim"]
        vecs = np.fromfile(run / "embeddings.f32", dtype=np.float32) \
            .reshape(-1, dim)
        labels = None
        if (run / "clusters.json").exists():
            cj = json.loads((run / "clusters.json").read_text())
            labels = np.array(cj["labels"], "int32")
            cents = np.array(cj["centroids"], "float32")
        t = pa.table({
            "ts": pa.array([w["t0_ns"] for w in win["windows"]], pa.int64()),
            "t1": pa.array([w["t1_ns"] for w in win["windows"]], pa.int64()),
            "stream": pa.array([w["stream_id"] for w in win["windows"]]),
            "vector": pa.array([v.tolist() for v in vecs],
                               pa.list_(pa.float32(), dim)),
            **({"cluster": pa.array(labels)} if labels is not None else {}),
        })
        meta = json.loads((run / "meta.json").read_text())
        db.table("embeddings").append(
            t, kind="embeddings",
            meta={"model": meta["model"], "dim": dim,
                  "window_s": meta.get("window_s"), "migrated_from": run_id})
        if labels is not None and len(cents):
            ct = pa.table({
                "ts": pa.array([0] * len(cents), pa.int64()),
                "cluster": pa.array(range(len(cents)), pa.int32()),
                "vector": pa.array([c.tolist() for c in cents],
                                   pa.list_(pa.float32(), dim)),
            })
            db.table("centroids").append(ct, kind="centroids")
        if verbose:
            print(f"  embeddings << {len(t)} windows (dim {dim})")
    return db


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("v1_store")
    ap.add_argument("out")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    db = migrate_v1(args.v1_store, args.out, args.name)
    for d in db.describe():
        print(f"{d['table']:14s} {d['kind']:12s} {d['rows']:>12,} rows "
              f"{d['bytes'] / 1e6:9.1f} MB  v{d['version']}")
