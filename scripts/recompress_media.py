"""Rebuild a store's media renditions: drop unreferenced frames, and
align GOP boundaries with the RETRIEVAL UNIT.

Three ideas, all measurable:

1. **Only what the store can address.** A subset store copied whole
   media files: lake/bench references 39,026 of 70,436 frames, so 45%
   of its media bytes serve no query. Frames outside every episode are
   unaddressable, and unaddressable bytes are not data.

2. **A GOP the length of an episode.** The old rendition forced a
   keyframe every second (gop_s=1.0), which is a guess; episodes are
   ~7 s, so most of those keyframes bought nothing and cost bytes.

3. **Keyframe at every episode start.** The classic compression-vs-
   random-access tension is decided by putting the GOP boundary where
   the reads are: each episode opens with an IDR and nothing else is
   forced, so a window read fetches that episode's bytes and never
   reaches back into its neighbour. GOP length becomes a property of
   the data's own structure instead of a guessed constant.

The rendition is re-derived from the ORIGINAL source (recorded in the
frames table's commit meta), not from the existing transcode, so the
result carries one generation of loss rather than two.

Timestamps are authoritative in the store: the new packets are paired
with the SAME ts values in order, so episodes, vectors and every
fitted artifact stay valid. The frames table gets a new version;
nothing else is touched.

Usage:
  python scripts/recompress_media.py <store> [--crf 28] [--codec hevc]
                                     [--only file-129] [--dry-run]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.fftools import find  # noqa: E402
from elidedb.log import FileEntry  # noqa: E402
from elidedb.store import Store  # noqa: E402
from elidedb.video import scan_video_packets  # noqa: E402

FPS = 5.0          # within-episode rate of this corpus (measured)
LONG_GOP = 10_000  # only the forced episode keyframes should appear


def originals(store: Store, table: str = "frames") -> dict[str, str]:
    """media stem -> original source path, from the frames commit metas.

    Keyed by the media file's stem (`file-129-f788e50f`) so a store that
    inherited media from another store can resolve provenance through
    `--from-store`: lake/bench was built by copying rows and records NO
    original for its media, which is its own loophole — the recompress
    commit below writes the resolved provenance back in."""
    out = {}
    for h in store.table(table).history():
        m = h.get("meta", {})
        if "source" in m and "original" in m:
            stem = Path(m["source"]).stem
            out[stem] = m["original"]
            out[stem.rsplit("-", 1)[0]] = m["original"]   # file-129
    return out


def runs_of(idx: list[int]) -> list[tuple[int, int]]:
    """Contiguous [start, end] runs over sorted frame indices."""
    out = []
    s = p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
            continue
        out.append((s, p))
        s = p = i
    out.append((s, p))
    return out


RUNS_PER_PASS = 40   # ffmpeg's expression parser OOMs past ~100 terms

# CODEC CHOICE IS MEASURED, NOT ASSUMED (2026-07-28). HEVC compresses
# this corpus better (5.9x vs 3.6x on file-129), but this ffmpeg's
# libx265 silently IGNORES -force_key_frames: the output carried ONE
# IDR NAL in 148 frames (checked at the NAL level, not by trusting
# ffprobe's flags), so an episode read would decode from the top of the
# file. Random access is the product; the smaller file that cannot be
# randomly read is worthless here. libx264 honours forced keyframes
# exactly (107 runs -> 107 IDRs, verified below on every rebuild).


def encode(src: Path, keep_runs, dst: Path, codec: str, crf: int,
           key_out_idx: set[int] | None = None):
    """Select the kept frames, renumber to a constant rate, force a
    keyframe at every requested OUTPUT index.

    `key_out_idx` is the set of output positions that must be IDRs —
    the episode starts. Forcing them at RUN starts instead was measured
    wrong: adjacent episodes with no gap merge into one run (file-129:
    280 episodes, 107 runs), so an episode in the middle of a run had to
    decode from the run's head and the rebuilt store read MORE bytes per
    episode than the one it replaced (395 KB vs 352 KB). The seek target
    is the episode, so the keyframe goes at the episode.

    Done in passes of RUNS_PER_PASS runs: a `select` expression of ~107
    `between()` terms makes ffmpeg's expression parser fail with
    "Cannot allocate memory" (measured; 48 terms is fine). Each pass
    emits an Annex-B elementary stream that BEGINS with parameter sets
    and an IDR, so concatenating the passes yields one valid stream.
    Byte offsets are re-derived from the concatenated file afterwards,
    so the split leaves no trace in the index."""
    enc = "libx265" if codec == "hevc" else "libx264"
    total = 0
    parts = []
    base = 0
    for p in range(0, len(keep_runs), RUNS_PER_PASS):
        chunk = keep_runs[p:p + RUNS_PER_PASS]
        sel = "+".join(f"between(n,{a},{b})" for a, b in chunk)
        n = sum(b - a + 1 for a, b in chunk)
        if key_out_idx is None:
            starts, off = [], 0
            for a, b in chunk:
                starts.append(off / FPS)
                off += b - a + 1
        else:
            starts = sorted((i - base) / FPS for i in key_out_idx
                            if base <= i < base + n)
        base += n
        part = dst.with_suffix(f".part{p}.{codec}")
        cmd = [find("ffmpeg"), "-v", "error", "-y", "-i", str(src),
               "-vf", f"select='{sel}',setpts=N/{FPS}/TB",
               "-r", str(FPS), "-an",
               "-c:v", enc, "-preset", "medium", "-crf", str(crf),
               # NO B-FRAMES. The frame index pairs one ts with one
               # packet offset, which is only meaningful when packet
               # (decode) order equals presentation order. The rendition
               # being replaced here HAS B-frames (I,B,P,B,P measured),
               # so its rows pair presentation-ordered timestamps with
               # decode-ordered packets — the latent bug that made this
               # rebuild decode the wrong scenes until it was found.
               "-bf", "0",
               "-g", str(LONG_GOP), "-keyint_min", str(LONG_GOP),
               "-force_key_frames", ",".join(f"{t:.4f}" for t in starts)]
        if codec == "hevc":
            cmd += ["-x265-params", "log-level=error"]
        cmd += ["-f", codec, str(part)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exit {r.returncode}\n{r.stderr[-2000:]}")
        parts.append(part)
        total += n
    with open(dst, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
            part.unlink()
    return total


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    crf = int(flags[flags.index("--crf") + 1]) if "--crf" in flags else 26
    codec = flags[flags.index("--codec") + 1] if "--codec" in flags else "h264"
    only = flags[flags.index("--only") + 1] if "--only" in flags else None
    dry = "--dry-run" in flags

    store = Store.open(args[0])
    frames = store.table("frames")
    tbl = frames.scan()
    orig = originals(store)
    all_ts: dict[str, list[int]] = {}
    if "--from-store" in flags:
        parent = Store.open(flags[flags.index("--from-store") + 1])
        orig = {**originals(parent), **orig}
        pf = parent.table("frames").scan()
        for s, t in zip(pf.column("source").to_pylist(),
                        pf.column("ts").to_pylist()):
            all_ts.setdefault(s, []).append(int(t))
    cols = {c: tbl.column(c).to_pylist() for c in tbl.schema.names}
    sources = sorted(set(cols["source"]))
    eps = store.table("episodes").scan()
    ep_t0 = [int(v) for v in eps.column("ts").to_pylist()]
    ep_t1 = [int(v) for v in eps.column("t1").to_pylist()]

    rebuilt, before, after = {}, 0, 0
    provenance: dict[str, str] = {}
    for sref in sources:
        if only and only not in sref:
            continue
        media = store.dir / sref.replace("@media/", "media/")
        stem = Path(sref).stem
        found = orig.get(stem) or orig.get(stem.rsplit("-", 1)[0])
        if not found:
            print(f"  SKIP {sref}: no recorded original "
                  f"(try --from-store <parent>)")
            continue
        src = Path(found)
        if not src.exists():
            print(f"  SKIP {sref}: original missing ({src})")
            continue
        # Source frame index = the RANK OF THE TIMESTAMP among all of
        # that file's frames, taken from the store that ingested the
        # whole file. Deriving it from packet positions instead was
        # measured wrong: the rendition carries B-frames, so packet
        # (decode) order is not presentation order, and `select`'s `n`
        # counts presentation frames — 11 of 16 sampled episodes came
        # back as visibly different scenes.
        full_ts = all_ts.get(sref)
        if not full_ts:
            print(f"  SKIP {sref}: need the whole file's frame list "
                  f"(pass --from-store <parent>)")
            continue
        ts_to_n = {t: n for n, t in enumerate(sorted(full_ts))}
        rows = [i for i, s in enumerate(cols["source"]) if s == sref]
        rows.sort(key=lambda i: cols["ts"][i])
        try:
            idx = [ts_to_n[int(cols["ts"][i])] for i in rows]
        except KeyError as e:
            print(f"  SKIP {sref}: ts {e} absent from the parent's "
                  f"frame list")
            continue
        assert idx == sorted(idx), "frame order disagrees with ts order"
        keep = runs_of(idx)
        # the seek target is the EPISODE, so that is where the IDRs go
        ts_out = [int(cols["ts"][i]) for i in rows]
        pos_of_ts = {t: j for j, t in enumerate(ts_out)}
        key_out = set()
        for e0, e1 in zip(ep_t0, ep_t1):
            if e0 in pos_of_ts:
                key_out.add(pos_of_ts[e0])
                continue
            nxt = [t for t in ts_out if e0 <= t <= e1]
            if nxt:
                key_out.add(pos_of_ts[min(nxt)])
        print(f"{sref}: {len(rows)} of {len(full_ts)} frames "
              f"referenced, {len(keep)} runs, {len(key_out)} episode starts")
        if dry:
            continue
        dst = media.with_name(media.stem.split("-")[0] + "-" +
                              media.stem.split("-")[1] + "-" +
                              uuid.uuid4().hex[:8] + f".{codec}")
        provenance[f"@media/{dst.name}"] = str(src)
        n_out = encode(src, keep, dst, codec, crf, key_out)
        assert n_out == len(rows), f"{n_out} encoded vs {len(rows)} rows"
        new = scan_video_packets(dst)
        assert len(new["ts"]) == len(rows), (
            f"{len(new['ts'])} packets in rendition vs {len(rows)} rows")
        # CORRECTNESS GATE: every episode must open with a keyframe, or
        # a window read has to decode backwards into its neighbour and
        # the whole point of the rendition is lost.
        got = {i for i, k in enumerate(new["keyframe"]) if k}
        missing = key_out - got
        assert not missing, (
            f"{len(missing)} of {len(starts)} episode starts are not "
            f"keyframes (first: {sorted(missing)[:5]}) — the encoder "
            f"ignored -force_key_frames")
        before += os.path.getsize(media)
        after += os.path.getsize(dst)
        print(f"  {os.path.getsize(media)/1e6:.1f} MB -> "
              f"{os.path.getsize(dst)/1e6:.1f} MB  "
              f"({os.path.getsize(media)/os.path.getsize(dst):.2f}x)  "
              f"keyframes {sum(1 for k in new['keyframe'] if k)}")
        rebuilt[sref] = (dst, rows, new)

    if dry or not rebuilt:
        return

    # rewrite the frames table: rebuilt sources take their new byte
    # ranges, untouched sources keep theirs
    out = {c: list(v) for c, v in cols.items()}
    for sref, (dst, rows, new) in rebuilt.items():
        for j, i in enumerate(rows):
            out["byte_offset"][i] = new["byte_offset"][j]
            out["packet_size"][i] = new["packet_size"][j]
            out["keyframe"][i] = new["keyframe"][j]
            out["codec"][i] = new["codec"][j]
            out["source"][i] = f"@media/{dst.name}"
    t = pa.table({c: pa.array(v, type=tbl.schema.field(c).type)
                  for c, v in out.items()})
    t = t.take(pa.compute.sort_indices(t.column("ts")))
    new_name = f"part-{uuid.uuid4().hex[:12]}.parquet"
    import pyarrow.parquet as pq
    pq.write_table(t, frames.dir / new_name, row_group_size=8192,
                   compression="zstd", write_statistics=True)
    st = frames.state()
    version = frames.log.commit(
        # a frame index registered as kind="timeseries" (what stores
        # assembled by filtering another store ended up with) makes
        # window() return a bare Arrow table instead of a FrameSet, the
        # quirk desk.py carries a workaround for. Correct it here.
        op="recompress_media", kind="frame_index",
        schema="\n".join(f"{f.name}: {f.type}" for f in t.schema),
        add=[FileEntry(new_name, len(t),
                       (frames.dir / new_name).stat().st_size,
                       int(pa.compute.min(t.column("ts")).as_py()),
                       int(pa.compute.max(t.column("ts")).as_py()))],
        remove=[f.path for f in st.files],
        meta={**st.meta, "media_policy":
              f"{codec}-crf{crf}-idr-per-episode",
              "media_note": "unreferenced frames dropped; keyframe at "
                            "every episode start",
              # provenance this store previously did not record at all
              "originals": provenance})
    for sref, (dst, _, _) in rebuilt.items():
        old = store.dir / sref.replace("@media/", "media/")
        old.unlink(missing_ok=True)
    print(f"\nframes v{version}: media {before/1e6:.1f} MB -> "
          f"{after/1e6:.1f} MB ({before/max(after,1):.2f}x)")
    print(json.dumps({"codec": codec, "crf": crf,
                      "policy": "idr-per-episode"}))


if __name__ == "__main__":
    main()
