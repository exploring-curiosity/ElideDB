# FORMAT.md — StreetDex On-Disk Formats

Two formats. Both little-endian, both immutable once written, both designed around one
rule: **a reader must be able to decide what NOT to read using only a small, cheap-to-load
directory.** Every field below has a WHY. If a field's why is unclear, ask before coding.

- **SDX** — columnar sensor/scalar data with zone maps (our mini-Parquet)
- **SFI** — video frame index enabling GOP-granular byte-range decode (our answer to
  "file formats that respect random access into clips")

Conventions: `u32`/`i64` etc. are fixed-width little-endian. `pad` bytes are zero.
All timestamps are `i64` nanoseconds since Unix epoch, UTC. All offsets are absolute
file offsets unless stated. Strings are `u16 length + UTF-8 bytes` (no null terminator).

---

## 1. SDX — sensor columnar format, v1

### 1.1 File layout (top to bottom)

```
offset 0
┌────────────────────────────┐
│ magic  "SDX1"     (4 B)    │  detect wrong file / non-SDX quickly
├────────────────────────────┤
│ chunk 0:                   │
│   col 0 data (ts)          │  ← fixed-width arrays, contiguous per column,
│   col 1 data               │    8-byte aligned (pad between as needed)
│   ...                      │
│ chunk 1: ...               │
│ ...                        │
├────────────────────────────┤
│ FOOTER (see 1.3)           │
├────────────────────────────┤
│ footer_length      u32     │
│ magic  "SDX1"      (4 B)   │
└────────────────────────────┘  ← EOF
```

**WHY footer at the end (same as Parquet):** stats (min/max) and chunk offsets are
unknown until data is written; footer-at-end allows a single-pass, no-seek writer.
Reader cost: one read of the last 8 bytes, then one read of `footer_length` bytes.
Two I/Os to learn everything needed for pruning.

**WHY magic at BOTH ends:** leading magic rejects wrong files cheaply; trailing magic
detects truncated writes (a torn file won't end in "SDX1").

### 1.2 Data region

- One SDX file = one stream (e.g., one IMU, one GPS trace) over one time range.
- Rows are **sorted by timestamp, non-decreasing**. This is an invariant, enforced at
  write time. WHY: sorted ts ⇒ chunk time-ranges are ordered ⇒ pruning is a binary
  search over the directory instead of a full directory scan, and ⇒ zone maps on ts are
  perfectly tight (no overlap between chunks except boundary duplicates).
- Chunk = up to `chunk_target_rows` rows (default **4096**). WHY 4096: at 8 B/value
  that's 32 KiB per column chunk — big enough to amortize a read, small enough that a
  2-second window rarely drags in more than one or two chunks. (Tunable; benchmark it —
  this is a classic row-group-size tradeoff and a great thing to have MEASURED.)
- Within a chunk, columns are stored back-to-back (col 0 fully, then col 1, ...).
  Column 0 is ALWAYS the timestamp column.
- v1 encodings: raw fixed-width only. No compression, no varlen types.
  WHY: mmap + pointer cast = zero-copy reads; random access is the design goal, and
  compression vs. seekability is the central tension in current format research
  (Lance/Vortex). v1 takes the extreme seekability position; v2 may add per-chunk LZ4
  and must then justify the decode cost. Being able to argue this tradeoff is the point.

Column types (`type` byte):

| code | name    | width | notes                          |
|------|---------|-------|--------------------------------|
| 0    | TS_NS   | 8     | i64 ns since epoch; col 0 only |
| 1    | F64     | 8     |                                |
| 2    | F32     | 4     |                                |
| 3    | I64     | 8     |                                |
| 4    | I16     | 2     | added for raw 16-bit PCM audio: the stored bytes are byte-identical to the source WAV samples (immutability extends into the columnar copy), and 40 B/row beats 72 B/row for ts + 16 channels |

Zone-map `min`/`max` are always 8-byte fields; narrow types occupy the low
bytes (high bytes zero). WHY: fixed-width CHUNK_ENTRY regardless of schema.

### 1.3 Footer layout

```
FOOTER :=
  version        u16      (=1)
  flags          u16      (=0, reserved)
  stream_id      string   (e.g. "imu0")
  units          string   (e.g. "m/s^2", may be empty)
  clock_offset_ns i64     (offset of this stream's clock vs. reference clock;
                           measured at ingest — the TV-clock calibration lives here)
  chunk_target_rows u32
  column_count   u16
  COLUMN_DESC × column_count
  chunk_count    u32
  CHUNK_ENTRY × chunk_count

COLUMN_DESC :=
  name   string
  type   u8
  pad    u8

CHUNK_ENTRY :=                       // fixed width once schema is known
  row_count  u32
  pad        u32
  COLSPAN × column_count

COLSPAN :=
  data_offset u64      // absolute file offset of this column chunk
  data_length u64      // bytes
  min         8 B      // raw bit pattern of the column's type
  max         8 B      // (F32 stored in low 4 bytes, high 4 zero)
```

**WHY min/max live in the footer, not next to the data:** pruning must never touch the
data region. The whole directory loads in one read; the decision "skip chunk k" costs
zero data-region I/O. These min/max pairs are our **zone maps** — the same mechanism as
Parquet column statistics and ClickHouse skip indexes. (Say "zone map" in the interview.)

**WHY fixed-width CHUNK_ENTRY:** the directory itself is binary-searchable in place
after mmap — no parsing pass, no allocation.

### 1.4 Read path (normative)

Query: rows with `ts ∈ [t0, t1]`, columns C.

1. Read last 8 bytes → verify magic, get `footer_length`. (I/O #1)
2. Read footer. Verify version. (I/O #2)
3. Binary-search CHUNK_ENTRYs on ts-column (min, max) for the overlap range
   [first_chunk, last_chunk]. Chunks outside → **elided, counted**.
4. For each surviving chunk, for each requested column: read exactly
   (data_offset, data_length). Chunks are checked against [t0,t1] at row level only in
   the first and last surviving chunk (interior chunks are fully inside by sortedness).
5. Return column spans (zero-copy if mmapped).

Bytes read = 8 + footer_length + Σ selected COLSPAN lengths. Report it.

### 1.5 Worked example (do this math again yourself)

Stream: 10,000,000 rows, columns = ts (TS_NS) + accel (F64).
- Data region = 10M × 16 B = **160 MB**.
- Chunks = ceil(10M / 4096) = 2,442.
- CHUNK_ENTRY size = 8 + 2 × 32 = 72 B → directory ≈ **176 KB** (0.11% of file).
- Query: 2-second window at 100 Hz ⇒ 200 rows ⇒ 1 chunk (2 at a boundary).
- Bytes read ≈ 8 + 176 KB + 64 KB ≈ **240 KB** of 160 MB ⇒ **99.85% elided**.

If a number like this doesn't come out of `sdx bench`, something is wrong — the format
guarantees it by construction.

### 1.6 Invariants (each one becomes a test)

1. ts column non-decreasing across the whole file.
2. Every COLSPAN's [offset, offset+length) lies inside the data region; no overlaps.
3. min ≤ max per COLSPAN; chunk k's ts.max ≤ chunk k+1's ts.min (boundary equality OK).
4. row_count ≤ chunk_target_rows; only the final chunk may be short.
5. Trailing magic present ⇔ file is complete (torn-write detection).
6. Pruned read ≡ full-scan read for randomized [t0,t1] (property test).

---

## 2. SFI — StreetDex Frame Index, v1

One SFI file per source video file. The video file itself is **never modified**.
SFI is built once at ingest by a packet-level scan (libavformat, no decode) and makes
this query cheap forever after: *"give me decoded frames for [t0,t1] while reading the
fewest possible bytes of the video file."*

**pts provenance (implemented):** for REIP captures the container's own pts is a
lie — frame_number/fps on the sensor's unsynced local clock. At build time the
packet scan is JOINED with the `time/` JSON sidecar (packet order ↔ frame_id order
per file_index), so `pts_ns` in the FRAME_TABLE is *canonical-timeline* ns from the
fitted global clock. For plain videos with no sidecar, container pts is converted
to ns and used as-is. The join tolerates truncated tail segments (index the known
prefix) and timing bundles that outlive the copied video files.

**A discovery that vindicates the design:** in the lab captures only segment 0 of
each camera is a real RIFF/AVI; segments ≥1 are *headerless concatenations of raw
JPEGs*. Generic tools must linearly parse such a file to find frame k; with SFI the
packet scan pays that cost once, and every later read is `pread(byte_offset,
packet_size)`. The decode path does not even demux: FRAME_TABLE stores exact packet
framing, so packets are fed to the codec straight from the indexed byte ranges
(container opened once per decoder only when a codec keeps parameters in container
extradata, e.g. H.264-in-MP4 — not needed for MJPEG).

**WHY GOP granularity:** compressed video is only independently decodable from a
keyframe (IDR). The minimal correct read unit for a time window is therefore the set of
GOPs overlapping it: seek to the keyframe's byte offset, decode forward, discard frames
before t0. Frame-exact byte access is impossible in H.264/H.265 by design; GOP-exact is
optimal. This is the video analog of a chunk + zone map: the GOP table IS a zone map
over (pts range → byte range).

### 2.1 File layout

```
offset 0
┌──────────────────────────────┐
│ magic "SFI1"        (4 B)    │
│ version   u16   flags u16    │
│ HEADER (fixed, 2.2)          │
│ source_path  string          │
│ pad to 8-byte boundary       │
├──────────────────────────────┤
│ GOP_TABLE   (2.3)            │   gop_count × 48 B, sorted by first_pts
├──────────────────────────────┤
│ FRAME_TABLE (2.4)            │   frame_count × 40 B, sorted by pts
├──────────────────────────────┤
│ gop_table_offset    u64      │
│ frame_table_offset  u64      │
│ magic "SFI1"        (4 B)    │
└──────────────────────────────┘
```

Header first (unlike SDX): SFI is written after the scan completes with all counts known,
so a two-pass writer is trivial and a reader can mmap and jump via the trailing offsets.

### 2.2 HEADER (fixed part)

```
codec_id        u32     // libav AVCodecID, informational
width           u32
height          u32
timebase_num    u32     // source stream time_base
timebase_den    u32
pad             u32
first_pts_ns    i64     // after conversion to ns
last_pts_ns     i64
clock_offset_ns i64     // this camera's clock vs. reference (TV-clock calibrated)
frame_count     u64
gop_count       u64
source_size     u64     // bytes; the elision denominator for this file
```

**WHY store pts in ns, not native time_base:** every consumer of StreetDex speaks
ns-since-epoch; converting once at ingest (pts × num/den × 1e9 + stream start offset)
keeps the align layer free of per-stream unit handling. Keep time_base anyway for
debugging and for seeking via libav if needed.

### 2.3 GOP_TABLE entry (48 B, 8-aligned)

```
gop_id          u32
frame_count     u32     // frames in this GOP
start_byte      u64     // byte offset of keyframe packet in source file
end_byte        u64     // one past last packet byte of this GOP
first_pts_ns    i64
last_pts_ns     i64
first_frame_row u32     // row index into FRAME_TABLE
pad             u32
```

Query [t0,t1] → binary search on (first_pts_ns, last_pts_ns) → list of (start_byte,
end_byte) ranges → those are the ONLY bytes of the video file that get read.

### 2.4 FRAME_TABLE entry (40 B, 8-aligned)

```
pts_ns          i64
dts_ns          i64
byte_offset     u64     // packet position (av_packet->pos)
packet_size     u32
gop_id          u32
flags           u32     // bit0 = keyframe
pad             u32
```

**WHY keep a per-frame table at all if decode is GOP-granular:** (a) frame-exact
timestamps for the align layer (nearest-frame lookup is a binary search here, never a
decode); (b) per-frame provenance in results; (c) future partial-GOP early-stop
(stop decoding once pts > t1 — implement this, it's free elision inside the last GOP).

### 2.5 Read path (normative)

1. mmap SFI; verify magics; jump to tables via trailing offsets. (No video-file I/O yet.)
2. Binary search GOP_TABLE for [t0,t1] → GOP list → byte ranges.
3. Open source video with libavformat, seek to each GOP's start_byte, feed packets
   [start_byte, end_byte) to the decoder; drop decoded frames with pts < t0; stop when
   pts > t1.
4. Emit frames (HWC uint8 + pts_ns). Count: SFI bytes + Σ (end_byte − start_byte) read
   vs. source_size.

### 2.6 Invariants (tests)

1. FRAME_TABLE sorted by pts_ns; GOP_TABLE sorted by first_pts_ns; ranges non-overlapping
   and contiguous in bytes (gop k end_byte == gop k+1 start_byte for typical files —
   assert-warn, not assert-fail; some containers interleave audio).
2. Every frame's gop_id points to a GOP whose [first_pts, last_pts] contains its pts.
3. Exactly one keyframe-flagged frame per GOP, at first_frame_row.
4. Decoded frame count and pts for a full-file query ≡ a naive full decode (golden test
   on a short clip).
5. Sync test: OCR the TV clock on frames returned for known wall-times; |error| < one
   frame interval after clock_offset_ns correction. Record the number.

---

## 3. How SDX/SFI map to the real world (interview table)

| Ours | Real-world analog | The idea |
|---|---|---|
| SDX footer + trailing length | Parquet footer + "PAR1" | single-pass write, 2-I/O metadata load |
| COLSPAN min/max | Parquet column stats / zone maps / CH skip idx | prune before reading data |
| chunk_target_rows | Parquet row-group size | the amortize-vs-overfetch dial |
| no compression in v1 | Lance's random-access-first stance | seekability vs. ratio tension |
| SFI GOP table | Lance structural encodings / an index over opaque blobs | make blob-internal structure visible to the query layer |
| clock_offset_ns in footer | (usually ad-hoc glue code) | alignment metadata belongs IN the format |
| snapshot manifest (CLAUDE.md M5) | Iceberg metadata tree | reproducibility via immutable files + versioned pointers |

## 4. Explicit non-goals of v1 (know these cold — they're follow-up questions)

- **Checksums** (CRC32C per chunk) — v2. Say why they matter (silent corruption on
  object stores) and why v1 skips them (local NVMe, scope).
- **Compression** — v2 option: per-chunk LZ4 with decompressed-size in COLSPAN.
  Tradeoff: ~2-4× smaller, but chunk becomes the atomic read (no zero-copy, no intra-
  chunk offset math). For ts columns, delta + bitpacking would beat LZ4 — mention it.
- **Varlen / string columns** — needs offsets arrays (Arrow-style), out of scope.
- **Appends/updates** — files are immutable; new data = new file + new snapshot.
  This is a feature (versioning, cacheability), not a limitation. Same stance as
  lakehouse table formats.
- **Bloom filters / secondary value indices** — zone maps on sorted ts are enough for
  time queries; value-predicate pruning on unsorted columns would want more. Know the
  limitation: our min/max on a noisy accel column prunes almost nothing — WHY? (Because
  unsorted data ⇒ every chunk's range spans nearly the full domain. Sortedness is what
  makes zone maps sharp. This question WILL come up in some form.)

## 5. Self-test (answer from memory before the interview)

1. Trace every I/O for `get_window(t0, t1)` across one SDX and one SFI stream — how many
   reads, what bytes, in what order?
2. Why must the ts column be sorted, and exactly which two mechanisms break without it?
3. Your accel column's zone maps prune nothing. Why, and what are two fixes?
4. Why can't we read a single frame's bytes and decode just it?
5. What breaks if chunk_target_rows = 64? = 1,000,000?
6. Where does a torn write get detected, and what does the reader do?
7. Why is clock_offset_ns per-stream metadata rather than baked into stored timestamps?
   (Hint: immutability + recalibration.)
8. What would you change first to run this against S3 instead of NVMe? (Hint: I/O count
   vs. I/O size; footer+directory caching; range-request coalescing.)
