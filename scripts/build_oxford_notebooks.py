#!/usr/bin/env python3
"""Build notebooks/oxford_upload.ipynb (NOT executed — the user runs it) and
notebooks/database_operations.ipynb (executed now, outputs baked in)."""
import sys

import nbformat as nbf
from nbclient import NotebookClient

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ═════════════════════════════ oxford_upload ═════════════════════════════
U = []
U.append(md("""# Oxford RobotCar → ElideDB, step by step

This notebook builds the `lake/oxford` database **from the raw dataset**, one
step per cell, and ends with a cell that clears the datastore so you can
rebuild from scratch any time.

**Prerequisites:** `pip install -e ".[ml]"` from the repo root, `brew install
ffmpeg`, and the raw sample at `Data/oxfordDataset/` with this layout:

```
Data/oxfordDataset/
  stereo/{centre,left,right}/   1280×960 raw Bayer PNGs (filename = epoch µs)
  mono_{left,right,rear}/       1024×1024 raw Bayer PNGs
  gps/{gps.csv, ins.csv}        GPS ~5 Hz, INS 50 Hz (timestamp in µs)
  lms_front/, lms_rear/, ldmrs/ lidar scans as float64 .bin files
```

### Where the gigabyte goes (raw 1.04 GB → database ≈ 0.37 GB)

Nothing is dropped — every one of the 1,443 frames and every sensor row ends
up in the database. The size shrinks because the **representation** changes:

| raw form | stored form | why smaller |
|---|---|---|
| Bayer-mosaic PNGs (~250 KB/frame, lossless mosaic) | demosaiced JPEG q92 packed into one AVI per camera | JPEG on natural images ≈ 0.3× lossless mosaic |
| CSV text (gps/ins) | Parquet + zstd, columnar | ~10× smaller than text |
| float64 lidar scans | per-scan summary rows | v1 keeps summaries, not point clouds |

If you need bit-identical raw frames retained, keep the raw folder (ElideDB
never touches it) or ingest with `copy=False` to reference files in place."""))

U.append(code("""# ── 0 · setup ────────────────────────────────────────────────────────────
import sys, shutil, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "python"))

import cv2, numpy as np, pandas as pd
from elidedb import Store, cluster
from elidedb.fftools import find as find_tool

DATASET = Path("../Data/oxfordDataset")   # ← the raw dataset (read-only)
STORE   = Path("../lake/oxford")          # ← the database this notebook builds
assert DATASET.is_dir(), f"raw dataset not found at {DATASET.resolve()}"
print("raw dataset :", DATASET.resolve())
print("database at :", STORE.resolve())"""))

U.append(md("""## 1 · Create the database

A database is just a directory. If this cell says it already exists, either
keep using it or jump to the **clear** cell at the bottom first."""))

U.append(code("""if (STORE / "_store.json").exists():
    db = Store.open(STORE)
    print(f"store already exists — {len(db.tables())} tables. "
          "Run the CLEAR cell at the bottom first for a fresh build.")
else:
    db = Store.create(STORE, "oxford-robotcar-sample")
    print("created empty database:", db.name)"""))

U.append(md("""## 2 · Upload the six cameras

RobotCar cameras store **raw Bayer-mosaic** PNGs — one file per frame, the
filename is the capture time in epoch microseconds. The helper below:

1. demosaics each frame (stereo = GBRG pattern, mono = RGGB),
2. JPEG-encodes (q92) and packs all frames into one AVI per camera
   (a pile of small files has no efficient random-access story; one packed
   file + a byte-range index does),
3. hands ElideDB the AVI **plus the exact per-frame nanosecond timestamps**.

`copy=True` (the default) moves the packed AVI into the store's `media/`
directory, so the store stays a single self-contained folder."""))

U.append(code("""def pack_camera(cam_dir: Path, bayer_code: int, workdir: str):
    \"\"\"raw Bayer PNGs -> (packed .avi path, per-frame ns timestamps)\"\"\"
    frames = sorted(cam_dir.glob("*.png"))
    raw = Path(workdir) / (cam_dir.name + ".mjpeg")
    ts_ns = []
    with open(raw, "wb") as out:
        for p in frames:
            img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), bayer_code)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            assert ok, p
            out.write(jpg.tobytes())
            ts_ns.append(int(p.stem) * 1000)          # µs filename -> ns
    avi = raw.with_suffix(".avi")                      # remux for exact packet
    subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y",  # byte offsets
                    "-f", "mjpeg", "-i", str(raw), "-c:v", "copy", str(avi)],
                   check=True)
    raw.unlink()
    return avi, ts_ns

CAMERAS = {  # dir -> (bayer pattern, stream name)
    "stereo/centre": (cv2.COLOR_BayerGR2BGR, "stereo/centre"),
    "stereo/left":   (cv2.COLOR_BayerGR2BGR, "stereo/left"),
    "stereo/right":  (cv2.COLOR_BayerGR2BGR, "stereo/right"),
    "mono_left":     (cv2.COLOR_BayerBG2BGR, "mono/left"),
    "mono_right":    (cv2.COLOR_BayerBG2BGR, "mono/right"),
    "mono_rear":     (cv2.COLOR_BayerBG2BGR, "mono/rear"),
}

with tempfile.TemporaryDirectory() as wd:
    for rel, (code_, stream) in CAMERAS.items():
        avi, ts_ns = pack_camera(DATASET / rel, code_, wd)
        v = db.ingest_video("frames", avi, timestamps_ns=ts_ns, stream=stream)
        print(f"  {stream:<15} {len(ts_ns):>4} frames  -> frames table v{v}")
print("cameras done — media now lives inside", STORE / "media")"""))

U.append(md("""## 3 · Upload GPS and INS

Straight from the raw CSVs — no preprocessing. `ts_column` names the
dataset's timestamp column; the µs epoch unit is auto-detected. String
columns (`utm_zone`, `ins_status`) are kept as-is: any column type rides
along, only the timestamp is mandatory."""))

U.append(code("""for name in ("gps", "ins"):
    v = db.ingest_rows(name, DATASET / "gps" / f"{name}.csv", ts_column="timestamp")
    print(f"  {name}: {db.table(name).state().rows:,} rows -> v{v}")"""))

U.append(md("""## 4 · Upload lidar activity

The `.bin` scans are float64 arrays — `lms_*` are 2-D (x, y, reflectance),
`ldmrs` is 3-D (x, y, z). We store one summary row per scan (count +
min/mean/max range) so lidar activity is queryable and alignable like any
other sensor. (Full point-cloud storage would just be another table with
more columns.)"""))

U.append(code("""def lidar_summary(scan_dir: Path, dims: int) -> pd.DataFrame:
    rows = []
    for p in sorted(scan_dir.glob("*.bin")):
        a = np.fromfile(p, np.float64).reshape(3, -1)
        r = np.hypot(a[0], a[1]) if dims == 2 else np.sqrt((a ** 2).sum(0))
        rows.append({"ts": int(p.stem) * 1000, "n_points": r.size,
                     "mean_range": r.mean(), "min_range": r.min(),
                     "max_range": r.max()})
    return pd.DataFrame(rows)

for name, dims in (("lms_front", 2), ("lms_rear", 2), ("ldmrs", 3)):
    v = db.ingest_rows(name, lidar_summary(DATASET / name, dims), ts_unit="ns")
    print(f"  {name}: {db.table(name).state().rows:,} scans -> v{v}")"""))

U.append(md("""## 5 · Verify what you uploaded"""))

U.append(code("""import pandas as pd
inv = pd.DataFrame(db.describe())[["table", "kind", "rows", "bytes", "files", "version"]]
media = sum(p.stat().st_size for p in (STORE / "media").glob("*"))
total = sum(p.stat().st_size for p in STORE.rglob("*") if p.is_file())
print(f"database size: {total/1e9:.2f} GB standalone "
      f"(media {media/1e9:.2f} GB, tables {(total-media)/1e6:.1f} MB)")
inv"""))

U.append(md("""## 6 · Embed for semantic search

Local SigLIP (first run downloads the model, ~2 GB, then cached). Windows
are the *indexing* granularity — search results merge them into dynamic
segments — so 1 s is a good default here. Re-run this cell whenever you add
more video."""))

U.append(code("""print(db.embed_windows(window_s=1.0, frames_per_window=2))
print(cluster(db, min_cluster_size=5))"""))

U.append(md("""## 7 · First query"""))

U.append(code("""import matplotlib.pyplot as plt
hits, stats = db.search_text("pedestrians walking on the sidewalk", k=4)
print(stats)
fig, axes = plt.subplots(1, len(hits), figsize=(16, 3.4))
for ax, h in zip(axes, hits):
    w, _ = db.window(h["t0"], h["t1"], tables=["frames"])
    dec = w["frames"].decode(stream=h["stream"], width=400, limit=1)
    if dec:
        ax.imshow(dec[0][1])
    ax.axis("off")
    ax.set_title(f"{h['score']:.3f} · {h['stream']} · "
                 f"{(h['t1']-h['t0'])/1e9:.1f}s", fontsize=9)
plt.tight_layout()"""))

U.append(md("""Browse it visually: `elidedb desk` (or the macOS app) — Overview,
Timeline, **Storage** (the Parquet files and row groups), the semantic map,
and click-to-play clips.

---

## ⚠ Clear the datastore

Deletes `lake/oxford` **entirely** (database only — the raw dataset under
`Data/` is never touched). Set `CONFIRM = True` and run."""))

U.append(code("""CONFIRM = False   # ← set to True, then run this cell

if CONFIRM:
    import shutil
    shutil.rmtree(STORE, ignore_errors=True)
    print("cleared:", STORE.resolve(),
          "— re-run this notebook from the top to rebuild.")
else:
    print("not cleared — set CONFIRM = True first.")"""))

nbu = nbf.v4.new_notebook()
nbu.cells = U
nbf.write(nbu, "notebooks/oxford_upload.ipynb")
print("wrote notebooks/oxford_upload.ipynb (NOT executed — user-run)")

# ═════════════════════════ database_operations ═══════════════════════════
O = []
O.append(md("""# ElideDB — database operations for analysis

Everything you do with a store from Python, on the Oxford database
(`lake/oxford` — build it with `oxford_upload.ipynb` if it doesn't exist):
**inventory · SQL · DataFrames · time windows · frame decode · alignment ·
semantic search (dynamic segments) · clip similarity · time travel ·
export**."""))

O.append(code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "python"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from elidedb import Store

db = Store.open("../lake/oxford")
BASE = min(d["min_ts"] for d in db.describe()
           if d["rows"] and d["table"] != "centroids")
def T(sec):  # capture-relative seconds -> absolute ns
    return BASE + int(sec * 1e9)
def rel(ns):
    return (ns - BASE) / 1e9
print(db.name, "— opened")"""))

O.append(md("""## Inventory — the database as a DataFrame"""))

O.append(code("""pd.DataFrame(db.describe())[["table","kind","rows","bytes","files","version"]]"""))

O.append(md("""## SQL (DuckDB over the store's own Parquet)

Table names in SQL are your table names. Any SELECT works — aggregates,
window functions, joins across tables."""))

O.append(code("""db.sql('''
  SELECT count(*)                                              AS ins_rows,
         round(avg(sqrt(velocity_north^2 + velocity_east^2)),2) AS avg_speed_ms,
         round(max(sqrt(velocity_north^2 + velocity_east^2)),2) AS max_speed_ms,
         round((max(ts) - min(ts)) / 1e9)                       AS span_s
  FROM ins''')"""))

O.append(code("""# join two sensors on 1-second buckets — lidar clutter vs vehicle speed
db.sql('''
  WITH l AS (SELECT ts // 1000000000 AS sec, avg(mean_range) AS lidar_m
             FROM lms_front GROUP BY 1),
       v AS (SELECT ts // 1000000000 AS sec,
                    avg(sqrt(velocity_north^2 + velocity_east^2)) AS speed
             FROM ins GROUP BY 1)
  SELECT l.sec - (SELECT min(sec) FROM l) AS t_s,
         round(lidar_m, 1) AS lidar_range_m, round(speed, 2) AS speed_ms
  FROM l JOIN v USING (sec) ORDER BY 1 LIMIT 10''')"""))

O.append(md("""## Tables → DataFrames, analysis-ready

`scan()` is the pruned columnar read: give it a time range and the columns
you want, get a pyarrow table — `.to_pandas()` and go."""))

O.append(code("""ins = db.table("ins").scan(T(600), T(900),
                           columns=["ts","velocity_north","velocity_east","yaw"]
                           ).to_pandas()
ins["t_s"] = (ins.ts - BASE) / 1e9
ins["speed"] = np.hypot(ins.velocity_north, ins.velocity_east)
ins.describe().round(3)"""))

O.append(code("""# the whole GPS track as a DataFrame + an analysis plot: route colored by speed
gps = db.table("gps").scan().to_pandas()
ins_all = db.table("ins").scan(columns=["ts","velocity_north","velocity_east"]).to_pandas()
ins_all["speed"] = np.hypot(ins_all.velocity_north, ins_all.velocity_east)
gps["speed"] = np.interp(gps.ts, ins_all.ts, ins_all.speed)
fig, ax = plt.subplots(figsize=(7.5, 7))
sc = ax.scatter(gps.longitude, gps.latitude, c=gps.speed, s=3, cmap="viridis")
fig.colorbar(sc, label="speed m/s"); ax.set_title("42-minute route, colored by speed")
ax.set_xlabel("lon"); ax.set_ylabel("lat"); plt.tight_layout()"""))

O.append(md("""## Time-window reads (the core multimodal query)

Everything overlapping [t0, t1], with honest byte accounting. Video comes
back as a lazy `FrameSet`; decoding preads exactly the frames you ask for."""))

O.append(code("""w, stats = db.window(T(651), T(653))
print(stats)
frames = w["frames"].decode(stream="stereo/centre", width=440, stride=6)
fig, axes = plt.subplots(1, len(frames), figsize=(16, 3))
for ax, (ts, img) in zip(axes, frames):
    ax.imshow(img); ax.set_title(f"+{rel(ts):.2f}s", fontsize=9); ax.axis("off")
plt.tight_layout()"""))

O.append(md("""## Query-time alignment

Resample every numeric stream onto one timeline — a tidy DataFrame for
modeling. Alignment is a query option, never a storage commitment."""))

O.append(code("""al, _ = db.aligned(T(645), T(664), rate_hz=20, tables=["ins", "lms_front"],
                   interp="linear")
df = pd.DataFrame({"t_s": (al["timeline_ns"] - BASE) / 1e9})
for tab, cols in al.items():
    if tab == "timeline_ns": continue
    for c, v in cols.items():
        df[f"{tab}.{c}"] = v
df["speed"] = np.hypot(df["ins.velocity_north"], df["ins.velocity_east"])
ax = df.plot(x="t_s", y=["speed", "lms_front.mean_range"], secondary_y=["lms_front.mean_range"],
             figsize=(12, 3), lw=1.4, title="speed vs lidar mean range, one 20 Hz timeline")
df.head(4).round(3)"""))

O.append(md("""## Semantic search — dynamic segments

Results are **merged segments**: consecutive matching windows fuse into one
hit of the event's true duration (no fixed chunks, no sub-clip duplicates).
`stats` reports the per-query threshold and how many windows merged."""))

O.append(code("""hits, stats = db.search_text("pedestrians walking on the sidewalk past buildings", k=5)
print(stats)
hdf = pd.DataFrame(hits)
hdf["duration_s"] = (hdf.t1 - hdf.t0) / 1e9
hdf["start"] = hdf.t0.map(rel).round(1)
hdf[["score","stream","start","duration_s","windows"]]"""))

O.append(code("""fig, axes = plt.subplots(1, len(hits), figsize=(16, 3.2))
for ax, h in zip(axes, hits):
    w, _ = db.window(h["t0"], h["t1"], tables=["frames"])
    dec = w["frames"].decode(stream=h["stream"], width=400, limit=1)
    if dec: ax.imshow(dec[0][1])
    ax.axis("off")
    ax.set_title(f"{h['score']:.3f} · {h['stream']} · {(h['t1']-h['t0'])/1e9:.1f}s",
                 fontsize=9)
plt.tight_layout()"""))

O.append(md("""## Clip similarity — “find moments like this one”"""))

O.append(code("""sim, sstats = db.search_clip("stereo/centre", T(652), T(654), k=4)
pd.DataFrame(sim).assign(start=lambda d: d.t0.map(rel).round(1),
                         duration_s=lambda d: (d.t1-d.t0)/1e9)[
    ["score","stream","start","duration_s","windows"]]"""))

O.append(md("""## Time travel

Every commit is a version; every old version stays readable forever."""))

O.append(code("""print("frames table history:")
hist = pd.DataFrame(db.table("frames").history())
print(f"rows @v2: {db.table('frames').state(2).rows:,}   "
      f"rows now (v{db.table('frames').state().version}): {db.table('frames').state().rows:,}")
hist[["version","op","added_rows","ts_utc"]]"""))

O.append(md("""## Export — take data anywhere

A windowed, projected slice out to Parquet/CSV for any other tool. (Other
engines can also read the store's Parquet files directly — nothing is
locked in.)"""))

O.append(code("""slice_df = db.table("ins").scan(T(600), T(660)).to_pandas()
slice_df.to_parquet("/tmp/oxford_ins_600_660.parquet")
slice_df.to_csv("/tmp/oxford_ins_600_660.csv", index=False)
print(f"exported {len(slice_df):,} rows ->",
      "/tmp/oxford_ins_600_660.parquet + .csv")

# training-data example: dump every frame of a search hit as numpy arrays
h = hits[0]
w, _ = db.window(h["t0"], h["t1"], tables=["frames"])
clip = w["frames"].decode(stream=h["stream"], width=640)
arr = np.stack([f for _, f in clip])
print("clip tensor for training:", arr.shape, arr.dtype)"""))

O.append(md("""---
Also see: `oxford_upload.ipynb` (build/clear this database) ·
`elidedb desk` (visual browser) · `docs/API.md` (full reference)."""))

nbo = nbf.v4.new_notebook()
nbo.cells = O
client = NotebookClient(nbo, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": "notebooks"}})
client.execute()
nbf.write(nbo, "notebooks/database_operations.ipynb")
print("wrote notebooks/database_operations.ipynb (EXECUTED)")
