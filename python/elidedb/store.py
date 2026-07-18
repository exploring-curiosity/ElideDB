"""Store: a directory of Parquet tables under a transaction log.

Everything is Parquet — sensor rows, the video frame index, embeddings,
centroids. No custom byte formats. The design borrows one idea from each
system it wants to be judged against:

- warehouses (Vertica/BigQuery): columnar + statistics pruning, projection
  pushdown — Parquet row groups and column chunks give both natively.
- Spark: predicate pushdown to the scan; a query touches the minimum files,
  row groups, and columns.
- Delta/Iceberg: the table is a log fold; snapshots, time travel, atomic
  appends (log.py).
- C-Store: late materialization — video pixels are produced last, from byte
  ranges the frame-index table points at; the raw media file is never copied
  into the store.
- Kafka/streaming: time is the primary axis; every table MUST carry `ts`
  (int64 ns, sorted within a file) — that is the one schema law here.
- lakehouse: open format means other engines read the store for free;
  `Store.sql()` is DuckDB pointed at the very same files.
"""
from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .log import FileEntry, TableLog

ROW_GROUP_ROWS = 64 * 1024  # the amortize-vs-overfetch dial (Parquet's
                            # row-group size == SDX's chunk_target_rows)


class QueryStats:
    """Bytes accounting: the elision number, file-granular.

    Log-level pruning is exact (a pruned file contributes 0 bytes). Inside a
    surviving file, Parquet's own row-group pruning + column projection cut
    further; we report the surviving files' bytes as the upper bound actually
    mapped, plus rows returned."""

    def __init__(self):
        self.corpus_bytes = 0
        self.files_total = 0
        self.files_touched = 0
        self.bytes_touched = 0
        self.rows_returned = 0
        self.wall_ms = 0.0

    @property
    def elided_pct(self):
        if not self.corpus_bytes:
            return 0.0
        b = min(self.bytes_touched, self.corpus_bytes)
        return 100.0 * (self.corpus_bytes - b) / self.corpus_bytes

    def __repr__(self):
        return (f"<{self.files_touched}/{self.files_total} files, "
                f"{self.bytes_touched:,} B touched of {self.corpus_bytes:,} B "
                f"({self.elided_pct:.3f}% elided), {self.rows_returned:,} rows, "
                f"{self.wall_ms:.1f} ms>")


class Table:
    def __init__(self, store: "Store", name: str):
        self.store = store
        self.name = name
        self.dir = store.dir / "tables" / name
        self.log = TableLog(self.dir)

    def state(self, version=None):
        return self.log.read_state(version)

    def history(self):
        return self.log.history()

    # ---- write path -------------------------------------------------------
    def append(self, table: pa.Table, *, kind="timeseries", meta=None) -> int:
        if "ts" not in table.column_names:
            raise ValueError(f"table '{self.name}': a `ts` int64-ns column is "
                             "required — timestamps are the one schema law")
        ts = table.column("ts")
        if ts.type != pa.int64():
            raise ValueError("`ts` must be int64 nanoseconds since epoch")
        order = pc.sort_indices(ts)
        if not pc.all(pc.equal(order, pa.array(range(len(table))))).as_py():
            table = table.take(order)  # ts-sorted files ⇒ tight zone maps
        self.dir.mkdir(parents=True, exist_ok=True)
        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        path = self.dir / fname
        pq.write_table(table, path, row_group_size=ROW_GROUP_ROWS,
                       compression="zstd")
        tsv = table.column("ts").to_numpy()
        entry = FileEntry(fname, len(table), path.stat().st_size,
                          int(tsv[0]), int(tsv[-1]))
        return self.log.commit(op="append", kind=kind,
                               schema=str(table.schema), add=[entry],
                               meta=meta)

    # ---- read path --------------------------------------------------------
    def scan(self, t0=None, t1=None, columns=None, version=None,
             stats: QueryStats | None = None) -> pa.Table:
        st = self.state(version)
        stats = stats if stats is not None else QueryStats()
        stats.files_total += len(st.files)
        stats.corpus_bytes += st.bytes
        if columns is not None and "ts" not in columns:
            columns = ["ts", *columns]  # ts always rides along: it is the
                                        # sort key and the alignment axis
        # layer 1: log-level file pruning (zone maps in the commit entries)
        files = [f for f in st.files
                 if not (t0 is not None and f.max_ts < t0)
                 and not (t1 is not None and f.min_ts > t1)]
        stats.files_touched += len(files)
        # layer 2 accounting: row-group zone maps from the Parquet footer.
        # A surviving file is charged its footer + only the row groups whose
        # ts min/max overlap the window — which is exactly what the reader
        # below will materialize. Same math as warehouse skip-indexes.
        for f in files:
            pf = pq.ParquetFile(self.dir / f.path)
            md = pf.metadata
            footer_bytes = md.serialized_size
            ts_idx = md.schema.names.index("ts") if "ts" in md.schema.names else 0
            touched = 0
            for rg in range(md.num_row_groups):
                g = md.row_group(rg)
                st_ts = g.column(ts_idx).statistics
                if st_ts is not None and t0 is not None and st_ts.max < t0:
                    continue
                if st_ts is not None and t1 is not None and st_ts.min > t1:
                    continue
                for c in range(g.num_columns):
                    col = g.column(c)
                    name = md.schema.names[c] if c < len(md.schema.names) else ""
                    if columns is None or name in columns or c == ts_idx:
                        touched += col.total_compressed_size
            stats.bytes_touched += footer_bytes + touched
        if not files:
            empty = pa.schema([("ts", pa.int64())])
            return pa.table({"ts": pa.array([], pa.int64())}).cast(empty)
        # layer 2: Parquet row-group pruning + projection pushdown
        filt = None
        if t0 is not None:
            filt = pc.field("ts") >= t0
        if t1 is not None:
            c = pc.field("ts") <= t1
            filt = c if filt is None else filt & c
        parts = [pq.read_table(self.dir / f.path, columns=columns,
                               filters=filt) for f in files]
        out = pa.concat_tables(parts, promote_options="permissive")
        if len(parts) > 1:  # files may interleave in time across streams
            out = out.take(pc.sort_indices(out.column("ts")))
        stats.rows_returned += len(out)
        return out


class Store:
    FORMAT = "elidedb/2"

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        meta_path = self.dir / "_store.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{path} is not a store (no _store.json) — Store.create() it")
        self.meta = json.loads(meta_path.read_text())

    # ---- lifecycle --------------------------------------------------------
    @staticmethod
    def create(path: str | Path, name: str) -> "Store":
        p = Path(path)
        (p / "tables").mkdir(parents=True, exist_ok=True)
        meta = {"format": Store.FORMAT, "name": name,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (p / "_store.json").write_text(json.dumps(meta, indent=1))
        return Store(p)

    @staticmethod
    def open(path: str | Path) -> "Store":
        return Store(path)

    @property
    def name(self):
        return self.meta["name"]

    def tables(self) -> list[str]:
        root = self.dir / "tables"
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "_log").is_dir())

    def table(self, name: str) -> Table:
        return Table(self, name)

    def describe(self) -> list[dict]:
        out = []
        for name in self.tables():
            st = self.table(name).state()
            out.append({"table": name, "kind": st.kind, "version": st.version,
                        "rows": st.rows, "bytes": st.bytes,
                        "min_ts": st.min_ts, "max_ts": st.max_ts,
                        "files": len(st.files), "meta": st.meta})
        return out

    # ---- friendly ingest --------------------------------------------------
    def ingest_rows(self, table: str, data, ts_column="ts", ts_unit="auto",
                    meta=None) -> int:
        """Append rows from a pandas DataFrame / dict of arrays / pyarrow
        Table / CSV / Parquet path.

        Friendly on purpose: `ts_column` may be a datetime column, an ISO-8601
        string column, or epoch numbers in s/ms/us/ns — `ts_unit="auto"`
        detects the epoch unit by magnitude (an explicit unit always wins)."""
        import pandas as pd
        if isinstance(data, (str, Path)):
            p = str(data)
            data = (pd.read_parquet(p) if p.endswith((".parquet", ".pq"))
                    else pd.read_csv(p))
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            if ts_column not in data.columns:
                raise ValueError(
                    f"no column '{ts_column}' — available: "
                    f"{list(data.columns)} (pass ts_column=...)")
            df = data.rename(columns={ts_column: "ts"}).copy()
            col = df["ts"]
            if pd.api.types.is_datetime64_any_dtype(col):
                df["ts"] = col.astype("int64")  # datetime64 is already ns
            elif col.dtype == object or pd.api.types.is_string_dtype(col):
                df["ts"] = pd.to_datetime(col).astype("int64")  # ISO strings
            else:
                if ts_unit == "auto":
                    # epoch magnitude: seconds ~1e9, ms ~1e12, us ~1e15, ns ~1e18
                    m = float(pd.Series(col).abs().median())
                    ts_unit = ("s" if m < 1e11 else "ms" if m < 1e14
                               else "us" if m < 1e17 else "ns")
                mult = {"ns": 1, "us": 1_000, "ms": 1_000_000,
                        "s": 1_000_000_000}[ts_unit]
                df["ts"] = (col.astype("float64") * mult).round().astype("int64")
            data = pa.Table.from_pandas(df, preserve_index=False)
        return self.table(table).append(data, meta=meta)

    def ingest_video(self, table: str, video_path, timestamps_ns=None,
                     stream=None, meta=None) -> int:
        """Index a video file: packet scan → frame_index Parquet rows. The
        media file itself is never copied — the table stores byte ranges."""
        from .video import scan_video_packets
        rows = scan_video_packets(video_path, timestamps_ns)
        rows["stream"] = [stream or Path(video_path).stem] * len(rows["ts"])
        t = pa.table(rows)
        return self.table(table).append(
            t, kind="frame_index",
            meta={"source": str(Path(video_path).resolve()), **(meta or {})})

    # ---- queries ----------------------------------------------------------
    def window(self, t0: int, t1: int, tables=None, columns=None,
               version=None):
        """The multimodal read: every requested table filtered to [t0, t1].
        frame_index tables come back as FrameSet (lazy byte-range decode)."""
        from .video import FrameSet
        stats = QueryStats()
        start = time.perf_counter()
        out = {}
        for name in (tables or self.tables()):
            tab = self.table(name)
            st = tab.state(version)
            cols = columns.get(name) if isinstance(columns, dict) else columns
            data = tab.scan(t0, t1, columns=cols, version=version, stats=stats)
            out[name] = FrameSet(self, name, data) if st.kind == "frame_index" \
                else data
        stats.wall_ms = (time.perf_counter() - start) * 1e3
        return out, stats

    def aligned(self, t0, t1, rate_hz, tables=None, interp="nearest",
                version=None):
        """Query-time alignment: resample numeric columns of the requested
        timeseries tables onto one [t0, t1] timeline at rate_hz."""
        timeline = np.arange(t0, t1 + 1, int(1e9 / rate_hz), dtype=np.int64)
        guard = int(1e9)  # neighbors just outside the window make edges exact
        out = {"timeline_ns": timeline}
        stats = QueryStats()
        for name in (tables or self.tables()):
            tab = self.table(name)
            if tab.state(version).kind != "timeseries":
                continue
            data = tab.scan(t0 - guard, t1 + guard, version=version, stats=stats)
            if len(data) == 0:
                continue
            ts = data.column("ts").to_numpy()
            cols = {}
            for cname in data.column_names:
                if cname == "ts":
                    continue
                arr = data.column(cname)
                if not pa.types.is_floating(arr.type) and \
                   not pa.types.is_integer(arr.type):
                    continue
                v = arr.to_numpy().astype(np.float64)
                if interp == "linear":
                    cols[cname] = np.interp(timeline, ts, v)
                else:  # nearest
                    idx = np.searchsorted(ts, timeline)
                    idx = np.clip(idx, 0, len(ts) - 1)
                    prev = np.clip(idx - 1, 0, len(ts) - 1)
                    use_prev = (timeline - ts[prev]) <= (ts[idx] - timeline)
                    cols[cname] = v[np.where(use_prev, prev, idx)]
            out[name] = cols
        return out, stats

    def sql(self, query: str, version=None):
        """DuckDB over the store's own Parquet files — the lakehouse dividend:
        because the format is open, a whole second engine comes for free.
        Table names in the query = store table names."""
        import duckdb
        con = duckdb.connect()
        for name in self.tables():
            st = self.table(name).state(version)
            files = [str(self.dir / "tables" / name / f.path) for f in st.files]
            if files:
                quoted = ", ".join(f"'{f}'" for f in files)
                con.execute(
                    f'CREATE VIEW "{name}" AS SELECT * FROM '
                    f"read_parquet([{quoted}])")
        return con.execute(query).fetchdf()

    # ---- semantic layer (see embeddings.py) --------------------------------
    def embed_windows(self, frame_table="frames", window_s=2.0,
                      frames_per_window=2, model=None, batch=16):
        from .embeddings import embed_windows
        return embed_windows(self, frame_table, window_s, frames_per_window,
                             model, batch)

    def search_text(self, text: str, k=10, nprobe=3):
        from .embeddings import search_text
        return search_text(self, text, k, nprobe)

    def search_clip(self, stream: str, t0: int, t1: int, k=10, nprobe=3):
        from .embeddings import search_clip
        return search_clip(self, stream, t0, t1, k, nprobe)
