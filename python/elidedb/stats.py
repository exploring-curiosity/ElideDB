"""Read accounting: the elision number, and the rule that it stay honest.

Kept in its own module because every claim this project makes is a byte
count, so the thing producing byte counts should not sit buried in table
semantics where a bug in it is invisible. One was: a nested-column
naming mismatch meant vector columns were never charged, and a full
vector scan reported 5,493 bytes where it actually read 43,705,404.
"""
from __future__ import annotations


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


def _fold(dst, src):
    """Accumulate one QueryStats into another.

    corpus_bytes takes a MAX, not a sum: it is a denominator, and adding
    denominators across calls would inflate it until the elision figure
    became meaningless. The measure() block seeds it with the whole
    store, which is the largest and the correct one; a caller's own
    stats keeps whatever per-table denominator it had.
    """
    if dst is None:
        return
    dst.files_total += src.files_total
    dst.files_touched += src.files_touched
    dst.bytes_touched += src.bytes_touched
    dst.rows_returned += src.rows_returned
    dst.corpus_bytes = max(dst.corpus_bytes, src.corpus_bytes)

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
                    meta=None, evolve=False) -> int:
        """Append rows from a pandas DataFrame / dict of arrays / pyarrow
        Table / CSV / Parquet path.

        Friendly on purpose: `ts_column` may be a datetime column, an ISO-8601
        string column, or epoch numbers in s/ms/us/ns — `ts_unit="auto"`
        detects the epoch unit by magnitude (an explicit unit always wins)."""
        import os as _os

        import pandas as pd
        if isinstance(data, (str, Path)):
            p = str(data)
            if p.endswith((".parquet", ".pq")):
                data = pd.read_parquet(p)
            elif _os.path.getsize(p) > 128 * 1024 * 1024:
                # memory-bounded load: stream the CSV in chunks, one file per
                # chunk, ONE atomic commit for the whole load
                def gen():
                    for chunk in pd.read_csv(p, chunksize=2_000_000):
                        yield self._normalize_ts(chunk, ts_column, ts_unit)
                return self.table(table).append_batches(gen(), meta=meta)
            else:
                data = pd.read_csv(p)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            data = self._normalize_ts(data, ts_column, ts_unit)
        return self.table(table).append(data, meta=meta, evolve=evolve)

    @staticmethod
    def _normalize_ts(df, ts_column, ts_unit) -> pa.Table:
        import pandas as pd
        if ts_column not in df.columns:
            raise ValueError(
                f"no column '{ts_column}' — available: "
                f"{list(df.columns)} (pass ts_column=...)")
        df = df.rename(columns={ts_column: "ts"}).copy()
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
        return pa.Table.from_pandas(df, preserve_index=False)

    def _media_dest(self, src: Path) -> Path:
        import hashlib
        h = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
        media = self.dir / "media"
        media.mkdir(exist_ok=True)
        return media / f"{src.stem}-{h}{src.suffix}"

    def ingest_video(self, table: str, video_path, timestamps_ns=None,
                     stream=None, meta=None, copy=True, transcode=None,
                     gop_s: float = 1.0, crf: int = 26) -> int:
        """Index a video file: packet scan → frame_index Parquet rows.

        copy=True (default): the media file is copied into the store's
        `media/` directory first, so the store directory IS the complete,
        portable database. copy=False indexes the file in place.

        transcode="hevc"|"h264": re-encode the managed copy as a compressed
        elementary stream (≈10-25x smaller than MJPEG at like quality) with a
        forced keyframe every `gop_s` seconds. Random access becomes
        GOP-granular instead of frame-exact — `gop_s` IS the seekability-vs-
        compression dial, chosen per table at ingest, and decode reads
        exactly one GOP span per window."""
        import shutil
        import subprocess

        from .fftools import find
        from .video import scan_video_packets
        src = Path(video_path)
        if transcode:
            assert transcode in ("hevc", "h264")
            if timestamps_ns is None:  # take pts from the source container
                probe = scan_video_packets(src)
                timestamps_ns = probe["ts"].to_pylist()
            n_in = len(timestamps_ns)
            span_s = max((timestamps_ns[-1] - timestamps_ns[0]) / 1e9, 0.1)
            fps = max((n_in - 1) / span_s, 1.0)
            g = max(1, round(gop_s * fps))
            dest = self._media_dest(src).with_suffix(f".{transcode}")
            if not dest.exists():
                enc = "libx265" if transcode == "hevc" else "libx264"
                subprocess.run(
                    [find("ffmpeg"), "-v", "error", "-y", "-i", str(src),
                     "-c:v", enc, "-preset", "fast", "-crf", str(crf),
                     "-g", str(g), "-keyint_min", str(g), "-an",
                     "-f", transcode, str(dest)], check=True)
            scanned_path, source_ref = dest, f"@media/{dest.name}"
        elif copy:
            dest = self._media_dest(src)
            if not dest.exists():
                shutil.copy2(src, dest)
            scanned_path, source_ref = dest, f"@media/{dest.name}"
        else:
            scanned_path = src
            source_ref = str(src.resolve())
        rows = scan_video_packets(scanned_path, timestamps_ns)
        n = len(rows["ts"])
        rows["source"] = pa.array([source_ref] * n)
        rows["stream"] = [stream or src.stem] * n
        t = pa.table(rows)
        return self.table(table).append(
            t, kind="frame_index",
            meta={"source": source_ref, "original": str(src.resolve()),
                  **({"transcode": transcode, "gop_s": gop_s, "crf": crf}
                     if transcode else {}),
                  **(meta or {})})

    def adopt_media(self, table: str = "frames", verbose=True) -> dict:
        """Make the store standalone: copy every externally-referenced media
        file into `media/` and rewrite the frame index to store-relative
        paths. One replace-commit per call — old index versions still resolve
        (the external files are not deleted)."""
        import shutil
        import uuid as _uuid
        from .log import FileEntry
        tab = self.table(table)
        st = tab.state()
        if st.kind != "frame_index":
            raise ValueError(f"{table} is not a frame_index table")
        t = tab.scan()
        srcs = t.column("source").to_pylist()
        external = sorted({s for s in srcs if not s.startswith("@")})
        if not external:
            return {"adopted": 0, "bytes": 0}
        mapping, copied = {}, 0
        for s in external:
            p = Path(s)
            if not p.exists():
                raise FileNotFoundError(f"referenced media missing: {s}")
            dest = self._media_dest(p)
            if not dest.exists():
                shutil.copy2(p, dest)
            copied += dest.stat().st_size
            mapping[s] = f"@media/{dest.name}"
            if verbose:
                print(f"  adopted {p.name} -> media/{dest.name}")
        new_src = pa.array([mapping.get(s, s) for s in srcs])
        t = t.set_column(t.column_names.index("source"), "source", new_src)
        fname = f"part-{_uuid.uuid4().hex[:12]}.parquet"
        path = self.dir / "tables" / table / fname
        write_parquet(t, path)
        tsv = t.column("ts").to_numpy()
        tab.log.commit(op="adopt-media", kind="frame_index",
                       schema=str(t.schema),
                       add=[FileEntry(fname, len(t), path.stat().st_size,
                                      int(tsv.min()), int(tsv.max()))],
                       remove=[f.path for f in st.files],
                       meta={"media_files": len(external)})
        return {"adopted": len(external), "bytes": copied}

    # ---- queries ----------------------------------------------------------
    def window(self, t0: int, t1: int, tables=None, columns=None,
               version=None):
        """The multimodal read: every requested table filtered to [t0, t1].
        frame_index tables come back as FrameSet (lazy byte-range decode)."""
        from .video import FrameSet
        stats = QueryStats()
        start = time.perf_counter()
        out = {}
        names = tables
        if names is None:
            # Default to the DATA tables. Index artifacts (embeddings,
            # centroids, frame_vectors, context, ...) are timestamped too, so
            # they would otherwise be dragged into every window read and drag
            # thousands of 1152-d vectors with them. Ask for them by name and
            # you still get them.
            names = [n for n in self.tables()
                     if self.table(n).state(self._ver(version, n)).kind
                     not in ("embeddings", "centroids")]
        for name in names:
            tab = self.table(name)
            v = self._ver(version, name)
            st = tab.state(v)
            cols = columns.get(name) if isinstance(columns, dict) else columns
            data = tab.scan(t0, t1, columns=cols, version=v, stats=stats)
            out[name] = FrameSet(self, name, data) if st.kind == "frame_index" \
                else data
        stats.wall_ms = (time.perf_counter() - start) * 1e3
        return out, stats

    def aligned(self, t0, t1, rate_hz, tables=None, interp="nearest",
                version=None, edge_guard_s: float = 1.0):
        """Query-time alignment: resample numeric columns of the requested
        timeseries tables onto one [t0, t1] timeline at rate_hz."""
        timeline = np.arange(t0, t1 + 1, int(1e9 / rate_hz), dtype=np.int64)
        guard = int(edge_guard_s * 1e9)  # neighbors just outside the window
                                         # make edge interpolation exact
        out = {"timeline_ns": timeline}
        stats = QueryStats()
        for name in (tables or self.tables()):
            tab = self.table(name)
            v = self._ver(version, name)
            if tab.state(v).kind != "timeseries":
                continue
            data = tab.scan(t0 - guard, t1 + guard, version=v, stats=stats)
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
            st = self.table(name).state(self._ver(version, name))
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

    def search(self, text: str, k=10, nprobe=3, merge=True, t0=None, t1=None,
               streams=None, method="auto", neg_weight=0.5, min_score=None,
               percentile=None, rerank=False, rerank_top=12,
               rerank_alpha=0.7):
        """Compositional text search. `text` supports AND / NOT / -term;
        `min_score`/`percentile` add a precision floor. See embeddings.search."""
        from .embeddings import search
        return search(self, text, k=k, nprobe=nprobe, merge=merge, t0=t0,
                      t1=t1, streams=streams, method=method,
                      neg_weight=neg_weight, min_score=min_score,
                      percentile=percentile, rerank=rerank,
                      rerank_top=rerank_top, rerank_alpha=rerank_alpha)

    def search_text(self, text: str, k=10, nprobe=3, **kw):
        from .embeddings import search_text
        return search_text(self, text, k=k, nprobe=nprobe, **kw)

    def search_clip(self, stream: str, t0: int, t1: int, k=10, nprobe=3, **kw):
        """Query-by-example. When the store carries a V-JEPA clip index
        the neighbor space is the WORLD MODEL's (video-native, motion-
        structured, no text anywhere); otherwise appearance windows.
        Measured (bridge4h): V-JEPA beats appearance on action-class
        neighbor purity for 'open' (0.51 vs 0.40) and ties elsewhere."""
        try:
            from .embeddings import _vec_table
            import numpy as np
            tbl, vecs = _vec_table(self, "vjepa_vectors")
            ss = tbl.column("stream").to_pylist()
            sa = [int(v) for v in tbl.column("ts").to_pylist()]
            sb = [int(v) for v in tbl.column("t1").to_pylist()]
            mid = (t0 + t1) // 2
            qi = next((i for i in range(len(ss))
                       if ss[i] == stream and sa[i] <= mid <= sb[i]), None)
            if qi is not None:
                sc = vecs @ np.asarray(vecs[qi])
                sc[qi] = -9
                order = np.argsort(-sc)[:k]
                hits = [{"stream": ss[i], "t0": sa[i], "t1": sb[i],
                         "score": float(sc[i])} for i in order]
                return hits, {"method": "vjepa-qbe", "k": k}
        except Exception:
            pass
        from .embeddings import search_clip
        return search_clip(self, stream, t0, t1, k=k, nprobe=nprobe, **kw)

    # ---- context retrieval -------------------------------------------------
    def index_context(self, window_s=2.0, stride_s=0.5, label_fraction=1.0,
                      prune=True, epochs=300, verbose=True, model=None,
                      frame_stride=1, prompt="scene"):
        """Build the context index end to end.

        frames -> per-frame vectors -> VLM captions on `label_fraction` of
        windows -> caption-LSA space -> temporal tower -> cellular turnover
        -> materialised `context` table.

        Cost is dominated by the image encoder, so the knobs that matter are:

          model="fast"      3.3x faster encoder, same 1152-d space
          frame_stride=N    embed every Nth frame (5 Hz video rarely needs all)
          label_fraction<1  caption only part of the corpus; the tower covers
                            the rest
          prompt=           "scene" or "manipulation" — the caption IS the
                            index, so it has to use the words a user would

        Measured: decode 2.7 ms/frame, encode 90.3 ms/frame (quality) or
        27.7 ms/frame (fast). Embedding a large corpus is a batch job measured
        in hours; nothing here hides that.
        """
        from . import context as C
        out = {"frame_vectors": C.embed_frames(self, verbose=verbose,
                                               model=model,
                                               stride=frame_stride)}
        windows = C.plan_windows(self, window_s, stride_s)
        if label_fraction < 1.0:
            # Label a TIME PREFIX, not a random sample: the realistic shape of
            # this problem is "we captioned what we had, then more footage
            # arrived", and a random sample would quietly hand the tower
            # neighbours of every held-out window.
            n = max(int(len(windows) * label_fraction), 16)
            windows = sorted(windows, key=lambda w: w[1])[:n]
        out["captions"] = C.caption_windows(self, windows, verbose=verbose,
                                            prompt=prompt)
        _, _, out["train"] = C.train_context(self, window_s, stride_s,
                                             epochs=epochs, verbose=verbose)
        if prune:
            _, rec = C.prune_context(self, verbose=verbose)
            out["prune"] = rec.get("selected")
        out["build"] = C.build_context(self, verbose=verbose)
        return out

    def search_context(self, text: str, k=8, pool=48, deep=0, t0=None,
                       t1=None, streams=None, rerank=False, verify="async",
                       **_legacy):
        """THE search: any query, action or not, on any store.

        Union recall over every tier the store has (appearance embeddings,
        caption words, caption-LSA vectors) proposes candidates; a VLM
        reading each clip's start/end frames verifies WHAT IS HAPPENING;
        verdicts are cached into the store so hot queries get cheap.
        `deep=N` (or rerank=True) re-judges the top N with the larger VLM
        over 4 ordered frames. Legacy RRF-only search remains at
        elidedb.context.search for stores where a model-free path matters."""
        from .verified import search_verified
        if rerank and not deep:
            deep = 6
        if deep and verify == "async":
            verify = "sync"          # deep judging is an explicit wait
        return search_verified(self, text, k=k, pool=pool, deep=deep,
                               t0=t0, t1=t1, streams=streams, verify=verify)

    def search_verified(self, text: str, k=8, pool=48, deep=0):
        """Any query, action or not: union recall proposes, a VLM shown
        frames IN TIME ORDER disposes, verdicts are cached into the store.
        The only path that can enforce 'the green object is the one being
        moved' or '...and close it'. See elidedb.verified."""
        from .verified import search_verified
        return search_verified(self, text, k=k, pool=pool, deep=deep)

    def search_sharp(self, text: str, k=10, shortlist=48):
        """Text search at teacher quality, student price: the student ranks
        every window (~1 ms), the teacher re-scores only the shortlist, and
        every teacher vector is cached into the store — quality accumulates
        where users query (database cracking). See elidedb.cracked."""
        from .cracked import search_sharp
        return search_sharp(self, text, k=k, shortlist=shortlist)

    def explain(self, t0: int, t1: int, stream=None):
        """The teacher's own description of what happens in a window."""
        from .context import explain
        return explain(self, t0, t1, stream=stream)

