"""elidedb — the command line. One verb per lifecycle step:

  elidedb create  lake/mydb --name "my dataset"
  elidedb add     lake/mydb readings data.csv --ts-col time
  elidedb video   lake/mydb cam.mp4 --stream cam0
  elidedb ls      lake/mydb
  elidedb embed   lake/mydb
  elidedb search  lake/mydb "a person crossing the street"
  elidedb sql     lake/mydb "SELECT count(*) FROM readings"
  elidedb desk

Times print as +SECONDS relative to the store's start; pass the same +N form
back into `window`.
"""
from __future__ import annotations

import argparse
import sys

from .store import Store


def _rel(store, ns):
    lo = min((d["min_ts"] for d in store.describe()
              if d["rows"] and d["table"] != "centroids"), default=0)
    return (ns - lo) / 1e9, lo


def _parse_t(store, s: str) -> int:
    if s.startswith("+"):
        lo = min((d["min_ts"] for d in store.describe()
                  if d["rows"] and d["table"] != "centroids"), default=0)
        return lo + int(float(s[1:]) * 1e9)
    return int(s)


def cmd_create(a):
    Store.create(a.store, a.name or a.store.rstrip("/").split("/")[-1])
    print(f"created database at {a.store}")


def cmd_ls(a):
    db = Store.open(a.store)
    print(f"{db.name}  ({db.dir})")
    rows = db.describe()
    if not rows:
        print("  (empty — add data with `elidedb add` or `elidedb video`)")
        return
    w = max(len(r["table"]) for r in rows)
    for r in rows:
        span = ""
        if r["rows"] and r["table"] != "centroids":
            lo, base = _rel(db, r["min_ts"])
            hi, _ = _rel(db, r["max_ts"])
            span = f"  [+{lo:.1f} .. +{hi:.1f}]s"
        print(f"  {r['table']:<{w}}  {r['kind']:<12} {r['rows']:>12,} rows "
              f"{r['bytes'] / 1e6:9.1f} MB  v{r['version']}{span}")


def cmd_add(a):
    db = Store.open(a.store)
    v = db.ingest_rows(a.table, a.file, ts_column=a.ts_col, ts_unit=a.ts_unit)
    st = db.table(a.table).state()
    print(f"{a.table}: +{st.files[-1].rows:,} rows -> version {v}")


def cmd_video(a):
    db = Store.open(a.store)
    ts = None
    if a.timestamps:
        ts = [int(line) for line in open(a.timestamps)]
    v = db.ingest_video(a.table, a.file, timestamps_ns=ts, stream=a.stream)
    print(f"{a.table}: indexed {a.file} as stream "
          f"'{a.stream or a.file}' -> version {v}")


def cmd_adopt(a):
    db = Store.open(a.store)
    for d in db.describe():
        if d["kind"] == "frame_index":
            r = db.adopt_media(d["table"])
            print(f"{d['table']}: adopted {r['adopted']} media files "
                  f"({r['bytes'] / 1e6:.1f} MB) into {a.store}/media/")


def cmd_embed(a):
    db = Store.open(a.store)
    print(db.embed_windows(window_s=a.window_s,
                           frames_per_window=a.frames_per_window))
    if not a.no_cluster:
        from .embeddings import cluster
        print(cluster(db, min_cluster_size=a.min_cluster_size))


def cmd_search(a):
    db = Store.open(a.store)
    hits, stats = db.search_text(a.text, k=a.k)
    print(f"scanned {stats['scanned']}/{stats['total']} vectors, "
          f"probed {stats['clusters_probed']}/{stats['clusters_total']} clusters")
    for h in hits:
        lo, _ = _rel(db, h["t0"])
        hi, _ = _rel(db, h["t1"])
        print(f"  {h['score']:.4f}  {h['stream']:<24} [+{lo:.2f} .. +{hi:.2f}]s"
              f"   (elidedb window {a.store} +{lo:.2f} +{hi:.2f})")


def cmd_sql(a):
    db = Store.open(a.store)
    df = db.sql(a.query)
    print(df.to_string(index=False, max_rows=50))


def cmd_window(a):
    db = Store.open(a.store)
    t0, t1 = _parse_t(db, a.t0), _parse_t(db, a.t1)
    w, stats = db.window(t0, t1)
    print(stats)
    from .video import FrameSet
    for name, v in w.items():
        if isinstance(v, FrameSet):
            print(f"  {name}: {len(v)} frames, streams {v.streams()}")
            if a.dump:
                from PIL import Image
                import pathlib
                out = pathlib.Path(a.dump)
                out.mkdir(parents=True, exist_ok=True)
                n = 0
                for s in v.streams():
                    for ts, img in v.decode(stream=s, width=a.width):
                        lo, _ = _rel(db, ts)
                        safe = s.replace("/", "_").replace(" ", "_")
                        Image.fromarray(img).save(out / f"{safe}_{lo:.3f}s.jpg")
                        n += 1
                print(f"  wrote {n} frames to {a.dump}")
        elif len(v):
            print(f"  {name}: {len(v):,} rows x {len(v.column_names)} cols")


def cmd_desk(a):
    from . import desk
    sys.argv = ["elidedb-desk", "--root", a.root, "--port", str(a.port)] + \
        (["--open"] if not a.no_open else [])
    desk.main()


def main():
    ap = argparse.ArgumentParser(
        prog="elidedb",
        description="ElideDB — Parquet-native, timestamp-first multimodal "
                    "database. The best read is the read elided.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create", help="create an empty database")
    p.add_argument("store")
    p.add_argument("--name")
    p.set_defaults(f=cmd_create)

    p = sub.add_parser("ls", help="list a database's tables")
    p.add_argument("store")
    p.set_defaults(f=cmd_ls)

    p = sub.add_parser("add", help="add timestamped rows (CSV/Parquet)")
    p.add_argument("store")
    p.add_argument("table")
    p.add_argument("file")
    p.add_argument("--ts-col", default="ts",
                   help="timestamp column name (default: ts)")
    p.add_argument("--ts-unit", default="auto",
                   choices=["auto", "s", "ms", "us", "ns"],
                   help="timestamp unit (default: auto-detect)")
    p.set_defaults(f=cmd_add)

    p = sub.add_parser("video", help="index a video file (media stays put)")
    p.add_argument("store")
    p.add_argument("file")
    p.add_argument("--table", default="frames")
    p.add_argument("--stream", help="stream name (default: file stem)")
    p.add_argument("--timestamps",
                   help="file with one ns timestamp per frame; omitted = "
                        "container timestamps")
    p.set_defaults(f=cmd_video)

    p = sub.add_parser("adopt", help="copy referenced media into the store "
                                     "(makes it standalone)")
    p.add_argument("store")
    p.set_defaults(f=cmd_adopt)

    p = sub.add_parser("embed", help="embed video windows + cluster (local ML)")
    p.add_argument("store")
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--frames-per-window", type=int, default=2)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--no-cluster", action="store_true")
    p.set_defaults(f=cmd_embed)

    p = sub.add_parser("search", help="semantic text search")
    p.add_argument("store")
    p.add_argument("text")
    p.add_argument("-k", type=int, default=8)
    p.set_defaults(f=cmd_search)

    p = sub.add_parser("sql", help="run SQL (DuckDB) over the database")
    p.add_argument("store")
    p.add_argument("query")
    p.set_defaults(f=cmd_sql)

    p = sub.add_parser("window", help="read a time window (+SECS or raw ns)")
    p.add_argument("store")
    p.add_argument("t0")
    p.add_argument("t1")
    p.add_argument("--dump", help="write frames as JPEGs to this directory")
    p.add_argument("--width", type=int, default=640)
    p.set_defaults(f=cmd_window)

    p = sub.add_parser("desk", help="open the browser UI")
    p.add_argument("--root", default="lake")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(f=cmd_desk)

    a = ap.parse_args()
    try:
        a.f(a)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
