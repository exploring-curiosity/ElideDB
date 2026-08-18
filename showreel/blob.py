#!/usr/bin/env python3
"""Where the bytes live: local disk or S3, decided by one environment variable.

    PRECEDENT_BUCKET=""              everything is on this laptop  (default)
    PRECEDENT_BUCKET=precedent-...   traces and clips live in S3

Two kinds of large object sit outside the database, for two different reasons.

    traces/{rec_id}.npy   the DTW representation of one recording. Read by the
                          SIDECAR, on the server side, and only for recordings
                          stage 1 already shortlisted.
    clips/{rec_id}.mp4    the video. Read by the BROWSER, never by the server.

They are fetched differently on purpose. A trace is pulled into a local cache
because the ranker needs the array in memory. A clip is handed to the browser as
a presigned URL and never passes through this process at all: proxying 1.34 GB
of video through the app would make the app the bottleneck for bytes it does not
look at, and would cost a second copy of every byte.

The cache is the point. A query touches 48 recordings out of 3,556, so a cold
Space downloads about 15 MB to answer its first query rather than the 1.1 GB the
corpus weighs. That is the same principle the retrieval path is built on, applied
one level down: the best read is the read elided.
"""

from __future__ import annotations

import os
import pathlib
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]

BUCKET = os.environ.get("PRECEDENT_BUCKET", "").strip()
REGION = os.environ.get("AWS_REGION", "us-east-1")
CACHE = pathlib.Path(os.environ.get("PRECEDENT_TRACES", ROOT / "showreel" / "traces"))
URL_TTL = int(os.environ.get("PRECEDENT_URL_TTL", "3600"))

_s3 = None
_lock = threading.Lock()
hits = misses = fetched_bytes = 0


def enabled() -> bool:
    return bool(BUCKET)


def client():
    """One boto3 client, made on first use.

    Built lazily because the local path must not require boto3 to be installed,
    and because importing botocore costs about 0.4 s that a laptop run never
    needs to pay.
    """
    global _s3
    if _s3 is None:
        with _lock:
            if _s3 is None:
                import boto3
                from botocore.config import Config

                _s3 = boto3.client(
                    "s3", region_name=REGION,
                    config=Config(retries={"max_attempts": 3, "mode": "adaptive"},
                                  max_pool_connections=32))
    return _s3


# ---- traces: server side, cached ------------------------------------------

def trace_path(rid: str) -> pathlib.Path | None:
    """Local path to one recording's trace, pulling it from S3 if absent.

    Writes through a temporary name and renames, because two concurrent queries
    can want the same trace and a half-written .npy loads as a corrupt array
    rather than as an error.
    """
    global hits, misses, fetched_bytes
    p = CACHE / f"{rid}.npy"
    if p.exists():
        hits += 1
        return p
    if not BUCKET:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".npy.{os.getpid()}.part")
    try:
        client().download_file(BUCKET, f"traces/{rid}.npy", str(tmp))
        os.replace(tmp, p)
    except Exception:                                             # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None
    misses += 1
    fetched_bytes += p.stat().st_size
    return p


def prefetch(rids) -> int:
    """Pull every trace a query will need, at once.

    Stage 2 knows all 48 candidates before it ranks any of them, so fetching
    them one at a time inside the loop turns one query into 48 sequential round
    trips. Measured from a laptop against us-east-1 that was 37.3 s for a query
    whose actual work is under a second: the retrieval was never the cost, the
    serialisation was. A thread pool makes the depth the pool size instead of
    the candidate count, and S3 is happy to serve them in parallel.

    Returns the number actually fetched, so a caller can log the miss rate.
    """
    todo = [r for r in dict.fromkeys(rids) if not (CACHE / f"{r}.npy").exists()]
    if not todo or not BUCKET:
        return 0
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(min(16, len(todo))) as ex:
        return sum(1 for p in ex.map(trace_path, todo) if p is not None)


def stats() -> dict:
    return dict(bucket=BUCKET or None, region=REGION if BUCKET else None,
                cache=str(CACHE), cache_hits=hits, s3_fetches=misses,
                s3_bytes=fetched_bytes,
                cached_traces=len(list(CACHE.glob("*.npy"))) if CACHE.exists() else 0)


# ---- clips: browser side, never through this process -----------------------

def video_url(video: str) -> tuple[str, bool]:
    """(location, is_url) for the `video` column.

    The column holds either a local path from ingest or an s3:// key after
    upload_assets.sh has run, so both deployments read the same column and the
    server does not need to know which one it is in.
    """
    if video and video.startswith("s3://"):
        bucket, _, key = video[5:].partition("/")
        return client().generate_presigned_url(
            "get_object", Params=dict(Bucket=bucket, Key=key),
            ExpiresIn=URL_TTL), True
    return video, False
