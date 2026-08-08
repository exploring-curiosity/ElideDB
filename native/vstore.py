"""THE STORE. Columnar, memory-mapped, and honest about every byte read.

    stores/vision/<corpus>/
        manifest.json    what was written, with what code, at what settings
        windows.parquet  media, t0, t1, scale, energy, row   (the zone map)
        c1.f32 c2.f32 c3.f32   one column per channel, (N,d) row-major
        hub.f32          per-row hubness, see below

Channels are SEPARATE FILES, not one fused matrix, for the same reason
they are separate at write time: a query that decides c2 is the
informative channel should read c2 and pay for c2. Fusing at write would
force every query to read everything, and the whole point of this
database is that the best read is the read elided.

HUBNESS is computed once per store, from the store's own vectors, and
stored as a column. In high dimensions a few vectors are everyone's
nearest neighbour and dominate every result list. Radovanovic et al.
named the effect; CSLS (Conneau et al., ICLR 2018) fixes it by
subtracting each candidate's own mean similarity to its neighbourhood,
which turns "this vector is close to everything" from an advantage into
nothing. It costs one float per row and it is corpus-derived, not fitted.

RESUMABILITY is per media: an interrupted build re-encodes at most one
media, never the corpus. A build that exits cleanly having written no
rows is a failed build and says so - exit status is not evidence.

    python native/vstore.py --corpus sim --media 8 --cap 60
    python native/vstore.py --stat
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "native"))

import vcore                                             # noqa: E402
import vsrc                                              # noqa: E402

# The store ROOT is overridable so an A/B can run WITHOUT destroying the
# artifact it is being compared against. build() opens with rmtree(out),
# and every arm of an A/B rebuilds the same four corpus names - so an A/B
# queued behind a full build deletes the stores the headline was measured
# on, minutes after it is written. Writer and reader both read this one
# variable, so an arm cannot half-move between roots.
STORES = ROOT / os.environ.get("SDX_STORES", "stores/vision")
CACHE = Path(os.environ.get("SDX_VCACHE",
                            "/private/tmp/claude-501/vstore"))
# MEASURED: c2 ALONE beats all three channels on every store. c1 and c3
# are not merely dead weight, they actively hurt.
#
#   real vgrade, 96 q/store, identical stores:
#     three channels  yield 0.713  prec 0.726
#     c2 only         yield 0.803  prec 0.846
#   ablation (5 variants sharing one query encode):
#     none 0.909/0.787  drop c1 0.915/0.782  drop c3 0.948/0.843
#     drop c2 0.720/0.697   <- c2 is the one that matters
#     c2 only 0.965/0.867
#
# Two caveats that survive the result:
#  1. c3 was justified by Junejo et al. on VIEWPOINT stability, and the
#     benchmark's viewpoint test is a parallax-free homography. This
#     shows c3 does not help against the proxy; it cannot show c3 would
#     not help against a real second camera. That question is open, not
#     settled - cross-camera testing is out of scope by instruction.
#  2. The per-query weighting was supposed to SELECT the best channel and
#     instead diluted c2 with two that hurt, handing c3 ~0.40 weight. Its
#     rule rewards self-consistency across adjacent windows, not
#     discrimination. If channels are ever re-added, that rule has to be
#     rebuilt first.
CHANNELS = tuple(x for x in os.environ.get(
    "SDX_CHANNELS", "c2").split(",") if x)
HUB_SAMPLE = 4096          # rows sampled to estimate hubness
HUB_K = 16                 # neighbourhood size in the CSLS sense


class Counted:
    """Every read of a vector column goes through here.

    Counting is row-granular: a scan that touches R rows of a d-wide
    float32 column is charged R*d*4 bytes. That is the honest unit for a
    memory-mapped columnar read - the OS pages in more, and a page-exact
    number would only ever be larger, so this never flatters the store.
    """

    def __init__(self, path, d):
        self.path = Path(path)
        self.d = d
        n = self.path.stat().st_size // (4 * d)
        self.mm = np.memmap(self.path, np.float32, "r", shape=(n, d))
        self.bytes = 0
        self.rows = 0

    def take(self, idx=None):
        v = self.mm if idx is None else self.mm[idx]
        self.rows += len(v)
        self.bytes += int(len(v)) * self.d * 4
        return np.asarray(v)

    def reset(self):
        self.bytes = self.rows = 0


class Store:
    def __init__(self, path):
        import pyarrow.parquet as pq
        self.path = Path(path)
        self.man = json.loads((self.path / "manifest.json").read_text())
        t = pq.read_table(self.path / "windows.parquet").to_pydict()
        self.media = np.array(t["media"])
        self.t0 = np.asarray(t["t0"], np.float64)
        self.t1 = np.asarray(t["t1"], np.float64)
        self.scale = np.asarray(t["scale"], np.float64)
        self.energy = np.asarray(t["energy"], np.float32)
        self.n = len(self.t0)
        self.col = {c: Counted(self.path / f"{c}.f32",
                               self.man["dims"][c]) for c in CHANNELS}
        hp = self.path / "hub.f32"
        self.hub = (np.fromfile(hp, np.float32).reshape(self.n, -1)
                    if hp.exists()
                    else np.zeros((self.n, len(CHANNELS)), np.float32))
        self.cell, self.centroids = {}, {}
        for c in CHANNELS:
            cp, kp = self.path / f"cells_{c}.i32", self.path / f"cent_{c}.f32"
            if cp.exists() and kp.exists():
                self.cell[c] = np.fromfile(cp, np.int32)
                self.centroids[c] = np.fromfile(kp, np.float32).reshape(
                    -1, self.man["dims"][c])

    def reset_bytes(self):
        for c in self.col.values():
            c.reset()

    @property
    def bytes_read(self):
        return sum(c.bytes for c in self.col.values())

    def corpus_bytes(self):
        return int(self.man.get("source_bytes", 0))

    def __repr__(self):
        return (f"<store {self.man['corpus']} {self.n} windows "
                f"{len(set(self.media.tolist()))} media "
                f"{self.man['seconds'] / 60:.1f} min>")


# ------------------------------------------------------------------ write

def _cache_key(src, cap):
    return (f"{src.id.replace('/', '_')}__{vcore.ENCODER.split('/')[-1]}"
            f"_{vsrc.FPS:g}_{vsrc.WIDTH}_{vcore.CTX_MULT:g}_"
            f"r{'-'.join(str(r) for r in vcore.RES)}_{vcore.POOL}_"
            f"{'-'.join(f'{s:g}' for s in vcore.SCALES)}_{cap:g}.npz")


def encode_media(src, cap, use_cache=True):
    """Windows + channels for one media, resumable via a disk cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / _cache_key(src, cap)
    if use_cache and fp.exists():
        try:
            z = np.load(fp, allow_pickle=False)
            return (z["wins"], {c: z[c] for c in CHANNELS},
                    z["energy"], z["valid"])
        except Exception:                                # noqa: BLE001
            fp.unlink(missing_ok=True)
    dur = min(src.dur, cap) if cap else src.dur
    if dur < min(vcore.SCALES) + 2.0:
        return None
    F = vcore.media_features(_Clipped(src, dur))
    wins = vcore.windows(dur)
    if not wins or len(F) < 4:
        return None
    e = vcore.encode_windows(F, wins)
    W = np.asarray([[a, b, w] for a, b, w in wins], np.float64)
    np.savez(fp, wins=W, energy=e["energy"], valid=e["valid"],
             **{c: e[c] for c in CHANNELS})
    return W, {c: e[c] for c in CHANNELS}, e["energy"], e["valid"]


class _Clipped:
    """A source truncated to `dur` - the same cap rule for every corpus."""

    def __init__(self, s, dur):
        self.s, self.dur, self.id = s, float(dur), s.id

    def blocks(self, block_s=vsrc.BLOCK_S):
        t = 0.0
        while t < self.dur - 1e-6:
            b = min(block_s, self.dur - t)
            F = self.s.cut(t, t + b)
            if len(F):
                yield t, F
            t += b


def build(corpus, media=None, cap=0.0, out=None, use_cache=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    out = Path(out or (STORES / corpus))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    srcs = vsrc.sources(corpus, media)
    if not srcs:
        raise SystemExit(f"{corpus}: no media found")

    # asked of the encoder, never assumed
    fd = vcore.feature_dim()
    # ONLY the channels actually written. This was hardcoded to all
    # three, so a c2-only store shipped a manifest claiming c1 and c3
    # that do not exist on disk. Harmless to the reader, which only
    # looks up CHANNELS, and still wrong: a snapshot has to describe
    # what it contains.
    _all = {"c1": fd, "c2": fd, "c3": vcore.NS * (vcore.NS - 1) // 2}
    dims = {c: _all[c] for c in CHANNELS}
    fh = {c: open(out / f"{c}.f32", "wb") for c in CHANNELS}
    rows = {k: [] for k in ("media", "t0", "t1", "scale", "energy")}
    n, secs, sbytes, t_start = 0, 0.0, 0, time.time()
    est = sum(min(s.dur, cap) if cap else s.dur for s in srcs)
    bar = tqdm(srcs, desc=f"{corpus} write", unit="media")
    for s in bar:
        try:
            got = encode_media(s, cap, use_cache)
        except Exception as e:                           # noqa: BLE001
            bar.write(f"  skip {s.id}: {type(e).__name__} {e}")
            continue
        if got is None:
            continue
        W, ch, en, ok = got
        if not ok.any():
            continue
        for c in CHANNELS:
            fh[c].write(np.ascontiguousarray(ch[c][ok],
                                             np.float32).tobytes())
        rows["media"] += [s.id] * int(ok.sum())
        rows["t0"] += W[ok, 0].tolist()
        rows["t1"] += W[ok, 1].tolist()
        rows["scale"] += W[ok, 2].tolist()
        rows["energy"] += en[ok].astype(float).tolist()
        n += int(ok.sum())
        secs += min(s.dur, cap) if cap else s.dur
        sbytes += s.source_bytes()
        bar.set_postfix(rows=n, min=f"{secs / 60:.0f}")
    for f in fh.values():
        f.close()

    if n == 0:
        raise SystemExit(f"{corpus}: build wrote ZERO rows - "
                         f"a clean exit is not evidence")

    # zone maps come free: the table is written in (media, t0) order, so
    # every parquet row group carries a contiguous time range per media
    pq.write_table(pa.table(rows), out / "windows.parquet",
                   compression="zstd", row_group_size=8192)
    (out / "manifest.json").write_text(json.dumps(dict(
        corpus=corpus, encoder=vcore.ENCODER, fps=vsrc.FPS,
        # RES decides how every vector in this store was produced, so it
        # is pinned here. A snapshot that does not record the flags
        # deciding its own contents lets one build's number be read as
        # another's.
        res=list(vcore.RES), pool=vcore.POOL,
        width=vsrc.WIDTH, scales=list(vcore.SCALES),
        ctx_mult=vcore.CTX_MULT, ns=vcore.NS, dims=dims,
        media=len(set(rows["media"])), windows=n, seconds=secs,
        source_bytes=sbytes, build_s=round(time.time() - t_start, 1),
        est_seconds=est), indent=2))
    st = Store(out)
    _write_hubness(st)
    _write_cells(st)
    return Store(out)


def _write_hubness(st, sample=HUB_SAMPLE, k=HUB_K):
    """Per row, PER CHANNEL: mean similarity to its top-k neighbours in
    a random sample of the store.

    Per channel and not averaged: a row can be a hub in the shape
    channel (whose unrelated-pair p99 was measured at 0.898) while being
    perfectly ordinary in the order channel, and averaging the two
    throws away exactly the distinction the correction needs.
    """
    rs = np.random.RandomState(0)
    m = min(sample, st.n)
    sel = np.sort(rs.choice(st.n, m, replace=False))
    hub = np.zeros((st.n, len(CHANNELS)), np.float32)
    for ci, c in enumerate(CHANNELS):
        A = st.col[c].take()
        B = A[sel]
        kk = max(min(k, m - 1), 1)
        for i in range(0, st.n, 2048):
            S = A[i:i + 2048] @ B.T
            hub[i:i + 2048, ci] = np.partition(
                S, -kk, axis=1)[:, -kk:].mean(1)
    hub.astype(np.float32).tofile(st.path / "hub.f32")
    st.hub = hub
    st.reset_bytes()
    return hub


def _write_cells(st):
    """Coarse cells PER CHANNEL: a PRUNE STRUCTURE, never a label.

    Per channel, and that is a correction. The first version partitioned
    on c1 alone - and c1 is the channel queries give almost no weight to,
    so the prune was discarding rows on evidence the ranking does not
    use. Measured recall against an exact scan: 0.55-0.79, and no probe
    rule fixed it because the partition itself was on the wrong axis.

    A query now probes the cells of the channels it actually weights.
    The cells are k-means cells and carry no meaning - nothing reads a
    cell id as a class - and every probed cell is re-ranked exactly, so
    a bad partition can only cost recall, which is measured.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    ncell = max(4, min(int(np.sqrt(st.n)), 4096))
    for c in CHANNELS:
        A = st.col[c].take()
        # Above ~20k rows exact k-means costs more than the encode it is
        # meant to accelerate. The cells are a pure speed structure with
        # an exact rerank inside every probed cell, so an approximate
        # partition can only cost recall - which gate P measures.
        km = (MiniBatchKMeans(ncell, n_init=3, random_state=0,
                              batch_size=4096) if st.n > 20000
              else KMeans(ncell, n_init=4, random_state=0)).fit(A)
        km.cluster_centers_.astype(np.float32).tofile(
            st.path / f"cent_{c}.f32")
        km.labels_.astype(np.int32).tofile(st.path / f"cells_{c}.i32")
    st.reset_bytes()


def main():
    from flowgebd import arg
    if "--stat" in sys.argv:
        if not STORES.exists():
            print("no stores")
            return
        print(f"{'corpus':<9}{'media':<7}{'windows':<9}{'minutes':<9}"
              f"{'store MB':<10}{'raw MB':<10}{'build s'}")
        for d in sorted(STORES.iterdir()):
            if not (d / "manifest.json").exists():
                continue
            m = json.loads((d / "manifest.json").read_text())
            sz = sum(f.stat().st_size for f in d.rglob("*")) / 1e6
            print(f"{m['corpus']:<9}{m['media']:<7}{m['windows']:<9}"
                  f"{m['seconds'] / 60:<9.1f}{sz:<10.1f}"
                  f"{m.get('source_bytes', 0) / 1e6:<10.1f}"
                  f"{m.get('build_s', 0):.0f}")
        return
    corpus = arg("--corpus", "sim")
    media = arg("--media", 0, int) or None
    cap = arg("--cap", 0.0, float)
    for name in (list(vsrc.CORPORA) if corpus == "all"
                 else corpus.split(",")):
        st = build(name, media, cap)
        print(f"{st!r}  {st.man['build_s']}s", flush=True)


if __name__ == "__main__":
    main()
