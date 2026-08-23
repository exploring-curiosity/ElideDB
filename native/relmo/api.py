"""ElideDB — video memory. Public API.

Give it video. Ask it "when did something like this happen?". It answers with
timestamps. No labels, no captions, no fine-tuning, no per-dataset setup.

    from relmo.api import Memory

    mem = Memory.open("kitchen")          # a store = one deployment's memory
    mem.add("/videos/robot_runs")         # ingest a folder of video
    for hit in mem.query("/clips/spill.mp4"):
        print(hit.video, hit.start, hit.end, hit.score)

Everything below is the supported surface. Anything else in `relmo` is
internal and may change.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

_HERE = Path(__file__).resolve().parents[1]
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm")


@dataclass(frozen=True)
class Hit:
    """One retrieved memory. `score` is a similarity in [0,1], higher = closer."""
    id: str
    score: float
    video: str
    start: float          # matched span, seconds into the source video
    end: float            # (the whole recording when it is no longer than the query)

    @property
    def label(self):
        """A source name a human can tell apart. Many pipelines write every
        clip as `frames.mp4` in its own folder, so the filename alone is
        useless; fall back to the parent directory, then the record id."""
        p = Path(self.video) if self.video else None
        if p is None or not p.name:
            return self.id
        return p.name if p.stem not in ("frames", "video", "clip") \
            else f"{p.parent.name}/{p.name}"

    def __repr__(self):
        return (f"Hit({self.label[:48]}, {self.start:.1f}-{self.end:.1f}s, "
                f"score={self.score:.3f})")


class Memory:
    """One store: one robot, one site, one deployment.

    Stores never mix. Every statistic the read path uses is computed inside
    the store, so two deployments cannot leak into each other's results.
    """

    def __init__(self, name):
        self.name = name
        self._store = None
        self._enc = None

    # ---------------------------------------------------------------- open
    @classmethod
    def open(cls, name):
        """Open (or name) a store. Lazy: nothing loads until you query."""
        return cls(name)

    @classmethod
    def list(cls):
        """-> {store: n_recordings}. Stores, not the datasets inside them:
        a store may be assembled from several ingests, and the split is an
        internal detail nobody querying should have to know."""
        from relmo.vjstore import STORES
        d = R.BASE / "vjrec7"
        if not d.exists():
            return {}
        have = {p.name[:-3]: len([x for x in p.glob("*.npz")
                                  if not x.name.startswith(".")])
                for p in sorted(d.glob("*_L6"))}
        out, claimed = {}, set()
        for store, members in STORES.items():
            n = sum(have.get(m, 0) for m in members)
            claimed |= set(members)
            if n:
                out[store] = n
        for k, v in have.items():                # stores created by add()
            if k not in claimed and v:
                out[k] = v
        return out

    # -------------------------------------------------------------- ingest
    def add(self, source, fps=None, recursive=True):
        """Ingest video into this store.

        source: a directory of video files, a single video, or a list.
        Returns the number of recordings now in the store.

        Cost is ~14 compute-minutes per hour of video (4x real time) and is
        one pass: decode, encode, write. Re-running skips anything already
        ingested, so an interrupted ingest resumes.
        """
        vids = _collect(source, recursive)
        if not vids:
            raise ValueError(f"no video found under {source!r}")
        ds = f"{self.name}"
        _write_manifest(ds, vids, fps)
        print(f"ingesting {len(vids)} videos into store {self.name!r} "
              f"(~{len(vids) * 0.25:.0f} min, resumable)", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "relmo.vjrec8", "--dataset", ds,
             "--fp16", "--pca-suffix", ""],
            cwd=_HERE)
        if r.returncode != 0:
            raise RuntimeError("ingest failed; see output above")
        self._store = None                       # force reload
        return self.size

    # --------------------------------------------------------------- query
    def query(self, video, start=None, end=None, top_k=10, fast=True):
        """Find moments like this clip. -> [Hit], best first.

        video: path to a video file (any length; use start/end to slice it).
        fast:  True  -> ~0.5 s, P@10 0.752 (default, recommended)
               False -> exact scan, P@10 0.765, ~20 s on a 3.5k store
        """
        fix, sig, secs = self._encode(video, start, end)
        return self._search(fix, sig, top_k, fast, exclude=None, q_seconds=secs)

    def query_recording(self, rec_id, top_k=10, fast=True):
        """Find moments like one ALREADY in the store — no re-encoding.

        This is the robot's own case: its live trace is already in memory, so
        a self-query costs search time only.
        """
        st = self._st()
        if rec_id not in st.raw:
            raise KeyError(f"{rec_id!r} not in store {self.name!r}")
        fix, sig = st.raw[rec_id]
        return self._search(fix, sig, top_k, fast, exclude={rec_id},
                            q_seconds=self._span(rec_id)[1] or None)

    # ---------------------------------------------------------------- info
    @property
    def size(self):
        """Number of recordings in this store."""
        try:
            return len(self._st().ids)
        except SystemExit:
            return 0

    def stats(self):
        st = self._st()
        secs = [self._span(i)[1] for i in st.ids]
        return dict(store=self.name, recordings=len(st.ids),
                    hours=round(float(np.sum(secs)) / 3600, 2),
                    median_seconds=round(float(np.median(secs)), 1))

    # ------------------------------------------------------------ internal
    def _st(self):
        if self._store is None:
            from relmo.vjstore import Store, STORES
            STORES.setdefault(self.name, (self.name,))
            self._store = Store(self.name)
        return self._store

    def _search(self, fix, sig, top_k, fast, exclude, q_seconds=None):
        st = self._st()
        kw = dict(prefilter_m=100, band=0.25) if fast else \
            dict(prefilter_m=len(st.ids), band=0.0)
        raw = st.query(fix, sig, top_k=top_k, exclude=exclude, **kw)
        q = st.query_vec(fix, sig)
        out = []
        for rid, sc in raw:
            v, dur = self._span(rid)
            a, b, L = st.localize(q, rid)
            if b - a >= L:                       # matched whole
                t0, t1 = 0.0, dur
            else:
                # the START is where the best-matching window begins. The
                # END runs for the query's real duration: arc-step counts
                # are not comparable between a clip and a region inside a
                # longer recording (gate energy is median-normalised over
                # the whole recording at write), so an arc-based end could
                # under-cover the event by half. Clamped to the recording.
                t0, t_arc = st.span_seconds(rid, a, b)
                t1 = t0 + q_seconds if q_seconds else t_arc
                if dur:
                    t1 = min(dur, t1)
            # the matcher returns NEGATED length-normalised DTW cost, where
            # cost is mean (1 - cosine). Users get a similarity, not an
            # internal distance with a sign flip on it.
            out.append(Hit(id=rid, score=round(max(0.0, 1.0 + float(sc)), 4),
                           video=v, start=round(t0, 2), end=round(t1, 2)))
        return out

    def _span(self, rec_id):
        """-> (source video path, duration seconds) from the manifest."""
        if not hasattr(self, "_meta"):
            self._meta = {}
            from relmo.vjstore import STORES
            for ds in STORES.get(self.name, (self.name,)):
                try:
                    m = R.read_manifest(ds)
                except Exception:                          # noqa: BLE001
                    continue
                f = float(m.get("fps") or 20)
                for e in m["episodes"]:
                    self._meta[e["id"]] = (
                        str(e.get("video", "")),
                        float(e.get("T", 0)) / float(e.get("fps", f) or f))
        return self._meta.get(rec_id, ("", 0.0))

    def _encode(self, video, start, end):
        """One clip -> (fix, sig) traces. Loads the encoders once per process."""
        import torch
        from relmo.vjrec6 import STREAM_FPS
        from relmo.vjrec8 import record_all
        from relmo.vjeval import REC
        from relmo.vjrec7 import OUT7
        from relmo.vjs import MODEL, probe_dims, read_frames
        from relmo.vjsig import MODEL as SIGM
        from relmo import vjz

        if self._enc is None:
            from transformers import AutoModel, VJEPA2Model
            from relmo.device import pick as _pick_device  # cuda > mps > cpu
            dev = _pick_device()
            print(f"loading encoders onto {dev} (once per process)...",
                  flush=True)
            vj = VJEPA2Model.from_pretrained(
                MODEL, dtype=torch.float16).to(dev).eval()
            sg = AutoModel.from_pretrained(
                SIGM, dtype=torch.float16).to(dev).eval()
            z = np.load(REC / "rcasa" / "_calib.npz")
            cal = (float(z["alpha"]), z["b"].astype(np.float32))
            proj = np.load(OUT7 / "_token_pca.npz")["W"].astype(np.float32)
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            self._enc = (vj, sg, dev, cal, proj, mean, std)
        vj, sg, dev, cal, proj, mean, std = self._enc

        p = Path(video)
        if not p.exists():
            raise FileNotFoundError(p)
        w, h = probe_dims(p)
        F = read_frames(p, w, h)                 # the real frames, once each
        dur = _probe_dur(p)
        # time the decoded array by ITS OWN length over the file's duration:
        # exact for the frames we hold, immune to a wrong rate tag
        fps = len(F) / dur if dur > 0 and len(F) else _probe_fps(p)
        if start is not None or end is not None:
            a = int((start or 0) * fps)
            b = int((end or len(F) / fps) * fps)
            F = F[max(0, a):min(len(F), b)]
        if len(F) / fps < 4.0:
            raise ValueError(
                f"clip is {len(F)/fps:.1f}s; the encoder window is 4.0s. "
                f"Pass a longer clip or widen start/end.")
        out = record_all(vj, sg, torch, dev, F, fps, cal, 6, torch.float16,
                         proj, mean=mean, std=std)
        if out is None:
            raise ValueError("clip too short to encode")
        rec6, _rec7, sig = out
        fix, _g = vjz.channels(rec6, primary="b")
        fix, sig = fix.astype(np.float32), np.asarray(sig, np.float32)
        # The store's traces are re-indexed by cumulative change (arc length)
        # at load; a query must ride the SAME grid or the matcher compares
        # time steps against change steps. This is exactly what load_corpus
        # does to every stored recording.
        from relmo.vjmatch import ARC_DS, arc_resample
        gate = rec6["where_map"].reshape(len(fix), -1).sum(1)
        fix, (sig,) = arc_resample(fix, gate, ARC_DS, aux=[sig], max_len=256)
        return fix, sig, len(F) / fps


# ------------------------------------------------------------------ helpers
def _collect(source, recursive):
    if isinstance(source, (list, tuple)):
        return [Path(x) for x in source]
    p = Path(source)
    if p.is_file():
        return [p]
    it = p.rglob("*") if recursive else p.glob("*")
    return sorted(q for q in it if q.suffix.lower() in VIDEO_EXT)


def _probe_fps(p):
    """The CONTENT frame rate: frames the file holds / seconds it lasts.

    The container's r_frame_rate tag is not that - research files here carry
    25/1 over 10 Hz and 16 Hz content - and timing a decode by the tag
    stretched their traces 2.5x. Prefer frame count over duration; fall
    back to avg_frame_rate, then the tag, then 30.
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,avg_frame_rate,r_frame_rate",
             "-show_entries", "format=duration", "-of", "json", str(p)],
            text=True)
        j = json.loads(out)
        st = (j.get("streams") or [{}])[0]
        dur = float((j.get("format") or {}).get("duration") or 0)
        nb = int(st.get("nb_frames") or 0)
        if nb > 0 and dur > 0:
            return nb / dur
        for key in ("avg_frame_rate", "r_frame_rate"):
            v = st.get(key) or ""
            if "/" in v:
                n, d = v.split("/")
                if float(d or 0) > 0 and float(n) > 0:
                    return float(n) / float(d)
    except Exception:                                          # noqa: BLE001
        pass
    return 30.0


def _probe_dur(p):
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)], text=True).strip())
    except Exception:                                          # noqa: BLE001
        return 0.0


def _write_manifest(ds, vids, fps=None):
    """Manifest is derived from the FILES, never trusted from a prior run."""
    out = R.BASE / "datasets" / ds
    out.mkdir(parents=True, exist_ok=True)
    # Recording ids come from filenames, but customer pipelines routinely
    # name every clip the same thing (frames.mp4 in per-run folders). A bare
    # stem would then collide: the second file matches the first one's
    # already-written trace and is SILENTLY skipped as done. Disambiguate
    # colliding stems with the parent folder, then a path hash - stable
    # across re-runs, so resume still works.
    from collections import Counter
    import hashlib
    stems = Counter(v.stem for v in vids)
    seen, eps, skipped = set(), [], 0
    for v in vids:
        f = fps or _probe_fps(v)
        dur = _probe_dur(v)
        if dur < 4.0:                       # shorter than one encoder window
            skipped += 1
            continue
        rid = v.stem if stems[v.stem] == 1 else f"{v.parent.name}_{v.stem}"
        if rid in seen:
            rid = f"{rid}_{hashlib.sha1(str(v.resolve()).encode()).hexdigest()[:8]}"
        seen.add(rid)
        eps.append(dict(id=rid, video=str(v.resolve()), fps=f,
                        T=int(round(dur * f))))
    if skipped:
        print(f"  skipping {skipped} clips shorter than the 4.0s window",
              flush=True)
    (out / "manifest.json").write_text(
        json.dumps(dict(fps=fps or 20, episodes=eps), indent=1))
    return len(eps)
