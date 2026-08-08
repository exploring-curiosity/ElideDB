"""ONE media abstraction over all four corpora. Single view. Bounded RAM.

Everything below is I/O plumbing - how to open a file - and nothing else.
There is no per-corpus parameter, no per-corpus branch downstream of this
module, and nothing here knows what any corpus contains. The four
registries differ only in where the pixels live on disk:

    sim     data/sim_chains/ep*/cam*.mp4         first camera only
    bridge  data/bridge/.../file-*.mp4           the one stream we kept
    car     data/kitti/.../image_02/data/*.png   left colour camera only
    drone   data/drone/agz_full/.../MAV Images   the flight camera

SINGLE VIEW is a rule, not an accident: each media is treated alone, and
no comparison is ever made between two views of the same moment.

RAM is bounded by construction. A source never materialises a whole
media - it yields blocks of BLOCK_S seconds, so a 45-minute continuous
flight costs the same memory as a 7-second robot episode.

    python native/vsrc.py --list
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

FPS = 4.0          # decode rate, identical for every corpus
WIDTH = 640        # decode width; sources are 640-1920 native, see vback
BLOCK_S = 30.0     # streaming block; the RAM ceiling of the whole system


def _probe(path):
    """(duration_s, width, height) of a video file."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    vals = [v for v in r.stdout.split() if v.strip()]
    try:
        w, h, d = int(vals[0]), int(vals[1]), float(vals[2])
    except Exception:                                    # noqa: BLE001
        return 0.0, 0, 0
    return d, w, h


class Source:
    """A media. Yields (t0, frames) blocks at FPS; can also cut one range.

    `cut` is the read path's late materialisation: it decodes only the
    requested seconds, never the whole file.
    """

    kind = "abstract"

    def __init__(self, mid, dur):
        self.id = mid
        self.dur = float(dur)

    def blocks(self, block_s=BLOCK_S):
        t = 0.0
        while t < self.dur - 1e-6:
            b = min(block_s, self.dur - t)
            F = self.cut(t, t + b)
            if len(F):
                yield t, F
            t += b

    def cut(self, t0, t1):
        raise NotImplementedError

    def source_bytes(self):
        """Bytes of raw media this source represents.

        Every source must answer this, because it is the DENOMINATOR of
        the elision metric. The first version only implemented it for
        video files, so the two image-sequence corpora reported a corpus
        size of 0 - which would have made their elision percentage
        silently meaningless rather than visibly wrong.
        """
        return 0

    def __repr__(self):
        return f"<{self.kind} {self.id} {self.dur:.1f}s>"


class VideoSource(Source):
    kind = "video"

    def __init__(self, mid, path, t_off=0.0, dur=None):
        self.path = Path(path)
        d, w, h = _probe(self.path)
        self.t_off = float(t_off)          # offset inside a packed file
        self._w, self._h0 = w, h
        super().__init__(mid, d - t_off if dur is None else dur)

    def source_bytes(self):
        # a packed file shared by several sources is charged pro rata
        try:
            tot = self.path.stat().st_size
        except OSError:
            return 0
        full = max(self.dur + self.t_off, 1e-6)
        return int(tot * min(self.dur / full, 1.0))

    def cut(self, t0, t1):
        if self._w == 0:
            return np.zeros((0, 8, WIDTH, 3), np.uint8)
        H = int(round(self._h0 * WIDTH / self._w / 2) * 2)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{self.t_off + t0:.3f}",
             "-i", str(self.path), "-t", f"{max(t1 - t0, 0.0):.3f}",
             "-vf", f"fps={FPS},scale={WIDTH}:{H}", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
        n = len(r.stdout) // (WIDTH * H * 3)
        if n == 0:
            return np.zeros((0, H, WIDTH, 3), np.uint8)
        return np.frombuffer(r.stdout[:n * WIDTH * H * 3],
                             np.uint8).reshape(n, H, WIDTH, 3)


class ImageSeqSource(Source):
    """A directory or zip of numbered stills, treated as a video at
    `native_fps`. The corpus decides its own capture rate; we resample to
    the same FPS as every other corpus so nothing downstream differs."""

    kind = "images"

    def __init__(self, mid, items, native_fps, zf=None):
        self.items = items                 # list of paths, or zip names
        self.zf = zf
        self.native = float(native_fps)
        super().__init__(mid, len(items) / float(native_fps))

    def _read(self, i):
        from PIL import Image
        if i < 0 or i >= len(self.items):
            return None
        try:
            if self.zf is not None:
                im = Image.open(io.BytesIO(self.zf.read(self.items[i])))
            else:
                im = Image.open(self.items[i])
            im = im.convert("RGB")
        except Exception:                                # noqa: BLE001
            return None
        h = max(int(round(im.height * WIDTH / im.width / 2) * 2), 8)
        return np.asarray(im.resize((WIDTH, h)))

    def source_bytes(self):
        if self.zf is not None:
            info = {i.filename: i.compress_size
                    for i in self.zf.infolist()}
            return int(sum(info.get(n, 0) for n in self.items))
        tot = 0
        for p in self.items:
            try:
                tot += p.stat().st_size
            except OSError:
                pass
        return int(tot)

    def cut(self, t0, t1):
        n = max(int(round((t1 - t0) * FPS)), 1)
        idx = (np.arange(n) / FPS + t0) * self.native
        out = [f for f in (self._read(int(round(j))) for j in idx)
               if f is not None]
        if not out:
            return np.zeros((0, 8, WIDTH, 3), np.uint8)
        h = min(f.shape[0] for f in out)
        return np.stack([f[:h] for f in out])


# ---------------------------------------------------------------- corpora

def sim(limit=None):
    """One camera per episode. The episode has several; we take the
    first and never look at the others - single view is the rule."""
    out = []
    base = ROOT / "data/sim_chains"
    for d in sorted(p for p in base.iterdir()
                    if p.is_dir() and p.name.startswith("ep")):
        cams = sorted(d.glob("cam*.mp4"))
        if not cams:
            continue
        out.append(VideoSource(f"sim/{d.name}", cams[0]))
        if limit and len(out) >= limit:
            break
    return out


def bridge(limit=None):
    """The packed video files as they sit on disk. Episode boundaries
    are metadata, so they are not used: a file is a media."""
    base = (ROOT / "data/bridge/videos/observation.images.image_0/"
            "chunk-000")
    out = []
    for p in sorted(base.glob("file-*.mp4")):
        s = VideoSource(f"bridge/{p.stem}", p)
        if s.dur > 5.0:
            out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def car(limit=None):
    base = ROOT / "data/kitti/2011_09_26"
    out = []
    for dr in sorted(p for p in base.iterdir() if p.is_dir()):
        d = dr / "image_02/data"
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.png"))
        if len(files) < 60:
            continue
        out.append(ImageSeqSource(f"car/{dr.name}", files, 10.0))
        if limit and len(out) >= limit:
            break
    return out


def drone(limit=None):
    """One continuous 45-minute flight, cut into equal media so that no
    single media is disproportionately long. The cut length is a fixed
    constant used for every corpus that needs it, not a drone choice."""
    out = []
    base = ROOT / "data/drone/agz_full/AGZ/MAV Images"
    if base.is_dir():
        files = sorted(base.glob("*.jpg"))
        SEG = 3000                       # frames per media, ~100 s @30Hz
        for i in range(0, len(files) - SEG // 2, SEG):
            out.append(ImageSeqSource(f"drone/agz{i // SEG:03d}",
                                      files[i:i + SEG], 30.0))
            if limit and len(out) >= limit:
                return out
    for zp in sorted((ROOT / "data/drone").glob("uzhfpv_*.zip")):
        try:
            z = zipfile.ZipFile(zp)
        except Exception:                                # noqa: BLE001
            continue
        names = sorted((n for n in z.namelist()
                        if n.startswith("img/image_0_")),
                       key=lambda n: int(n.split("_")[-1].split(".")[0]))
        if len(names) < 200:
            continue
        out.append(ImageSeqSource(f"drone/{zp.stem}", names, 30.0, zf=z))
        if limit and len(out) >= limit:
            break
    return out


CORPORA = {"sim": sim, "bridge": bridge, "car": car, "drone": drone}


def sources(name, limit=None, skip=None):
    """`skip` drops the first N media, which is how a HELD-OUT set is
    made here.

    Every architectural choice in this system - encoder, resolution,
    pooling, cut rule, channel set, weighting, coarse cells - was
    selected by measuring on these same corpora. That is seven
    sequential selections with no held-out data, so the headline is a
    selected-best number rather than an out-of-sample one. Skipping past
    the media the A/Bs used gives a set none of those decisions ever saw.
    """
    if skip is None:
        skip = int(os.environ.get("SDX_SKIP", "0") or 0)
    ss = CORPORA[name](None)
    ss = ss[skip:]
    return ss[:limit] if limit else ss


def main():
    from flowgebd import arg
    lim = arg("--limit", 3, int)
    for name in CORPORA:
        try:
            ss = sources(name, lim)
        except Exception as e:                           # noqa: BLE001
            print(f"{name:<8} ERROR {e}")
            continue
        tot = sum(s.dur for s in ss)
        print(f"{name:<8}{len(ss):>4} media (capped {lim})  "
              f"{tot / 60:.1f} min")
        for s in ss[:2]:
            F = s.cut(0.0, 2.0)
            print(f"         {s!r:<46} cut(0,2) -> {F.shape}")


if __name__ == "__main__":
    main()
