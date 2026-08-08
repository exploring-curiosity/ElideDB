"""Contact sheets: what the system returns, as pixels, for a human to judge.

The automatic benchmark can only measure a necessary condition - the same
event, re-filmed, must come back - because no label-free answer key for
"like that one" exists. The sufficient condition has exactly one
admissible judge: a person looking at the query and looking at what came
back.

So this renders it. Top row is the example. Every row below is one
returned span, in rank order, as a filmstrip. Nothing is captioned,
scored on the image, or sorted by anything but the system's own ranking,
because a label on the picture would tell the judge what to think.

Frames are decoded ONLY for the spans that survived the search - the
same late materialisation the query path uses. A sheet costs a few
hundred KB of video decode, not a corpus scan.

    python native/vsheet.py --store sim --n 6 --k 8
    open stores/vision/sim/sheets/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "native"))

import vcore                                             # noqa: E402
import vgrade                                            # noqa: E402
import vqbe                                              # noqa: E402
import vsrc                                              # noqa: E402
from vstore import CHANNELS, STORES, Store               # noqa: E402

NF = 6             # frames per filmstrip
TH = 128           # thumbnail height
PAD = 4


def strip(frames, nf=NF, th=TH):
    from PIL import Image
    if len(frames) == 0:
        return None
    idx = np.linspace(0, len(frames) - 1, nf).round().astype(int)
    ims = []
    for i in idx:
        im = Image.fromarray(np.ascontiguousarray(frames[int(i)]))
        w = max(int(im.width * th / im.height), 8)
        ims.append(im.resize((w, th), Image.BILINEAR))
    W = sum(i.width for i in ims) + PAD * (len(ims) - 1)
    out = Image.new("RGB", (W, th), (16, 16, 16))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + PAD
    return out


def sheet(st, srcs, mid, t0, t1, k=8, probe=vqbe.CAND_FRAC, ref=None,
          exclude_sep=None):
    from PIL import Image
    src = srcs[mid]
    Q = vqbe.Query.from_media(src, t0, t1)
    ex = None if exclude_sep is None else (mid, t0, t1, exclude_sep)
    res, info = vqbe.search(st, Q, k=k, abstain=True, probe=probe,
                            ref=ref, exclude=ex)
    rows = [strip(src.cut(t0, t1))]
    for m, a, b, _s in res:
        if m in srcs:
            rows.append(strip(srcs[m].cut(a, b)))
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return None, res, info
    W = max(r.width for r in rows)
    # a bright band under the example separates query from answers
    H = sum(r.height for r in rows) + PAD * len(rows) + 6
    out = Image.new("RGB", (W, H), (16, 16, 16))
    y = 0
    for i, r in enumerate(rows):
        out.paste(r, (0, y))
        y += r.height + PAD
        if i == 0:
            out.paste(Image.new("RGB", (W, 6), (210, 120, 40)), (0, y))
            y += 6
    return out, res, info


def main():
    from tqdm import tqdm
    from flowgebd import arg
    name = arg("--store", "sim")
    n = arg("--n", 6, int)
    k = arg("--k", 8, int)
    # A product query should not retrieve the seconds either side of
    # itself; that is not an answer, it is the same moment again.
    sep = arg("--sep", vcore.CTX_MULT * max(vcore.SCALES), float)
    st = Store(STORES / name)
    srcs = {s.id: s for s in vsrc.sources(name)}
    rs = np.random.RandomState(0)
    qs = [q for q in vgrade.sample_queries(st, n, rs) if q[0] in srcs]
    sel = np.sort(rs.choice(st.n, min(vqbe.REF_SAMPLE, st.n),
                            replace=False))
    ref = {c: st.col[c].take(sel) for c in CHANNELS}
    d = st.path / "sheets"
    d.mkdir(exist_ok=True)
    print(f"{st!r}\n{'query':<34}{'returned':<10}{'cross':<8}{'weights'}")
    xm = []
    for i, (mid, t0, t1) in enumerate(tqdm(qs, desc="sheets", unit="q",
                                           leave=False)):
        img, res, info = sheet(st, srcs, mid, t0, t1, k, ref=ref,
                               exclude_sep=sep)
        if img is None:
            continue
        fp = d / f"{i:02d}_{mid.replace('/', '_')}_{t0:.0f}.jpg"
        img.save(fp, "JPEG", quality=88)
        w = "  ".join(f"{c} {info['w'][c]:.2f}" for c in CHANNELS)
        # CROSS-MEDIA FRACTION. Answers from the same recording as the
        # query are legitimate product results, but only cross-media
        # hits are evidence the representation generalises past one
        # session - the exact distinction that inflated every earlier
        # number in this project. Reported, never silently averaged in.
        x = (sum(1 for m, *_ in res if m != mid) / len(res)) if res else 0.0
        xm.append(x)
        print(f"{mid + f' [{t0:.0f},{t1:.0f}]':<34}{len(res):<10}"
              f"{x:<8.2f}{w}")
    print(f"\ncross-media fraction of answers: "
          f"{float(np.mean(xm)) if xm else 0:.2f}   "
          f"(same-media answers are valid results but are not evidence "
          f"of generalisation)")
    print(f"{len(list(d.glob('*.jpg')))} sheets in {d}")
    print("top strip is the example; every strip below is an answer, "
          "in the system's own rank order.")


if __name__ == "__main__":
    main()
