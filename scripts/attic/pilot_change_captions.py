"""Pilot: can the 2B answer "what changed?" on event segments correctly?

Open-ended scene captions failed at scale (measured: 'close' mentions occur
MORE in open episodes than close episodes — the captioner narrates a
template, not the clip). The before/after DIRECT formulation is the one
place the 2B measurably works (AUC 0.75). This pilot asks for a free-form
CHANGE caption on first/last frames of events inside close/open episodes
and grades the verb against the episode's human task label. Decides whether
change-captioning the corpus is worth a full pass.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.rerank import DEFAULT_VLM, _load                # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

PROMPT = ("These are the first and last frames of a short robot "
          "manipulation clip. In one short sentence, state what CHANGED "
          "between them — the action that must have happened. Start with "
          "a verb. If a container or drawer changed state, say so.")


def main():
    from PIL import Image
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = {"close": [], "open": []}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        k = (k or "").lower()
        if "drawer" not in k:
            continue
        which = ("close" if "close" in k else
                 "open" if "open" in k and "put" not in k else None)
        if which:
            eps[which].append((stream_of.get(int(i)), int(a), int(b)))

    ev = db.table("context_events").scan()
    es = ev.column("stream").to_pylist()
    ea = [int(v) for v in ev.column("ts").to_pylist()]
    eb = [int(v) for v in ev.column("t1").to_pylist()]

    picks = []
    for kind, lst in eps.items():
        found = 0
        for s, a, b in lst:
            if found >= 15:
                break
            for s2, a2, b2 in zip(es, ea, eb):
                if s2 == s and a2 >= a and b2 <= b and b2 - a2 > 1e9:
                    picks.append((kind, s2, a2, b2))
                    found += 1
                    break

    vlm, processor, cfg, _, _ = _load(DEFAULT_VLM)
    prompt = apply_chat_template(processor, cfg, PROMPT, num_images=2)
    frames_tbl = db.table("frames").scan()
    tmp = Path(tempfile.mkdtemp(prefix="pilot_cc_"))

    ok = {"close": [0, 0], "open": [0, 0]}
    for kind, s, a, b in picks:
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            continue
        dec = FrameSet(db, "frames", sel.take(
            np.array([0, len(sel) - 1]))).decode(width=448)
        if len(dec) < 2:
            continue
        paths = []
        for j, (_, arr) in enumerate(sorted(dec)):
            p = tmp / f"{kind}_{a}_{j}.jpg"
            Image.fromarray(arr).save(p)
            paths.append(str(p))
        out = generate(vlm, processor, prompt, paths, max_tokens=48,
                       temperature=0.0, verbose=False)
        cap = (out.text if hasattr(out, "text") else str(out)).lower()
        hit = ("clos" in cap) if kind == "close" else ("open" in cap)
        inv = ("open" in cap) if kind == "close" else ("clos" in cap)
        ok[kind][0] += int(hit and not inv)
        ok[kind][1] += 1
        print(f"[{kind}] {'HIT ' if hit and not inv else 'miss'} {cap[:90]}",
              flush=True)

    for kind, (h, n) in ok.items():
        print(f"{kind}: {h}/{n} correct-verb change captions")


if __name__ == "__main__":
    main()
