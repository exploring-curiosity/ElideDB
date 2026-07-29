"""V1: the event extractor, v0 - state diff + agent + naming.

The answer schema's core made literal: an action's name is its
initial->final state diff. Per demo:

  P1 agent      flow-asymmetry track (probe_actors logic - validated
                40-100% persistence)
  P2 sites      robust before/after diff between the demo's first and
                last frames -> vanish-site (where something left) and
                appear-site (where something arrived). The agent is
                masked out of the diff where its track covers it.
  P3 naming     the VANISH-site crop from the FIRST frame is the
                cleanest look the whole demo offers at the moved
                object - pre-grasp, unoccluded (L3: context kept, the
                crop only focuses the namer). One constrained naming
                question per site (L5: generator, never judge) +
                SigLIP text vec of the name for query-time matching.
                The appear-site is named from the LAST frame (often a
                container: "drawer", "pot").
  P4 verb       topology, not classification:
                  vanish A + appear B          -> moved / put
                  appear inside articulated rgn-> put_into
                  articulated sign +/-         -> open / close
                  vanish only                  -> taken / put_away
                Articulation = a large wide region whose flow is
                coherent horizontal/vertical mid-demo, signed.

V1 mode prints event scripts for --n random demos and the cost/demo;
no store writes. Gates (plan): agent found >=90%, plausible
participant name >=80%, cost <= 8s/demo.

  python scripts/extract_events.py --n 20 [--seed 0] [--json out.json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                    # noqa: E402

JUDGE = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
MIN_BLOB = 120          # px in the before/after diff
PAD = 0.6               # crop padding around a site, fraction of box


def agent_track(frames):
    """Largest persistent coherent-motion track + per-frame agent mask."""
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    H, W = grey[0].shape
    masks = np.zeros((len(frames), H, W), bool)
    tracks, live = [], []
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-6
        mv = (mag > max(1.0, med + 4 * 1.4826 * mad)).astype(np.uint8)
        masks[i] |= mv.astype(bool)
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(mv, 8)
        blobs = [(tuple(cents[j]), float(stats[j, cv2.CC_STAT_AREA]))
                 for j in range(1, nlab)
                 if stats[j, cv2.CC_STAT_AREA] >= 40]
        nxt = []
        for c, a in blobs:
            best, bd = None, 1e9
            for tr in live:
                dd = float(np.hypot(*(np.array(tr[-1][1]) - np.array(c))))
                if dd < bd:
                    best, bd = tr, dd
            if best is not None and bd < 40:
                best.append((i, c, a))
                nxt.append(best)
            else:
                tr = [(i, c, a)]
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    if len(frames) > 1:
        masks[-1] = masks[-2]
    main = max(tracks, key=len) if tracks else None
    # a track may collect >1 blob per frame pair; span is coverage of
    # DISTINCT frame pairs, capped at 1
    span = (len({e[0] for e in main}) / max(len(frames) - 1, 1)
            if main else 0.0)
    return main, min(span, 1.0), masks


def state_diff(first, last, agent_union):
    """Changed blobs between first and last frame, agent regions
    excluded. Returns list of (x0,y0,x1,y1, kind) - kind is which side
    changed more locally: 'vanish' (content left) or 'appear'."""
    import cv2
    H, W = first.shape[:2]
    d = np.abs(first.astype(int) - last.astype(int)).mean(2)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    m = (d > max(12.0, med + 6 * 1.4826 * mad)).astype(np.uint8)
    m[agent_union] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    sites = []
    for j in range(1, nlab):
        if stats[j, cv2.CC_STAT_AREA] < MIN_BLOB:
            continue
        x, y, w, h = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP],
                      stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])
        blob = lab[y:y + h, x:x + w] == j
        # local texture: the side where the region has MORE structure is
        # the side where the object IS - high-variance in first => it
        # was there and left (vanish); high-variance in last => appear
        import cv2 as _cv
        g1 = _cv.cvtColor(first[y:y + h, x:x + w], _cv.COLOR_RGB2GRAY)
        g2 = _cv.cvtColor(last[y:y + h, x:x + w], _cv.COLOR_RGB2GRAY)
        v1 = float(_cv.Laplacian(g1, _cv.CV_64F)[blob].var())
        v2 = float(_cv.Laplacian(g2, _cv.CV_64F)[blob].var())
        kind = "vanish" if v1 > v2 else "appear"
        sites.append((int(x), int(y), int(x + w), int(y + h), kind,
                      int(stats[j, cv2.CC_STAT_AREA])))
    sites.sort(key=lambda s: -s[5])
    return sites[:4]


def articulation(frames, agent_union):
    """Signed CUMULATIVE displacement of large coherent non-agent
    regions across consecutive pairs. First-to-last flow alone missed
    the drawer demos (open registered +/-1.1-1.5, under threshold):
    Farneback across a 6s gap underestimates a slow slide; summing
    per-pair medians recovers it. Sign: dominant axis, positive =
    down/right in image space."""
    import cv2
    if len(frames) < 4:
        return 0.0
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tot_dx = tot_dy = 0.0
    votes = 0
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        med = float(np.median(mag))
        m = (mag > max(1.0, 3 * med)) & (~agent_union)
        if m.sum() < 800:
            continue
        tot_dx += float(np.median(flow[..., 0][m]))
        tot_dy += float(np.median(flow[..., 1][m]))
        votes += 1
    if votes == 0:
        return 0.0
    return float(tot_dy if abs(tot_dy) >= abs(tot_dx) else tot_dx)


def name_site(frames_idx_img, box, namer):
    x0, y0, x1, y1, kind, _ = box
    im = frames_idx_img
    H, W = im.shape[:2]
    px, py = int((x1 - x0) * PAD), int((y1 - y0) * PAD)
    cx0, cy0 = max(x0 - px, 0), max(y0 - py, 0)
    cx1, cy1 = min(x1 + px, W), min(y1 + py, H)
    return namer(im[cy0:cy1, cx0:cx1])


def make_namer():
    import tempfile

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    from elidedb.rerank import _load
    model, processor, cfg, _, _ = _load(JUDGE)

    def namer(crop_hwc):
        im = Image.fromarray(crop_hwc)
        if im.width < 48 or im.height < 48:
            im = im.resize((max(im.width * 3, 96), max(im.height * 3, 96)))
        q = ("What is the main object at the center of this image? "
             "Answer with only 1-3 words, no sentence.")
        prompt = apply_chat_template(processor, cfg, q, num_images=1)
        with tempfile.TemporaryDirectory() as td:
            p = f"{td}/c.jpg"
            im.save(p, "JPEG", quality=92)
            r = generate(model, processor, prompt, image=[p],
                         max_tokens=8, verbose=False)
        name = (r.text or "").strip().lower().strip(".").split("\n")[0]
        return name[:48]
    return namer


def extract(db, frames_tbl, key, namer):
    """One demo -> event script dict."""
    from elidedb.video import FrameSet
    s, a, b = key
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), s),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                pc.less_equal(frames_tbl.column("ts"), b))))
    n = len(sel)
    if n < 4:
        return None
    pick = np.unique(np.linspace(0, n - 1, min(12, n)).round().astype(int))
    dec = sorted(FrameSet(db, "frames", sel.take(pick)).decode())
    frames = [f for _, f in dec]
    tr, span, masks = agent_track(frames)
    agent_union = masks.any(0)
    first, last = frames[0], frames[-1]
    sites = state_diff(first, last, agent_union)
    art = articulation(frames, agent_union)

    parts = []
    for box in sites[:2]:
        # tiny diff blobs produced junk names ("triangle", "moon") -
        # a namer needs enough pixels to see an object at all
        if box[5] < 400:
            continue
        src_img = first if box[4] == "vanish" else last
        try:
            nm = name_site(src_img, box, namer)
        except Exception as e:
            nm = f"<namer failed: {type(e).__name__}>"
        parts.append({"kind": box[4], "box": box[:4], "name": nm})

    vanish = [p for p in parts if p["kind"] == "vanish"]
    appear = [p for p in parts if p["kind"] == "appear"]
    if abs(art) > 1.5 and not (vanish and appear):
        verb = "open" if art > 0 else "close"
    elif vanish and appear:
        verb = "move"
    elif vanish:
        verb = "put_away"
    elif appear:
        verb = "bring"
    else:
        verb = "adjust"
    return {"stream": s, "ts": a, "dur_s": round((b - a) / 1e9, 1),
            "agent_span": round(span, 2), "verb": verb,
            "articulation": round(art, 2), "participants": parts}


def main():
    argv = sys.argv
    n_want = int(argv[argv.index("--n") + 1]) if "--n" in argv else 20
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    jout = argv[argv.index("--json") + 1] if "--json" in argv else None

    db = Store.open("lake/bench")
    ep = db.table("episodes").scan().to_pydict()
    keys = list(zip(ep["stream"], (int(v) for v in ep["ts"]),
                    (int(v) for v in ep["t1"])))
    rng = np.random.default_rng(seed)
    sample = [keys[i] for i in rng.choice(len(keys), n_want, replace=False)]
    frames_tbl = db.table("frames").scan()
    namer = make_namer()

    out, t0 = [], time.time()
    ok_agent = ok_name = 0
    for k in sample:
        r = extract(db, frames_tbl, k, namer)
        if r is None:
            continue
        out.append(r)
        ok_agent += r["agent_span"] >= 0.4
        ok_name += any(p["name"] and not p["name"].startswith("<")
                       for p in r["participants"])
        pl = "; ".join(f"{p['kind']}:{p['name']}" for p in r["participants"])
        print(f"{r['stream'].split('/')[-1]} {str(r['ts'])[-8:]} "
              f"{r['dur_s']:>4}s  agent {r['agent_span']:.0%}  "
              f"verb {r['verb']:<8} art {r['articulation']:+.1f}  {pl}",
              flush=True)
    dt = (time.time() - t0) / max(len(out), 1)
    print(f"\n{len(out)} demos  agent-found {ok_agent}/{len(out)}  "
          f"named {ok_name}/{len(out)}  {dt:.1f}s/demo")
    if jout:
        Path(jout).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
