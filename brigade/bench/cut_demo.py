#!/usr/bin/env python3
"""Stitch the demo-act clips into one film with captions.

    python3 brigade/bench/cut_demo.py --out eval_logs/demo_act/BRIGADE_DEMO.mp4

Each beat gets a title card (act, instruction, measured success rate) followed by
the clip. Captions come from demo_act_results.json — the numbers on screen are
the numbers that were measured, not typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = H = 512          # upscale from the sim's 256 so text is legible
FPS = 30
CARD_S = 1.6         # seconds a title card holds


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def title_card(act, instruction, rate, seconds):
    img = Image.new("RGB", (W, H), (11, 11, 12))
    d = ImageDraw.Draw(img)
    f_act = _font(22)
    f_main = _font(30)
    f_sub = _font(20)

    d.text((36, 44), f"ACT {act}", font=f_act, fill=(255, 138, 30))
    d.line([(36, 78), (W - 36, 78)], fill=(42, 42, 49), width=2)

    lines = _wrap(d, f'"{instruction}"', f_main, W - 72)
    y = 130
    for ln in lines:
        d.text((36, y), ln, font=f_main, fill=(232, 228, 220))
        y += 40

    y = max(y + 26, H - 150)
    d.text((36, y), f"policy: pi0.5 (3.6B)   {rate}", font=f_sub, fill=(143, 191, 106))
    d.text((36, y + 30), f"{seconds:.0f}s  ·  no scripted motion",
           font=f_sub, fill=(138, 151, 166))
    return np.array(img)


def label_frames(frames, instruction):
    """Burn a one-line caption along the bottom of every frame."""
    out = []
    f = _font(17)
    for fr in frames:
        img = Image.fromarray(fr).resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        d.rectangle([(0, H - 42), (W, H)], fill=(11, 11, 12))
        txt = instruction if d.textlength(instruction, font=f) < W - 28 else instruction[:64] + "…"
        d.text((14, H - 30), txt, font=f, fill=(232, 228, 220))
        out.append(np.array(img))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_logs/demo_act/demo_act_results.json")
    ap.add_argument("--out", default="eval_logs/demo_act/BRIGADE_DEMO.mp4")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"no results at {args.results} — run demo_act.py first", file=sys.stderr)
        return 1
    beats = json.load(open(args.results))

    reel: list[np.ndarray] = []
    used = 0
    for b in beats:
        vid = b.get("video")
        if not vid or not os.path.exists(vid):
            print(f"  skip (no video): {b['instruction'][:50]}")
            continue
        frames = iio.imread(vid, plugin="pyav")
        rate = f"{b.get('n_ok', 0)}/{b.get('n', 0)} success"
        card = title_card(b["act"], b["instruction"], rate, b.get("seconds", 0))
        reel.extend([card] * int(CARD_S * FPS))
        reel.extend(label_frames(frames, b["instruction"]))
        used += 1
        print(f"  + [{b['act']}] {b['instruction'][:58]}  ({len(frames)} frames)")

    if not reel:
        print("nothing to stitch", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # pyav plugin spells it in_pixel_format/out_pixel_format, not pixelformat
    iio.imwrite(args.out, np.stack(reel), fps=FPS, codec="libx264",
                out_pixel_format="yuv420p", plugin="pyav")
    secs = len(reel) / FPS
    print(f"\nwrote {args.out}  —  {used} beats, {secs:.0f}s, {len(reel)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
