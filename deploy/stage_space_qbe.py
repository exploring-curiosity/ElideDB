"""Assemble deploy/space_qbe/: the query-by-example Hugging Face Space.

A SECOND Space, not a change to the first. The text demo
(SudharshanR/elidedb-demo) answers "describe what you want" and pays five
text towers and a fifteen minute first boot for it. This one answers the
question a text encoder cannot - which direction a thing went - and pays
nothing: every vector it ranks was computed at ingest, so the image holds
no model at all and the service is warm the moment the store lands.

Both read the SAME public store dataset. Nothing here writes to it, and
nothing here touches the other Space's repo.

  python deploy/stage_space_qbe.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "space_qbe"

SPACE_README = """\
---
title: ElideDB Query by Example
emoji: 🎬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ElideDB — query by example

1,122 robot episodes, and not one word of text. Pick a few clips that show
the same kind of moment; the archive returns its own kind.

**Why this exists.** A text encoder scores *"open the drawer"* against
*"close the drawer"* at cosine 0.977, and a pooled video embedding scores a
clip against its own reversal at 1.000000. Description destroys the
distinction before search begins. A clip needs no description: it is an
instance of what it shows.

**What runs when you click.** Nothing loads. Every vector was computed when
the video was written, so a query is arithmetic over parquet columns —
about 250 ms over the whole archive. Six channels vote, weighted by how
tightly each one pulls *your* picks together relative to how it holds the
archive: a statistic taken from the query itself, with no labels and no
training. The panel shows those weights, because which model recognised
your pick is the interesting part of the answer.

**How many come back.** The count is a ceiling, not a target. Each pick is
held out in turn to see how deep its own kind ranks, and the set stops
where that evidence stops — returned means believed.

Thumbnails and clips are decoded from the stored video's own byte ranges on
demand. There is no preview cache anywhere in the store.

The text-query demo of the same engine is at
[SudharshanR/elidedb-demo](https://huggingface.co/spaces/SudharshanR/elidedb-demo).
"""


def copy(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copy(ROOT / "python", OUT / "python")
    for f in ("requirements-qbe.txt", "entrypoint-qbe.sh", "qbe_serve.py",
              "qbe_ui.html"):
        copy(ROOT / "deploy" / f, OUT / "deploy" / f)
    # the Dockerfile builds from the repo root with this exact layout
    copy(ROOT / "deploy" / "Dockerfile.qbe", OUT / "Dockerfile")
    copy(ROOT / ".dockerignore", OUT / ".dockerignore")
    (OUT / "README.md").write_text(SPACE_README)
    (OUT / ".gitattributes").write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.h264 filter=lfs diff=lfs merge=lfs -text\n"
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
        "*.npy filter=lfs diff=lfs merge=lfs -text\n")

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"staged {OUT}  ({total/1e6:.1f} MB)")
    print()
    print("Then, from your terminal:")
    print("  hf repo create elidedb-qbe --repo-type space "
          "--space_sdk docker")
    print(f"  cd {OUT}")
    print("  git init -b main && git add . && git commit -m 'QbE Space'")
    print("  git remote add origin "
          "https://huggingface.co/spaces/<YOUR_USER>/elidedb-qbe")
    print("  git push -u origin main --force")


if __name__ == "__main__":
    main()
