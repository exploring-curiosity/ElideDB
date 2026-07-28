"""Assemble deploy/space/: a Hugging Face Space repo, ready to push.

Everything the Space needs, laid out so the repo-root Dockerfile
builds unchanged: the engine code, the standalone demo store, the
deploy scripts, LFS rules for the large files, and the Space README
with its required metadata. Run it, then follow the printed commands.

  python deploy/stage_space.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "space"

SPACE_README = """\
---
title: ElideDB
emoji: 🟢
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ElideDB live demo

The real product, read only: 1,122 robot manipulation episodes,
searchable by plain description. First boot downloads model weights
and warms every text tower (about 15 minutes); after that, queries
run in seconds.
"""

GITATTRS = """\
*.parquet filter=lfs diff=lfs merge=lfs -text
*.h264 filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
"""


def copy(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    # the store itself ships as a public dataset (Space repos cap at
    # 1 GB); the container downloads it at boot via DEMO_STORE_DATASET
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copy(ROOT / "python", OUT / "python")
    copy(ROOT / "scripts" / "get_iv2.py", OUT / "scripts" / "get_iv2.py")
    for f in ("requirements.txt", "entrypoint.sh", "warm.py"):
        copy(ROOT / "deploy" / f, OUT / "deploy" / f)
    # the Dockerfile builds from the repo root with this exact layout
    copy(ROOT / "deploy" / "Dockerfile", OUT / "Dockerfile")
    copy(ROOT / ".dockerignore", OUT / ".dockerignore")
    (OUT / "README.md").write_text(SPACE_README)
    (OUT / ".gitattributes").write_text(GITATTRS)

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"staged {OUT}  ({total/1e9:.2f} GB)")
    print()
    print("Next, from your terminal (needs a huggingface.co account):")
    print("  1. pip install -U huggingface_hub && hf auth login")
    print("  2. create the Space (public, Docker) once:")
    print("       hf repo create elidedb-demo --repo-type space "
          "--space_sdk docker")
    print("  3. push the staged repo:")
    print(f"       cd {OUT}")
    print("       git init -b main && git lfs install")
    print("       git add . && git commit -m 'ElideDB demo'")
    print("       git remote add origin "
          "https://huggingface.co/spaces/<YOUR_USER>/elidedb-demo")
    print("       git push -u origin main --force")
    print("  4. watch the Space build; first boot warms models "
          "(~15 min), then the console is live.")


if __name__ == "__main__":
    main()
