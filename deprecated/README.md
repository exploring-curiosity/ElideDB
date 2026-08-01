# deprecated/ — the C++ era

Nothing in this directory is on any code path. It is kept because it is
the *provenance* of the current design, not because it runs.

ElideDB v1 was a C++20 embedded engine with two hand-built formats:

* **SFI** — a per-video frame index: `(pts, dts, byte_offset, is_keyframe,
  gop_id)`, built once by scanning packets with libavformat, so that
  retrieving a 2 s window meant decoding only the overlapping GOPs.
* **SDX** — a columnar chunk format: header, column chunks, per-chunk zone
  maps (min/max), footer with the chunk directory at the END of the file,
  so a reader does one seek to learn the layout.

Both were replaced by Parquet in July 2026. **The mechanisms survived the
rewrite; only the file format changed.** Zone maps became Parquet row-group
statistics and the page index; the SDX footer became the Parquet footer; the
SFI frame index became the `frames` table with byte offsets into per-demo
H.264 segments. Writing them by hand first is why the Parquet layer is used
the way it is rather than as a black box.

## Contents

```
cpp/
  src/            libstreetdex: catalog, format, video, index, query, align, metrics
  bindings/       pybind11 surface
  tests/          Catch2: SDX round-trip + property tests, SFI, planner, align, B+ tree
  CMakeLists.txt  the build that produced `sdx`
  build/          CMake output (git-ignored, regenerable)
  store/          a v1 store in .sfi / .sdx (git-ignored data, kept for format archaeology)

sidecar/          the v1 Python ML sidecar
  embed.py        SigLIP per-window embeddings -> run files the C++ vector index mmapped
  cluster.py      PCA(50) -> HDBSCAN -> clusters.json (the coarse quantiser)
  register_run.py registered a semantic run as a manifest snapshot
  embed_text.py   one text query -> raw float32 on stdout, read by the C++ CLI
  sfi_reader.py   minimal SFI reader so Python could pread single packets
```

The sidecar is the same boundary the project still draws — models in
Python, hot path in the compiled core. Only the compiled core changed
language. See the root `CLAUDE.md` §2 for why that boundary exists.

## Why this is not simply deleted

Two of the interview-relevant claims in `BENCHMARKS.md` are about the
*difference* between the two eras, and the code behind the earlier number
should be readable. Deleting it would leave the comparison unfalsifiable.
