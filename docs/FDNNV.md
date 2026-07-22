# FDNN-V: the write-path video encoder

The database must embed **every frame, at write time**. This document is the
model built for that, the FDNN rules translated to vision+time, and the
iteration log — each design change justified by a measurement, including the
ones that failed.

---

## 1. Why SigLIP cannot be the write path

SigLIP is a 428M-parameter model that treats each frame as an unrelated
photograph: measured 90.3 ms/frame (384px) / 27.7 ms/frame (224px) on this
machine, against 2.2 ms/frame for byte-range decode. 97% of ingest is the
encoder, and no amount of storage engineering changes that — the encoder had
to change.

Video is not a pile of photographs. Frame *t* is almost entirely explained by
frame *t−1* ([Deep Feature Flow](https://arxiv.org/abs/1611.07715) built a
recognition system on exactly that), and a human watching video does not
re-parse the scene 30 times a second — they keep a scene model and update it.
An encoder built *for* video carries state, which is simultaneously the
"understands video, not frames" property and the thing that makes per-frame
cost collapse.

## 2. The FDNN rules, translated

| FDNN (MLP regression) | FDNN-V (vision + time) |
|---|---|
| **1. Each neuron is a sub-network** — KAN-style sum over FINER/Gabor/poly-phase sub-functions | Each temporal-core channel is a KAN sum over the same three bases applied to the *(glimpse, state)* signal: FINER = periodic motion, Gabor = temporal burst (a grasp, a brake light), poly-phase = chirp/acceleration. ω bands partition the temporal spectrum (slow = scene identity, mid = object motion, fast = transitions). Spatially, the conv stem's first layer is a Gabor bank — V1 simple cells *are* Gabor filters. |
| **2. Neurogenesis & apoptosis post-training** | Identical cycle in `scripts/fdnnv_prune.py` — and unlike the attempt to prune SigLIP itself (fidelity collapsed to 0.43 with nothing to fine-tune on), the re-settle step exists here: distillation pairs are free, every corpus frame has a teacher vector. |
| **3. PPO + reverse attention decide** | Utilization = Δ held-out embedding fidelity when a channel is silenced; reverse attention surfaces candidates; a bandit searches keep-fractions; compaction deletes channels for real FLOPs (verified equivalent to 7e-9). |

Architecture (`python/elidedb/fdnnvideo.py`):

```
frame (192x144) ──► stem (tiny ViT or Gabor-conv pyramid) ──► glimpse g_t
                                                                  │
        state h_{t-1} ──► FDNN temporal cell (KAN-sum bases, ──► h_t
                          GRU-style gate, persist-by-default)     │
                                                                  ▼
              e_t = normalize( head_g·g_t + head_h·h_t )   ∈ SigLIP space
```

`step(frame, h) → (embedding, h')` is causal and O(1)/frame — the encoder sits
*inside* the ingest loop. Output lands in the teacher's 1152-d space, so every
existing index, text query, and centroid works unchanged.

## 3. The iteration log (the method is the point)

Split is by time per stream, never random. Baseline: ridge from raw pixels =
fid 0.62 (p10 −0.66).

| iteration | change | result | verdict |
|---|---|---|---|
| 1a | stem race, 4 epochs each | conv 0.897 fid / 0.256 ms·f (97 s train); **ViT 0.900 / 0.153 ms·f (23 s)** | ViT: matmuls win on Apple GPU |
| 1b | stage-1 per-frame distill | fid **0.907**, but kNN agreement **0.044** | closeness ≠ retrieval |
| — | **calibration**: teacher-224 vs teacher-384 | fid 0.921, kNN **0.507**, text top-10 **4.3/10** | the ceiling is 4.3/10, not 10/10; fid→kNN curve is near-vertical at ~0.92 |
| 1c | stage-2 recurrence, ω ≤ 15 | fid **−0.019** — loss climbed during training | 32-step BPTT through fast sin bases is chaotic |
| 2 | centered (mean-free) loss + caption anchors (w=50) + ω ≤ 8 + grad clip + keep-best guard | stage 2 now **helps**: fid flat, kNN 0.028 → **0.059** | calmed recurrence earns its place |
| 3a | anchor weight 50 → 800 | stage-2 fidelity **collapsed to 0.63**; guard shipped stage 1 | louder loss ≠ better; capacity is binding |
| 3b | closed-form linear corrector (ridge, student→teacher) | fid +0.003, text overlap 0.2 → 0.5/10 | the residual is *absent*, not scrambled |
| 4 | scale stem: ViT dim 384, depth 6 (8.9M) | fid 0.892, kNN 0.047, 0.34 ms/f | **3.3× params bought nothing** — at 192px, capacity beyond ~3M is not the constraint |
| 5 (shipped) | iter-2 recipe on extended cache (+file-000 stride-8 teacher pairs) | fid **0.8943**, kNN **0.069**, temporal gain positive on ALL metrics (+0.005 fid, +0.009 p10, +0.017 kNN) | operating point: smallest model within tolerance of best |
| 6 | FDNN cycle (rules 2+3) on the shipped model | apoptosis 256→**70 channels**, full-eval fid **0.8943 → 0.9020**, params 2.71M → **1.95M**, 0.196 → **0.153 ms/f** | per-channel utilization ≈ 0 everywhere; the bandit found keep=27% *improves* fidelity — over-provisioned capacity was noise |

Two findings worth keeping:

- **The common-mode trap, again.** On a homogeneous corpus, pointwise cosine
  buys ~0.9 for free by matching the shared component; ranking runs on the
  thin residual. The same insight that made the context index mean-free
  applies to distillation losses.
- **Metrics need ceilings.** kNN overlap of 0.044 looks catastrophic until you
  measure that the teacher against itself at a different resolution scores
  0.507, and 4.3/10 on text queries. Every quality number in this doc is
  reported against that calibration, not against a fantasy of 1.0.

## 4. Speed, measured (shipped model, 1.95M params after the cycle)

| | per frame | vs SigLIP-224 | vs SigLIP-384 |
|---|---|---|---|
| FDNN-V embed (batched) | **0.270 ms** | **103×** | **334×** |
| full Bridge store, EVERY frame (58,011) | 135.7 s wall (427 fps) | — | — |
| full 477 h corpus, every frame, single process | **5.6 h** | was 38.5 h | was 112 h |

The write path is now **decode-bound** (2.0 ms/f of the 2.3 ms/f total);
decode parallelises across streams, embedding at 0.26 ms/f keeps up with
~200 live 5 fps streams on one machine. Embed-on-write is real:
`store.embed_frames(engine="fdnnv")` runs every frame through the streaming
encoder into `frame_vectors`.

## 5. Honest status

**Delivered:** every-frame embed-on-write at 0.27 ms/frame (103× SigLIP-224);
pointwise fidelity 0.9020 against a teacher-self ceiling of 0.9206; stream
generalisation 0.897 with stride-8 teacher coverage; the temporal state
measurably improving all metrics; the full FDNN cycle improving fidelity
while deleting 73% of the temporal core.

**Not delivered, stated plainly:** text-query top-10 overlap is 0.5/10
against a ceiling of 4.3/10, essentially unmoved across four iterations.
Everything that was tried — anchor weighting at two scales, a closed-form
corrector, 3.3× capacity — is in the table above with its measurement. The
residual that text ranking runs on sits below what a 192px student trained on
17k pairs captures. The un-tried levers, in expected order of payoff:
distill on the FULL un-labelled corpus (58k+ frames of pixels are free; only
the loss's anchor term needs teacher pairs), higher input resolution, and a
query-side calibration that re-ranks the student's top-100 with teacher
embeddings of just those frames (bounded teacher cost per query).

Until that lands, the honest deployment is: **student vectors serve
image-image similarity, clip search, and the context index; text search
keeps using teacher-built window embeddings** — both tables in the same
store, both in the same space, each doing what it is measured to do.

The teacher stays in the system as the offline labeller (captions, sampled
ground-truth pairs); the student owns the write path. That division —
expensive model labels once, cheap model serves forever — is the same shape
as the context index, applied one level deeper.
