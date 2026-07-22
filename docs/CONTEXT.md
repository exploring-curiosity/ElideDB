# Context retrieval

Semantic search answers *what is in frame*. Context search answers *what is
happening*. This document is the design, the measurements that chose it, and
the parts that do not work yet.

---

## 1. The problem

`embeddings` holds one mean-pooled SigLIP vector per window. Mean pooling is
order-blind — reverse the frames and the vector is identical — so the index
cannot represent motion or relation. "Two people working together on a laptop"
therefore matches every clip containing two people and a laptop.

The previous fix was to rerank with a VLM at query time. It works, and it costs
**2.5 seconds per query**. That is the wrong place for a database to spend
time: the expensive operator runs on every query forever, instead of once at
ingest.

## 2. The shape of the fix

Precompute the expensive operator into an index and make the query a lookup.

```
INGEST (once)                          QUERY (every time)
─────────────────────────────          ──────────────────────────
frames ─► SigLIP per-frame vectors     text ─► TF-IDF ─► LSA-48
             │                                    │
             ├─► VLM reads 3 frames of            └─► one matmul over a
             │   a window, writes a                   Parquet column
             │   relational caption
             │        │                            no VLM. no decode.
             │        ├─► TF-IDF ─► LSA-48  ──────► `context` table
             │        │   (exact, for captioned windows)
             │        │
             └────────┴─► temporal tower learns to predict
                          that vector from frame vectors alone
                          (estimated, for everything else)
```

The VLM labels what you can afford. The tower covers the rest. The query does
not care which it got, and the `estimated` column says which it was.

Three tables come out, all ordinary Parquet:

| table | one row per | columns |
|---|---|---|
| `frame_vectors` | frame | `ts, stream, vector[1152]` |
| `context_captions` | labelled window | `ts, t1, stream, caption, vector[1152]` |
| `context` | window | `ts, t1, stream, vector[48], appearance[1152], estimated` |

`appearance` lives in the same row as `context` on purpose: fusing the two
indexes is then two matmuls over one scan, with no join.

## 3. Choosing the context space — by measurement, not by taste

The obvious design is to embed the caption with SigLIP's **text** tower and
store that vector, so query and caption meet in one space. It was built, and it
lost. Measured against an independent VLM judge on 66 held-out windows
(mean yes/no logprob margin over each method's top-5, six relational queries):

| context space | judge |
|---|---|
| appearance only (no context) | +0.276 |
| caption **embedding**, SigLIP text tower — *oracle* | **+0.218** |
| caption **text**, raw TF-IDF | +0.331 |
| caption **text**, LSA-48 | **+0.339** |

The embedding route loses **even with perfect captions**, which rules out the
student as the cause. SigLIP is trained so that text sits near *images*; its
text tower was never optimised for text-to-text comparison, and that is exactly
what "query embedding vs caption embedding" asks of it. Matching the caption as
**text** sidesteps the problem entirely.

LSA-48 is used over raw TF-IDF despite scoring 0.006 lower, for two reasons
that matter more than 0.006: it is dense and fixed-width, so it is a target a
small tower can actually regress, and it is one more `fixed_size_list` column,
so every index already in the store consumes it unchanged.

## 4. The tower

A custom SigLIP video tower over frozen frame features — the temporal half of
an R(2+1)D factorisation, where the spatial half is frozen SigLIP.

```
(T, 1152) frame vectors
   │  frozen PCA (no learned params — see below)
(T, 48)
   │  dilated temporal conv, learned taps           ← the CNN
   │  each output neuron = KAN sum over k=4
   │  heterogeneous bases (FINER / Gabor / poly)
(T, n)
   │  bi-GRU                                        ← the RNN
(T, n + 2h)
   │  attention pooling over time
   │  head
(48,)  ≈ the caption's LSA coordinates
```

**Why a conv *and* a GRU.** They fail differently, which is the only good
reason to have both. The conv is translation-equivariant with a fixed receptive
field: it detects "something moved left-to-right" wherever in the window it
happens. The GRU carries unbounded state: "stationary the whole time, then
braked" is not expressible by any fixed-width kernel. CLIP4Clip
([arXiv 2104.08860](https://arxiv.org/abs/2104.08860)) measured exactly this
axis on frozen CLIP features (meanP vs seqLSTM vs seqTransf). TSM
([ICCV 2019](https://arxiv.org/abs/1811.08383)) was the alternative for the
conv slot — shifting channels along time is free — but its shift is a fixed ±1
tap, and the point here is that the taps are frequency-selective.

**Why the FDNN bases.** A window's feature trajectory is a signal, and these
bases were built for signals. After a temporal convolution, a Gabor
sub-function is a temporal wavelet (a burst: a brake light, a door), a FINER
sub-function is a variable-period oscillator (gait, wipers), and a
polynomial-phase sub-function is a chirp (a vehicle accelerating away). The
frequency bands then partition the **temporal** spectrum instead of every
neuron competing for it. Each neuron being a sum over `k` of these is FDNN
thesis 1, carried over intact.

**Why the PCA is frozen.** There are ~150 labelled windows. A learned
1152→48 encoder is ~55k parameters fitted from 150 examples, which is not
learning. PCA costs zero training examples, so the whole sample budget goes to
the temporal dynamics — the only part mean pooling cannot already do.

**Loss.** SigLIP's own pairwise sigmoid loss
([arXiv 2303.15343](https://arxiv.org/abs/2303.15343)), chosen because the
paper's ablation shows sigmoid beats softmax below ~16k batch and our batch is
the entire labelled set. Pairs whose windows overlap in time on the same stream
are masked out of the negatives — windows slide with 75% overlap, so window
`i+1` genuinely depicts the same moment as window `i`, and calling it a
negative would train the tower to separate identical content.

## 5. Cellular turnover (FDNN theses 2 and 3)

Post-training: `apoptosis → re-settle → neurogenesis → re-settle`, with the
kill/keep decision made by a clipped-PPO contextual bandit over per-channel
features including a reverse-attention score. Utilization is the increase in
**validation retrieval loss** when a channel is silenced — not weight
magnitude.

**What changed from FDNN.** FDNN masks per (layer, neuron). This tower is a
*residual* stack, where channel *j* is a wire running the whole depth: masking
it in one layer does nothing, because the skip connection keeps carrying it. So
the prune unit is the residual-stream **channel**, decided once and applied to
every block. That is the correct form of the idea for this topology, and it has
a property per-layer masking does not — a dead channel can be physically
deleted. `compact()` rebuilds the tower without it (verified numerically
equivalent to 1.5e-7), so the mask becomes wall-clock speed rather than a
multiply by zero.

Result on the caption-LSA target:

| | channels | params | µs/window | val loss |
|---|---|---|---|---|
| before | 48 | 82,770 | 66.5 | 0.3242 |
| after | **17** | **30,504** | **42.4** | **0.2939** |

63% fewer parameters, 36% faster, and validation loss **improved** 9%.
Over-provisioned capacity on a small corpus is memorisation, so apoptosis buys
accuracy and speed at once — which is the regime cellular turnover was designed
for.

Two guardrails were necessary and both came from measured failures:

- **Re-settling is validation-aware.** Survivors re-settle on the train set,
  and an unguarded 400-epoch re-settle took a pruned tower from 0.311 back up
  to 0.357 — undoing everything apoptosis gained.
- **Selection is an operating point, not a minimum.** "Keep the lowest loss"
  can never prune, because the unpruned tower is usually the most accurate and
  the search then returns its own input. Instead: take the best loss seen,
  allow 3% slack, and among everything inside that band keep the fewest
  channels.

## 6. Results

66 held-out windows (last 30% of the timeline, never trained on), six
relational queries, independent VLM judge.

| method | ms/query | judge | judge-top5 overlap |
|---|---|---|---|
| appearance (plain semantic search) | 12.9 | +0.276 | 1.2/5 |
| **context, captions materialised** | **0.6** | **+0.347** | **2.0/5** |
| context, tower-estimated | 0.4 | +0.259 | 1.5/5 |
| fused (0.6 ctx + 0.4 appearance), materialised | 12.5 | +0.323 | 1.8/5 |
| fused, tower-estimated | 12.0 | +0.283 | 1.5/5 |
| VLM rerank at query time | 2508.5 | +0.306 | 1.5/5 |

**The headline: materialised context is 4,000× faster than query-time VLM
reranking and scores higher (+0.347 vs +0.306).** It is also faster than plain
semantic search, because the lexical path needs no neural text encoder at query
time at all — the 12.9 ms of "appearance" is almost entirely SigLIP's text
tower.

## 7. What does not work yet — stated plainly

**The student is weak.** Tower-estimated context scores +0.259, *below* plain
appearance at +0.276. It only earns its place when fused (+0.283). On
validation it reaches cos 0.283 against a corpus-prior baseline of 0.248 — a
real lift, but a small one.

The cause is the corpus, not obviously the architecture: the Oxford RobotCar
sample is **19 seconds** of one continuous drive, giving 153 training windows
whose captions are near-duplicates (mean pairwise cosine 0.68 — every window is
"a city street with buildings and people"). A student cannot learn to
distinguish what the teacher does not distinguish. The honest reading is that
the distillation path is *built and measured* but not yet *demonstrated*; that
needs a corpus with hours of varied footage, not this sample.

**Practical consequence.** At this corpus size, caption everything — it costs
0.75 s/window and wins outright. The tower matters when the corpus is large
enough that captioning all of it is not an option, and that regime is exactly
the one this sample cannot exercise.

**The judge shares a model family with the teacher.** It is asked a different
question and never sees a ranking, so it cannot favour a method, but a fully
independent judge would need a second VLM.

## 8. Using it

```python
from elidedb import Store
db = Store.open("lake/oxford")

db.index_context()                       # whole pipeline, any timestamped video
db.index_context(label_fraction=0.5)     # caption half, let the tower cover the rest

hits, stats = db.search_context("a pedestrian crossing the road", k=5)
hits, _ = db.search_context("...", alpha=1.0)   # context only
hits, _ = db.search_context("...", alpha=0.0)   # appearance only

db.explain(hits[0]["t0"], hits[0]["t1"], hits[0]["stream"])
# -> the teacher's own words for that window, so a result can be checked
```

Reproduce the table above with `python scripts/bench_context.py`.
