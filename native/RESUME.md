# State — 2026-08-08 01:10

## Settled configuration (all measured, all recorded in vcore/vstore/vqbe)

    encoder   DINOv3 ViT-L/16, frozen, fp32     SDX_ENC=vitl
    decode    640 wide, 4 Hz                    vsrc.WIDTH / FPS
    pooling   GeM p=3 over patch tokens         SDX_POOL=gem
    ladder    RES 256 / 320 / 384               SDX_RES
    windows   2 / 4 / 8 s, stride half, scale-relative leave-one-out basis
    channel   c2 ONLY (rank pooling)            SDX_CHANNELS=c2
    cut       vqbe.score_cut, self-stopping recursive Otsu

Defaults in code now MATCH this. They did not until 2026-08-07: the code
defaulted to ConvNeXt-Tiny while every measured run set SDX_ENC=vitl, so
anyone without the environment got a different system than the numbers.

## Numbers

FULL SCALE, all four stores, 40 queries each — native/V_FINAL_c2.json,
the clean c2-only artifact (2026-08-08 06:37, 188 min):

    store    media  windows  yield  prec   ret   chance  rank1  ident  D-marg  P-rec
    sim      150    6006     0.713  0.594  14.3  0.007   0.971  0.907  0.736   0.992
    bridge    60    6120     0.693  0.732  11.5  0.007   0.988  0.852  0.589   0.925
    car       22    1194     0.803  0.881  10.2  0.034   1.000  0.930  0.657   0.867
    drone     29    4923     0.796  0.820  11.4  0.009   0.979  0.939  0.757   0.967
    mean                     0.751  0.757

    18,243 windows   76 MB store   48.8 GB raw media   1:641

SCALE IS PART OF THE NUMBER, so both scales get quoted together:

    A/B scale (~1,100 windows/store, 16 q)   0.803 / 0.846   V_c2only
    full scale (~4,600 windows/store, 40 q)  0.751 / 0.757   V_FINAL_c2

That is not a regression. car is the control: its A/B build and its full
build are the SAME 22 media, and it reads 0.849/0.883 at 16 q against
0.803/0.881 at 40 q — precision unmoved, yield off 0.046 on a larger
query sample. So the gap is corpus size, not a code change: sim and
bridge grew ~6x in windows while support stayed at 11, and precision
falls as the distractor pool grows. Against the previous full-scale run
(V_FINAL.json, single resolution, old cut) the same two stores were
0.387 and 0.379 precision, so full scale nearly doubled.

HELD OUT AND AT FULL SCALE — the gap that used to be open here. Same
stores, same protocol; only the QUERY sampler moves past every media
architecture selection ever saw. Building a store is unsupervised and
label-free, so all media belong in the distractor pool; what has to be
held out is what SELECTION saw, and that is a property of the sampler.

    store    dev -> held out (yield)      dev -> held out (prec)
    sim      0.713 -> 0.701  -0.012       0.594 -> 0.557  -0.037
    bridge   0.693 -> 0.713  +0.020       0.732 -> 0.688  -0.044
    drone    0.796 -> 0.788  -0.008       0.820 -> 0.812  -0.008
    MEAN(3)  0.734 -> 0.734  +0.000       0.715 -> 0.686  -0.030

SEVEN SEQUENTIAL ARCHITECTURE CHOICES COST 0.000 YIELD AND 0.030
PRECISION. And it replicates: the A/B-scale held-out run found the same
shape (yield -0.004, precision -0.054), so the pattern is not an artifact
of one draw, and the precision cost is SMALLER at full scale, not larger.

Yield holding while precision slips is the expected signature: selection
tuned where the abstention cut lands, which trades returned-count against
correctness, and it barely touched whether the right rows rank at all.

    A/B scale, held out    0.784 / 0.779   (selection set 0.788 / 0.833)

car has no held-out set - all 22 drives were used in the A/Bs - so every
held-out mean here is three corpora, not four.

THE THREE NUMBERS TO QUOTE, and never one alone:

    capability   0.751 / 0.757   4 corpora, full scale, development
    generalises  0.734 / 0.686   3 corpora, full scale, held out
    cost of selection            yield 0.000, precision 0.030

## Per-transform, full scale — where the loss actually is

    warp  0.783   photo 0.892   codec 0.805   tempo 0.760
    crop  0.638   codec_hard 0.633

crop and codec_hard are tied at the bottom, and crop degrades FASTEST
with corpus size: 0.734 -> 0.638 (-0.096) between A/B and full scale,
against codec_hard's -0.048. So crop is still the right target, and the
reason it is the target got stronger, not weaker.

No weights were ever trained or fine-tuned anywhere in this pipeline.
Every model is frozen and pretrained. The only "tuning" was architecture
SELECTION on the benchmark, which is what the held-out run measures.

## Backup — 2026-08-08 06:45

    stores/_backup/c2_2026-08-08/vision     4 stores, verified byte-identical
                                 /vframes   732 frame-feature .npy (512 MB)
                                 /artifacts V_*.json, RESUME, DESIGN, logs

vframes is only a cache, but it is HOURS of GPU work and it lives in
/private/tmp, which macOS may purge. stores/ is gitignored, so the
backup sits under it and stays out of the repo.

## The A/B footgun, found before it fired

vstore.build() opens with shutil.rmtree(out), and every arm of an A/B
rebuilds the SAME four corpus names. The queued crop A/B was waiting on
ALL_DONE and would have deleted the four full stores minutes after the
artifact above was written. Any future A/B runs with SDX_STORES pointed
at a scratch root, and never against stores/vision.

    status:  /private/tmp/claude-501/status.sh
    tracker: /private/tmp/claude-501/STATUS.tsv   stage / status / eta / actual

## crop: ATTACKED AND FAILED — 2026-08-08, and the reason matters

R-MAC was the standard answer and it is now measured dead.

    probe, held-out clips, margin = self - mean_cross
      gem 0.494   rmac2 0.477   rmac3 0.478   <- uniform grid, WORSE
      true R-MAC (overlapping, L=1..3)  0.532  <- +0.038, 6/6 transforms
      (crop +0.043, warp +0.028, photo/codec < 0.007 - the signature of
       a genuinely SPATIAL fix, which is what made it convincing)

    store A/B, 16 q, identical media, gem -> rmac3
      yield  0.803 -> 0.764  -0.039     crop  0.734 -> 0.674  -0.060
      prec   0.846 -> 0.819  -0.027     warp  0.854 -> 0.818  -0.036

The proxy predicted +0.043 on crop and delivered -0.060. THE PROXY WAS
MEASURING THE WRONG COMPETITION, and this is the transferable part:

    pool    HARD neg   EASY neg
    gem     0.620      0.001
    rmac3   0.627      0.001

HARD = the best-matching other window in the SAME media, which is what
the ranker must actually beat. EASY = windows of other media, which is
what the probe's cross term sampled. Easy negatives sit at ~0.001 for
both poolings, so that term cannot separate two descriptors that both
separate media perfectly - the probe's margin was almost entirely its
self term, which R-MAC really does improve. Ranking is decided at ~0.62
among same-media windows, and the probe never sampled that regime.

RULE: a cheap screen must sample negatives at the operating point where
ranking is decided. A screen whose negatives sit near zero similarity
will rank descriptors by self-consistency, and self-consistency is not
retrieval. (Same shape as the 80-episode pool that lied by 10x.)

Not claimed: that a corrected screen would have caught this. That is one
data point, and one data point does not validate a screen.

## Original reasoning, kept for the record

`crop` is the weakest transform left (0.693-0.807). The descriptor
GeM-pools the WHOLE frame into one vector, so removing a third of the
field of view shifts every frame vector and rank pooling faithfully
preserves that shift. Nothing in it is spatially local.

`vcore._rmac` is implemented and verified (correct dim, no NaN, no silent
fallback): pool each grid region, L2-norm it, SUM the regions. Summed and
not concatenated on purpose - concatenation is worse than the status quo
under a crop, because content slides between cells and every component
changes.

## Rules that must not be re-broken

  - vision only. No text, labels, sensors, class lists, or any encoder
    whose space is shaped by a vocabulary. SigLIP2 measured BETTER on
    actions (0.691 vs 0.538) and is still disqualified.
  - single view. Each media alone; never compare two views of one moment.
  - one approach on all four corpora. No per-dataset anything.
  - metrics are yield and precision only, at k = ceil(1.5 x support) as a
    MAX bound with abstention.
  - A/B <= 10 min video per store; initial stores <= 1 h each.

## THREE CUT RULES REJECTED FOR CONTAMINATION — do not reintroduce

Each BEAT the shipped rule. Documented inside `vqbe.score_cut`.

    otsu over top k*4, k from SUPPORT   0.873/0.815  support IS the answer key
    otsu over top 5%                    0.836/0.901  5% won a sweep on eval
    recursive otsu depth 3              0.820/0.866  depth is that same constant

The contamination moved UP ONE LEVEL each time it was removed, and every
version looked clean in code and produced a believable number. Refusing
to fit costs ~0.12 precision on the diagnostic harness. Accepted.

## Open, not solved

  - write path is 17-21 min per hour of video vs a 1 min/h budget. The
    lever is DECODE, not the encoder; that is where the time goes.
  - query encoding is UNCACHED and now dominates grade time. The frame
    cache covers the write path only.
  - the frame cache keys on media DURATION, so a 40 s slice cannot serve
    a 60 s request. Cache full media and slice on read.
  - c3 was dropped on a parallax-free homography proxy. That shows it
    does not help against the proxy, NOT that it would not help against a
    real second camera. Open by instruction (single view).
  - the per-query channel weighting was the fault in two consecutive
    iterations. It rewards self-consistency across adjacent windows, not
    discrimination. Rebuild it before ever adding a channel back.
