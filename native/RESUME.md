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

    held out   0.784 yield / 0.779 prec   3 corpora, media no A/B saw
    selection  0.788 / 0.833              same 3 corpora

QUOTE THE HELD-OUT NUMBER for generalisation and the FULL-SCALE number
for capability. car has no held-out set (all 22 drives used in the
A/Bs), so the held-out mean is three corpora, not four. The held-out run
is at A/B scale, so it does not yet answer "does it hold out AND scale".

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

## Next question: crop

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
