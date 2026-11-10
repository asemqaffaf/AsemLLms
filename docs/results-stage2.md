# AsemLLM Stage 2: supervised fine-tuning for passage QA (Common Lisp, CPU only)

**Status: target not reached.** No fine-tuned checkpoint has beaten chance on the development set, so the locked
1,000-question test set has **not** been evaluated (it stays locked; see "Locked test set" below). The main result of
this stage is a diagnosis: the Stage 1 checkpoints cannot copy anything from their context, and none of the fine-tuning
variants tried overnight created that ability. Everything needed to keep going (benchmark, fine-tuning pipeline,
controls, diagnostics) is built, tested and committed on `stage2-qa`.

<!-- RESULTS-TABLE -->

## What was built

| Piece | Where | Tested by |
|---|---|---|
| QA benchmark generator (5 categories, held-out entities and wording, solver-checked keys) | `lisp/src/qagen.lisp`, `lisp/bin/qa-build.lisp` | `tests/test-qagen.lisp` |
| Locked test set (1,000) and dev set (500), SHA-256 lock, audit manifest | `lisp/eval/qa/` (see its README) | `tests/test-qagen.lisp` |
| Answer-only loss: output head and softmax computed only at answer rows | `forward-rows` / `backward-rows` in `lisp/src/tf.lisp` | `tests/test-tf.lisp` (finite differences, padding invariance, no gradient from masked positions) |
| SFT trainer: letter or text targets, padding buckets, data-parallel threads, gradient accumulation, AdamW, warmup + cosine, clipping, deterministic order, dev-selected best checkpoint, exact resume, shuffled-label control, optional auxiliary prompt loss and copy-task mixing | `lisp/src/sft.lisp` (FP32 derived: `sft32.lisp`), `lisp/bin/sft.lisp` | `tests/test-sft.lisp` (alignment, no truncation, fitting, bit-exact FP32 resume, FP32/FP64 agreement, dev selection, auxiliary loss, copy task) |
| Evaluation with controls and 95% Wilson intervals; refuses the locked test file unless explicitly asked and logs every locked run | `lisp/bin/qa-eval.lisp` | used on dev |
| Optional tied copy path in the output head (gated from 0, old checkpoints unchanged) | `head-fwd` / `head-bwd` in `lisp/src/tf.lisp` | `tests/test-tf.lisp` (no-op at gate 0; gradients for full and row heads) |
| Induction (in-context copying) probe | `lisp/bin/induction-probe.lisp` | |
| Dependency-free SVG charts | `lisp/bin/qa-chart.lisp` | |

The whole existing suite still passes (11 Stage 1 test files plus `test-qagen` and `test-sft`).

## Benchmark

See [`lisp/eval/qa/README.md`](../lisp/eval/qa/README.md). In short: 1,000 locked test and 500 dev items, 200/100 per
category (attribute extraction, reverse lookup, directed relations, numeric comparison, temporal order), all four options
mentioned in the passage equally often, answer positions exactly balanced, entities split 3/1/1 with no overlap, every
sentence and question frame owned by one split, no near-duplicates across splits. Every prompt fits the 128-token window
(maximum 124 positions with the longest option); nothing is ever truncated. Each answer letter is one token (` A` 316,
` B` 367, ` C` 338, ` D` 443) in its real context.

Shortcut baselines on the test set: one fixed letter 25.0%; first-mentioned option 25.3%; longest 26.7%; a lexical rule
(option in the passage sentence sharing most words with the question) 59.1%.

## Experiments

All runs start from the Stage 1 10.52M checkpoint (`data/runs/final/10m/best.ckpt`, SHA-256 `10e6e08e…`) unless noted, FP32 with
Accelerate, 8 worker threads, Adam(0.9, 0.95), clip 1.0, dev evaluation every 100 or 250 steps. "Probe" is accuracy on 500
training items (an overfitting indicator). Dev accuracy has a 95% half-width of about ±4 points at 500 items; chance is 25%.

| Run | Change from the previous | Steps run | Dev accuracy (best / last) | Train probe (last) | Outcome |
|---|---|---|---|---|---|
| E1 | letter target, 16k items with real-name pools, lr 3e-4, batch 32 | 900 | 26.0 / 24.6 | 25.2 | collapsed onto one letter at every eval |
| E2 | text target (copy the answer), option-likelihood scoring | 500 | 28.2 / 26.0 | 31.0 | learns which strings are answers in training; dev loss rises |
| E3 | training people/towns from ~4,000 generated names (same 16k count) | 700 | 28.4 / 25.4 | 35.0 | same pattern |
| E4 | + auxiliary next-token loss on the prompt (weight 1.0) | 600 | 27.8 / 26.0 | 35.6 | first trace of copying (probe below), dev flat |
| E5 | + 8,000 repeated-random-sequence copy examples | 1,000 | 27.6 / 24.8 | 36.4 | dev flat |
| E6 | letter target on generated-name data, batch 64 | 200 | 25.0 / 25.0 | 26.8 | collapsed onto one letter |
| E7 | letter, 32k items, aux 0.5, lr 5e-4, planned 8,000 steps | 1,000 | 25.0 / 24.6 | 27.4 | collapsed; stopped |
| E8 | text + tied copy path + aux 0.5, 32k items, planned 8,000 | 1,250 | 27.6 / 26.0 | 29.6 | tied gate stayed at 0.01; probe > dev (memorizing) |
| E9 | E8 on 256k never-repeated items | see below | | | |

The pretrained model itself scores 27.6% (text scoring) and 23.4% (letter scoring, mostly answering "B") on dev.

![Development accuracy of the early runs](results/stage2/dev-accuracy-early.svg)

### Why nothing transfers: the model cannot copy from its context

`bin/induction-probe.lisp` feeds 50 sequences of 40 random tokens repeated twice and compares next-token loss on the two copies.
A model with induction heads predicts the second copy almost for free.

| Checkpoint | Loss, first copy | Loss, second copy |
|---|---|---|
| Stage 1 10.52M | 13.09 | 13.06 |
| Stage 1 1.16M | 13.66 | 13.69 |
| E3 after 700 steps | 11.99 | 11.94 |
| E4 after 600 steps (aux loss) | 14.47 | 13.42 |

The Stage 1 models never learned to copy (30M training tokens is far below where induction heads usually appear). Every
formulation of this task needs some in-context copying or matching: the text target needs the answer's tokens copied, the
letter target needs the option text matched against the passage. Training-set accuracy rising while dev stays at chance is
what memorizing training strings without a general copy mechanism looks like.

Direct attempts to create copying (loss on the second copy of random sequences; chance = 8.95 nats):

| Diagnostic | Steps | Copy loss start → end |
|---|---|---|
| D1 10.52M, copy task only, lr 1e-3, 8–30 tokens | 600 | 11.89 → 8.61 |
| D2 1.16M from scratch, lr 3e-3 | 600 | 9.12 → 8.86 |
| D3 1.16M pretrained, lr 3e-3 | 600 | 11.00 → 8.71 |
| D4 1.16M pretrained, frequent tokens only (chance 7.09) | 600 | 9.64 → 6.83 |
| D5 10.52M, 40–63-token copies, batch 64, untied / tied head | ~440 | 11.87 → 8.70 / 8.74 |

![Copy-task loss in the diagnostics](results/stage2/copy-loss-diagnostics.svg)

A small model does learn to copy when the copied tokens come from a 60-token range (7.16 → 2.81 nats, below the 4.09 chance), with
or without Accelerate, so the trainer and the attention code are not at fault; the obstacle is copying arbitrary tokens from an
8,192-token vocabulary, which none of the runs achieved within ~600 steps. With untied input and output embeddings every token's
input-to-output association has to be learned separately, which is why a gated tied path was added; in the runs so far the model
did not open the gate (0.010 after 1,000 steps of E8).

<!-- E9 -->

## Locked test set

Not evaluated. `bin/qa-eval.lisp` refuses `lisp/eval/qa/test.jsonl` unless `locked-test=1` is passed, and records every such run
in `docs/results/stage2/locked-test-runs.jsonl`; that file does not exist, i.e. the test set has never been scored. Running it
on a checkpoint at chance on dev would only spend the test set.

## Corpus for additional pretraining (Phase 8)

<!-- CORPUS -->

## Training speed and memory

Answer-only SFT on the 10.52M model: about 5,500 prompt tokens/s (median over E1–E3), 4,700 with the auxiliary loss (it computes
the full output head), peak RSS 0.56–1.9 GB depending on the number of padding buckets in use. Dev evaluation takes about 5 s
(letter) or 20 s (text, four forward passes per item).

## Recommendation

<!-- RECOMMENDATION -->

## Reproduce

    sbcl --dynamic-space-size 4GB --script lisp/bin/qa-build.lisp data=data/fineweb/stage1 out=data/qa/v1
    sbcl --dynamic-space-size 8GB --script lisp/bin/qa-build.lisp data=data/fineweb/stage1 out=data/qa/v3-synth-256k train-entities=synthetic train=256000
    TLM_BLAS=1 sbcl --dynamic-space-size 10GB --script lisp/bin/sft.lisp data=data/fineweb/stage1 qa=data/qa/v3-synth-256k \
         out=data/runs/sft/e9 mode=text tie=1 aux-lm=0.5 steps=8000 batch=32 workers=8 lr=5e-4 warmup=200 eval-every=250
    TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script lisp/bin/qa-eval.lisp data=data/fineweb/stage1 ckpt=data/runs/sft/e9/best.ckpt \
         items=lisp/eval/qa/dev.jsonl mode=text out=dev-eval.json
    sbcl --dynamic-space-size 4GB --script lisp/bin/induction-probe.lisp ckpt=data/runs/final/10m/best.ckpt

Exact arguments of every run are in the `argv` field of each run's `log.jsonl` start event (copied to `docs/results/stage2/runs/`).
