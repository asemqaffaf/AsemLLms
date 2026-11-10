# AsemLLM in Common Lisp (`lisp/`)

SBCL code written without third-party libraries (no Quicklisp): a small transformer with hand-written backprop,
a trainer, the chat-data cleaner and a corpus pipeline.

    brew install sbcl            # tested with SBCL 2.6.10
    ./run-tests.sh               # 6 test files: cleaner, transformer, trainer, data pipeline, tuner, memory benchmark (~1 min)

| File | What it is | Erlang counterpart |
|---|---|---|
| `src/clean.lisp`, `bin/clean-chat.lisp` | Chat-data cleaner. **Reproduces `dataset/chat_clean.txt` byte for byte** (checked in the tests). | none |
| `src/tf.lisp` | Pre-RMSNorm transformer, multi-head causal attention, ReLU MLP; forward + backward over `double-float` arrays. Gradients checked against numerical differentiation for every tensor. | `tinyllm_tf.erl` |
| `src/train.lisp`, `bin/train.lisp`, `bin/generate.lisp` | Word tokenizer, thread-parallel (SBCL threads) trainer with Adam, clipping, validation, unigram/bigram baselines, sampling, text save/load. | `tinyllm_train.erl`, `tinyllm_tok.erl` |
| `src/sha256.lisp`, `json.lisp`, `bpe.lisp`, `data.lisp`, `bin/build-data.lisp` | JSONL → filters → exact + MinHash dedup → hash split → byte-level BPE → uint16 shards + `manifest.json` (SHA-256, dataset version, `verify`). | none |
| `src/tune.lisp`, `bin/tune.lisp`, `tune/kernel_bench.escript` | **Autotuner (code that writes code).** See below. | none |
| `src/membench.lisp`, `bin/membench-gen.lisp` | Long-conversation **memory benchmark generator**: seeded, deterministic cases with facts planted early, later updates, third-party distractors and never-mentioned (absent) questions, with exact ground truth. `sbcl --script bin/membench-gen.lisp --out cases.jsonl --cases 20 --lengths 50,200,800` | none |
| `bin/bench.lisp` | Throughput benchmark, same JSON as the Erlang one. | `bench.escript` |

## Use

    sbcl --script bin/clean-chat.lisp ../dataset/chat.txt /tmp/clean.txt
    sbcl --script bin/train.lisp ../dataset/chat_clean.txt 1500 /tmp/chat.model 10     # corpus steps out workers
    sbcl --script bin/generate.lisp /tmp/chat.model "User: Hello!
    Assistant:" 60 0.6

    # data: input is JSONL, one document per line with "text", "score" (edu score) and "language_score".
    # There is no downloader here (no Parquet/HTTP in plain Lisp); bring your own file.
    sbcl --script bin/build-data.lisp build docs.jsonl out-dir 8192 20000
    sbcl --script bin/build-data.lisp verify out-dir

## Self-tuning kernels (code as data)

The three matrix kernels in `src/tf.lisp` are not written by hand: `gen-mm-nn`, `gen-mm-nt` and `gen-mm-tn` build their source as a Lisp
list for a given unroll factor, and the `def-kernels` macro expands into the factors recorded in `src/tuned.sexp`.

    sbcl --script bin/tune.lisp             # Lisp kernels -> src/tuned.sexp
    sbcl --script bin/tune.lisp --erlang    # also tinyllm's dot/3 and axpy/3 (needs escript and rebar3)

The tuner `compile`s every variant at runtime, **rejects any whose result differs from a plain reference by even one bit**, times the rest on the
shapes the trainer uses, and keeps the fastest (a larger factor must win by 3% to beat a smaller one). On this M-series Mac it chose unroll 16
over the hand-picked 4: training went from ~3.1k to ~3.5k tokens/s on one thread (~12.6k to ~13.0k on 10), with identical results.
Unroll 12 turned out slower than 8 and 16, which is why it measures instead of guessing.

With `--erlang` it writes `tinyllm/apps/tinyllm/src/tinyllm_kernels.hrl` (the unrolled `dot/3` and `axpy/3` that `tinyllm_nn` includes), picking each
factor separately, and **keeps the file only if `rebar3 eunit` still passes**; otherwise it restores the previous one. The gain for Erlang training is small
(~6%: about 420 to 445 tokens/s). The tuner stops at 16: a 32-wide unroll crashes the BEAM JIT here (and a crashing candidate is rejected anyway).
`tuned.sexp` and the `.hrl` are machine-specific results; re-run the tuner on another machine.

## Optional C kernels and the shared weight file

`make -C ../native` builds `libtlm.so`; `tf.lisp` loads it at start-up through `sb-alien` (no Quicklisp, no CFFI) and sends products
with `m*k*n >= 4096` (env `TLM_MIN_WORK`) to C. Results are bit-identical to the Lisp kernels (tested), and without the library nothing changes.
`bin/bench-native.lisp` compares the two; on an x86 Linux box the C kernels took 0.33 to 0.46 of the Lisp time for every shape from 8x16x16 to 512x512x512.
`src/tlm.lisp` (`save-tlm`) writes a model as a TLM1 file (see `native/FORMAT.md`) that the C library and the Erlang NIF mmap.

## Accelerate (BLAS) and single precision

Off by default; turn on with `TLM_BLAS=1` (or call `(blas-init)`). Products with `m*k*n >= TLM_BLAS_MIN_WORK` (16384) go to
`cblas_dgemm` from Apple's Accelerate through `sb-alien`, and so does per-head causal attention (forward and backward, strided
`dgemm`, no transposed copies, zeros above the diagonal). The pure-Lisp kernels remain the reference and the fallback. BLAS adds in a
different order, so results are close (rtol 1e-10, atol 1e-12 in double), not bit-identical; `tests/test-blas.lisp` checks that.

`src/tf32.lisp` and `src/train32.lisp` give the same API in the packages `asem-tf32` / `asem-train32` with `single-float` tensors,
`sgemm`, and FP32 weights, gradients and Adam state. They are not copies: `src/derive.lisp` rewrites `tf.lisp` / `train.lisp` at load time
(double to single, `cblas_dgemm` to `cblas_sgemm`, package names), so FP32 follows the FP64 reference automatically. Host-side scalars (gradient
norm, Adam bias terms, baselines) are written as `long-float` / `0l0`, which stay double. `tests/test-fp32.lisp` compares FP32 with FP64
(loss within 1e-5 relative, gradients within 1e-4 relative L2, sgemm products rtol 1e-4 / atol 1e-5), runs a finite-difference check, NaN
propagation, convergence of both precisions to the same loss, and checkpoint round trips. FP64 and FP32 text checkpoints load into either.
`bin/bench.lisp` / `bin/bench32.lisp` measure full training steps (`TLM_BLAS=1`). Use `sbcl --dynamic-space-size 12GB` for large
configurations: the default 1 GiB heap is exhausted by 10 workers at dim 512.

Full training steps, tokens/s (this M1 Pro, 10 workers unless noted):

| Config | Pure Lisp | BLAS FP64 | BLAS FP32 |
|---|---|---|---|
| dim 64, 2 layers, 1 worker | 2,847 | 14,126 | 18,500 |
| dim 256, 4 layers, ctx 128 | 916 | 7,501 | 13,423 (219 GFLOP/s) |
| dim 512, 8 layers, ctx 256 (18.4M params) | n/a | 1,256 (148 GFLOP/s) | 3,144 (370 GFLOP/s) |

## Corpus, pretraining and evaluation

    # 1. FineWeb-Edu straight from the official Parquet shards, no Python (HTTP range requests, one 3 MB row group at a time)
    sbcl --script bin/fetch-fineweb.lisp ../data/fineweb/stage1.jsonl max-tokens=10000000 \
         shards=sample/10BT/000_00000.parquet,sample/10BT/000_00001.parquet      # score >= 4.0, language_score >= 0.95; resume=1 continues
    # 2. filter, dedup, document-level train/val/test split, 8k BPE trained on the training split only, uint16 shards + manifest
    sbcl --dynamic-space-size 8GB --script bin/build-data.lisp build ../data/fineweb/stage1.jsonl ../data/fineweb/stage1 8192 100000 4.0 5
    # 3. is the evaluation set clean?
    sbcl --dynamic-space-size 4GB --script bin/check-contamination.lisp eval/asem_eval.jsonl ../data/fineweb/stage1.jsonl
    # 4. train from scratch (FP32 by default; TLM_BLAS=1 for Accelerate), log to OUT/log.jsonl, checkpoints OUT/best.ckpt, OUT/last.ckpt
    TLM_BLAS=1 sbcl --dynamic-space-size 12GB --script bin/pretrain.lisp data=../data/fineweb/stage1 out=../data/runs/1m size=1m steps=3000

`src/parquet.lisp` is a small Parquet reader (Thrift compact footer, Snappy, PLAIN and dictionary encodings, RLE/bit-packed hybrid, data pages v1 and v2)
checked against a real FineWeb-Edu row: same id, scores and text as the dataset server returns. The `/filter` endpoint of the dataset server was tried first and
abandoned (it returns HTTP 500 authentication errors under load and only indexes part of each shard). About 0.8% of rows pass score >= 4 and language_score >= 0.95.

`eval/asem_eval.jsonl` is 90 four-option questions (30 simple facts, 30 reading comprehension with the passage in the prompt, 30 basic reasoning), correct answers
balanced over the four positions. Each option is scored by the mean log-probability of its tokens after `Question: ... Answer:`, so a model that cannot answer
still produces a continuous score (`answer_margin`: correct minus best wrong); chance is 25%. The log also records validation loss and perplexity, tokens/s, peak RSS and
the SHA-256 of the checkpoint (same seed and data give identical bytes with one worker; `tests/test-pretrain.lisp` checks it for both precisions).

### Is the model using the question? `bin/diagnose.lisp`

    sbcl --dynamic-space-size 4GB --script bin/diagnose.lisp data=../data/fineweb/stage1 ckpt=../data/runs/1m/best.ckpt text=1 > diagnosis.json

Deterministic (forward passes only), one JSON object. Accuracy is reported four ways (with the question, without it with the passage kept, with no input at all, and with
another item's prompt), plus a calibrated score, log P(option | prompt) minus log P(option | "Answer:"), which cancels length and frequency priors. `length_bias` and
`frequency_bias` show whether scores follow answer length or token frequency (summed log-probability on this set picks the unique shortest option ~86% of the time,
mean-per-token correlates +0.45 with length: neither raw rule is length-neutral). `question_sensitivity` scores each item's fixed options under mismatched questions:
`pmi_*` is how much the correct answer's probability changes, which can be positive from question type alone (names vs numbers); `margin_gain_*` is whether the correct answer
gains on the wrong ones when its own question is present, which is the evidence of conditioning on the answer-relevant content. `by_category` repeats it for facts,
reading comprehension and reasoning. `tests/test-pretrain.lisp` has a positive control (a model trained on learnable facts must be flagged as using the question, an untrained
one must not). 1M model on the 1.7M-token snapshot: accuracy 24% (chance), calibrated 22%, margin gain z +0.4 (summed) and +2.1 (mean): it reacts to question type, not content.

## What was measured

- Data pipeline: on 20,000 FineWeb-Edu documents it kept 7,479 (11,282 low edu score; 979 low language score; 257 too short;
  1 duplicate; 2 too long) in ~4 minutes. That run used the since-removed Python pipeline for the download.
- Training throughput for the same model shape as the Erlang benchmark: ~3.5k tokens/s on one thread, ~13k with 10
  threads, vs ~0.4k for Erlang (10 workers) on the same machine. PyTorch on the same CPU did ~35k (measured before it was removed).
- A 1,500-step run on `chat_clean.txt` (1,401-word vocabulary, 247k parameters) reached validation loss 3.70, against
  4.14 for a bigram model and 5.14 for unigram. Like the Erlang model it learns the `User:`/`Assistant:` format, not meaning.

## Limits

- No GPU, no BLAS: the same wall as Erlang, just higher. This cannot train a 3B model.
- The data pipeline's near-duplicate index is in memory.
- `tokenize` works on characters, the Erlang one on bytes; only non-ASCII text differs.
- Model files are a text format (readable, slow for large models).
