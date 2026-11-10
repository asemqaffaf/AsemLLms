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
