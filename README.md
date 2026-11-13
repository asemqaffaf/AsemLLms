# AsemLLM

Small language models written from scratch in Erlang and Common Lisp (SBCL), with no deep-learning framework, plus the tooling and experiments around them.

It is a learning project: the models are tiny and their answers are mostly not meaningful. Trained weights of the Erlang models: https://huggingface.co/asemqaffaf/AsemLLM

| Folder | What it is |
|---|---|
| [`tinyllm/`](tinyllm/) | **AsemLLM**: a small neural language model (embedding-MLP or transformer) written and trained in pure Erlang, with parallel gradient workers, a web UI, an OpenAI-compatible API, web search, and Hugging Face packaging. |
| [`agents/`](agents/) | Supervised Erlang agent workers that call a local LM Studio model (tool use, parallel jobs). Also generates the training dataset. |
| [`dataset/`](dataset/) | `chat.txt`: ~138 KB of synthetic `User:` / `Assistant:` dialogue written by a locally run LLM, and `chat_clean.txt`, the same data after `lisp/bin/clean-chat.lisp` removed malformed, duplicate and context-free dialogues (743 → 543). |
| [`lisp/`](lisp/) | Common Lisp (SBCL): the chat cleaner, a hand-written transformer + thread-parallel trainer, a FineWeb-style data pipeline (filters, dedup, BPE, shards, manifest), and the Stage 1 and Stage 2 experiment code, including the QA benchmark in `lisp/eval/qa/`. No third-party libraries. |
| [`native/`](native/) | C library under the Erlang and Lisp code, for inference only: the three matrix products and a KV-cache transformer step, as a Lisp `sb-alien` library and an Erlang NIF. Optional BLAS. Without BLAS it reproduces the Erlang and Lisp numbers exactly. |
| [`docs/`](docs/) | Results and design notes: [Stage 1](docs/results-stage1.md), [Stage 2](docs/results-stage2.md), the [R7 analysis](docs/results-stage2-r7.md) and the [Layer 2 design note](docs/layer2-design.md). |

## Results so far

**Stage 1** ([details](docs/results-stage1.md)): three transformers of 1.16M, 2.90M and 10.52M parameters, trained on the same FineWeb-Edu corpus (29.9M tokens, FP32, one M1 Pro CPU). Test perplexity falls with size: 112.4, 87.8 and 80.8. On a 90-question four-option set none of them answers better than chance (25%): 28.9%, 28.9% and 25.6%.

**Stage 2** ([details](docs/results-stage2.md), [R7 analysis](docs/results-stage2-r7.md)): supervised fine-tuning of the 10.52M model for passage question answering. The best model (R7) scores 45.8% (229/500) on the development set with four options (chance is 25%). A simple word-overlap rule scores 62.0% on the same set, and the 90% target was not reached. Four controlled follow-up pilots did not improve on R7, and the line was stopped on 2026-11-13 ([final report](docs/results/stage2/mission-20261111/FINAL_REPORT.md)). The 1,000-item test set has not been used to score any model, but its items are in `lisp/eval/qa/`, so they are public.

**3B parameters is a goal, not a result.** Measured here without BLAS: Erlang trains at ~0.4 GFLOP/s and Lisp at ~14 GFLOP/s on a 10-core laptop (no GPU or SIMD in either). A Chinchilla-sized 3B run needs ~1.4·10²¹ FLOPs, i.e. thousands of years at those speeds, so neither implementation can get there. The Lisp trainer can call Apple's Accelerate BLAS (`TLM_BLAS=1`), which the Stage 1 runs and the Stage 2 training engine use; the Erlang code uses no BLAS.

## Quickstart

Erlang (OTP 27 or newer and rebar3; CI uses OTP 28):

    cd tinyllm
    rebar3 compile && rebar3 eunit
    escript scripts/train.escript 3000 chat_word.bin ../dataset/chat_clean.txt word   # first run only: no model is committed
    scripts/run.sh        # web UI at http://localhost:8080

Common Lisp (SBCL):

    sh lisp/run-tests.sh
    cd lisp && sbcl --script bin/train.lisp ../dataset/chat_clean.txt 1500 /tmp/chat.model 10

Native C library (optional; the native tests skip themselves without it):

    make -C native            # build/libtlm.so for Lisp
    make -C native install    # also builds the Erlang NIF and copies it into tinyllm/
    sh native/interop_test.sh # Lisp writes a model, the Erlang NIF must reproduce its logits exactly

More detail: [`tinyllm/README.md`](tinyllm/README.md), [`lisp/README.md`](lisp/README.md), [`native/README.md`](native/README.md).

License: Apache-2.0, see [`LICENSE`](LICENSE).
