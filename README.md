# AsemLLM

Language models built in Erlang, plus the tooling around them.

| Folder | What it is |
|---|---|
| [`tinyllm/`](tinyllm/) | **AsemLLM**: a small neural language model (embedding-MLP or transformer) written and trained in pure Erlang, with parallel gradient workers, a web UI, an OpenAI-compatible API, web search, and Hugging Face packaging. |
| [`agents/`](agents/) | Supervised Erlang agent workers that call a local LM Studio model (tool use, parallel jobs). Also generates the training dataset. |
| [`dataset/`](dataset/) | `chat.txt`: ~138 KB of synthetic `User:` / `Assistant:` dialogue written by a locally run LLM, and `chat_clean.txt`, the same data after `lisp/bin/clean-chat.lisp` removed malformed, duplicate and context-free dialogues (743 → 543). |
| [`lisp/`](lisp/) | Common Lisp (SBCL): the chat cleaner, a hand-written transformer + thread-parallel trainer, and a FineWeb-style data pipeline (filters, dedup, BPE, shards, manifest). No third-party libraries. |

Trained weights: https://huggingface.co/asemqaffaf/AsemLLM

It is a learning project: the models are tiny and their answers are mostly not meaningful. See `tinyllm/README.md`.

**3B parameters is a goal, not a result.** Measured here: Erlang trains at ~0.4 GFLOP/s and Lisp at ~14 GFLOP/s on a 10-core laptop
(no GPU, BLAS or SIMD in either). A Chinchilla-sized 3B run needs ~1.4·10²¹ FLOPs, i.e. thousands of years at those speeds, so neither
implementation can get there. The earlier PyTorch/GPU pipeline (FineWeb-Edu streaming, Llama-style model, memory planner, benchmarks)
was removed from `main`; it is kept on the branch `archive/python-cobol-lisp`, together with the COBOL cleaner.
