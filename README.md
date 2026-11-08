# AsemLLM

Language models built in Erlang, plus the tooling around them.

| Folder | What it is |
|---|---|
| [`tinyllm/`](tinyllm/) | **AsemLLM**: a small neural language model (embedding-MLP or transformer) written and trained in pure Erlang, with parallel gradient workers, a web UI, an OpenAI-compatible API, web search, and Hugging Face packaging. |
| [`agents/`](agents/) | Supervised Erlang agent workers that call a local LM Studio model (tool use, parallel jobs). Also generates the training dataset. |
| [`dataset/`](dataset/) | `chat.txt`: ~138 KB of synthetic `User:` / `Assistant:` dialogue written by a locally run LLM. |

Trained weights: https://huggingface.co/asemqaffaf/AsemLLM

It is a learning project: the models are tiny and their answers are mostly not meaningful. See `tinyllm/README.md`.
