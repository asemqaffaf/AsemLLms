# AsemLLM

A tiny neural language model (character- or word-level), written and trained in pure Erlang
(no outside APIs). Supervised worker processes compute gradients in parallel.

## Try it in the browser

    cd tinyllm
    scripts/run.sh        # starts the app + web UI (fixed node name tinyllm@127.0.0.1)
    # open http://localhost:8080

The page is rendered by Erlang (plain HTML form, no JavaScript). It listens on
127.0.0.1 only and loads `model.bin` (train one with the script below).

## Train

    rebar3 compile && escript scripts/train.escript 3000 model.bin   # ~4 min on 10 cores

or from the shell: `{ok, M} = tinyllm:train(Binary, #{steps => 2000}), tinyllm:save(M, "mine.bin").`
Point the UI at it with `{model_file, "mine.bin"}` in `config/sys.config`.

## Use from Erlang

    M = tinyllm:load("model.bin"),
    tinyllm:generate(M, <<"-module(">>, 300, #{temperature => 0.7, seed => 1}).

## JSON API

    curl -s localhost:8080/api/generate -d '{"prompt":"-module(","length":200,"temperature":0.7}'
    curl -s localhost:8080/api/info

## Layout

| Module | Role |
|---|---|
| `tinyllm_nn` | the model: embeddings -> tanh hidden -> softmax; hand-written backprop |
| `tinyllm_train` | data-parallel training: shard batch over workers, sum gradients, Adam |
| `tinyllm_worker`, `tinyllm_worker_sup` | gradient workers; crashed workers restart and their shard is retried |
| `tinyllm_web` | Cowboy handler: server-rendered page + JSON API |
| `tinyllm` | public API: train, generate, save, load |

`rebar3 eunit` runs the tests (gradient check, learning, worker-crash recovery, web UI).

## Web search (optional, no API key)

Tick "Also search" on the chat page (or `GET /api/search?q=...`). Providers, set in `config/sys.config`:

- `{search_provider, wikipedia}` (default): Wikipedia's search API, Wikipedia only.
- `{search_provider, searxng}`: your own [SearXNG](https://docs.searxng.org) instance, which queries Google and
  other engines for you. Run it with the bundled settings (JSON on, localhost only):
  `docker run -d --name tinyllm-searxng -p 127.0.0.1:8888:8080 -v "$PWD/searxng/settings.yml:/etc/searxng/settings.yml:ro" searxng/searxng`
  (`searxng/settings.yml` needs a `server.secret_key`; `openssl rand -hex 32`), and set `{searxng_url, "http://127.0.0.1:8888"}`.

Every search result also gets an "open in Google" link that opens the query in your own browser.
Google does not allow automated scraping, and its official API needs a key, so tinyllm does neither.
tinyllm itself cannot read the results; they are shown beside its reply.

## Let Qwen answer from the results (local, optional)

On the chat page, tick "Let Qwen (LM Studio, local) answer from the search results". The page searches, then asks
your LM Studio model to answer using only those numbered results (citing `[1]`, `[2]`). Set the model in
`config/sys.config` (`lmstudio_url`, `lmstudio_model`; it must be a loaded chat model). If LM Studio is down, the
page says so and tinyllm answers instead. Search results are treated as untrusted text and the answer is only displayed.

## Transformer

`arch => transformer` trains a small GPT-style decoder (learned positions, pre-LayerNorm, causal multi-head
attention, ReLU MLP, residuals) in `tinyllm_tf.erl`, with hand-written backprop that the test suite checks against
numerical gradients for every tensor. Same parallel workers, same `tinyllm:train/2`:

    tinyllm:train(Text, #{tokenizer => word, arch => transformer, context => 32,
                          dim => 64, heads => 4, layers => 2, ff => 128, batch => 8, steps => 2500, lr => 0.003}).

On a task that needs a 7-token lookback, the transformer scores 0.25 against 0.41 for the 4-token-window MLP
(`tf_uses_long_range_context` in the tests). On the 138 KB chat dataset both models mostly memorize.

## Conversation memory (Mnesia)

Chats are saved in Mnesia, Erlang's built-in transactional database (`tinyllm_memory.erl`, table `tinyllm_conv`),
and survive restarts. Each question + answer is written in one transaction and synced to the log before the page returns.

- Chat page: `/chat` starts a conversation, `/chat?c=<id>` reopens one; recent conversations are listed at the top, with a delete button.
- JSON: `GET /api/conversations`, `GET|DELETE /api/conversations/<id>`.
- Erlang: `tinyllm_memory:append/2`, `get/1`, `list/0`, `transcript/1`, `delete/1`.
- Config (`config/sys.config`): `memory` (true/false) and `memory_dir` (default `data/mnesia`, git-ignored). If Mnesia cannot start,
  the chat still works from the transcript in the page, without saving.
- The disc schema belongs to the node name, so launch with `scripts/run.sh` (node `tinyllm@127.0.0.1`, same as `config/vm.args`).
  A plain `rebar3 shell` is `nonode@nohost` and would not see the saved chats.
- The OpenAI-compatible `/v1/chat/completions` endpoint is stateless and does not save anything.

## Publishing to Hugging Face

    escript scripts/export.escript chat_word.bin export/        # per model
    escript scripts/export.escript chat_tf.bin export_tf/
    python3 scripts/pack_release.py release/ <user>/AsemLLM export/ export_tf/   # checks the Python loader against Erlang

Published: https://huggingface.co/asemqaffaf/AsemLLM (MLP at the root, transformer in `transformer/`).
