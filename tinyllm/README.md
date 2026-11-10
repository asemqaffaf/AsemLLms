# AsemLLM

A tiny neural language model (character- or word-level), written and trained in pure Erlang
(no outside APIs). Supervised worker processes compute gradients in parallel.

## Try it in the browser

    cd tinyllm
    rebar3 compile && escript scripts/train.escript 3000 chat_word.bin ../dataset/chat_clean.txt word   # first run only: no model is committed
    scripts/run.sh        # starts the app + web UI (fixed node name tinyllm@127.0.0.1)
    # open http://localhost:8080

The page is rendered by Erlang (plain HTML form, no JavaScript). It listens on
127.0.0.1 only (requests with a non-local `Host` or `Origin` get 403) and loads the `model_file` from
`config/sys.config` (`chat_word.bin`; trained weights are not committed, so train one as above).

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
| `tinyllm_session`, `tinyllm_session_sup` | one supervised process per conversation holding the transformer's KV-cache |
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
- The OpenAI-compatible `/v1/chat/completions` endpoint does not save anything; it is stateless unless the request carries a `session_id` (see below).

## Stateful sessions (KV-cache)

`tinyllm:complete/4` rebuilds the whole forward pass from the prompt for every generated token. A *session*
(`tinyllm_session`, one process per conversation under `tinyllm_session_sup`) keeps the keys and values of every
position it has seen (`tinyllm_tf:step/4`), so a new token costs one row through the network instead of a pass over the
whole context. The logits are bit-for-bit those of the stateless path and sampling is shared, so the same seed gives
the same text (eunit: `tinyllm_session_tests`).

    {ok, Pid} = tinyllm_session:start(Model),
    {ok, _}   = tinyllm_session:feed(Pid, <<"User: hi\nAssistant:">>),     %% only these tokens are computed
    {Text, _} = tinyllm_session:complete(Pid, 20, #{temperature => 0.7, stop => <<"\n">>}),
    {Reply, _} = tinyllm_session:chat(Pid, Messages, #{}),                 %% like tinyllm:chat/3; reuses the cache

`set_context/2` replaces the context and reuses the cache for the longest common prefix; `info/1` reports `computed`
(single-token passes) and `reused` (positions served from the cache); `get_or_start/3` finds or starts the session for a
conversation id. Sessions are `temporary` (a crash loses only a rebuildable cache and the next turn starts a fresh one),
stop after 5 idle minutes (`session_idle_ms`), and are capped at `max_sessions` (64). Both archs work; the MLP has no
cache to keep, so it only gets the conversation bookkeeping.

- Web chat: turns of a saved conversation run in that conversation's session (deleting the conversation ends it). If
  sessions are off (`{sessions, false}`), memory is off, or the cap is hit, it falls back to the stateless call.
- OpenAI endpoint: add `"session_id": "abc-1"` to a `/v1/chat/completions` request; requests repeating it share one cache and
  the response carries `tinyllm_session: {computed, reused}`. The reply is the same as without it.

**Known limits.** It still forgets everything older than the context window (32 tokens for the shipped transformer).
And the transformer uses *absolute* learned position embeddings: once the window is full, every new token shifts all
positions, so the cache cannot be reused and the window is recomputed (the work the stateless path does anyway). The cache
therefore pays off while the window is filling and across turns that extend it. `evict => K` (config `session_evict`,
default 1 = exact) drops the K oldest tokens at once when full, so the rebuild happens every K tokens; K > 1 is faster but
does **not** reproduce the stateless text. Carrying state beyond the window (segment recurrence / a recurrent state) is
a model change and not part of this.

Measured (`escript scripts/session_bench.escript <model.bin | train:CONTEXT> 12 3`: 8 chat turns, 12 new tokens per reply,
same prompts and seed, best of 3, 10-core laptop; ms per turn):

| model | variant | total | speedup | ms per turn |
|---|---|---|---|---|
| `chat_tf.bin`, context 32 | stateless | 2588 ms | 1.0x | 106 251 446 248 433 435 283 386 |
| | session, exact | 2464 ms | 1.05x | 34 34 464 288 456 453 315 420 |
| | session, evict=8 | 539 ms | 4.8x | 36 33 45 80 78 84 83 100 |
| 64-dim 2-layer, context 128 | stateless | 10134 ms | 1.0x | 172 458 794 1165 1546 1900 2050 2049 |
| | session, exact | 4888 ms | 2.07x | 44 45 51 52 54 54 2318 2270 |
| | session, evict=8 | 1234 ms | 8.2x | 44 46 52 57 56 55 553 371 |

Reading it: while the window is not full the exact session costs a few tens of ms per turn whatever the history length
(3x faster at turn 1, 20-40x by turn 6 at context 128). Once the window is full (turn 7 at context 128; from turn 3 at
context 32, which the shipped model hits almost at once) exact mode is at parity with the stateless path, so with the
32-token model the exact speedup is about 1x overall and the real win needs `evict` (or a longer context). The first
version of the cache was 2x *slower* when full because every prefill token also computed the vocabulary projection;
`tinyllm_tf:advance/4` skips it for all but the last token.

## Benchmark and limits

    rebar3 compile && escript scripts/bench.escript 64 2 4 32 8 30    # dim layers heads ctx batch steps

Prints tokens/s and achieved FLOP/s as JSON. On a 10-core laptop the transformer trains at a few hundred tokens/s
(~0.4 GFLOP/s); the Lisp port in `../lisp` runs the same shape ~8× faster on one thread (~33× with 10 threads). Both stay small-scale:
no GPU, no BLAS.
The cleaned chat set for training is `../dataset/chat_clean.txt` (`sbcl --script ../lisp/bin/clean-chat.lisp chat.txt chat_clean.txt`).

## Publishing to Hugging Face

    escript scripts/export.escript chat_word.bin export/        # per model
    escript scripts/export.escript chat_tf.bin export_tf/
    python3 scripts/pack_release.py release/ <user>/AsemLLM export/ export_tf/   # checks the Python loader against Erlang

Published: https://huggingface.co/asemqaffaf/AsemLLM (MLP at the root, transformer in `transformer/`).
