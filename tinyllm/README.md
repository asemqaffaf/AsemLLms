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
| `tinyllm_tuner` | propose, score, gate, commit or roll back: supervises any training engine (see "Tuner") |
| `tinyllm_engine`, `_fake`, `_port` | the engine behaviour; a deterministic test engine; a driver for an external worker (MLX, llama.cpp) |
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
| `chat_tf.bin`, context 32 | stateless | 2765 ms | 1.0x | 105 260 469 270 465 466 308 423 |
| | session, exact | 1643 ms | 1.68x | 33 31 301 192 299 302 208 276 |
| | session, evict=8 | 413 ms | 6.7x | 33 30 36 59 58 65 60 73 |
| 64-dim 2-layer, context 128 | stateless | 10147 ms | 1.0x | 168 453 889 1165 1520 1881 2041 2031 |
| | session, exact | 2984 ms | 3.4x | 41 40 43 45 46 47 1365 1357 |
| | session, evict=8 | 823 ms | 12.3x | 41 40 43 44 45 46 337 228 |

Reading it: while the window is not full the exact session costs a few tens of ms per turn whatever the history length
(3x faster at turn 1, ~40x by turn 6 at context 128). Once the window is full (turn 7 at context 128; from turn 3 at
context 32, which the shipped model hits almost at once) every token needs the whole window recomputed, because of the
absolute positions. Exact mode is still ~1.5x faster than the stateless path there: prefill tokens skip the vocabulary
projection and, in the last layer, everything except their keys and values (`tinyllm_tf:advance/4`), which nothing later
can see. Beyond that, speed needs `evict` (or a longer context). The first version of the cache was 2x *slower* when full
because every prefill token also computed the vocabulary projection.

## Tuner (supervised fine-tuning of any engine)

`tinyllm_tuner` decides what to learn and whether to keep it; a `tinyllm_engine` does the numerical work.
One `learn/2` round trains one candidate per learning rate, one at a time by default (`parallel => true` runs them
concurrently and needs memory for all of them), every candidate starting from the same current version, then scores each on a held-out set and
a probe set (a reference set that must not get worse), and measures its drift from version 0. A candidate is
kept only if heldout loss improves by more than `min_gain`, probe loss stays within `probe_tolerance`, and drift
is within `drift_cap`; the best passing one becomes the new current version, otherwise nothing changes and the
reason comes back. With `control => shuffle` (or a `fun(Batch) -> Batch`) a passing candidate must also beat, by
`control_margin`, a control trained at the same learning rate on a word-shuffled copy of the batch: part of any heldout
gain is generic (the model adapting to short raw text), and without the control garbage passes. The last `retain` (5) versions plus version 0 are kept, and `rollback/1,2` steps back.
Candidates run in their own processes with a per-candidate `timeout`, so an engine crash or hang costs one
candidate, not the tuner. Only one round runs at a time; `current/1`, `versions/1` and `log/1` answer meanwhile.

    {ok, T} = tinyllm_tuner:start_link(#{
        engine => {tinyllm_engine_port, #{cmd => "python3", args => ["scripts/mlx_worker.py"],
                                          root => "/tmp/tuner", base_model => "Qwen/Qwen2.5-0.5B-Instruct",
                                          heldout_file => "heldout.txt", probe_file => "probe.txt"}},
        candidates => [1.0e-5, 5.0e-5, 1.0e-4], steps => 50, rank => 8, probe_tolerance => 0.02, drift_cap => 5.0}),
    {committed, 1, _Info} = tinyllm_tuner:learn(T, [<<"a training sample">>, <<"another">>]),
    {ok, 0} = tinyllm_tuner:rollback(T).

`tinyllm_engine_fake` (a 1-D quadratic) and a shell stand-in worker behind `tinyllm_engine_port` are what
`tinyllm_tuner_tests` run on, so the commit, reject, drift, retention, rollback, timeout and crash logic is
tested without a GPU or a model. The worker protocol (command line in, one `RESULT k=v` line out) is documented
at the top of `src/tinyllm_engine_port.erl`.

`scripts/mlx_worker.py` is the MLX LoRA worker for a Mac. It was checked on Linux CPU (mlx-lm 0.32.2, mlx 0.32.4) and on
an M1 Pro (mlx-lm 0.31.3), both with a tiny random model, and then on a real model, Qwen3.5-4B (MLX 4-bit): 8 LoRA layers,
rank 4, 20 steps, learning rates 1e-5 and 5e-5. There heldout/probe loss went 5.682/5.388 to 2.307/4.092 (v1) and
1.161/3.972 (v2), a further round was rejected as `no_improvement`, and rollback to the base reproduced the baseline
scores. That run committed a "garbage" round, because its loss fell too (word salad lowered heldout by 0.24 vs 1.16 for real
data), and its 2-line heldout/probe sets were near-copies of the training text. Fixed with disjoint, larger sets
(`scripts/make_eval_sets.py`: 40 train / 60 heldout dialogues sharing no user question, a shuffled-vocabulary garbage
set, and `eval/probe_general.txt`, 51 varied general lines) and the control above. `scripts/tuner_gate.escript` checks
all of it on a real model (all PASS on Qwen3.5-4B; baseline heldout/probe 2.971/2.409, real data 1.808/1.789, garbage
rejected from the base and after training, rollback to base exact). The probe still falls with training (it measures
the generic shift too), so it only catches regressions. On a 16 GB Mac use `TLM_MLX_BATCH=1 TLM_MLX_GRAD_CKPT=1
TLM_MLX_MAX_SEQ=256`, or training runs out of Metal memory. Not covered: rollback to a middle version on a real model,
and whether a gain on this chat data means better answers. The tiny-model check found and fixed four bugs (no parseable "Train loss" line, `--adapter-path ""` needed to score the base, the default
`--num-layers 16` failing on small models, a deprecated entry point). To repeat it with no download:

    pip install mlx-lm                    # on Linux also: pip install "mlx[cpu]"
    rebar3 compile
    python3 scripts/tiny_mlx_model.py /tmp/tiny
    TLM_MLX_NUM_LAYERS=2 escript scripts/tuner_smoke.escript $(which python3) scripts/mlx_worker.py /tmp/tuner_root /tmp/tiny

On a real model (set `TLM_MLX_NUM_LAYERS=8` and the memory variables above):

    python3 scripts/make_eval_sets.py /tmp/sets
    escript scripts/tuner_gate.escript $(which python3) scripts/mlx_worker.py /tmp/gate_root /tmp/sets /path/to/mlx/model

mlx-lm prints losses with 3 decimals, so use `min_gain` of at least 0.002. Set `TLM_MLX_NUM_LAYERS` for models
with fewer than 16 layers. Candidates run one at a time by default, so one MLX job holds the GPU at once.

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
