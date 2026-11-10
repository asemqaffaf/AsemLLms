# agents

Supervised Erlang workers that run Claude agents. One agent = one process
(conversation state + tool-use loop); agents run in parallel and a failure in
one never reaches another.

## Run with the real Claude API

```sh
export ANTHROPIC_API_KEY=sk-ant-...      # in your own terminal; never commit it
rebar3 shell
```

```erlang
{ok, A} = agents:start_agent(#{
    system => <<"You are a concise assistant.">>,
    tools  => [agents_tools:calculate(), agents_tools:get_time()]}),
agents:ask(A, <<"What is (17 * 23) + 4, and what time is it in UTC?">>).

%% many agents in parallel
agents:run_all([{N, #{tools => [agents_tools:calculate()]}, P} || {N, P} <- [
    {a, <<"What is 12*12?">>}, {b, <<"What is 99/3?">>}]]).
```

Default model is `claude-opus-5-5` (override: `llm_opts => #{model => <<"...">>}`).
Requests send `fallbacks: "default"` (beta header `server-side-fallback-2026-07-01`);
turn it off with `llm_opts => #{fallbacks => off}`.

## Layout

| Module | Role |
|---|---|
| `agent` | gen_server: conversation history + tool-use loop |
| `agent_sup` | dynamic supervisor, one `temporary` child per agent |
| `agents_tools` | tool definitions; runs tool calls in parallel with timeouts, errors become `is_error` results |
| `agents_llm_claude` | Claude Messages API over `httpc` (TLS verified, retries 429/5xx) |
| `agents_llm_mock` | offline backend used by the tests only |
| `agents_calc` | safe arithmetic parser behind the `calculate` tool |

## Test

`rebar3 eunit` (offline; uses the mock backend and a local fake HTTP server).

## Memory benchmark baselines

`agents_membench` runs the benchmark written by `lisp/bin/membench-gen.lisp` against LM Studio and scores
each answer against the generator's ground truth. Setups: `none`, `truncated` (most recent `window` tokens),
`retrieval` (top `k` exchanges by BM25, or embeddings from an LM Studio embedding model).

```sh
sbcl --script ../lisp/bin/membench-gen.lisp --out cases.jsonl --cases 10 --lengths 50,200,800
rebar3 compile && escript scripts/membench.escript cases.jsonl window=2048 k=6 out=rows.jsonl
escript scripts/membench.escript cases.jsonl retriever=embedding embed_model=<embedding model id>
```

Outcomes per question: `correct`, `stale` (answered with the replaced value), `distractor` (a third party's value),
`unknown`, `hallucinated` (answered a never-mentioned fact), `wrong`. Hedging between two values counts as wrong.
Needs LM Studio with a chat model loaded; the tests (`rebar3 eunit`) use the mock backend.
