%% A stateful chat session: one supervised process per conversation that keeps the model's KV-cache.
%%
%% tinyllm:complete/4 rebuilds the whole forward pass from the prompt for every generated token. A
%% session keeps the keys/values of every position it has already seen (tinyllm_tf:step/4), so each new
%% token costs one row through the network instead of a pass over the whole context. Logits and sampled
%% tokens are exactly those of the stateless path (same operations in the same order, same sampler).
%%
%% API (all calls run inside the session process, so concurrent callers are serialized):
%%   {ok, Pid} = tinyllm_session:start(Model, #{id => ConvId}),
%%   tinyllm_session:feed(Pid, <<"User: hi\nAssistant:">>),        %% only these new tokens are computed
%%   {Text, Why} = tinyllm_session:complete(Pid, 20, #{temperature => 0.7, stop => <<"\n">>}),
%%   {Reply, Why} = tinyllm_session:chat(Pid, Messages, Opts).     %% like tinyllm:chat/3, reusing the cache
%%
%% Limits (state them wherever this is used):
%%  * It forgets everything older than the model's context window (32 tokens for the shipped transformer).
%%  * The transformer has learned ABSOLUTE position embeddings, so once the window is full a new token
%%    shifts every position and the cache cannot be reused: the window is recomputed (the same work the
%%    stateless path does). The cache pays off while the context is still filling and across turns that
%%    extend it. `evict => K` (default 1 = exact) trades exactness for speed: a full window drops its
%%    oldest K tokens at once, so a rebuild happens every K tokens instead of every token. K > 1 does NOT
%%    reproduce the stateless output.
%%
%% Model: a loaded model map, or {pt, Key} naming a prepared model published with publish/2 (shared, not
%% copied into each session).
-module(tinyllm_session).
-behaviour(gen_server).
-export([prepare/1, publish/2, unpublish/1, start/1, start/2, start_link/2, get_or_start/3, whereis/1,
         feed/2, set_context/2, complete/3, chat/3, logits/1, info/1, reset/1, stop/1, close/1, count/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(CALL_TIMEOUT, 120000).
-define(DEFAULT_IDLE_MS, 300000).
-define(DEFAULT_MAX_SESSIONS, 64).

-record(s, {m, win = [], cache, logits, evict = 1, id, idle, computed = 0, reused = 0}).

%% ------------------------------------------------------------------- model

%% Everything a session needs from a model, computed once.
prepare(Model0) ->
    #{cfg := #{context := N} = Cfg, vocab := Vocab, params := Params, tokenizer := Tok} = tinyllm:normalize(Model0),
    VMap = maps:from_list(lists:zip(Vocab, lists:seq(0, length(Vocab) - 1))),
    Arch = maps:get(arch, Cfg),
    #{cfg => Cfg, arch => Arch, n => N, tok => Tok, vmap => VMap, inv => list_to_tuple(Vocab),
      params => case Arch of mlp -> tinyllm_nn:prepare(Params); transformer -> tinyllm_tf:prepare(Params) end,
      %% unknown prompt tokens: <unk> for word models, newline for char models (as tinyllm:complete/4)
      pad => case Tok of word -> 0; char -> maps:get(<<"\n">>, VMap, 0) end,
      padctx => maps:get(<<"\n">>, VMap, 0)}.

%% Publish a prepared model under Key so sessions can share it: start(Model-as-{pt, Key}).
publish(Key, Model) -> persistent_term:put({?MODULE, Key}, prepare(Model)).
unpublish(Key) -> _ = persistent_term:erase({?MODULE, Key}), ok.

resolve({pt, Key}) -> persistent_term:get({?MODULE, Key});
resolve(Model) when is_map(Model) -> prepare(Model).

%% --------------------------------------------------------------- lifecycle

start(Model) -> start(Model, #{}).

%% Opts: id (binary; registers the session globally so get_or_start/3 finds it), evict (default 1),
%%       idle_ms (default 5 min: the session stops after that long without a call).
%% -> {ok, Pid} | {error, {already_started, Pid}} | {error, too_many_sessions}
start(Model, Opts) ->
    Max = application:get_env(tinyllm, max_sessions, ?DEFAULT_MAX_SESSIONS),
    case proplists:get_value(active, supervisor:count_children(tinyllm_session_sup)) >= Max of
        true -> {error, too_many_sessions};
        false -> supervisor:start_child(tinyllm_session_sup, [Model, Opts])
    end.

start_link(Model, Opts) ->
    case Opts of
        #{id := Id} -> gen_server:start_link({global, {?MODULE, Id}}, ?MODULE, {Model, Opts}, []);
        _ -> gen_server:start_link(?MODULE, {Model, Opts}, [])
    end.

get_or_start(Id, Model, Opts) ->
    case ?MODULE:whereis(Id) of
        Pid when is_pid(Pid) -> {ok, Pid};
        undefined ->
            case start(Model, Opts#{id => Id}) of
                {error, {already_started, Pid}} -> {ok, Pid};
                Other -> Other
            end
    end.

whereis(Id) -> global:whereis_name({?MODULE, Id}).

count() -> proplists:get_value(active, supervisor:count_children(tinyllm_session_sup)).

stop(Pid) -> gen_server:stop(Pid).

%% Stop the session registered under Id, if there is one.
close(Id) ->
    case ?MODULE:whereis(Id) of
        Pid when is_pid(Pid) -> catch stop(Pid), ok;
        undefined -> ok
    end.

%% ---------------------------------------------------------------------- API

%% Append text (or token ids) to the context; computes only the new tokens. -> {ok, Info}
feed(Pid, TextOrIds) -> call(Pid, {feed, TextOrIds}).

%% Replace the context by this text (or token ids), reusing the cache for the longest common prefix.
set_context(Pid, TextOrIds) -> call(Pid, {set_context, TextOrIds}).

%% Sample up to Len tokens after the current context; they join the context. -> {Text, stop | length}
%% Opts: temperature (0.8), seed, stop (binary | [binary]), as tinyllm:complete/4.
complete(Pid, Len, Opts) -> call(Pid, {complete, Len, Opts}).

%% Like tinyllm:chat/3 (same prompt, same stops, same trimming) but reusing what the session already holds.
chat(Pid, Messages, Opts) -> call(Pid, {chat, Messages, Opts}).

%% Logits for the token after the current context.
logits(Pid) -> call(Pid, logits).

%% #{window, cached, computed, reused, ...}: computed counts single-token forward passes done so far,
%% reused counts positions served from the cache instead.
info(Pid) -> call(Pid, info).

reset(Pid) -> call(Pid, reset).

call(Pid, Req) -> gen_server:call(Pid, Req, ?CALL_TIMEOUT).

%% ------------------------------------------------------------- gen_server

init({Model, Opts}) ->
    M = resolve(Model),
    Evict = max(1, min(maps:get(evict, Opts, 1), maps:get(n, M))),
    Idle = maps:get(idle_ms, Opts, application:get_env(tinyllm, session_idle_ms, ?DEFAULT_IDLE_MS)),
    {ok, fresh(#s{m = M, evict = Evict, id = maps:get(id, Opts, undefined), idle = Idle}), Idle}.

handle_call(Req, _From, #s{idle = Idle} = S) ->
    {Reply, S1} = try req(Req, S)
                  catch throw:{bad, Why} -> {{error, Why}, S} end,
    {reply, Reply, S1, Idle}.

handle_cast(_, #s{idle = Idle} = S) -> {noreply, S, Idle}.
handle_info(timeout, S) -> {stop, normal, S};
handle_info(_, #s{idle = Idle} = S) -> {noreply, S, Idle}.

req({feed, X}, #s{win = Win} = S) ->
    S1 = set_window(S, window(S, Win, ids(S, X))),
    {{ok, summary(S1)}, S1};
req({set_context, X}, S) ->
    S1 = set_window(S, window(S, [], ids(S, X))),
    {{ok, summary(S1)}, S1};
req({complete, Len, Opts}, S) ->
    {Text, Why, S1} = run(S, Len, Opts),
    {{Text, Why}, S1};
req({chat, Messages, Opts}, S) ->
    Prompt = tinyllm:chat_prompt(Messages),
    S1 = set_window(S, window(S, [], ids(S, Prompt))),
    {Reply, Why, S2} = run(S1, maps:get(max_chars, Opts, 200), Opts#{stop => [<<"\n">>, <<"User:">>]}),
    {{string:trim(Reply), Why}, S2};
req(logits, S) ->
    S1 = ensure_context(S),
    {logits_of(S1), S1};
req(info, S) -> {summary(S), S};
req(reset, S) -> {ok, fresh(S)}.

fresh(#s{m = #{arch := Arch} = M} = S) ->
    S#s{win = [], logits = undefined,
        cache = case Arch of transformer -> tinyllm_tf:cache_new(maps:get(cfg, M)); mlp -> undefined end}.

summary(#s{m = #{arch := Arch, n := N}, win = Win, cache = C, computed = Cp, reused = Ru, evict = E, id = Id}) ->
    #{id => Id, arch => Arch, context => N, window => length(Win),
      cached => case C of undefined -> 0; _ -> tinyllm_tf:cache_len(C) end,
      computed => Cp, reused => Ru, evict => E}.

%% ------------------------------------------------------------------ tokens

ids(_, Ids) when is_list(Ids) -> Ids;
ids(#s{m = #{tok := Tok, vmap := VMap, pad := Pad}}, Text) when is_binary(Text) ->
    [maps:get(T, VMap, Pad) || T <- tinyllm_tok:tokenize(Tok, Text)];
ids(_, _) -> throw({bad, <<"expected text or a list of token ids">>}).

%% The window after appending Ids to Win. Exact mode (evict 1) keeps the last N tokens, which is what the
%% stateless path conditions on; with evict K > 1 a full window drops its K oldest tokens first.
window(#s{m = #{n := N}, evict = 1}, Win, Ids) ->
    last_n(Win ++ Ids, N);
window(#s{m = #{n := N, arch := mlp}}, Win, Ids) -> last_n(Win ++ Ids, N);
window(#s{m = #{n := N}, evict = E}, Win, Ids) ->
    lists:foldl(fun(I, W) when length(W) >= N -> lists:nthtail(E, W) ++ [I];
                   (I, W) -> W ++ [I]
                end, Win, Ids).

last_n(L, N) when length(L) > N -> lists:nthtail(length(L) - N, L);
last_n(L, _) -> L.

%% --------------------------------------------------------- cache management

%% Make the cached context exactly Win. Reuses the cache for the longest common prefix; computes the rest.
set_window(#s{m = #{arch := mlp}} = S, Win) -> S#s{win = Win, logits = undefined};
set_window(#s{win = Win} = S, Win) -> S;
set_window(S, []) -> fresh(S);
set_window(#s{m = #{cfg := Cfg, params := P}, win = Old, cache = C, computed = Cp, reused = Ru} = S, New) ->
    %% logits are only held for the last cached position, so at least one token is always recomputed
    Keep = min(lcp(Old, New, 0), length(New) - 1),
    Todo = lists:nthtail(Keep, New),
    %% only the last token needs logits; the others just extend the cache
    {Prefill, [Last]} = lists:split(length(Todo) - 1, Todo),
    C1 = lists:foldl(fun(I, Ci) -> tinyllm_tf:advance(P, Cfg, Ci, I) end, tinyllm_tf:cache_truncate(C, Keep), Prefill),
    {Logits, C2} = tinyllm_tf:step(P, Cfg, C1, Last),
    S#s{win = New, cache = C2, logits = Logits, computed = Cp + length(Todo), reused = Ru + Keep}.

lcp([X | A], [X | B], N) -> lcp(A, B, N + 1);
lcp(_, _, N) -> N.

%% The transformer needs at least one token of context (the stateless path starts from a newline).
ensure_context(#s{win = [], m = #{arch := transformer, padctx := Pad}} = S) -> set_window(S, [Pad]);
ensure_context(S) -> S.

logits_of(#s{m = #{arch := transformer}, logits = L}) -> L;
logits_of(#s{m = #{arch := mlp, n := N, padctx := Pad, params := P}, win = Win}) ->
    tinyllm_nn:logits(P, lists:duplicate(max(0, N - length(Win)), Pad) ++ Win).

%% ----------------------------------------------------------------- sampling

run(S0, Len, Opts) ->
    #s{m = #{tok := Tok, inv := Inv}} = S0,
    case maps:find(seed, Opts) of
        {ok, Seed} -> rand:seed(exsss, {Seed, Seed, Seed});
        error -> ok
    end,
    Temp = maps:get(temperature, Opts, 0.8),
    Stops = case maps:get(stop, Opts, []) of B when is_binary(B) -> [B]; L -> L end,
    S1 = ensure_context(S0),
    tinyllm:sample_loop(fun logits_of/1,
                        fun(S, I) -> set_window(S, window(S, S#s.win, [I])) end,
                        S1, Len, Temp, Inv, Tok, Stops).
