%% The native backend: a TLM1 file mmap'd by native/tinyllm_nif.c.
%%
%%   {ok, Native} = tinyllm_backend_nif:to_native(Model, "/tmp/asemllm.tlm"),   %% write + mmap
%%   {ok, Native} = tinyllm_backend_nif:open("/tmp/asemllm.tlm", Meta),         %% mmap an existing file
%%
%% Native is the same model map as Model (cfg, vocab, tokenizer) with params => #{nif := Resource, ...}.
%% tinyllm_session, tinyllm:complete/4 and friends work on it unchanged.
%%
%% The KV cache is #{n := N, res := Resource}. Resources are mutable C buffers while Erlang values are
%% not, so the NIF copies on write: a buffer is extended in place only when N equals the number of positions
%% written so far, otherwise the first N positions are copied first. Using one cache linearly (what a session
%% does) never copies per token; using an old value again is always safe.
%% Only an Erlang-built empty cache (tinyllm_tf:cache_new/1) can start a native one.
-module(tinyllm_backend_nif).
-behaviour(tinyllm_backend).
-export([available/0, to_native/2, open/2, info/1]).
-export([cache_len/1, cache_truncate/2, advance/4, step/4, last_logits/3]).

%% Above this many parameters one call can take longer than a scheduler should be blocked (about 1 ms), so
%% the dirty CPU variants are used. Override with application env tinyllm:dirty_params.
-define(DIRTY_PARAMS, 2000000).

available() ->
    try tinyllm_nif:abi() of
        {1, _} -> true;
        _ -> false
    catch
        error:nif_library_not_loaded -> false
    end.

%% Write the model's params as a TLM1 file at Path and mmap it.
to_native(#{cfg := Cfg, params := Params} = Model, Path) when is_list(Path) ->
    ok = tinyllm_tlm:write(Path, Params, Cfg),
    open(Path, Model).

%% Meta: a model map (cfg, vocab, tokenizer, ...); its params, if any, are replaced by the native handle.
open(Path, Meta) when is_list(Path) ->
    case tinyllm_nif:load(unicode:characters_to_binary(Path)) of
        {ok, Res, #{params := N} = Info} ->
            case check_cfg(maps:get(cfg, Meta, #{}), Info) of
                ok ->
                    Dirty = N > application:get_env(tinyllm, dirty_params, ?DIRTY_PARAMS),
                    Cfg = maps:merge(maps:get(cfg, Meta, #{}), #{arch => transformer}),
                    {ok, Meta#{cfg => Cfg, params => #{nif => Res, info => Info, dirty => Dirty}}};
                {error, _} = E -> E
            end;
        {error, Why} -> {error, Why}
    end.

check_cfg(Cfg, Info) ->
    Keys = [vocab, context, dim, heads, layers, ff],
    case [K || K <- Keys, maps:is_key(K, Cfg), maps:get(K, Cfg) =/= maps:get(K, Info)] of
        [] -> ok;
        Bad -> {error, {cfg_mismatch, Bad}}
    end.

info(#{nif := _, info := Info}) -> Info.

%% ----------------------------------------------------------- tinyllm_backend

cache_len(#{n := N}) -> N.

cache_truncate(#{n := M} = C, N) when N >= M -> C;
cache_truncate(#{res := _} = C, N) -> C#{n := N}.

advance(#{nif := M} = P, _Cfg, Cache, Id) ->
    {Res, N} = native_cache(M, Cache),
    {Res1, undefined} = call(P, step, [M, Res, N, Id, kv_only]),
    #{n => N + 1, res => Res1}.

step(#{nif := M} = P, _Cfg, Cache, Id) ->
    {Res, N} = native_cache(M, Cache),
    {Res1, Logits} = call(P, step, [M, Res, N, Id, full]),
    {Logits, #{n => N + 1, res => Res1}}.

last_logits(#{nif := M} = P, _Cfg, Ids) -> call(P, last_logits, [M, Ids]).

native_cache(_, #{res := Res, n := N}) -> {Res, N};
native_cache(M, #{n := 0}) -> {tinyllm_nif:cache_new(M), 0};
native_cache(_, _) -> erlang:error(cache_backend_mismatch).

call(#{dirty := true}, step, [M, C, N, Id, Mode]) -> tinyllm_nif:step_dirty(M, C, N, Id, Mode);
call(_, step, [M, C, N, Id, Mode]) -> tinyllm_nif:step(M, C, N, Id, Mode);
call(#{dirty := true}, last_logits, [M, Ids]) -> tinyllm_nif:last_logits_dirty(M, Ids);
call(_, last_logits, [M, Ids]) -> tinyllm_nif:last_logits(M, Ids).
