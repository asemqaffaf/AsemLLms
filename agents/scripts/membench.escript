#!/usr/bin/env escript
%% Run the memory benchmark baselines against LM Studio.
%%   rebar3 compile && escript scripts/membench.escript CASES.jsonl [key=value ...]
%% Keys: setups=none,truncated,retrieval  window=2048  k=6  limit=N  concurrency=1  out=rows.jsonl
%%       model=NAME  base_url=http://127.0.0.1:1234  retriever=lexical|embedding  embed_model=NAME
main([]) ->
    io:format("usage: escript scripts/membench.escript CASES.jsonl [key=value ...]~n"), halt(2);
main([Cases | KVs]) ->
    true = code:add_pathz("_build/default/lib/agents/ebin"),
    {ok, _} = application:ensure_all_started(agents),
    O = maps:from_list([case string:split(KV, "=") of [K, V] -> {K, V}; _ -> error({bad_arg, KV}) end || KV <- KVs]),
    Get = fun(K, D) -> maps:get(K, O, D) end,
    Base = Get("base_url", "http://127.0.0.1:1234"),
    LlmOpts0 = #{base_url => Base},
    LlmOpts = case maps:find("model", O) of {ok, M} -> LlmOpts0#{model => list_to_binary(M)}; error -> LlmOpts0 end,
    Retriever = case Get("retriever", "lexical") of
                    "lexical" -> lexical;
                    "embedding" -> {embedding, agents_membench:lmstudio_embed(
                                                 #{base_url => Base, model => list_to_binary(Get("embed_model", ""))})}
                end,
    Setups = [list_to_atom(S) || S <- string:split(Get("setups", "none,truncated,retrieval"), ",", all)],
    Opts0 = #{setups => Setups, window => list_to_integer(Get("window", "2048")), k => list_to_integer(Get("k", "6")),
              concurrency => list_to_integer(Get("concurrency", "1")), llm_opts => LlmOpts, retriever => Retriever},
    Opts1 = case maps:find("limit", O) of {ok, L} -> Opts0#{limit => list_to_integer(L)}; error -> Opts0 end,
    Opts = case maps:find("out", O) of {ok, F} -> Opts1#{out => F}; error -> Opts1 end,
    Rows = agents_membench:run(Cases, Opts),
    io:format("~n"),
    agents_membench:print_summary(Rows).
