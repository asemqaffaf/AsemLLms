#!/usr/bin/env escript
%% Multi-turn chat latency: stateless tinyllm:chat/3 (rebuilds the forward pass for every token) against a
%% KV-cache session (tinyllm_session:chat/3), exact and with evict.
%%   rebar3 compile && escript scripts/session_bench.escript [model.bin | train:CONTEXT] [max_tokens] [repeats]
%% `train:128` builds a throwaway 3-step model with that context window (speed does not depend on training),
%% to show how the cache scales when the window is not already full.
%% Same prompts, same seed in every variant; the exact session must return the stateless replies.
main(Args) ->
    Path = case Args of [P | _] -> P; _ -> "chat_tf.bin" end,
    MaxTok = case Args of [_, T | _] -> list_to_integer(T); _ -> 12 end,
    Reps = case Args of [_, _, R | _] -> list_to_integer(R); _ -> 3 end,
    [true = code:add_pathz(P) || P <- filelib:wildcard("_build/default/lib/*/ebin")],
    application:set_env(tinyllm, memory, false),
    {ok, _} = application:ensure_all_started(tinyllm),
    M = case Path of
            "train:" ++ Ctx ->
                {ok, Corpus} = file:read_file("../dataset/chat_clean.txt"),
                {ok, Tmp} = tinyllm:train(Corpus, #{tokenizer => word, arch => transformer, context => list_to_integer(Ctx),
                                                    dim => 64, heads => 4, layers => 2, ff => 128, batch => 4, steps => 3,
                                                    lr => 0.003, max_vocab => 2000, min_count => 2, eval_every => 3,
                                                    val_examples => 4, verbose => false}),
                Tmp;
            _ -> tinyllm:load(Path)
        end,
    #{cfg := Cfg} = M,
    Questions = [<<"hello">>, <<"how are you today">>, <<"what do you like to do">>, <<"tell me about erlang">>,
                 <<"is it fast">>, <<"why">>, <<"ok thanks">>, <<"bye">>],
    Opts = #{temperature => 0.7, seed => 42, max_chars => MaxTok},
    %% stateless: also records the conversation every variant is replayed on
    {StateTimes, Convs} = stateless(M, Questions, Opts),
    io:format("model ~s: ~p, context ~p tokens, ~p new tokens per reply, ~p turns~n",
              [Path, maps:get(arch, Cfg), maps:get(context, Cfg), MaxTok, length(Questions)]),
    Variants = [{"stateless tinyllm:chat/3", StateTimes, undefined},
                session("session (exact, evict=1)", M, Convs, Opts, 1, Reps),
                session("session (evict=8, approximate)", M, Convs, Opts, 8, Reps)],
    Base = lists:sum(StateTimes),
    io:format("~n~-34s ~10s ~8s   ms per turn~n", ["variant", "total ms", "speedup"]),
    [io:format("~-34s ~10.1f ~7.2fx   ~s~n",
               [Name, lists:sum(Ts), Base / lists:sum(Ts), string:join([integer_to_list(round(T)) || T <- Ts], " ")])
     || {Name, Ts, _} <- Variants],
    [io:format("~s: computed ~p token passes, reused ~p cached positions~n", [Name, C, R])
     || {Name, _, #{computed := C, reused := R}} <- Variants],
    ok.

%% -> {[ms per turn], [Messages list sent at each turn]}  (best of one run; the first call also warms up)
stateless(M, Questions, Opts) ->
    {_, Ts, Convs} =
        lists:foldl(fun(Q, {Msgs0, Acc, Cs}) ->
                        Msgs = Msgs0 ++ [#{role => user, content => Q}],
                        {Us, {Reply, _}} = timer:tc(fun() -> tinyllm:chat(M, Msgs, Opts) end),
                        {Msgs ++ [#{role => assistant, content => Reply}], [Us / 1000 | Acc], [Msgs | Cs]}
                    end, {[], [], []}, Questions),
    {lists:reverse(Ts), lists:reverse(Convs)}.

session(Name, M, Convs, Opts, Evict, Reps) ->
    Runs = [begin
                {ok, Pid} = tinyllm_session:start(M, #{evict => Evict}),
                Ts = [begin
                          {Us, {Reply, _}} = timer:tc(fun() -> tinyllm_session:chat(Pid, Msgs, Opts) end),
                          Evict =:= 1 andalso begin  %% the exact session must reproduce the stateless reply
                              {Ref, _} = tinyllm:chat(M, Msgs, Opts),
                              Ref =:= Reply orelse error({differs, Msgs, Ref, Reply})
                          end,
                          Us / 1000
                      end || Msgs <- Convs],
                Info = tinyllm_session:info(Pid),
                tinyllm_session:stop(Pid),
                {Ts, Info}
            end || _ <- lists:seq(1, Reps)],
    {Ts, Info} = hd(lists:sort(fun({A, _}, {B, _}) -> lists:sum(A) =< lists:sum(B) end, Runs)),
    {Name, Ts, Info}.
