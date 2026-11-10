#!/usr/bin/env escript
%% End-to-end smoke test: the Erlang tuner driving the real MLX worker on the tiny random model.
%% Run from tinyllm/ after `rebar3 compile`, with TLM_MLX_NUM_LAYERS=2:
%%   python3 scripts/tiny_mlx_model.py /tmp/tiny
%%   escript scripts/tuner_smoke.escript $(which python3) scripts/mlx_worker.py /tmp/tuner_root /tmp/tiny
%% Expect: round 1 commits, the garbage round is rejected with the active weights unchanged, rollback restores
%% the exact round-1 weights (same sha), and rolling back to the base reproduces the baseline scores.
%%! -pa _build/default/lib/tinyllm/ebin
main([Py, Worker, Root, Tiny]) ->
    Read = fun(F) -> {ok, B} = file:read_file(filename:join(Tiny, F)),
                     [L || L <- binary:split(B, <<"\n">>, [global]), L =/= <<>>] end,
    Train = Read("train.txt"),
    Bad = [<<"zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz.">>],
    Eng = #{cmd => Py, args => [Worker], root => Root, base_model => filename:join(Tiny, "model"),
            heldout_file => filename:join(Tiny, "heldout.txt"), probe_file => filename:join(Tiny, "probe.txt")},
    {ok, T} = tinyllm_tuner:start_link(#{engine => {tinyllm_engine_port, Eng}, candidates => [0.005, 0.03],
                                         steps => 20, rank => 4, parallel => false, min_gain => 0.002,
                                         probe_tolerance => 0.0}),
    Show = fun(Tag) -> {ok, C} = tinyllm_tuner:current(T),
                       io:format("~-28s id=~p heldout=~.3f probe=~.3f lr=~p sha=~s~n",
                                 [Tag, maps:get(id, C), maps:get(heldout, C), maps:get(probe, C), maps:get(lr, C), sha(maps:get(handle, C))]) end,
    Dirs = fun() -> lists:sort([filename:basename(D) || D <- filelib:wildcard(filename:join(Root, "v*"))]) end,
    Show("baseline"),
    R1 = tinyllm_tuner:learn(T, Train),
    io:format("learn(train)      -> ~p~n", [summ(R1)]), Show("after round 1"),
    Sha1 = cur_sha(T), D1 = Dirs(),
    R2 = tinyllm_tuner:learn(T, Bad),
    io:format("learn(garbage)    -> ~p~n", [summ(R2)]), Show("after garbage round"),
    io:format("active weights unchanged by rejected/failed round: ~p~n", [cur_sha(T) =:= Sha1]),
    io:format("no stray candidate dirs left: ~p (~p)~n", [Dirs() =:= D1, length(Dirs())]),
    R3 = tinyllm_tuner:learn(T, Train),
    io:format("learn(train again)-> ~p~n", [summ(R3)]), Show("after round 3"),
    io:format("versions: ~p~n", [[maps:get(id, V) || V <- tinyllm_tuner:versions(T)]]),
    case R3 of {committed, _, _} ->
        io:format("rollback -> ~p~n", [tinyllm_tuner:rollback(T)]), Show("after rollback"),
        io:format("rollback restored the exact round-1 weights: ~p~n", [cur_sha(T) =:= Sha1]);
      _ -> ok end,
    io:format("rollback to base -> ~p~n", [tinyllm_tuner:rollback(T, length(tinyllm_tuner:versions(T)) - 1)]),
    Show("at base"),
    io:format("log events: ~p~n", [[maps:get(event, E) || E <- lists:reverse(tinyllm_tuner:log(T))]]).

cur_sha(T) -> {ok, #{handle := H}} = tinyllm_tuner:current(T), sha(H).
sha(none) -> "none";
sha(H) -> {ok, B} = file:read_file(filename:join(binary_to_list(H), "adapters.safetensors")),
          string:slice(binary_to_list(binary:encode_hex(crypto:hash(sha, B))), 0, 10).
summ({committed, Id, #{candidates := Cs, failures := F}}) ->
    {committed, Id, [{maps:get(lr, C), round3(maps:get(heldout, C)), round3(maps:get(probe, C))} || C <- Cs], F};
summ({rejected, R, #{candidates := Cs}}) ->
    {rejected, R, [{maps:get(lr, C), round3(maps:get(heldout, C)), round3(maps:get(probe, C))} || C <- Cs]};
summ(Other) -> Other.
round3(X) -> round(X * 1000) / 1000.
