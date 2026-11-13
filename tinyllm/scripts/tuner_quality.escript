#!/usr/bin/env escript
%%! -pa _build/default/lib/tinyllm/ebin
%% Beyond loss: do answers get better, is old knowledge kept, and does rollback restore exact weights?
%%   python3 scripts/make_eval_sets.py /tmp/sets --train 60 --seed 0
%%   escript scripts/tuner_quality.escript PYTHON scripts/mlx_worker.py ROOT /tmp/sets /path/to/mlx/model
%% Three rounds on three disjoint thirds of train.txt (with the shuffled control gate). scripts/capability_eval.py
%% runs on the base and after every committed version (reply F1 on heldout dialogues, knowledge accuracy and
%% margin); the results are ROOT/cap_*.json. Then it rolls back one version at a time and compares the sha of
%% adapters.safetensors with what that version had when it was committed. Prints CHECK lines; exit 1 on a FAIL.
main([Py, Worker, Root, Sets, Model]) ->
    Read = fun(F) -> {ok, B} = file:read_file(filename:join(Sets, F)),
                     [L || L <- binary:split(B, <<"\n">>, [global]), L =/= <<>>] end,
    Train = Read("train.txt"), N = length(Train) div 3,
    Chunks = [lists:sublist(Train, 1 + (I - 1) * N, N) || I <- [1, 2, 3]],
    Cap = filename:join(filename:dirname(Worker), "capability_eval.py"),
    ok = filelib:ensure_dir(filename:join(Root, "x")),
    Eval = fun(Label, Adapter) ->
        Out = filename:join(Root, "cap_" ++ Label ++ ".json"),
        Ad = case Adapter of none -> ""; H -> " --adapter " ++ binary_to_list(H) end,
        os:cmd(io_lib:format("~s ~s ~s~s --sets ~s --n 30 2>/dev/null | tail -1 > ~s", [Py, Cap, Model, Ad, Sets, Out])),
        io:format("capability ~s -> ~s~n", [Label, Out]) end,
    Eng = #{cmd => Py, args => [Worker], root => filename:join(Root, "tuner"), base_model => Model,
            heldout_file => filename:join(Sets, "heldout.txt"), probe_file => filename:join(Sets, "probe.txt")},
    {ok, T} = tinyllm_tuner:start_link(#{engine => {tinyllm_engine_port, Eng}, candidates => [1.0e-5, 5.0e-5],
                                         steps => 20, rank => 8, parallel => false, min_gain => 0.05,
                                         probe_tolerance => 0.3, control => shuffle, control_margin => 0.1}),
    put(failed, false),
    Check = fun(Name, Ok) -> io:format("~s  ~s~n", [case Ok of true -> "PASS"; false -> "FAIL" end, Name]),
                              put(failed, get(failed) orelse (not Ok)) end,
    {ok, C0} = tinyllm_tuner:current(T),
    io:format("baseline heldout=~.3f probe=~.3f~n", [maps:get(heldout, C0), maps:get(probe, C0)]),
    Eval("base", none),
    Recorded = lists:foldl(fun({I, Chunk}, Acc) ->
        R = tinyllm_tuner:learn(T, Chunk),
        {ok, C} = tinyllm_tuner:current(T),
        io:format("round ~p -> ~p  now id=~p heldout=~.3f probe=~.3f~n",
                  [I, element(1, R), maps:get(id, C), maps:get(heldout, C), maps:get(probe, C)]),
        case R of
            {committed, Id, Info} ->
                io:format("  candidates ~p controls ~p~n", [cands(maps:get(candidates, Info)), cands(maps:get(controls, Info, []))]),
                Eval("v" ++ integer_to_list(Id), maps:get(handle, C)),
                [{Id, sha(maps:get(handle, C))} | Acc];
            {rejected, Why, _} -> io:format("  rejected: ~p~n", [Why]), Acc;
            Other -> io:format("  ~p~n", [Other]), Acc
        end end, [], lists:zip([1, 2, 3], Chunks)),
    io:format("committed versions (id, sha): ~p~n", [lists:reverse(Recorded)]),
    %% walk back one version at a time; every step must land on a version with the sha it had when committed
    lists:foreach(fun({_Id, _Sha}) ->
        {ok, Now} = tinyllm_tuner:rollback(T),
        {ok, Cur} = tinyllm_tuner:current(T),
        case Now of
            0 -> Check("rollback to base reproduces baseline heldout", abs(maps:get(heldout, Cur) - maps:get(heldout, C0)) < 1.0e-6);
            _ -> Want = proplists:get_value(Now, Recorded),
                 Check(io_lib:format("rollback to v~p restores identical adapter sha ~s", [Now, Want]),
                       sha(maps:get(handle, Cur)) =:= Want)
        end end, Recorded),
    halt(case get(failed) of true -> 1; false -> 0 end).

sha(none) -> "none";
sha(H) -> {ok, B} = file:read_file(filename:join(binary_to_list(H), "adapters.safetensors")),
          string:slice(binary_to_list(binary:encode_hex(crypto:hash(sha, B))), 0, 12).
cands(Cs) -> [{maps:get(lr, C), round(maps:get(heldout, C) * 1000) / 1000, round(maps:get(probe, C) * 1000) / 1000} || C <- Cs].
