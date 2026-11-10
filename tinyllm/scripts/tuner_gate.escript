#!/usr/bin/env escript
%%! -pa _build/default/lib/tinyllm/ebin
%% Does the tuner's gate separate real signal from garbage on a REAL model? (tuner_smoke.escript only proves the
%% plumbing, on a random 2-layer model.)  Build the sets first:
%%   python3 scripts/make_eval_sets.py /tmp/sets
%%   escript scripts/tuner_gate.escript PYTHON scripts/mlx_worker.py ROOT /tmp/sets /path/to/mlx/model [LR,LR,..]
%% Set TLM_MLX_NUM_LAYERS (LoRA layers) for big models. Checks, each printed PASS/FAIL:
%%   1 garbage offered to the BASE model is rejected (the tuner trains a word-shuffled control and the batch
%%     must beat it by `control_margin`; without that, garbage passes because part of any gain is generic);
%%   2 real training data is committed and lowers held-out loss (held-out shares no user question with train);
%%   3 garbage offered after that is rejected and the active weights are untouched;
%%   4 the probe (general text, unlike the chat domain) did not get worse than 'probe_tolerance' vs baseline;
%%   5 rollback restores the committed weights exactly, then the base.
%% Exit status is 1 if any check FAILs.
main([Py, Worker, Root, Sets, Model | Rest]) ->
    Lrs = case Rest of [S] -> [list_to_float(string:trim(X)) || X <- string:split(S, ",", all)];
                       [] -> [1.0e-5, 5.0e-5] end,
    Read = fun(F) -> {ok, B} = file:read_file(filename:join(Sets, F)),
                     [L || L <- binary:split(B, <<"\n">>, [global]), L =/= <<>>] end,
    Train = Read("train.txt"), Garbage = Read("garbage.txt"),
    Tol = 0.3, Gain = 0.05, Margin = 0.1,
    Eng = fun(Dir) -> #{cmd => Py, args => [Worker], root => filename:join(Root, Dir), base_model => Model,
                        heldout_file => filename:join(Sets, "heldout.txt"),
                        probe_file => filename:join(Sets, "probe.txt")} end,
    Start = fun(Dir) -> {ok, T} = tinyllm_tuner:start_link(#{engine => {tinyllm_engine_port, Eng(Dir)},
                            candidates => Lrs, steps => 20, rank => 8, parallel => false,
                            min_gain => Gain, probe_tolerance => Tol,
                            control => shuffle, control_margin => Margin}), T end,
    Check = fun(Name, Ok) -> io:format("~s  ~s~n", [case Ok of true -> "PASS"; false -> "FAIL" end, Name]),
                              put(failed, get(failed) orelse (not Ok)), Ok end,
    put(failed, false),
    %% 1: garbage first, from the base model
    TA = Start("a"),
    {ok, A0} = tinyllm_tuner:current(TA),
    io:format("baseline  heldout=~.3f probe=~.3f~n", [maps:get(heldout, A0), maps:get(probe, A0)]),
    RA = tinyllm_tuner:learn(TA, Garbage), io:format("garbage from base -> ~p~n", [summ(RA)]),
    Check("1 garbage from base is rejected", element(1, RA) =:= rejected),
    %% 2-5: real data, then garbage, then rollback
    TB = Start("b"),
    R1 = tinyllm_tuner:learn(TB, Train), io:format("train -> ~p~n", [summ(R1)]),
    {ok, B1} = tinyllm_tuner:current(TB), Sha1 = sha(maps:get(handle, B1)),
    io:format("after train  heldout=~.3f probe=~.3f lr=~p~n", [maps:get(heldout, B1), maps:get(probe, B1), maps:get(lr, B1)]),
    Check("2 train data committed", element(1, R1) =:= committed),
    Check("2 held-out loss fell by more than min_gain", maps:get(heldout, B1) < maps:get(heldout, A0) - Gain),
    R2 = tinyllm_tuner:learn(TB, Garbage), io:format("garbage after train -> ~p~n", [summ(R2)]),
    {ok, B2} = tinyllm_tuner:current(TB),
    Check("3 garbage after train is rejected", element(1, R2) =:= rejected),
    Check("3 active weights unchanged", sha(maps:get(handle, B2)) =:= Sha1),
    Check(io_lib:format("4 probe within ~.2f of baseline (now ~.3f, base ~.3f)", [Tol, maps:get(probe, B1), maps:get(probe, A0)]),
          maps:get(probe, B1) =< maps:get(probe, A0) + Tol),
    case R1 of {committed, _, _} ->
        {ok, 0} = tinyllm_tuner:rollback(TB, length(tinyllm_tuner:versions(TB)) - 1),
        {ok, Cb} = tinyllm_tuner:current(TB),
        Check("5 rollback to base reproduces the baseline scores",
              abs(maps:get(heldout, Cb) - maps:get(heldout, A0)) < 1.0e-6);
      _ -> ok end,
    halt(case get(failed) of true -> 1; false -> 0 end).

sha(none) -> "none";
sha(H) -> {ok, B} = file:read_file(filename:join(binary_to_list(H), "adapters.safetensors")),
          string:slice(binary_to_list(binary:encode_hex(crypto:hash(sha, B))), 0, 10).
summ({committed, Id, #{candidates := Cs}}) -> {committed, Id, cands(Cs)};
summ({rejected, R, #{candidates := Cs}}) -> {rejected, R, cands(Cs)};
summ(Other) -> Other.
cands(Cs) -> [{maps:get(lr, C), r3(maps:get(heldout, C)), r3(maps:get(probe, C))} || C <- Cs].
r3(X) -> round(X * 1000) / 1000.
