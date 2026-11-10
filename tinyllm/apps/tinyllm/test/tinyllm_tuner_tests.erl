%% tinyllm_tuner: propose, score, gate, commit, roll back. Everything runs on stand-in engines (a 1-D quadratic
%% and a shell script behind tinyllm_engine_port), so it needs no GPU, no model and no Python.
-module(tinyllm_tuner_tests).
-include_lib("eunit/include/eunit.hrl").

-define(FAKE, tinyllm_engine_fake).

%% ---------------------------------------------------------------- helpers

start(Eng, Opts) ->
    released(), trained_from(),
    {ok, T} = tinyllm_tuner:start_link(Opts#{engine => {?FAKE, Eng#{notify => self()}}}),
    T.

close(T) -> tinyllm_tuner:stop(T).

near(A, B) -> ?assert(abs(A - B) < 1.0e-9).

weight(T) -> {ok, #{handle := H}} = tinyllm_tuner:current(T), ?FAKE:weight(H).
cur_id(T) -> {ok, #{id := Id}} = tinyllm_tuner:current(T), Id.

released() -> receive {released, H} -> [H | released()] after 0 -> [] end.
trained_from() -> receive {trained_from, F, Lr} -> [{F, Lr} | trained_from()] after 0 -> [] end.

%% ---------------------------------------------------------------- baseline and commit

baseline_is_version_zero_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1]}),
    {ok, C} = tinyllm_tuner:current(T),
    ?assertMatch(#{id := 0, handle := {fake, 0.0}}, C),
    near(maps:get(heldout, C), 1.0),
    near(maps:get(probe, C), 1.0),
    ?assertEqual([0], [maps:get(id, V) || V <- tinyllm_tuner:versions(T)]),
    close(T).

commit_when_all_gates_pass_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5}),
    {committed, 1, Info} = tinyllm_tuner:learn(T, [1.0]),
    ?assertEqual(1, cur_id(T)),
    near(weight(T), 1.0 - math:pow(0.8, 5)),
    {ok, C} = tinyllm_tuner:current(T),
    near(maps:get(heldout, C), math:pow(0.8, 10)),
    ?assertMatch(#{candidates := [#{lr := 0.1}], failures := []}, Info),
    close(T).

%% ---------------------------------------------------------------- gates

reject_when_heldout_gets_worse_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5}),
    ?assertMatch({rejected, no_improvement, _}, tinyllm_tuner:learn(T, [-5.0])),
    ?assertEqual(0, cur_id(T)),
    [{fake, W}] = released(),
    near(W, -5.0 * (1.0 - math:pow(0.8, 5))),
    close(T).

reject_when_probe_gets_worse_then_allow_with_tolerance_test() ->
    %% heldout wants W=1, the probe set wants W=0: learning toward 1 is forgetting the probe.
    T = start(#{}, #{candidates => [0.1], steps => 5}),
    ?assertMatch({rejected, probe_regressed, _}, tinyllm_tuner:learn(T, [1.0])),
    ?assertEqual(0, cur_id(T)),
    close(T),
    T2 = start(#{}, #{candidates => [0.1], steps => 5, probe_tolerance => 1.0}),
    ?assertMatch({committed, 1, _}, tinyllm_tuner:learn(T2, [1.0])),
    close(T2).

reject_when_drift_cap_exceeded_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.4], steps => 5, drift_cap => 0.1}),
    ?assertMatch({rejected, drift_cap, _}, tinyllm_tuner:learn(T, [1.0])),
    close(T).

min_gain_is_required_test() ->
    %% one tiny step gains less than 0.5 of heldout loss
    T = start(#{probe_target => 1.0}, #{candidates => [0.01], steps => 1, min_gain => 0.5}),
    ?assertMatch({rejected, no_improvement, _}, tinyllm_tuner:learn(T, [1.0])),
    close(T).

%% ---------------------------------------------------------------- control gate

%% heldout_target 1.0, start 0.0: the real batch [1.0] pulls W toward the target; a control batch [0.0] does not.
control_that_learns_nothing_lets_real_data_through_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5, control => fun(_) -> [0.0] end}),
    {committed, 1, Info} = tinyllm_tuner:learn(T, [1.0]),
    ?assertMatch(#{controls := [#{lr := 0.1}]}, Info),
    [#{heldout := KHo}] = maps:get(controls, Info),
    [#{heldout := Ho}] = maps:get(candidates, Info),
    ?assert(Ho < KHo),
    %% the control is released and never becomes the current version
    near(weight(T), 1.0 - math:pow(0.8, 5)),
    close(T).

control_as_good_as_the_data_rejects_it_test() ->
    %% the "garbage" teaches exactly what the real batch does, so the gain is not about the batch
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5, control => fun(B) -> B end}),
    ?assertMatch({rejected, no_better_than_control, #{controls := [_]}}, tinyllm_tuner:learn(T, [1.0])),
    ?assertEqual(0, cur_id(T)),
    close(T).

control_margin_is_required_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5, control => fun(_) -> [0.0] end, control_margin => 5.0}),
    ?assertMatch({rejected, no_better_than_control, _}, tinyllm_tuner:learn(T, [1.0])),
    close(T).

control_that_cannot_be_trained_is_not_evidence_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5, control => fun(_) -> [crash] end}),
    ?assertMatch({rejected, no_better_than_control, _}, tinyllm_tuner:learn(T, [1.0])),
    ?assertEqual(0, cur_id(T)),
    close(T).

control_is_not_trained_when_the_normal_gate_already_failed_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1], steps => 5, control => fun(_) -> [0.0] end}),
    trained_from(),
    ?assertMatch({rejected, no_improvement, Info} when not is_map_key(controls, Info), tinyllm_tuner:learn(T, [-5.0])),
    ?assertEqual(1, length(trained_from())),
    close(T).

shuffle_keeps_the_words_and_loses_the_order_test() ->
    In = <<"User: how much water does a plant need? Assistant: about an inch of water weekly.">>,
    [Out] = tinyllm_tuner:control_batch(shuffle, [In]),
    Words = fun(B) -> lists:sort(binary:split(B, <<" ">>, [global])) end,
    ?assertEqual(Words(In), Words(Out)),
    ?assertNotEqual(In, Out),
    ?assertEqual([Out], tinyllm_tuner:control_batch(shuffle, [In])),   %% deterministic
    ?assertEqual([7], tinyllm_tuner:control_batch(shuffle, [7])).

bad_control_option_is_refused_test() ->
    process_flag(trap_exit, true),
    ?assertMatch({error, {bad_options, _}},
                 tinyllm_tuner:start_link(#{engine => {?FAKE, #{}}, candidates => [0.1], control => nope})),
    flush_exits().

%% ---------------------------------------------------------------- candidate search

best_passing_candidate_wins_test_() ->
    [{atom_to_list(P), fun() -> best_candidate(P) end} || P <- [parallel, sequential]].

best_candidate(Mode) ->
    %% lr 0.3 converges fastest (factor 0.4 per step); 0.9 overshoots and oscillates; 0.01 barely moves
    T = start(#{probe_target => 1.0}, #{candidates => [0.01, 0.3, 0.9], steps => 5, parallel => Mode =:= parallel}),
    {committed, 1, #{candidates := Cs}} = tinyllm_tuner:learn(T, [1.0]),
    ?assertEqual(3, length(Cs)),
    {ok, #{lr := Lr}} = tinyllm_tuner:current(T),
    ?assertEqual(0.3, Lr),
    near(weight(T), 1.0 - math:pow(0.4, 5)),
    %% the two losers were released, the winner was not
    ?assertEqual(2, length(released())),
    close(T).

every_candidate_starts_from_the_same_checkpoint_test_() ->
    [{atom_to_list(P), fun() -> same_start(P) end} || P <- [parallel, sequential]].

same_start(Mode) ->
    %% learning rates are only comparable if every candidate of a round starts from the same weights
    T = start(#{probe_target => 1.0}, #{candidates => [0.01, 0.1, 0.3], steps => 2, parallel => Mode =:= parallel}),
    {ok, #{handle := Base}} = tinyllm_tuner:current(T),
    {committed, 1, _} = tinyllm_tuner:learn(T, [1.0]),
    Round1 = trained_from(),
    {ok, #{handle := H1}} = tinyllm_tuner:current(T),
    {committed, 2, _} = tinyllm_tuner:learn(T, [1.0]),
    Round2 = trained_from(),
    ?assertNotEqual(Base, H1),
    ?assertEqual([0.01, 0.1, 0.3], lists:sort([Lr || {_, Lr} <- Round1])),
    ?assertEqual([Base], lists:usort([F || {F, _} <- Round1])),
    ?assertEqual([0.01, 0.1, 0.3], lists:sort([Lr || {_, Lr} <- Round2])),
    ?assertEqual([H1], lists:usort([F || {F, _} <- Round2])),
    close(T).

a_candidate_that_breaks_a_gate_is_skipped_test() ->
    %% lr 0.4 gets closest to the target but drifts 0.99968 > 0.1; lr 0.01 drifts 0.096 and passes
    T = start(#{probe_target => 1.0}, #{candidates => [0.4, 0.01], steps => 5, drift_cap => 0.1}),
    {committed, 1, _} = tinyllm_tuner:learn(T, [1.0]),
    {ok, #{lr := Lr}} = tinyllm_tuner:current(T),
    ?assertEqual(0.01, Lr),
    close(T).

candidates_run_one_at_a_time_by_default_test() ->
    %% delay 150 ms x 3 candidates: sequential takes >= 450 ms, parallel would take ~150 ms
    T = start(#{probe_target => 1.0, delay => 150}, #{candidates => [0.01, 0.1, 0.3], steps => 1}),
    {Us, {committed, 1, _}} = timer:tc(fun() -> tinyllm_tuner:learn(T, [1.0]) end),
    ?assert(Us >= 440000),
    close(T),
    T2 = start(#{probe_target => 1.0, delay => 150}, #{candidates => [0.01, 0.1, 0.3], steps => 1, parallel => true}),
    {Us2, {committed, 1, _}} = timer:tc(fun() -> tinyllm_tuner:learn(T2, [1.0]) end),
    ?assert(Us2 < 400000),
    close(T2).

%% ---------------------------------------------------------------- engine failures

failed_candidates_are_reported_and_skipped_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [-1.0, 99.0, 0.3], steps => 5}),
    {committed, 1, #{failures := F}} = tinyllm_tuner:learn(T, [1.0]),
    ?assertEqual([-1.0, 99.0], lists:sort([Lr || {Lr, _} <- F])),
    ?assertEqual({-1.0, bad_lr}, lists:keyfind(-1.0, 1, F)),
    ?assertMatch({99.0, {error, boom}}, lists:keyfind(99.0, 1, F)),
    close(T).

all_candidates_failing_changes_nothing_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.1, 0.2], steps => 5}),
    ?assertMatch({error, {all_candidates_failed, [{0.1, _}, {0.2, _}]}}, tinyllm_tuner:learn(T, [crash])),
    ?assertMatch({error, {all_candidates_failed, [{0.1, bad_batch}, {0.2, bad_batch}]}}, tinyllm_tuner:learn(T, [bad])),
    ?assertEqual(0, cur_id(T)),
    ?assertMatch([#{event := error}, #{event := error}], tinyllm_tuner:log(T)),
    close(T).

a_hung_candidate_times_out_test() ->
    T = start(#{probe_target => 1.0, delay => 600}, #{candidates => [0.1], timeout => 100}),
    ?assertEqual({error, {all_candidates_failed, [{0.1, timeout}]}}, tinyllm_tuner:learn(T, [1.0])),
    ?assertEqual(0, cur_id(T)),
    close(T).

bad_options_and_bad_engine_test() ->
    process_flag(trap_exit, true),
    ?assertMatch({error, {bad_options, _}},
                 tinyllm_tuner:start_link(#{engine => {?FAKE, #{}}, candidates => []})),
    ?assertMatch({error, {bad_options, _}},
                 tinyllm_tuner:start_link(#{engine => {?FAKE, #{}}, retain => 0})),
    ?assertMatch({error, {engine_init_failed, _}},
                 tinyllm_tuner:start_link(#{engine => {tinyllm_engine_port, #{}}})),
    flush_exits().

flush_exits() -> receive {'EXIT', _, _} -> flush_exits() after 50 -> ok end.

%% ---------------------------------------------------------------- versions, retention, rollback

rollback_steps_back_and_releases_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.05], steps => 1}),
    [{committed, N, _} = tinyllm_tuner:learn(T, [1.0]) || N <- [1, 2, 3]],
    W3 = weight(T),
    ?assertEqual([3, 2, 1, 0], [maps:get(id, V) || V <- tinyllm_tuner:versions(T)]),
    ?assertEqual({ok, 2}, tinyllm_tuner:rollback(T)),
    ?assertEqual([{fake, W3}], released()),
    near(weight(T), 1.0 - math:pow(0.9, 2)),
    ?assertEqual({ok, 0}, tinyllm_tuner:rollback(T, 2)),
    ?assertEqual({fake, 0.0}, begin {ok, #{handle := H}} = tinyllm_tuner:current(T), H end),
    ?assertEqual({error, nothing_to_roll_back}, tinyllm_tuner:rollback(T)),
    ?assertMatch([#{event := rollback}, #{event := rollback}, #{event := commit} | _], tinyllm_tuner:log(T)),
    close(T).

rollback_refuses_more_than_it_has_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.05], steps => 1}),
    {committed, 1, _} = tinyllm_tuner:learn(T, [1.0]),
    ?assertEqual({error, not_enough_versions}, tinyllm_tuner:rollback(T, 2)),
    ?assertEqual(1, cur_id(T)),
    close(T).

only_the_last_n_versions_and_the_base_are_kept_test() ->
    T = start(#{probe_target => 1.0}, #{candidates => [0.05], steps => 1, retain => 2}),
    [{committed, N, _} = tinyllm_tuner:learn(T, [1.0]) || N <- [1, 2, 3, 4]],
    ?assertEqual([4, 3, 0], [maps:get(id, V) || V <- tinyllm_tuner:versions(T)]),
    Gone = released(),
    ?assertEqual(2, length(Gone)),
    ?assertNot(lists:member({fake, 0.0}, Gone)),
    ?assertEqual({ok, 3}, tinyllm_tuner:rollback(T)),
    ?assertEqual({ok, 0}, tinyllm_tuner:rollback(T)),
    close(T).

%% ---------------------------------------------------------------- one round at a time

reads_stay_answerable_during_a_long_round_test() ->
    T = start(#{probe_target => 1.0, delay => 400}, #{candidates => [0.1], steps => 1}),
    Me = self(),
    spawn_link(fun() -> Me ! {done, tinyllm_tuner:learn(T, [1.0])} end),
    timer:sleep(100),
    ?assertEqual({error, busy}, tinyllm_tuner:learn(T, [1.0])),
    ?assertEqual({error, busy}, tinyllm_tuner:rollback(T)),
    ?assertEqual(0, cur_id(T)),
    receive {done, R} -> ?assertMatch({committed, 1, _}, R) after 5000 -> ?assert(false) end,
    ?assertEqual(1, cur_id(T)),
    close(T).

%% ---------------------------------------------------------------- the port engine, against a shell worker

port_dir() ->
    D = filename:join(os:getenv("TMPDIR", "/tmp"),
                      "tinyllm_tuner_test_" ++ integer_to_list(erlang:unique_integer([positive]))),
    ok = filelib:ensure_path(D),
    D.

% The stand-in worker is written into the test's own temp dir, so the tests depend on no source path.
worker_script() ->
    lists:join("\n", [
      "#!/bin/sh",
      "# Deterministic stand-in for a real LoRA worker (MLX, llama.cpp ...), used by tinyllm_tuner_tests to test the",
      "# port plumbing. The \"adapter\" is a directory holding one number in file `w`; the first line of a data file is",
      "# its target T (a number, or the word `crash`).",
      "#   train: w <- w - lr * 2 * (w - T), repeated `steps` times; loss = (w - T)^2",
      "#   eval:  loss = (w - T)^2 for the data file's T          distance: |w_b - w_a|",
      "op=\"$1\"; shift",
      "adapter_in=none; adapter_out=\"\"; lr=0; steps=1; data=\"\"; a=none; b=none; adapter=none",
      "while [ $# -gt 1 ]; do",
      "  case \"$1\" in",
      "    --adapter-in) adapter_in=\"$2\" ;;",
      "    --adapter-out) adapter_out=\"$2\" ;;",
      "    --adapter) adapter=\"$2\" ;;",
      "    --lr) lr=\"$2\" ;;",
      "    --steps) steps=\"$2\" ;;",
      "    --data) data=\"$2\" ;;",
      "    --a) a=\"$2\" ;;",
      "    --b) b=\"$2\" ;;",
      "  esac",
      "  shift 2",
      "done",
      "weight() { if [ \"$1\" = none ] || [ ! -f \"$1/w\" ]; then echo 0; else cat \"$1/w\"; fi; }",
      "case \"$op\" in",
      "  train)",
      "    t=$(head -n 1 \"$data\")",
      "    if [ \"$t\" = crash ]; then echo \"worker exploded\" >&2; exit 3; fi",
      "    w1=$(awk -v w=\"$(weight \"$adapter_in\")\" -v t=\"$t\" -v lr=\"$lr\" -v n=\"$steps\" \\",
      "         'BEGIN { for (i = 0; i < n; i++) w = w - lr * 2 * (w - t); printf \"%.12g\\n\", w }')",
      "    echo \"$w1\" > \"$adapter_out/w\"",
      "    echo \"step noise that is not a result\"",
      "    awk -v w=\"$w1\" -v t=\"$t\" 'BEGIN { printf \"RESULT train_loss=%.12g\\n\", (w - t) ^ 2 }'",
      "    ;;",
      "  eval)",
      "    t=$(head -n 1 \"$data\")",
      "    awk -v w=\"$(weight \"$adapter\")\" -v t=\"$t\" 'BEGIN { printf \"RESULT loss=%.12g\\n\", (w - t) ^ 2 }'",
      "    ;;",
      "  distance)",
      "    awk -v x=\"$(weight \"$a\")\" -v y=\"$(weight \"$b\")\" 'BEGIN { d = y - x; if (d < 0) d = -d; printf \"RESULT dist=%.12g\\n\", d }'",
      "    ;;",
      "  *) echo \"unknown op $op\" >&2; exit 2 ;;",
      "esac"
    ]) ++ "\n".

port_opts(Dir) ->
    Worker = filename:join(Dir, "worker.sh"),
    ok = file:write_file(Worker, worker_script()),
    ok = file:write_file(filename:join(Dir, "heldout.txt"), "1.0\n"),
    ok = file:write_file(filename:join(Dir, "probe.txt"), "1.0\n"),
    #{cmd => "/bin/sh", args => [Worker], root => filename:join(Dir, "versions"),
      heldout_file => filename:join(Dir, "heldout.txt"), probe_file => filename:join(Dir, "probe.txt")}.

port_engine_train_eval_drift_release_test() ->
    Dir = port_dir(),
    {ok, St, none} = tinyllm_engine_port:init(port_opts(Dir)),
    {ok, L0} = tinyllm_engine_port:eval(St, none, heldout),
    near(L0, 1.0),
    {ok, H, #{train_loss := TL}} =
        tinyllm_engine_port:train(St, none, #{lr => 0.1, steps => 5, rank => 8, batch => [<<"1.0">>, "1.0"]}),
    near(TL, math:pow(0.8, 10)),
    {ok, L1} = tinyllm_engine_port:eval(St, H, heldout),
    near(L1, math:pow(0.8, 10)),
    {ok, D} = tinyllm_engine_port:drift(St, none, H),
    near(D, 1.0 - math:pow(0.8, 5)),
    ?assert(filelib:is_dir(H)),
    ok = tinyllm_engine_port:release(St, H),
    ?assertNot(filelib:is_dir(H)),
    ok = tinyllm_engine_port:release(St, none),
    file:del_dir_r(Dir).

port_engine_failures_test() ->
    Dir = port_dir(),
    {ok, St, none} = tinyllm_engine_port:init(port_opts(Dir)),
    Job = #{lr => 0.1, steps => 1, rank => 8},
    ?assertMatch({error, {exit, 3, [<<"worker exploded">>]}}, tinyllm_engine_port:train(St, none, Job#{batch => [<<"crash">>]})),
    ?assertMatch({error, {error, {newline_in_sample, _}}}, tinyllm_engine_port:train(St, none, Job#{batch => ["a\nb"]})),
    %% failed trains leave no half-written version directory behind
    ?assertEqual([], filelib:wildcard(filename:join(filename:join(Dir, "versions"), "*"))),
    ?assertMatch({error, {no_such_command, _}}, tinyllm_engine_port:init((port_opts(Dir))#{cmd => "no-such-worker-xyz"})),
    ?assertMatch({error, {missing_file, _}}, tinyllm_engine_port:init((port_opts(Dir))#{probe_file => "/no/such/file"})),
    file:del_dir_r(Dir).

port_engine_timeout_test() ->
    Dir = port_dir(),
    {ok, St, none} = tinyllm_engine_port:init((port_opts(Dir))#{args => ["-c", "sleep 5", "x"], op_timeout => 200}),
    ?assertEqual({error, timeout}, tinyllm_engine_port:eval(St, none, heldout)),
    file:del_dir_r(Dir).

port_engine_without_a_result_line_is_an_error_test() ->
    Dir = port_dir(),
    {ok, St, none} = tinyllm_engine_port:init((port_opts(Dir))#{args => ["-c", "echo no result here", "x"]}),
    ?assertEqual({error, no_result}, tinyllm_engine_port:eval(St, none, heldout)),
    {ok, St2, none} = tinyllm_engine_port:init((port_opts(Dir))#{args => ["-c", "echo RESULT loss=1e-3", "x"]}),
    {ok, F} = tinyllm_engine_port:eval(St2, none, heldout),
    near(F, 0.001),
    file:del_dir_r(Dir).

tuner_drives_the_port_engine_end_to_end_test() ->
    Dir = port_dir(),
    {ok, T} = tinyllm_tuner:start_link(#{engine => {tinyllm_engine_port, port_opts(Dir)},
                                         candidates => [0.05, 0.3], steps => 5, retain => 1}),
    {committed, 1, #{candidates := Cs}} = tinyllm_tuner:learn(T, [<<"1.0">>]),
    ?assertEqual(2, length(Cs)),
    {ok, #{lr := Lr, handle := H1}} = tinyllm_tuner:current(T),
    ?assertEqual(0.3, Lr),
    ?assert(filelib:is_dir(H1)),
    {committed, 2, _} = tinyllm_tuner:learn(T, [<<"1.0">>]),
    %% retain => 1 pruned version 1 from disk; only the winner of round 2 remains besides the base
    ?assertNot(filelib:is_dir(H1)),
    ?assertEqual(1, length(filelib:wildcard(filename:join(filename:join(Dir, "versions"), "*")))),
    ?assertEqual({ok, 0}, tinyllm_tuner:rollback(T, 1)),
    ?assertEqual([], filelib:wildcard(filename:join(filename:join(Dir, "versions"), "*"))),
    tinyllm_tuner:stop(T),
    file:del_dir_r(Dir).
