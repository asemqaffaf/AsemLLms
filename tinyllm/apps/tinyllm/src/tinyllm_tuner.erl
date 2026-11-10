%% Propose, score, commit or roll back: the supervisor around any tinyllm_engine.
%%
%% learn(Tuner, Batch) runs one round:
%%   1. propose  train one candidate per learning rate in `candidates`, each starting from the SAME handle (the
%%               current version), one after another by default (parallel => true runs them concurrently);
%%   2. score    each candidate gets a heldout loss, a probe loss and its drift from version 0;
%%   3. gate     a candidate is kept only if  heldout < current_heldout - min_gain
%%                                      and   probe   =< current_probe + probe_tolerance
%%                                      and   drift   =< drift_cap;
%%   4. control  (optional, `control`) a passing candidate must also beat a CONTROL trained from the same
%%               start, at the same learning rate, on a meaningless copy of the batch (`control => shuffle`
%%               shuffles the words of every sample): heldout < control_heldout - control_margin. Part of any
%%               heldout gain is generic (the model adapting to the style or format of short raw text), so
%%               "loss fell" alone also passes garbage; the control measures what the batch teaches beyond that.
%%   5. commit   the passing candidate with the lowest heldout loss becomes the new current version; the rest
%%               are released. If none passes, nothing changes and the reason of the best-heldout candidate
%%               is returned.
%% The last `retain` versions plus version 0 are kept; rollback/1,2 steps back through them.
%%
%% The engine calls run in their own processes, so a crash, a hang (`timeout`, ms per candidate) or a bad
%% return costs one candidate, never the tuner. A round runs in a worker, so current/1, versions/1, log/1 and
%% rollback attempts stay answerable while a long train job is going (a second learn gets {error, busy}).
-module(tinyllm_tuner).
-behaviour(gen_server).

-export([start_link/1, start_link/2, stop/1]).
-export([learn/2, learn/3, rollback/1, rollback/2, current/1, versions/1, log/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(LOG_MAX, 200).

-ifdef(TEST).
-export([control_batch/2]).
-endif.

-record(ver, {id, handle, heldout, probe, drift, lr, at}).
-record(st, {mod, es, cfg, vers, next = 1, busy, log = []}).

%% ---------------------------------------------------------------- API

%% Opts: engine := {Module, EngineOpts}
%%       candidates (learning rates, [1.0e-4, 3.0e-4, 1.0e-3]), steps (20), rank (8),
%%       retain (5), min_gain (0.0), probe_tolerance (0.0), drift_cap (infinity),
%%       control (none | shuffle | fun(Batch) -> Batch), control_margin (0.0),
%%       parallel (false: one candidate at a time, so one GPU/MLX job runs at once; true trains all candidates
%%       concurrently and needs memory for all of them), timeout (per candidate, 600000 ms)
start_link(Opts) -> gen_server:start_link(?MODULE, Opts, []).
start_link(Name, Opts) when is_atom(Name) -> gen_server:start_link({local, Name}, ?MODULE, Opts, []).
stop(T) -> gen_server:stop(T).

%% {committed, Id, Info} | {rejected, Reason, Info} | {error, Reason}
%% Reason of a rejection: no_improvement | probe_regressed | drift_cap | no_better_than_control.
%% Info: #{candidates := [#{lr, heldout, probe, drift, train_loss}], failures := [{Lr, Why}],
%%         controls => [#{lr, heldout, probe}]   (only with `control`)}
learn(T, Batch) -> learn(T, Batch, infinity).
learn(T, Batch, Timeout) when is_list(Batch), Batch =/= [] -> gen_server:call(T, {learn, Batch}, Timeout).

%% Drop the newest committed version(s). {ok, CurrentId} | {error, nothing_to_roll_back | not_enough_versions | busy}
rollback(T) -> rollback(T, 1).
rollback(T, N) when is_integer(N), N >= 1 -> gen_server:call(T, {rollback, N}).

current(T) -> gen_server:call(T, current).
versions(T) -> gen_server:call(T, versions).
log(T) -> gen_server:call(T, log).

%% ---------------------------------------------------------------- gen_server

init(Opts) ->
    {Mod, EOpts} = maps:get(engine, Opts),
    Cfg = maps:merge(defaults(), maps:remove(engine, Opts)),
    case validate(Cfg) of
        ok ->
            case catch Mod:init(EOpts) of
                {ok, ES, Base} ->
                    case catch baseline(Mod, ES, Base) of
                        {ok, Ho, Pr} ->
                            V0 = #ver{id = 0, handle = Base, heldout = Ho, probe = Pr, drift = 0.0, at = now_s()},
                            {ok, #st{mod = Mod, es = ES, cfg = Cfg, vers = [V0]}};
                        Err -> {stop, {baseline_failed, Err}}
                    end;
                Err -> {stop, {engine_init_failed, Err}}
            end;
        {error, R} -> {stop, {bad_options, R}}
    end.

handle_call({learn, _}, _From, #st{busy = B} = S) when B =/= undefined ->
    {reply, {error, busy}, S};
handle_call({learn, Batch}, From, #st{mod = Mod, es = ES, cfg = Cfg, vers = Vers} = S) ->
    Parent = self(),
    Cur = hd(Vers),
    Base = lists:last(Vers),
    {Pid, MRef} = spawn_monitor(fun() ->
        Result = try do_round(Mod, ES, Cfg, Cur, Base, Batch)
                 catch C:R -> {error, {worker_crashed, C, R}} end,
        Parent ! {learn_done, self(), Result}
    end),
    {noreply, S#st{busy = {Pid, MRef, From}}};

handle_call({rollback, _}, _From, #st{busy = B} = S) when B =/= undefined ->
    {reply, {error, busy}, S};
handle_call({rollback, N}, _From, #st{vers = Vers} = S) ->
    Rollable = length(Vers) - 1,
    if
        Rollable =:= 0 -> {reply, {error, nothing_to_roll_back}, S};
        N > Rollable -> {reply, {error, not_enough_versions}, S};
        true ->
            {Drop, Keep} = lists:split(N, Vers),
            [release(S, V#ver.handle) || V <- Drop],
            Id = (hd(Keep))#ver.id,
            S1 = add_log(#{event => rollback, dropped => [V#ver.id || V <- Drop], current => Id}, S),
            {reply, {ok, Id}, S1#st{vers = Keep}}
    end;

handle_call(current, _From, #st{vers = [Cur | _]} = S) -> {reply, {ok, summary(Cur)}, S};
handle_call(versions, _From, #st{vers = Vers} = S) -> {reply, [summary(V) || V <- Vers], S};
handle_call(log, _From, #st{log = L} = S) -> {reply, L, S}.

handle_cast(_, S) -> {noreply, S}.

handle_info({learn_done, Pid, Result}, #st{busy = {Pid, MRef, From}} = S) ->
    erlang:demonitor(MRef, [flush]),
    {Reply, S1} = finish(Result, S#st{busy = undefined}),
    gen_server:reply(From, Reply),
    {noreply, S1};
handle_info({'DOWN', MRef, process, Pid, Why}, #st{busy = {Pid, MRef, From}} = S) ->
    gen_server:reply(From, {error, {worker_down, Why}}),
    {noreply, add_log(#{event => error, reason => {worker_down, Why}}, S#st{busy = undefined})};
handle_info(_, S) -> {noreply, S}.

terminate(_, _) -> ok.

%% ---------------------------------------------------------------- finishing a round

finish({commit, Cand, Info}, #st{vers = Vers, next = Id, cfg = Cfg} = S) ->
    V = #ver{id = Id, handle = maps:get(handle, Cand), heldout = maps:get(heldout, Cand),
             probe = maps:get(probe, Cand), drift = maps:get(drift, Cand), lr = maps:get(lr, Cand), at = now_s()},
    Vers1 = prune([V | Vers], maps:get(retain, Cfg), S),
    S1 = add_log(#{event => commit, id => Id, lr => V#ver.lr, heldout => V#ver.heldout, probe => V#ver.probe,
                   drift => V#ver.drift}, S#st{vers = Vers1, next = Id + 1}),
    {{committed, Id, Info}, S1};
finish({reject, Reason, Info}, S) ->
    {{rejected, Reason, Info}, add_log(#{event => reject, reason => Reason}, S)};
finish({error, Reason}, S) ->
    {{error, Reason}, add_log(#{event => error, reason => Reason}, S)}.

%% Keep the newest `Retain` committed versions and version 0; release the rest.
prune(Vers, Retain, S) ->
    {NonBase, [Base]} = lists:split(length(Vers) - 1, Vers),
    {Keep, Drop} = case length(NonBase) > Retain of
                       true -> lists:split(Retain, NonBase);
                       false -> {NonBase, []}
                   end,
    [release(S, V#ver.handle) || V <- Drop],
    Keep ++ [Base].

%% ---------------------------------------------------------------- one round (runs in a worker process)

do_round(Mod, ES, Cfg, Cur, Base, Batch) ->
    #{candidates := Lrs, steps := Steps, rank := Rank, parallel := Par, timeout := TO} = Cfg,
    Job = #{steps => Steps, rank => Rank, batch => Batch},
    Funs = [fun() -> candidate(Mod, ES, Cur, Base, Job#{lr => Lr}) end || Lr <- Lrs],
    Results = lists:zip(Lrs, run_all(Funs, Par, TO)),
    Ok = [C || {_, {ok, C}} <- Results],
    Failures = [{Lr, Why} || {Lr, {error, Why}} <- Results],
    case Ok of
        [] -> {error, {all_candidates_failed, Failures}};
        _ ->
            Info0 = #{candidates => [strip(C) || C <- Ok], failures => Failures},
            Judged = [{gate(C, Cur, Cfg), C} || C <- sort_by_heldout(Ok)],
            case [C || {pass, C} <- Judged] of
                [] ->
                    [{{fail, Reason}, _} | _] = Judged,
                    [release_cand(Mod, ES, C) || C <- Ok],
                    {reject, Reason, Info0};
                Passing ->
                    case pick(Mod, ES, Cfg, Cur, Base, Job, Passing, []) of
                        {ok, Best, Controls} ->
                            [release_cand(Mod, ES, C) || C <- Ok, C =/= Best],
                            {commit, Best, with_controls(Info0, Controls)};
                        {no, Controls} ->
                            [release_cand(Mod, ES, C) || C <- Ok],
                            {reject, no_better_than_control, with_controls(Info0, Controls)}
                    end
            end
    end.

%% Passing candidates, best heldout first: the first one that also beats its control wins.
pick(_Mod, _ES, _Cfg, _Cur, _Base, _Job, [], Controls) -> {no, Controls};
pick(_Mod, _ES, #{control := none}, _Cur, _Base, _Job, [C | _], _) -> {ok, C, []};
pick(Mod, ES, #{control := Ctl, control_margin := Margin} = Cfg, Cur, Base, Job, [C | Rest], Controls) ->
    Lr = maps:get(lr, C),
    case candidate(Mod, ES, Cur, Base, Job#{lr => Lr, batch => control_batch(Ctl, maps:get(batch, Job))}) of
        {ok, K} ->
            release_cand(Mod, ES, K),
            Controls1 = Controls ++ [maps:with([lr, heldout, probe], K)],
            case maps:get(heldout, C) < maps:get(heldout, K) - Margin of
                true -> {ok, C, Controls1};
                false -> pick(Mod, ES, Cfg, Cur, Base, Job, Rest, Controls1)
            end;
        {error, _} ->
            %% a control that cannot be trained is no evidence: do not commit on it
            pick(Mod, ES, Cfg, Cur, Base, Job, Rest, Controls)
    end.

with_controls(Info, []) -> Info;
with_controls(Info, Controls) -> Info#{controls => Controls}.

control_batch(shuffle, Batch) -> [shuffle_words(X) || X <- Batch];
control_batch(Fun, Batch) when is_function(Fun, 1) -> Fun(Batch).

%% Deterministic word shuffle of a binary (other terms pass through): same words, no meaning.
shuffle_words(Bin) when is_binary(Bin) ->
    Words = binary:split(Bin, <<" ">>, [global]),
    {_, Keyed} = lists:foldl(fun(W, {St, Acc}) -> {R, St1} = rand:uniform_s(St), {St1, [{R, W} | Acc]} end,
                             {rand:seed_s(exsss, {1, 2, 3}), []}, Words),
    iolist_to_binary(lists:join(<<" ">>, [W || {_, W} <- lists:sort(Keyed)]));
shuffle_words(Other) -> Other.

candidate(Mod, ES, Cur, Base, Job) ->
    try Mod:train(ES, Cur#ver.handle, Job) of
        {ok, H, M} when is_map(M) ->
            try scores(Mod, ES, H, Base#ver.handle) of
                {ok, Ho, Pr, Dr} ->
                    {ok, #{handle => H, lr => maps:get(lr, Job), heldout => Ho, probe => Pr, drift => Dr,
                           train_loss => maps:get(train_loss, M, undefined)}};
                {error, R} -> catch Mod:release(ES, H), {error, R}
            catch C:R -> catch Mod:release(ES, H), {error, {C, R}}
            end;
        {error, R} -> {error, R};
        Other -> {error, {bad_return, Other}}
    catch C:R -> {error, {C, R}}
    end.

scores(Mod, ES, H, BaseH) ->
    with_number(Mod:eval(ES, H, heldout), fun(Ho) ->
    with_number(Mod:eval(ES, H, probe), fun(Pr) ->
    with_number(Mod:drift(ES, BaseH, H), fun(Dr) -> {ok, Ho, Pr, Dr} end) end) end).

with_number({ok, N}, K) when is_number(N) -> K(N);
with_number({ok, N}, _) -> {error, {bad_metric, N}};
with_number({error, R}, _) -> {error, R};
with_number(Other, _) -> {error, {bad_return, Other}}.

gate(#{heldout := Ho, probe := Pr, drift := Dr}, #ver{heldout = CHo, probe = CPr},
     #{min_gain := Gain, probe_tolerance := Tol, drift_cap := Cap}) ->
    if
        not (Ho < CHo - Gain) -> {fail, no_improvement};
        Pr > CPr + Tol -> {fail, probe_regressed};
        Cap =/= infinity andalso Dr > Cap -> {fail, drift_cap};
        true -> pass
    end.

sort_by_heldout(Cs) -> lists:sort(fun(A, B) -> maps:get(heldout, A) =< maps:get(heldout, B) end, Cs).

strip(C) -> maps:remove(handle, C).

release_cand(Mod, ES, C) -> catch Mod:release(ES, maps:get(handle, C)), ok.

%% ---------------------------------------------------------------- isolated candidate runs

run_all(Funs, true, TO) ->
    Deadline = mono() + TO,
    Started = [start(F) || F <- Funs],
    [collect(P, Deadline) || P <- Started];
run_all(Funs, false, TO) ->
    [collect(start(F), mono() + TO) || F <- Funs].

start(F) ->
    Self = self(),
    Ref = make_ref(),
    {Pid, MRef} = spawn_monitor(fun() -> Self ! {Ref, F()} end),
    {Ref, Pid, MRef}.

collect({Ref, Pid, MRef}, Deadline) ->
    receive
        {Ref, R} -> erlang:demonitor(MRef, [flush]), R;
        {'DOWN', MRef, process, Pid, Why} -> {error, {down, Why}}
    after max(0, Deadline - mono()) ->
        exit(Pid, kill),
        erlang:demonitor(MRef, [flush]),
        receive {Ref, _} -> ok after 0 -> ok end,
        {error, timeout}
    end.

%% ---------------------------------------------------------------- helpers

defaults() ->
    #{candidates => [1.0e-4, 3.0e-4, 1.0e-3], steps => 20, rank => 8, retain => 5, min_gain => 0.0,
      probe_tolerance => 0.0, drift_cap => infinity, control => none, control_margin => 0.0, parallel => false, timeout => 600000}.

validate(#{candidates := [_ | _] = Lrs, retain := R, steps := St, rank := Rk, timeout := T,
           control := Ctl, control_margin := M})
  when is_integer(R), R >= 1, is_integer(St), St >= 1, is_integer(Rk), Rk >= 1, is_integer(T), T >= 1,
       is_number(M) ->
    case {lists:all(fun is_number/1, Lrs), Ctl =:= none orelse Ctl =:= shuffle orelse is_function(Ctl, 1)} of
        {true, true} -> ok;
        {false, _} -> {error, candidates};
        {_, false} -> {error, control}
    end;
validate(_) -> {error, "need candidates [..], retain >= 1, steps >= 1, rank >= 1, timeout >= 1, "
                       "control none | shuffle | fun/1, control_margin a number"}.

baseline(Mod, ES, Base) ->
    with_number(Mod:eval(ES, Base, heldout), fun(Ho) ->
    with_number(Mod:eval(ES, Base, probe), fun(Pr) -> {ok, Ho, Pr} end) end).

release(#st{mod = Mod, es = ES}, Handle) -> catch Mod:release(ES, Handle), ok.

summary(#ver{} = V) ->
    #{id => V#ver.id, handle => V#ver.handle, heldout => V#ver.heldout, probe => V#ver.probe,
      drift => V#ver.drift, lr => V#ver.lr, at => V#ver.at}.

add_log(Ev, #st{log = L} = S) ->
    S#st{log = lists:sublist([Ev#{at => now_s()} | L], ?LOG_MAX)}.

now_s() -> erlang:system_time(second).
mono() -> erlang:monotonic_time(millisecond).
