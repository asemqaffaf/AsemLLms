%% Deterministic stand-in engine for tinyllm_tuner's tests: the "model" is one number W.
%%
%%   train   gradient descent on (W - mean(batch))^2, `steps` steps of size `lr`:  W <- W - lr * 2 * (W - M)
%%   heldout loss = (W - heldout_target)^2        probe loss = (W - probe_target)^2
%%   drift   = |W_to - W_from|
%%
%% Pulling W toward a batch mean that is far from probe_target makes the probe loss rise, which is exactly the
%% forgetting the tuner has to catch; a large lr overshoots, which is what its candidate search has to avoid.
%%
%% Failure injection: lr < 0 returns {error, bad_lr}; lr == 99.0 raises; a batch that contains the atom
%% `crash` raises; a batch that contains `bad` returns {error, bad_batch}.
%%
%% Opts: start (0.0), heldout_target (1.0), probe_target (0.0), delay (ms slept in train, 0),
%%       notify (pid that gets {released, Handle} from release/2 and {trained_from, FromHandle, Lr} from train/3).
-module(tinyllm_engine_fake).
-behaviour(tinyllm_engine).
-export([init/1, train/3, eval/3, drift/3, release/2, weight/1]).

init(Opts) ->
    W0 = float(maps:get(start, Opts, 0.0)),
    {ok, Opts#{start => W0}, {fake, W0}}.

train(St, {fake, W0} = From, #{lr := Lr, steps := Steps, batch := Batch}) ->
    notify(St, {trained_from, From, Lr}),
    case maps:get(delay, St, 0) of 0 -> ok; D -> timer:sleep(D) end,
    case {Lr, lists:member(crash, Batch), lists:member(bad, Batch)} of
        {99.0, _, _} -> error(boom);
        {_, true, _} -> error(boom);
        {_, _, true} -> {error, bad_batch};
        {L, _, _} when L < 0 -> {error, bad_lr};
        _ ->
            M = lists:sum(Batch) / length(Batch),
            W = lists:foldl(fun(_, Acc) -> Acc - Lr * 2 * (Acc - M) end, W0, lists:seq(1, Steps)),
            {ok, {fake, W}, #{train_loss => (W - M) * (W - M)}}
    end.

eval(St, {fake, W}, heldout) -> {ok, sq(W - float(maps:get(heldout_target, St, 1.0)))};
eval(St, {fake, W}, probe) -> {ok, sq(W - float(maps:get(probe_target, St, 0.0)))}.

drift(_St, {fake, A}, {fake, B}) -> {ok, abs(B - A)}.

release(St, Handle) -> notify(St, {released, Handle}), ok.

notify(St, Msg) ->
    case maps:get(notify, St, undefined) of
        Pid when is_pid(Pid) -> Pid ! Msg, ok;
        _ -> ok
    end.

weight({fake, W}) -> W.

sq(X) -> X * X.
