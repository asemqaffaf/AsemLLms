%% Data-parallel training coordinator.
%% Each step: sample a batch of positions, shard it across supervised worker
%% processes (each computes loss + gradient for its shard), sum the gradients,
%% apply an Adam update. A worker that crashes is restarted by its supervisor
%% and its shard is retried, so training survives worker failures.
%% The parameters with the best validation loss are the ones returned.
-module(tinyllm_train).
-export([train/2, defaults/0]).

defaults() ->
    #{tokenizer => char,            %% char | word
      arch => mlp,                  %% mlp | transformer
      context => 8, embed => 12, hidden => 64,       %% mlp (context = tokens of history)
      dim => 64, heads => 2, layers => 2, ff => 128, %% transformer (context = sequence length)
      max_vocab => 2000, min_count => 2,   %% word tokenizer only
      workers => erlang:system_info(schedulers_online),
      batch => 256, steps => 1500, lr => 0.005, seed => 1,
      eval_every => 100, val_examples => 1000, max_bytes => 300000,
      clip => 5.0, verbose => true}.

train(Corpus0, Opts0) ->
    O = maps:merge(defaults(), Opts0),
    #{context := N, workers := W, batch := B, arch := Arch,
      steps := Steps, seed := Seed, max_bytes := MaxBytes, tokenizer := Tok} = O,
    Corpus = binary:part(Corpus0, 0, min(byte_size(Corpus0), MaxBytes)),
    Tokens = tinyllm_tok:tokenize(Tok, Corpus),
    Vocab = tinyllm_tok:build_vocab(Tok, Tokens, O),
    V = length(Vocab),
    V =< 65535 orelse error(vocab_too_large),
    VMap = maps:from_list(lists:zip(Vocab, lists:seq(0, V - 1))),
    Ids = << <<(maps:get(T, VMap, 0)):16>> || T <- Tokens >>,
    Size = length(Tokens),
    Unk = case Tok of word -> length([T || T <- Tokens, not maps:is_key(T, VMap)]) / max(1, Size); char -> 0.0 end,
    TrainEnd = trunc(Size * 0.9),
    (TrainEnd > N + 10 andalso Size - TrainEnd > 10) orelse error(corpus_too_small),
    %% transformer validation sequences must lie wholly inside the held-out part (all T targets are scored)
    ValFrom = case Arch of transformer -> TrainEnd + N; mlp -> max(TrainEnd, N) end,
    Size - ValFrom > 10 orelse error(corpus_too_small),
    Cfg = case Arch of
              mlp -> #{arch => mlp, vocab => V, context => N, embed => maps:get(embed, O), hidden => maps:get(hidden, O)};
              transformer -> #{arch => transformer, vocab => V, context => N, dim => maps:get(dim, O),
                               heads => maps:get(heads, O), layers => maps:get(layers, O), ff => maps:get(ff, O)}
          end,
    rand:seed(exsss, {Seed, Seed, Seed}),
    ValPos = val_positions(ValFrom, Size, maps:get(val_examples, O)),
    Baselines = baselines(Ids, TrainEnd, ValPos, V),
    persistent_term:put(tinyllm_corpus, #{ids => Ids, cfg => Cfg}),
    Workers = tinyllm_worker_sup:start_workers(W),
    T0 = erlang:monotonic_time(millisecond),
    try
        Params0 = case Arch of mlp -> tinyllm_nn:init(Cfg, Seed); transformer -> tinyllm_tf:init(Cfg, Seed) end,
        log(O, "~p tokens (~p), vocab ~p (~.1f% unknown), ~p ~p params, ~p workers; baselines val loss: unigram ~.3f, bigram ~.3f",
            [Size, Tok, V, Unk * 100, Arch, count_params(Params0), W, maps:get(unigram, Baselines),
             maps:get(bigram, Baselines)]),
        St = #{step => 1, params => Params0, adam => adam_init(Params0), acc_loss => 0.0, acc_n => 0,
               best_val => infinity, best_params => Params0},
        #{best_val := BestVal, best_params := Best} =
            loop(St, Steps, B, N, TrainEnd, ValPos, Workers, O, T0),
        Secs = (erlang:monotonic_time(millisecond) - T0) / 1000,
        {ok, #{cfg => Cfg, vocab => Vocab, tokenizer => Tok, params => Best,
               stats => Baselines#{val_loss => BestVal, steps => Steps, seconds => Secs,
                                   examples_per_sec => Steps * B / Secs, tokens => Size}}}
    after
        tinyllm_worker_sup:stop_workers(W),
        persistent_term:erase(tinyllm_corpus)
    end.

loop(#{step := Step} = St, Steps, B, N, TrainEnd, ValPos, Workers, O, T0) ->
    case Step > Steps of
        true -> checkpoint(St, Workers, ValPos);
        false -> loop(do_step(St, B, N, TrainEnd, ValPos, Workers, O, T0, Steps), Steps, B, N, TrainEnd, ValPos, Workers, O, T0)
    end.

do_step(#{step := Step, params := Params, adam := Adam, acc_loss := AL, acc_n := AN} = St,
        B, N, TrainEnd, ValPos, Workers, O, T0, Steps) ->
    Positions = [N + rand:uniform(TrainEnd - N) - 1 || _ <- lists:seq(1, B)],
    Results = par(Workers, fun(W, S) -> tinyllm_worker:grad(W, Params, S) end, Positions),
    {LossSum, Count, GradSum} = combine(Results),
    Mean = maps:map(fun(_, G) -> tinyllm_nn:tmap(fun(X) -> X / Count end, G) end, GradSum),
    Clipped = clip(Mean, maps:get(clip, O)),
    LR = maps:get(lr, O) * (1.0 - 0.9 * (Step - 1) / max(1, Steps - 1)),
    {Params1, Adam1} = adam(Params, Clipped, Adam, LR),
    St1 = St#{step := Step + 1, params := Params1, adam := Adam1,
              acc_loss := AL + LossSum, acc_n := AN + Count},
    case Step rem maps:get(eval_every, O) =:= 0 of
        false -> St1;
        true ->
            St2 = checkpoint(St1, Workers, ValPos),
            Secs = max(0.001, (erlang:monotonic_time(millisecond) - T0) / 1000),
            Val = maps:get(last_val, St2),
            log(O, "step ~5w/~w  train ~.3f  val ~.3f  best ~.3f  ~.1f ex/s",
                [Step, Steps, maps:get(acc_loss, St1) / maps:get(acc_n, St1), Val,
                 maps:get(best_val, St2), Step * B / Secs]),
            St2#{acc_loss := 0.0, acc_n := 0}
    end.

%% Evaluate on the validation set and remember the best parameters so far.
checkpoint(#{params := Params, best_val := Best} = St, Workers, ValPos) ->
    Val = eval(Workers, Params, ValPos),
    case Best =:= infinity orelse Val < Best of
        true -> St#{best_val := Val, best_params := Params, last_val => Val};
        false -> St#{last_val => Val}
    end.

eval(Workers, Params, ValPos) ->
    Rs = par(Workers, fun(W, S) -> tinyllm_worker:loss(W, Params, S) end, ValPos),
    lists:sum([L || {L, _} <- Rs]) / lists:sum([C || {_, C} <- Rs]).

%% ---------------------------------------------------------- parallel step

par(Workers, Fun, Positions) ->
    Shards = shards(Positions, length(Workers)),
    Tasks = [spawn_monitor(fun() -> exit({ok, retry(fun() -> Fun(W, S) end, 8)}) end)
             || {W, S} <- lists:zip(Workers, Shards), S =/= []],
    [receive
         {'DOWN', M, process, P, {ok, R}} -> R;
         {'DOWN', M, process, P, Why} -> error({worker_failed, Why})
     end || {P, M} <- Tasks].

retry(F, 0) -> F();
retry(F, N) ->
    try F()
    catch exit:_ -> timer:sleep(300), retry(F, N - 1) end.

shards(Positions, W) ->
    Indexed = lists:zip(Positions, lists:seq(0, length(Positions) - 1)),
    [[X || {X, I} <- Indexed, I rem W =:= K] || K <- lists:seq(0, W - 1)].

combine([{L, C, G} | Rest]) ->
    lists:foldl(fun({L2, C2, G2}, {LA, CA, GA}) ->
                    {LA + L2, CA + C2,
                     maps:map(fun(K, V) -> tinyllm_nn:zw(fun(A, B) -> A + B end, V, maps:get(K, G2)) end, GA)}
                end, {L, C, G}, Rest).

%% --------------------------------------------------------------- optimizer

adam_init(Params) ->
    Z = maps:map(fun(_, V) -> tinyllm_nn:zeros_like(V) end, Params),
    {Z, Z, 0}.

-define(B1, 0.9).
-define(B2, 0.999).

adam(Params, Grads, {M, V, T}, LR) ->
    T1 = T + 1,
    M1 = maps:map(fun(K, Mk) -> tinyllm_nn:zw(fun(Mi, G) -> ?B1 * Mi + (1.0 - ?B1) * G end, Mk, maps:get(K, Grads)) end, M),
    V1 = maps:map(fun(K, Vk) -> tinyllm_nn:zw(fun(Vi, G) -> ?B2 * Vi + (1.0 - ?B2) * G * G end, Vk, maps:get(K, Grads)) end, V),
    LRt = LR * math:sqrt(1.0 - math:pow(?B2, T1)) / (1.0 - math:pow(?B1, T1)),
    P1 = maps:map(fun(K, Pk) ->
                      U = tinyllm_nn:zw(fun(Mi, Vi) -> Mi / (math:sqrt(Vi) + 1.0e-8) end, maps:get(K, M1), maps:get(K, V1)),
                      tinyllm_nn:zw(fun(P, Ui) -> P - LRt * Ui end, Pk, U)
                  end, Params),
    {P1, {M1, V1, T1}}.

clip(Grads, Max) ->
    Norm = math:sqrt(tinyllm_nn:sumsq(Grads)),
    case Norm > Max of
        true -> maps:map(fun(_, G) -> tinyllm_nn:tmap(fun(X) -> X * Max / Norm end, G) end, Grads);
        false -> Grads
    end.

count_params(Params) ->
    maps:fold(fun(_, V, A) -> A + count(V) end, 0, Params).
count([A | As]) when is_list(A) -> count(A) + count(As);
count([_ | As]) -> 1 + count(As);
count([]) -> 0.

%% -------------------------------------------------------------- baselines
%% Unigram / bigram models fitted on the train split, scored on the same
%% validation positions - the bar a neural model has to clear.

val_positions(From, Size, Count) ->
    %% random (but fixed for the run) sample - evenly spaced picks can alias with periodic text
    lists:usort([From + rand:uniform(Size - From) - 1 || _ <- lists:seq(1, Count)]).

baselines(Ids, TrainEnd, ValPos, V) ->
    Train = binary:part(Ids, 0, TrainEnd * 2),
    {Uni, Bi} = count_ngrams(Train),
    ULoss = lists:sum([-math:log((maps:get(tinyllm_nn:tok_at(Ids, I), Uni, 0) + 1) / (TrainEnd + V)) || I <- ValPos]),
    BLoss = lists:sum([begin
                           P = tinyllm_nn:tok_at(Ids, I - 1), T = tinyllm_nn:tok_at(Ids, I),
                           -math:log((maps:get({P, T}, Bi, 0) + 0.1) / (maps:get(P, Uni, 0) + 0.1 * V))
                       end || I <- ValPos]),
    N = length(ValPos),
    #{unigram => ULoss / N, bigram => BLoss / N}.

count_ngrams(Bin) -> count_ngrams(Bin, none, #{}, #{}).
count_ngrams(<<C:16, Rest/binary>>, Prev, Uni, Bi) ->
    Uni1 = maps:update_with(C, fun(X) -> X + 1 end, 1, Uni),
    Bi1 = case Prev of
              none -> Bi;
              _ -> maps:update_with({Prev, C}, fun(X) -> X + 1 end, 1, Bi)
          end,
    count_ngrams(Rest, C, Uni1, Bi1);
count_ngrams(<<>>, _, Uni, Bi) -> {Uni, Bi}.

log(#{verbose := true}, Fmt, Args) -> io:format(Fmt ++ "~n", Args);
log(_, _, _) -> ok.
