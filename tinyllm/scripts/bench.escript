#!/usr/bin/env escript
%% Throughput benchmark of the pure-Erlang transformer trainer.
%%   rebar3 compile && escript scripts/bench.escript [dim layers heads ctx batch steps] [workers]
%% Prints one line of JSON: params, tokens/s, achieved FLOP/s (6 * matmul params per token + attention).
%% Trains on ../dataset/chat_clean.txt (word tokenizer); throughput does not depend on the text.
main(Args) ->
    [Dim, Layers, Heads, Ctx, Batch, Steps] =
        case Args of
            [A, B, C, D, E, F | _] -> [list_to_integer(X) || X <- [A, B, C, D, E, F]];
            _ -> [64, 2, 4, 32, 8, 30]
        end,
    Workers = case Args of [_, _, _, _, _, _, W] -> list_to_integer(W); _ -> erlang:system_info(schedulers_online) end,
    [true = code:add_pathz(P) || P <- filelib:wildcard("_build/default/lib/*/ebin")],
    {ok, _} = application:ensure_all_started(tinyllm),
    {ok, Corpus} = file:read_file("../dataset/chat_clean.txt"),
    {ok, M} = tinyllm:train(Corpus, #{tokenizer => word, arch => transformer, context => Ctx, dim => Dim,
                                      heads => Heads, layers => Layers, ff => 2 * Dim, batch => Batch,
                                      steps => Steps, lr => 0.003, max_vocab => 2000, min_count => 2,
                                      eval_every => Steps, val_examples => 8, workers => Workers,
                                      verbose => false}),
    Stats = maps:get(stats, M),
    Params = tinyllm:param_count(M),
    V = length(maps:get(vocab, M)),
    TokPerSec = maps:get(examples_per_sec, Stats) * Ctx,
    %% the Erlang model has untied embeddings and positions; the head matmul is V*Dim, blocks are 4*Dim^2 + 2*Dim*FF per layer
    MatmulParams = Layers * (4 * Dim * Dim + 2 * Dim * (2 * Dim)) + V * Dim,
    FlopsPerTok = 6 * MatmulParams + 12 * Layers * Ctx * Dim,
    io:format("{\"impl\":\"erlang\",\"params\":~p,\"vocab\":~p,\"dim\":~p,\"layers\":~p,\"ctx\":~p,\"batch\":~p,\"workers\":~p,"
              "\"tokens_per_sec\":~.1f,\"flops_per_sec\":~.3e}~n",
              [Params, V, Dim, Layers, Ctx, Batch, Workers, TokPerSec, TokPerSec * FlopsPerTok]).
