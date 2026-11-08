#!/usr/bin/env escript
%% Train and save a model, then print samples.
%%   rebar3 compile && escript scripts/train.escript [steps] [model-file] [corpus-file] [char|word|transformer]
%% With no corpus file it trains on OTP's stdlib source; with one, on that text
%% (e.g. ../dataset/chat.txt from the LM Studio dataset generator).
main(Args) ->
    Steps = case Args of [S | _] -> list_to_integer(S); _ -> 3000 end,
    Out = case Args of [_, P | _] -> P; _ -> "model.bin" end,
    [true = code:add_pathz(P) || P <- filelib:wildcard("_build/default/lib/*/ebin")],
    {ok, _} = application:ensure_all_started(tinyllm),
    {Corpus, Opts, Prompts} =
        case Args of
            [_, _, File, "transformer" | _] ->
                {ok, B} = file:read_file(File),
                {B, #{tokenizer => word, arch => transformer, context => 32, dim => 64, heads => 4, layers => 2,
                      ff => 128, lr => 0.003, max_vocab => 2000, min_count => 2, batch => 8,
                      eval_every => 100, val_examples => 150},
                 [<<"User: How do I cook pasta?\nAssistant:">>, <<"User: Hello!\nAssistant:">>,
                  <<"User: What should I plant in my garden?\nAssistant:">>]};
            [_, _, File, "word" | _] ->
                {ok, B} = file:read_file(File),
                {B, #{tokenizer => word, context => 5, embed => 24, hidden => 64, lr => 0.005,
                      max_vocab => 2000, min_count => 2, batch => 128},
                 [<<"User: How do I cook pasta?\nAssistant:">>, <<"User: Hello!\nAssistant:">>,
                  <<"User: What should I plant in my garden?\nAssistant:">>]};
            [_, _, File | _] ->
                {ok, B} = file:read_file(File),
                {B, #{context => 16, embed => 16, hidden => 128, lr => 0.004},
                 [<<"User: How do I cook pasta?\nAssistant:">>, <<"User: Hello!\nAssistant:">>,
                  <<"User: What should I plant in my garden?\nAssistant:">>]};
            _ ->
                {tinyllm:erlang_corpus(), #{context => 10, embed => 16, hidden => 96, lr => 0.004},
                 [<<"-module(">>, <<"foo(X) ->\n    ">>]}
        end,
    io:format("corpus: ~p bytes~n", [byte_size(Corpus)]),
    {ok, M} = tinyllm:train(Corpus, maps:merge(#{batch => 256, eval_every => 250}, Opts#{steps => Steps, max_bytes => 2000000})),
    ok = tinyllm:save(M, Out),
    io:format("~nsaved ~s; stats: ~p~n", [Out, maps:get(stats, M)]),
    [io:format("~n--- prompt ~p ---~n~s~n",
               [P, tinyllm:generate(M, P, 250, #{temperature => 0.6, seed => 42})]) || P <- Prompts],
    ok.
