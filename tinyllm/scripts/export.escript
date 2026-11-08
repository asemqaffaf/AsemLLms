#!/usr/bin/env escript
%% Dump a trained model as plain JSON for packaging (see scripts/pack_release.py).
%%   rebar3 compile && escript scripts/export.escript chat_word.bin export/
main([ModelFile, OutDir]) ->
    [true = code:add_pathz(P) || P <- filelib:wildcard("_build/default/lib/*/ebin")],
    M = tinyllm:load(ModelFile),
    ok = filelib:ensure_dir(filename:join(OutDir, "x")),
    #{cfg := Cfg, vocab := Vocab, params := P, tokenizer := Tok} = M,
    Stats = maps:get(stats, M, #{}),
    %% tokens that are not valid UTF-8 on their own (byte fragments) are stored as {"hex": ...}
    Tokens = [case unicode:characters_to_binary(T) of
                  B when is_binary(B) -> B;
                  _ -> #{hex => binary:encode_hex(T, lowercase)}
              end || T <- Vocab],
    ok = file:write_file(filename:join(OutDir, "vocab.json"), json:encode(Tokens)),
    ok = file:write_file(filename:join(OutDir, "weights.json"), json:encode(P)),
    ok = file:write_file(filename:join(OutDir, "meta.json"),
                         json:encode(#{cfg => Cfg, tokenizer => Tok, stats => Stats})),
    %% a test vector so the Python loader can be checked against this Erlang implementation
    Ctx = lists:seq(1, maps:get(context, Cfg)),
    Logits = tinyllm_nn:logits(tinyllm_nn:prepare(P), Ctx),
    ok = file:write_file(filename:join(OutDir, "test_vector.json"), json:encode(#{context => Ctx, logits => Logits})),
    io:format("exported ~s to ~s (vocab ~p)~n", [ModelFile, OutDir, length(Vocab)]).
