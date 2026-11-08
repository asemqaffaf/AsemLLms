#!/usr/bin/env escript
%% Dump a trained model as plain JSON for packaging (see scripts/pack_release.py).
%%   rebar3 compile && escript scripts/export.escript chat_word.bin export/
%% Works for both architectures (mlp, transformer).
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
    %% parameter keys like {wq, 1} become "wq.1" (JSON keys must be strings)
    Flat = maps:from_list([{key(K), V} || {K, V} <- maps:to_list(P)]),
    ok = file:write_file(filename:join(OutDir, "weights.json"), json:encode(Flat)),
    ok = file:write_file(filename:join(OutDir, "meta.json"),
                         json:encode(#{cfg => Cfg, tokenizer => Tok, stats => Stats})),
    %% test vectors so the Python loader can be checked against this Erlang implementation
    Vectors = case maps:get(arch, Cfg, mlp) of
                  mlp ->
                      Ctx = lists:seq(1, maps:get(context, Cfg)),
                      [#{context => Ctx, logits => tinyllm_nn:logits(tinyllm_nn:prepare(P), Ctx)}];
                  transformer ->
                      T = maps:get(context, Cfg), V = maps:get(vocab, Cfg),
                      [begin
                           Ctx = [(I * 7 + 3) rem V || I <- lists:seq(1, Len)],
                           #{context => Ctx, logits => tinyllm_tf:last_logits(tinyllm_tf:prepare(P), Cfg, Ctx)}
                       end || Len <- [1, 5, T]]
              end,
    ok = file:write_file(filename:join(OutDir, "test_vectors.json"), json:encode(Vectors)),
    io:format("exported ~s to ~s (vocab ~p)~n", [ModelFile, OutDir, length(Vocab)]).

key({Name, I}) -> iolist_to_binary([atom_to_binary(Name), ".", integer_to_list(I)]);
key(Name) -> atom_to_binary(Name).
