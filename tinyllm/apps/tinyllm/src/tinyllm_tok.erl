%% Tokenizers. `char`: one token per byte. `word`: words (with their leading
%% space), punctuation marks and newlines, so detokenize(tokenize(T)) =:= T.
-module(tinyllm_tok).
-export([tokenize/2, detokenize/1, build_vocab/3, unk/0]).

unk() -> <<"<unk>">>.

tokenize(char, Bin) -> [<<B>> || <<B>> <= Bin];
tokenize(word, Bin) ->
    case re:run(Bin, " ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']|\\n|\\s", [global, {capture, first, binary}]) of
        {match, Ms} -> [M || [M] <- Ms];
        nomatch -> []
    end.

detokenize(Tokens) -> iolist_to_binary(Tokens).

%% char: every distinct token. word: <unk> (id 0) + the most frequent tokens
%% seen at least `min_count` times, at most `max_vocab` in total.
build_vocab(char, Tokens, _) -> lists:usort(Tokens);
build_vocab(word, Tokens, #{max_vocab := Max, min_count := Min}) ->
    Counts = lists:foldl(fun(T, M) -> maps:update_with(T, fun(X) -> X + 1 end, 1, M) end, #{}, Tokens),
    Sorted = lists:sort(fun({T1, C1}, {T2, C2}) -> {C1, T2} >= {C2, T1} end, maps:to_list(Counts)),
    [unk() | [T || {T, C} <- lists:sublist([X || {_, C} = X <- Sorted, C >= Min], Max - 1), C >= Min]].
