%% Safe arithmetic evaluator: numbers, + - * /, parentheses, unary minus.
%% A hand-written parser, so model-supplied text is never evaluated as code.
-module(agents_calc).
-export([eval/1]).

-spec eval(binary() | string()) -> {ok, number()} | {error, term()}.
eval(Bin) when is_binary(Bin) -> eval(binary_to_list(Bin));
eval(Str) ->
    try
        {V, []} = expr(tokens(Str, [])),
        {ok, V}
    catch
        throw:{calc, Reason} -> {error, Reason};
        error:{badmatch, _} -> {error, unexpected_input}
    end.

tokens([], Acc) -> lists:reverse(Acc);
tokens([$\s | T], Acc) -> tokens(T, Acc);
tokens([C | T], Acc) when C =:= $+; C =:= $-; C =:= $*; C =:= $/; C =:= $(; C =:= $) ->
    tokens(T, [{op, C} | Acc]);
tokens([C | _] = S, Acc) when (C >= $0 andalso C =< $9) orelse C =:= $. ->
    {Num, Rest} = lists:splitwith(fun(X) -> (X >= $0 andalso X =< $9) orelse X =:= $. end, S),
    tokens(Rest, [{num, number(Num)} | Acc]);
tokens([C | _], _) -> throw({calc, {unexpected_char, [C]}}).

number(S) ->
    try
        case lists:member($., S) of
            true -> list_to_float(S);
            false -> list_to_integer(S)
        end
    catch error:badarg -> throw({calc, bad_number}) end.

expr(Ts) -> {L, Rest} = term(Ts), expr_loop(L, Rest).
expr_loop(L, [{op, $+} | T]) -> {R, Rest} = term(T), expr_loop(L + R, Rest);
expr_loop(L, [{op, $-} | T]) -> {R, Rest} = term(T), expr_loop(L - R, Rest);
expr_loop(L, Rest) -> {L, Rest}.

term(Ts) -> {L, Rest} = factor(Ts), term_loop(L, Rest).
term_loop(L, [{op, $*} | T]) -> {R, Rest} = factor(T), term_loop(L * R, Rest);
term_loop(L, [{op, $/} | T]) ->
    {R, Rest} = factor(T),
    case R == 0 of
        true -> throw({calc, division_by_zero});
        false -> term_loop(L / R, Rest)
    end;
term_loop(L, Rest) -> {L, Rest}.

factor([{op, $-} | T]) -> {V, Rest} = factor(T), {-V, Rest};
factor([{num, N} | T]) -> {N, T};
factor([{op, $(} | T]) ->
    {V, Rest} = expr(T),
    case Rest of
        [{op, $)} | T2] -> {V, T2};
        _ -> throw({calc, missing_close_paren})
    end;
factor(_) -> throw({calc, unexpected_input}).
