#!/usr/bin/env escript
%% Compile a generated candidate module (exports dot/3, axpy/3), check it against plain reference
%% implementations bit for bit, then time it. Prints: ok DotMicros AxpyMicros   (or exits non-zero).
main([Path]) ->
    {ok, Mod, Bin} = compile:file(Path, [binary, return_errors]),
    {module, Mod} = code:load_binary(Mod, Path, Bin),
    rand:seed(exsss, {1, 2, 3}),
    Rnd = fun(N) -> [rand:uniform() * 2 - 1 || _ <- lists:seq(1, N)] end,
    [begin
         A = Rnd(N), B = Rnd(N),
         true = Mod:dot(A, B, 0.5) =:= ref_dot(A, B, 0.5),
         true = Mod:axpy(A, 0.37, B) =:= ref_axpy(A, 0.37, B)
     end || N <- lists:seq(0, 41) ++ [64, 1401]],
    {A64, B64, A1k, B1k} = {Rnd(64), Rnd(64), Rnd(1401), Rnd(1401)},
    Dot = best(fun() -> [begin Mod:dot(A64, B64, 0.0), Mod:dot(A1k, B1k, 0.0) end || _ <- lists:seq(1, 200)] end),
    Axpy = best(fun() -> [begin Mod:axpy(A64, 0.5, B64), Mod:axpy(A1k, 0.5, B1k) end || _ <- lists:seq(1, 200)] end),
    io:format("ok ~.2f ~.2f~n", [Dot / 200, Axpy / 200]).

best(F) -> F(), F(), lists:min([element(1, timer:tc(F)) || _ <- lists:seq(1, 7)]).

ref_dot([A | As], [B | Bs], Acc) -> ref_dot(As, Bs, Acc + A * B);
ref_dot([], [], Acc) -> Acc.
ref_axpy([Y | Ys], A, [X | Xs]) -> [Y + A * X | ref_axpy(Ys, A, Xs)];
ref_axpy([], _, []) -> [].
