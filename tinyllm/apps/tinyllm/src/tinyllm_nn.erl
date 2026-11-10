%% The model: a character-level neural language model (Bengio-style MLP).
%%   context of N chars -> embeddings (D each) -> concat -> tanh hidden (H) -> softmax over V
%% Plain Erlang lists, hand-written backprop. Tensors are nested lists.
-module(tinyllm_nn).
-export([init/2, prepare/1, examples/3, tok_at/2, loss_grad/3, loss_only/2, logits/2,
         softmax/2, sample/2, zw/3, tmap/2, zeros_like/1, sumsq/1,
         %% building blocks shared with tinyllm_tf
         matrix/4, zeros/1, dot/3, axpy/3, vadd/2, lincomb/3, acc_rows/3, sub_at/2, matvec/3]).

%% ------------------------------------------------------------------ init

init(#{vocab := V, context := N, embed := D, hidden := H}, Seed) ->
    S0 = rand:seed_s(exsss, {Seed, Seed + 1, Seed + 2}),
    In = N * D,
    {E, S1} = matrix(V, D, 0.1, S0),
    {W1, S2} = matrix(H, In, 1.0 / math:sqrt(In), S1),
    {W2, _} = matrix(V, H, 0.5 / math:sqrt(H), S2),
    #{e => E, w1 => W1, b1 => zeros(H), w2 => W2, b2 => zeros(V)}.

matrix(Rows, Cols, Scale, S) ->
    lists:mapfoldl(fun(_, Acc) -> row(Cols, Scale, Acc) end, S, lists:seq(1, Rows)).

row(Cols, Scale, S) ->
    lists:mapfoldl(fun(_, Acc) -> {Z, Acc1} = rand:normal_s(Acc), {Z * Scale, Acc1} end,
                   S, lists:seq(1, Cols)).

zeros(N) -> lists:duplicate(N, 0.0).

%% E as a tuple of rows for O(1) lookup.
prepare(#{e := E} = P) -> P#{et => list_to_tuple(E)}.

%% ------------------------------------------------------------ data helpers

%% Token ids are stored as 16-bit big-endian integers.
tok_at(Ids, J) -> <<V:16>> = binary:part(Ids, J * 2, 2), V.

%% Examples for target positions I: context = ids[I-N .. I-1], target = ids[I].
examples(Ids, N, Positions) ->
    [{[tok_at(Ids, I - N + K) || K <- lists:seq(0, N - 1)], tok_at(Ids, I)}
     || I <- Positions].

%% ---------------------------------------------------------------- forward

forward(#{et := Et, w1 := W1, b1 := B1, w2 := W2, b2 := B2}, Ctx) ->
    X = lists:append([element(C + 1, Et) || C <- Ctx]),
    H = [math:tanh(Z) || Z <- matvec(W1, B1, X)],
    {X, H, matvec(W2, B2, H)}.

logits(P, Ctx) -> {_, _, L} = forward(P, Ctx), L.

matvec([R | Rs], [B | Bs], X) -> [dot(R, X, B) | matvec(Rs, Bs, X)];
matvec([], [], _) -> [].

%% dot/3 and axpy/3 live in a generated file: lisp/bin/tune.lisp writes them (unrolled by whatever is
%% fastest on this machine, checked bit-for-bit against the plain loops) and keeps them only if eunit passes.
-include("tinyllm_kernels.hrl").

softmax(L, Temp) ->
    M = lists:max(L),
    Es = [math:exp((X - M) / Temp) || X <- L],
    S = lists:sum(Es),
    [E / S || E <- Es].

%% Sample an index from a probability list.
sample(Probs, _) ->
    sample(Probs, rand:uniform(), 0).
sample([P], _, I) -> _ = P, I;
sample([P | Ps], R, I) -> case R < P of true -> I; false -> sample(Ps, R - P, I + 1) end.

loss_only(P, Examples) ->
    lists:foldl(fun({Ctx, T}, Acc) ->
                    Probs = softmax(logits(P, Ctx), 1.0),
                    Acc - math:log(max(lists:nth(T + 1, Probs), 1.0e-12))
                end, 0.0, Examples).

%% --------------------------------------------------------------- backward

%% Returns {SumLoss, GradMap} for the given examples (gradients are SUMS).
loss_grad(#{e := E, w1 := W1, b1 := B1, w2 := W2, b2 := B2} = P, #{embed := D}, Examples) ->
    Acc0 = {0.0, zeros_like(W2), zeros_like(B2), zeros_like(W1), zeros_like(B1), #{}},
    {Loss, GW2, Gb2, GW1, Gb1, GE} =
        lists:foldl(fun(Ex, Acc) -> step(Ex, Acc, P, D) end, Acc0, Examples),
    ZeroRow = zeros(D),
    GEdense = [maps:get(I, GE, ZeroRow) || I <- lists:seq(0, length(E) - 1)],
    _ = {W1, B1, W2, B2},
    {Loss, #{e => GEdense, w1 => GW1, b1 => Gb1, w2 => GW2, b2 => Gb2}}.

step({Ctx, T}, {Loss, GW2, Gb2, GW1, Gb1, GE}, #{w1 := W1, w2 := W2} = P, D) ->
    {X, H, L} = forward(P, Ctx),
    Probs = softmax(L, 1.0),
    Loss1 = Loss - math:log(max(lists:nth(T + 1, Probs), 1.0e-12)),
    DL = sub_at(Probs, T),
    DH = lincomb(DL, W2, length(H)),
    DZ = zipmul(DH, H),
    DX = lincomb(DZ, W1, length(X)),
    {Loss1, acc_rows(GW2, DL, H), vadd(Gb2, DL), acc_rows(GW1, DZ, X), vadd(Gb1, DZ),
     add_emb(Ctx, chunks(DX, D), GE)}.

%% dL/dlogits = probs - onehot(target)
sub_at(Probs, T) -> sub_at(Probs, T, 0).
sub_at([P | Ps], T, T) -> [P - 1.0 | Ps];
sub_at([P | Ps], T, I) -> [P | sub_at(Ps, T, I + 1)];
sub_at([], _, _) -> [].

%% dz = dh * (1 - h^2)
zipmul([D | Ds], [H | Hs]) -> [D * (1.0 - H * H) | zipmul(Ds, Hs)];
zipmul([], []) -> [].

%% sum_i coef_i * row_i  (vector of length Len)
lincomb(Coefs, Rows, Len) -> lincomb_acc(Coefs, Rows, zeros(Len)).
lincomb_acc([C | Cs], [R | Rs], Acc) -> lincomb_acc(Cs, Rs, axpy(Acc, C, R));
lincomb_acc([], [], Acc) -> Acc.

%% acc_i = acc_i + coef_i * x   for each row i
acc_rows([Row | Rows], [C | Cs], X) -> [axpy(Row, C, X) | acc_rows(Rows, Cs, X)];
acc_rows([], [], _) -> [].


vadd([A | As], [B | Bs]) -> [A + B | vadd(As, Bs)];
vadd([], []) -> [].

chunks([], _) -> [];
chunks(L, D) -> {C, Rest} = lists:split(D, L), [C | chunks(Rest, D)].

add_emb([C | Cs], [Ch | Chs], GE) ->
    GE1 = case GE of
              #{C := Old} -> GE#{C := vadd(Old, Ch)};
              _ -> GE#{C => Ch}
          end,
    add_emb(Cs, Chs, GE1);
add_emb([], [], GE) -> GE.

%% ----------------------------------------------------- generic tensor tools

zw(F, [A | As], [B | Bs]) when is_list(A) -> [zw(F, A, B) | zw(F, As, Bs)];
zw(F, [A | As], [B | Bs]) -> [F(A, B) | zw(F, As, Bs)];
zw(_, [], []) -> [].

tmap(F, [A | As]) when is_list(A) -> [tmap(F, A) | tmap(F, As)];
tmap(F, [A | As]) -> [F(A) | tmap(F, As)];
tmap(_, []) -> [].

zeros_like(T) -> tmap(fun(_) -> 0.0 end, T).

sumsq(T) when is_map(T) -> maps:fold(fun(_, V, A) -> A + sumsq(V) end, 0.0, T);
sumsq([A | As]) when is_list(A) -> sumsq(A) + sumsq(As);
sumsq([A | As]) -> A * A + sumsq(As);
sumsq([]) -> 0.0.
