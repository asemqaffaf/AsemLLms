%% A small decoder-only transformer (GPT-style), pure Erlang, hand-written backprop.
%%   tokens -> embedding + learned position embedding
%%   L x [ x += Attn(LN(x)) ; x += MLP(LN(x)) ]     (pre-LN, causal multi-head attention, ReLU MLP)
%%   final LN -> output projection -> softmax over the vocabulary, at every position
%% Tensors are nested lists; matrices are lists of rows (row j = weights of output unit j).
%% Parameter keys are atoms or {Name, LayerIndex} tuples, so saved models load without new atoms.
-module(tinyllm_tf).
-export([init/2, prepare/1, sequences/3, loss_grad/3, loss_only/3, last_logits/3, param_count/1]).

-import(tinyllm_nn, [dot/3, axpy/3, vadd/2, lincomb/3, acc_rows/3, zeros/1, matrix/4,
                     softmax/2, sub_at/2, matvec/3, zeros_like/1]).

-define(EPS, 1.0e-5).

%% ------------------------------------------------------------------- init

init(#{vocab := V, context := T, dim := D, heads := H, layers := L, ff := F}, Seed) ->
    0 = D rem H,
    S0 = rand:seed_s(exsss, {Seed, Seed + 1, Seed + 2}),
    {E, S1} = matrix(V, D, 0.1, S0),
    {P, S2} = matrix(T, D, 0.1, S1),
    {Layers, S3} = lists:mapfoldl(fun(I, S) -> layer_init(I, D, F, L, S) end, S2, lists:seq(1, L)),
    {Wout, _} = matrix(V, D, 0.5 / math:sqrt(D), S3),
    maps:from_list([{e, E}, {p, P}, {lnfg, ones(D)}, {lnfb, zeros(D)}, {wout, Wout}, {bout, zeros(V)}]
                   ++ lists:append(Layers)).

layer_init(I, D, F, L, S0) ->
    Sc = 1.0 / math:sqrt(D),
    Res = 1.0 / math:sqrt(2 * L),          %% smaller init on the residual-branch outputs
    {Wq, S1} = matrix(D, D, Sc, S0),
    {Wk, S2} = matrix(D, D, Sc, S1),
    {Wv, S3} = matrix(D, D, Sc, S2),
    {Wo, S4} = matrix(D, D, Sc * Res, S3),
    {W1, S5} = matrix(F, D, Sc, S4),
    {W2, S6} = matrix(D, F, Res / math:sqrt(F), S5),
    {[{{ln1g, I}, ones(D)}, {{ln1b, I}, zeros(D)}, {{wq, I}, Wq}, {{wk, I}, Wk}, {{wv, I}, Wv}, {{wo, I}, Wo},
      {{ln2g, I}, ones(D)}, {{ln2b, I}, zeros(D)}, {{w1, I}, W1}, {{b1, I}, zeros(F)}, {{w2, I}, W2}, {{b2, I}, zeros(D)}], S6}.

ones(N) -> lists:duplicate(N, 1.0).

prepare(#{e := E} = P) -> P#{et => list_to_tuple(E)}.

param_count(Params) -> maps:fold(fun(_, V, A) -> A + count(V) end, 0, maps:without([et], Params)).
count([A | As]) when is_list(A) -> count(A) + count(As);
count([_ | As]) -> 1 + count(As);
count([]) -> 0.

%% ------------------------------------------------------------ data helpers

%% Sequences ending at target position I: inputs ids[I-T .. I-1], targets ids[I-T+1 .. I].
sequences(Ids, T, Positions) ->
    [begin
         Toks = [tinyllm_nn:tok_at(Ids, J) || J <- lists:seq(I - T, I)],
         {lists:droplast(Toks), tl(Toks)}
     end || I <- Positions].

%% ---------------------------------------------------------------- forward

%% -> {OutputRows (T x D, after final LN), Cache}
forward(P, #{layers := L}, Ids) ->
    #{et := Et, p := Pe} = P,
    PeT = list_to_tuple(Pe),
    X0 = [vadd(element(Id + 1, Et), element(Pos, PeT)) || {Id, Pos} <- lists:zip(Ids, lists:seq(1, length(Ids)))],
    {XL, RevCaches} = lists:foldl(fun(I, {X, Cs}) -> {X1, C} = layer_fwd(P, I, X, P), {X1, [C | Cs]} end,
                                  {X0, []}, lists:seq(1, L)),
    {Xf, Cf} = ln_fwd(XL, maps:get(lnfg, P), maps:get(lnfb, P)),
    {Xf, #{ids => Ids, caches => lists:reverse(RevCaches), cf => Cf}}.

layer_fwd(P, I, Xin, _) ->
    Heads = heads_of(P, I),
    {Xn1, C1} = ln_fwd(Xin, maps:get({ln1g, I}, P), maps:get({ln1b, I}, P)),
    Wq = maps:get({wq, I}, P), Wk = maps:get({wk, I}, P), Wv = maps:get({wv, I}, P), Wo = maps:get({wo, I}, P),
    Q = [mv(Wq, X) || X <- Xn1], K = [mv(Wk, X) || X <- Xn1], V = [mv(Wv, X) || X <- Xn1],
    {Ctx, Att} = attention(Q, K, V, Heads),
    X2 = [vadd(A, B) || {A, B} <- lists:zip(Xin, [mv(Wo, C) || C <- Ctx])],
    {Xn2, C2} = ln_fwd(X2, maps:get({ln2g, I}, P), maps:get({ln2b, I}, P)),
    W1 = maps:get({w1, I}, P), B1 = maps:get({b1, I}, P), W2 = maps:get({w2, I}, P), B2 = maps:get({b2, I}, P),
    Hact = [[max(0.0, Z) || Z <- matvec(W1, B1, X)] || X <- Xn2],
    X3 = [vadd(A, B) || {A, B} <- lists:zip(X2, [matvec(W2, B2, Hh) || Hh <- Hact])],
    {X3, #{xn1 => Xn1, c1 => C1, q => Q, k => K, v => V, att => Att, ctx => Ctx,
           xn2 => Xn2, c2 => C2, hact => Hact}}.

%% number of heads is recoverable from the shapes only if stored; keep it in the process
%% dictionary-free way: it is carried inside the prepared params under `heads`.
heads_of(#{heads := H}, _) -> H.

mv(W, X) -> [dot(R, X, 0.0) || R <- W].

%% ---- causal multi-head self-attention
attention(Q, K, V, Heads) ->
    D = length(hd(Q)), Dh = D div Heads,
    Qh = split(Q, Heads, Dh), Kh = split(K, Heads, Dh), Vh = split(V, Heads, Dh),
    Scale = 1.0 / math:sqrt(Dh),
    PerHead = [attn_head(Qi, Ki, Vi, Scale, Dh) || {Qi, Ki, Vi} <- lists:zip3(Qh, Kh, Vh)],  %% H x T x {A, O}
    PerPos = transpose(PerHead),                                                              %% T x H x {A, O}
    {[lists:append([O || {_, O} <- Row]) || Row <- PerPos],
     [[A || {A, _} <- Row] || Row <- PerPos]}.                                               %% Att: T x H x A

%% split a list of T vectors (length D) into H lists of T vectors (length Dh)
split(Vecs, Heads, Dh) ->
    transpose([[lists:sublist(Vec, H * Dh + 1, Dh) || H <- lists:seq(0, Heads - 1)] || Vec <- Vecs]).

transpose([]) -> [];
transpose([Row | Rows]) ->
    lists:foldl(fun(R, Acc) -> lists:zipwith(fun(X, Y) -> Y ++ [X] end, R, Acc) end,
                [[X] || X <- Row], Rows).

attn_head(QH, KH, VH, Scale, Dh) ->
    KT = list_to_tuple(KH), VT = list_to_tuple(VH),
    [begin
         Idx = lists:seq(1, Tt),
         A = softmax([dot(Qt, element(S, KT), 0.0) * Scale || S <- Idx], 1.0),
         {A, lincomb(A, [element(S, VT) || S <- Idx], Dh)}
     end || {Qt, Tt} <- lists:zip(QH, lists:seq(1, length(QH)))].

%% ---- layer norm
ln_fwd(Xs, G, B) ->
    Rows = [ln_row(X, G, B) || X <- Xs],
    {[Y || {Y, _} <- Rows], [C || {_, C} <- Rows]}.

ln_row(X, G, B) ->
    N = length(X),
    Mean = lists:sum(X) / N,
    Cen = [V - Mean || V <- X],
    Inv = 1.0 / math:sqrt(lists:sum([C * C || C <- Cen]) / N + ?EPS),
    Xh = [C * Inv || C <- Cen],
    {[Gi * Xi + Bi || {Gi, Xi, Bi} <- lists:zip3(G, Xh, B)], {Xh, Inv}}.

%% ---------------------------------------------------------------- backward

vmul(A, B) -> [X * Y || {X, Y} <- lists:zip(A, B)].

gadd_vec(G, Key, Vec) -> G#{Key := vadd(maps:get(Key, G), Vec)}.
gadd_vecs(G, Key, Vecs) -> lists:foldl(fun(V, Acc) -> gadd_vec(Acc, Key, V) end, G, Vecs).
%% matrix gradient: dW_j += sum_t dY_t[j] * x_t
gadd_mat(G, Key, DYs, Xs) ->
    G#{Key := lists:foldl(fun({DY, X}, M) -> acc_rows(M, DY, X) end, maps:get(Key, G), lists:zip(DYs, Xs))}.

ln_bwd(DYs, Cache, Gamma, G, GKey, BKey) ->
    G1 = gadd_vecs(G, BKey, DYs),
    Xhs = [Xh || {Xh, _} <- Cache],
    G2 = gadd_vecs(G1, GKey, [vmul(DY, Xh) || {DY, Xh} <- lists:zip(DYs, Xhs)]),
    {[begin
          DXh = vmul(DY, Gamma),
          N = length(Xh),
          M1 = lists:sum(DXh) / N,
          M2 = lists:sum(vmul(DXh, Xh)) / N,
          [Inv * (A - M1 - Xi * M2) || {A, Xi} <- lists:zip(DXh, Xh)]
      end || {DY, {Xh, Inv}} <- lists:zip(DYs, Cache)], G2}.

%% projections: given dOut_t (rows of the output), accumulate dW and return dX_t = W^T dOut_t
proj_bwd(DOuts, Xs, W, G, Key) ->
    D = length(hd(Xs)),
    {[lincomb(DO, W, D) || DO <- DOuts], gadd_mat(G, Key, DOuts, Xs)}.

layer_bwd(P, I, C, DXout, G0) ->
    Heads = heads_of(P, I),
    #{xn1 := Xn1, c1 := C1, q := Q, k := K, v := V, att := Att, ctx := Ctx, xn2 := Xn2, c2 := C2, hact := Hact} = C,
    W1 = maps:get({w1, I}, P), W2 = maps:get({w2, I}, P), Wo = maps:get({wo, I}, P),
    D = length(hd(Xn1)),
    %% MLP branch: X3 = X2 + W2 relu(W1 LN2(X2) + b1) + b2
    G1 = gadd_vecs(G0, {b2, I}, DXout),
    {DHact, G2} = proj_bwd(DXout, Hact, W2, G1, {w2, I}),
    DHpre = [[case A > 0.0 of true -> Dh; false -> 0.0 end || {Dh, A} <- lists:zip(Dh0, Ac)]
             || {Dh0, Ac} <- lists:zip(DHact, Hact)],
    G3 = gadd_vecs(G2, {b1, I}, DHpre),
    {DXn2, G4} = proj_bwd(DHpre, Xn2, W1, G3, {w1, I}),
    {DX2ln, G5} = ln_bwd(DXn2, C2, maps:get({ln2g, I}, P), G4, {ln2g, I}, {ln2b, I}),
    DX2 = [vadd(A, B) || {A, B} <- lists:zip(DXout, DX2ln)],
    %% attention branch: X2 = Xin + Wo Ctx
    {DCtx, G6} = proj_bwd(DX2, Ctx, Wo, G5, {wo, I}),
    {DQ, DK, DV} = attn_bwd(Q, K, V, Att, DCtx, Heads, D),
    {DXq, G7} = proj_bwd(DQ, Xn1, maps:get({wq, I}, P), G6, {wq, I}),
    {DXk, G8} = proj_bwd(DK, Xn1, maps:get({wk, I}, P), G7, {wk, I}),
    {DXv, G9} = proj_bwd(DV, Xn1, maps:get({wv, I}, P), G8, {wv, I}),
    DXn1 = [vadd(vadd(A, B), Cc) || {A, B, Cc} <- lists:zip3(DXq, DXk, DXv)],
    {DXin_ln, G10} = ln_bwd(DXn1, C1, maps:get({ln1g, I}, P), G9, {ln1g, I}, {ln1b, I}),
    {[vadd(A, B) || {A, B} <- lists:zip(DX2, DXin_ln)], G10}.

%% Backward through causal softmax attention. Inputs/outputs are T x D (heads side by side).
attn_bwd(Q, K, V, Att, DCtx, Heads, D) ->
    Dh = D div Heads, Scale = 1.0 / math:sqrt(Dh), T = length(Q),
    Qh = split(Q, Heads, Dh), Kh = split(K, Heads, Dh), Vh = split(V, Heads, Dh),
    DCh = split(DCtx, Heads, Dh),
    AttH = transpose(Att),                                         %% H x T x A
    Res = [attn_head_bwd(Qi, Ki, Vi, Ai, Di, Scale, Dh, T)
           || {{Qi, Ki, Vi}, {Ai, Di}} <- lists:zip(lists:zip3(Qh, Kh, Vh), lists:zip(AttH, DCh))],
    {merge([DQ || {DQ, _, _} <- Res]), merge([DK || {_, DK, _} <- Res]), merge([DV || {_, _, DV} <- Res])}.

%% H lists of T vectors (Dh) -> T vectors (D)
merge(PerHead) -> [lists:append(Row) || Row <- transpose(PerHead)].

attn_head_bwd(QH, KH, VH, AH, DOH, Scale, Dh, T) ->
    KT = list_to_tuple(KH), VT = list_to_tuple(VH),
    Z = zeros(Dh),
    {DQrev, DKm, DVm} =
        lists:foldl(
          fun({{Qt, A, DO}, Tt}, {DQacc, DKm0, DVm0}) ->
              Idx = lists:seq(1, Tt),
              Vs = [element(S, VT) || S <- Idx], Ks = [element(S, KT) || S <- Idx],
              Da = [dot(DO, Vv, 0.0) || Vv <- Vs],
              SumAda = lists:sum([Ai * Dai || {Ai, Dai} <- lists:zip(A, Da)]),
              Ds = [Ai * (Dai - SumAda) * Scale || {Ai, Dai} <- lists:zip(A, Da)],
              DVm1 = lists:foldl(fun({S, Ai}, M) -> M#{S => axpy(maps:get(S, M, Z), Ai, DO)} end, DVm0, lists:zip(Idx, A)),
              DKm1 = lists:foldl(fun({S, Dsi}, M) -> M#{S => axpy(maps:get(S, M, Z), Dsi, Qt)} end, DKm0, lists:zip(Idx, Ds)),
              {[lincomb(Ds, Ks, Dh) | DQacc], DKm1, DVm1}
          end, {[], #{}, #{}}, lists:zip(lists:zip3(QH, AH, DOH), lists:seq(1, T))),
    {lists:reverse(DQrev), [maps:get(S, DKm, Z) || S <- lists:seq(1, T)], [maps:get(S, DVm, Z) || S <- lists:seq(1, T)]}.

%% ------------------------------------------------------------ loss + grads

%% Sum of next-token losses over all positions of all sequences, and the gradient of that sum.
loss_grad(P0, Cfg, Seqs) ->
    P = with_heads(P0, Cfg),
    G0 = (maps:map(fun(_, V) -> zeros_like(V) end, maps:without([et, e, heads], P)))#{emap => #{}},
    {Loss, G} = lists:foldl(fun({Ids, Tgts}, {L, Gacc}) ->
                                {L1, G1} = seq_loss_grad(P, Cfg, Ids, Tgts, Gacc),
                                {L + L1, G1}
                            end, {0.0, G0}, Seqs),
    #{emap := EM} = G,
    D = maps:get(dim, Cfg), Z = zeros(D),
    Grad = (maps:remove(emap, G))#{e => [maps:get(I, EM, Z) || I <- lists:seq(0, maps:get(vocab, Cfg) - 1)]},
    {Loss, Grad}.

with_heads(P, #{heads := H}) -> P#{heads => H}.

seq_loss_grad(P, #{layers := L, dim := D} = Cfg, Ids, Tgts, G0) ->
    {Xf, FC} = forward(P, Cfg, Ids),
    Wout = maps:get(wout, P), Bout = maps:get(bout, P),
    {Loss, DLs} = lists:foldl(
                    fun({X, Tg}, {Ls, Acc}) ->
                        Probs = softmax(matvec(Wout, Bout, X), 1.0),
                        {Ls - math:log(max(lists:nth(Tg + 1, Probs), 1.0e-12)), [sub_at(Probs, Tg) | Acc]}
                    end, {0.0, []}, lists:zip(Xf, Tgts)),
    DL = lists:reverse(DLs),
    G1 = gadd_vecs(G0, bout, DL),
    {DXf, G2} = proj_bwd(DL, Xf, Wout, G1, wout),
    {DXL, G3} = ln_bwd(DXf, maps:get(cf, FC), maps:get(lnfg, P), G2, lnfg, lnfb),
    {DX0, G4} = lists:foldl(fun({I, C}, {DX, G}) -> layer_bwd(P, I, C, DX, G) end,
                            {DXL, G3}, lists:reverse(lists:zip(lists:seq(1, L), maps:get(caches, FC)))),
    %% embeddings: x0_t = E[id_t] + Pos[t]
    GP = [vadd(R, DX) || {R, DX} <- lists:zip(lists:sublist(maps:get(p, G4), length(DX0)), DX0)]
         ++ lists:nthtail(length(DX0), maps:get(p, G4)),
    EM = lists:foldl(fun({Id, DX}, M) -> M#{Id => vadd(maps:get(Id, M, zeros(D)), DX)} end,
                     maps:get(emap, G4), lists:zip(Ids, DX0)),
    {Loss, G4#{p := GP, emap := EM}}.

loss_only(P0, Cfg, Seqs) ->
    P = with_heads(P0, Cfg),
    Wout = maps:get(wout, P), Bout = maps:get(bout, P),
    lists:sum([begin
                   {Xf, _} = forward(P, Cfg, Ids),
                   lists:sum([-math:log(max(lists:nth(Tg + 1, softmax(matvec(Wout, Bout, X), 1.0)), 1.0e-12))
                              || {X, Tg} <- lists:zip(Xf, Tgts)])
               end || {Ids, Tgts} <- Seqs]).

%% Logits for the token after the last of Ids (at most `context` of them).
last_logits(P0, Cfg, Ids) ->
    P = with_heads(P0, Cfg),
    {Xf, _} = forward(P, Cfg, Ids),
    matvec(maps:get(wout, P), maps:get(bout, P), lists:last(Xf)).
