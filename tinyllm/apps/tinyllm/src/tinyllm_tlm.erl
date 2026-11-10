%% TLM1: the flat, page-aligned weight file shared by the Erlang NIF and the Lisp code (see native/FORMAT.md).
%% This module writes and reads it for a tinyllm_tf model (pre-LayerNorm, with biases).
%%
%%   write(Path, Params, Cfg) -> ok          Params as tinyllm_tf:init/2 builds them (nested float lists)
%%   read(Path) -> {ok, #{cfg, norm, biases, params}}
%%
%% Matrices are stored (out, in) row-major, which is exactly the "list of rows, row j = weights of output j"
%% layout of tinyllm_tf, so nothing is transposed. All numbers are little-endian IEEE doubles.
%% The file is written to Path ++ ".tmp" and renamed: a running VM may have the old file mmap'd, and
%% overwriting a mapped file in place would crash it.
-module(tinyllm_tlm).
-export([write/3, read/1, header/1, spec/1]).

-define(PAGE, 4096).
-define(ALIGN, 256).
-define(HDR, 128).
-define(ENTRY, 40).
-define(NAME, 24).
-define(DTYPE_F64, 1).
-define(NORM_LAYER, 0).
-define(FLAG_BIASES, 1).

%% {FileName, ParamKey, Rows, Cols, vec | mat}
spec(#{vocab := V, context := T, dim := D, layers := L, ff := F}) ->
    [{"tok", e, V, D, mat}, {"pos", p, T, D, mat}, {"lnf.g", lnfg, 1, D, vec}, {"lnf.b", lnfb, 1, D, vec},
     {"out.w", wout, V, D, mat}, {"out.b", bout, 1, V, vec}]
    ++ lists:append([layer_spec(I, D, F) || I <- lists:seq(1, L)]).

layer_spec(I, D, F) ->
    P = "l" ++ integer_to_list(I - 1) ++ ".",
    [{P ++ "ln1.g", {ln1g, I}, 1, D, vec}, {P ++ "ln1.b", {ln1b, I}, 1, D, vec},
     {P ++ "wq", {wq, I}, D, D, mat}, {P ++ "wk", {wk, I}, D, D, mat},
     {P ++ "wv", {wv, I}, D, D, mat}, {P ++ "wo", {wo, I}, D, D, mat},
     {P ++ "ln2.g", {ln2g, I}, 1, D, vec}, {P ++ "ln2.b", {ln2b, I}, 1, D, vec},
     {P ++ "w1", {w1, I}, F, D, mat}, {P ++ "b1", {b1, I}, 1, F, vec},
     {P ++ "w2", {w2, I}, D, F, mat}, {P ++ "b2", {b2, I}, 1, D, vec}].

align(N, A) -> (N + A - 1) div A * A.

write(Path, Params, #{vocab := V, context := T, dim := D, heads := H, layers := L, ff := F} = Cfg) ->
    Spec = spec(Cfg),
    N = length(Spec),
    DataOff = align(?HDR + N * ?ENTRY, ?PAGE),
    {Entries, Blobs, End} = layout(Spec, Params, DataOff),
    FSize = align(End, ?PAGE),
    Hdr = <<"TLM1", 1:32/little, ?DTYPE_F64:32/little, ?NORM_LAYER:32/little, ?FLAG_BIASES:32/little,
            V:32/little, T:32/little, D:32/little, H:32/little, L:32/little, F:32/little, N:32/little,
            ?HDR:64/little, DataOff:64/little, FSize:64/little>>,
    Head = [Hdr, zeros(?HDR - byte_size(Hdr)), Entries, zeros(DataOff - ?HDR - N * ?ENTRY)],
    Tmp = Path ++ ".tmp",
    ok = file:write_file(Tmp, [Head, Blobs, zeros(FSize - End)]),
    ok = file:rename(Tmp, Path).

layout(Spec, Params, DataOff) ->
    {Es, Bs, End} =
        lists:foldl(
          fun({Name, Key, R, C, Kind}, {EAcc, BAcc, Cur}) ->
                  Flat = flat(Kind, maps:get(Key, Params), Name),
                  case length(Flat) of
                      Len when Len =:= R * C -> ok;
                      _ -> erlang:error({bad_shape, Name})
                  end,
                  Off = align(Cur, ?ALIGN),
                  Bin = << <<X:64/float-little>> || X <- Flat >>,
                  NameBin = list_to_binary(Name),
                  Entry = <<NameBin/binary, 0:((?NAME - byte_size(NameBin)) * 8), R:32/little, C:32/little, Off:64/little>>,
                  {[Entry | EAcc], [Bin, zeros(Off - Cur) | BAcc], Off + byte_size(Bin)}   %% reversed below
          end, {[], [], DataOff}, Spec),
    {lists:reverse(Es), lists:reverse(Bs), End}.

flat(vec, Vec, _) when is_list(Vec) -> Vec;
flat(mat, Rows, _) when is_list(Rows) -> lists:append(Rows);
flat(_, _, Name) -> erlang:error({bad_tensor, Name}).

zeros(N) when N >= 0 -> binary:copy(<<0>>, N).

%% ------------------------------------------------------------------ reading

header(Path) ->
    case file:read_file(Path) of
        {ok, B} -> parse_header(B);
        {error, _} = E -> E
    end.

parse_header(<<"TLM1", Ver:32/little, Dt:32/little, Norm:32/little, Flags:32/little, V:32/little, T:32/little,
               D:32/little, H:32/little, L:32/little, F:32/little, N:32/little, Table:64/little, DataOff:64/little,
               FSize:64/little, _/binary>> = B) when byte_size(B) =:= FSize ->
    {ok, #{version => Ver, dtype => Dt, norm => case Norm of 0 -> layer; 1 -> rms end,
           biases => Flags band ?FLAG_BIASES =/= 0, n_tensors => N, table => Table, data_off => DataOff,
           cfg => #{vocab => V, context => T, dim => D, heads => H, layers => L, ff => F}}};
parse_header(_) -> {error, bad_header}.

read(Path) ->
    maybe_read(file:read_file(Path)).

maybe_read({ok, B}) ->
    case parse_header(B) of
        {ok, #{cfg := Cfg, norm := layer, biases := true, n_tensors := N, table := Table} = Hd} ->
            Entries = entries(B, Table, N),
            Params = maps:from_list(
                       [begin
                            {R, C, Off} = maps:get(list_to_binary(Name), Entries),
                            {R, C} = {Rows, Cols},
                            Flat = [X || <<X:64/float-little>> <= binary:part(B, Off, R * C * 8)],
                            {Key, case Kind of vec -> Flat; mat -> rows(Flat, C) end}
                        end || {Name, Key, Rows, Cols, Kind} <- spec(Cfg)]),
            {ok, Hd#{params => Params}};
        {ok, _} -> {error, unsupported_variant};
        {error, _} = E -> E
    end;
maybe_read({error, _} = E) -> E.

entries(B, Table, N) ->
    maps:from_list(
      [begin
           <<Name:?NAME/binary, R:32/little, C:32/little, Off:64/little>> = binary:part(B, Table + I * ?ENTRY, ?ENTRY),
           {hd(binary:split(Name, <<0>>)), {R, C, Off}}
       end || I <- lists:seq(0, N - 1)]).

rows([], _) -> [];
rows(L, C) -> {Row, Rest} = lists:split(C, L), [Row | rows(Rest, C)].
