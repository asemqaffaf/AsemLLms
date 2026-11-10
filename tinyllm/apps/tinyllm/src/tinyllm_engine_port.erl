%% Engine that drives an external worker, one OS process per operation: the bridge to MLX LoRA on a Mac,
%% llama.cpp / QVAC Fabric LoRA, or anything else that can train and score an adapter from the command line.
%% Erlang holds no tensors; it passes file and directory names and reads one result line back.
%%
%% Command line of every call:   Cmd Args... OP --flag value ...
%%   train     --base-model M --adapter-in H|none --adapter-out DIR --data FILE --lr X --steps N --rank R
%%   eval      --base-model M --adapter H|none --data FILE
%%   distance  --a H|none --b H|none
%% FILE for train is DIR/train.txt, one training sample per line. The worker prints anything it likes and, as
%% its LAST line starting with `RESULT `, space-separated k=v pairs:
%%   train: RESULT train_loss=1.93      eval: RESULT loss=2.10      distance: RESULT dist=0.37
%% and exits 0. A non-zero exit, a missing RESULT line or the op timeout is {error, ...}.
%%
%% Handles are adapter directories (binary paths) under `root`; `none` is the untouched base model.
%%
%% Opts: cmd (executable, required), args ([] fixed leading args, e.g. the script), root (work dir, required),
%%       base_model ("" passed as --base-model), heldout_file, probe_file (required),
%%       op_timeout (ms per call, 3600000).
-module(tinyllm_engine_port).
-behaviour(tinyllm_engine).
-export([init/1, train/3, eval/3, drift/3, release/2]).

init(#{cmd := Cmd0, root := Root, heldout_file := H, probe_file := P} = Opts) ->
    Cmd = case filename:pathtype(Cmd0) of absolute -> Cmd0; _ -> os:find_executable(Cmd0) end,
    case {Cmd, filelib:is_regular(H), filelib:is_regular(P), filelib:ensure_path(Root)} of
        {false, _, _, _} -> {error, {no_such_command, Cmd0}};
        {_, true, true, ok} ->
            {ok, Opts#{args => maps:get(args, Opts, []), base_model => maps:get(base_model, Opts, ""),
                       op_timeout => maps:get(op_timeout, Opts, 3600000), cmd => Cmd}, none};
        {_, false, _, _} -> {error, {missing_file, H}};
        {_, _, false, _} -> {error, {missing_file, P}};
        {_, _, _, {error, R}} -> {error, {root, R}}
    end;
init(_) -> {error, "need cmd, root, heldout_file, probe_file"}.

train(St, From, #{lr := Lr, steps := Steps, rank := Rank, batch := Batch}) ->
    Out = new_dir(St),
    Data = filename:join(Out, "train.txt"),
    try
        ok = file:write_file(Data, [[sample(S), $\n] || S <- Batch]),
        Args = ["train", "--base-model", maps:get(base_model, St), "--adapter-in", path(From),
                "--adapter-out", Out, "--data", Data, "--lr", num(Lr), "--steps", integer_to_list(Steps),
                "--rank", integer_to_list(Rank)],
        case run(St, Args) of
            {ok, KV} ->
                case number(train_loss, KV) of
                    {ok, L} -> {ok, bin(Out), #{train_loss => L}};
                    {error, _} = E -> cleanup(Out), E
                end;
            {error, _} = E -> cleanup(Out), E
        end
    catch C:R -> cleanup(Out), {error, {C, R}}
    end.

eval(St, H, Set) ->
    File = case Set of heldout -> maps:get(heldout_file, St); probe -> maps:get(probe_file, St) end,
    case run(St, ["eval", "--base-model", maps:get(base_model, St), "--adapter", path(H), "--data", File]) of
        {ok, KV} -> number(loss, KV);
        {error, _} = E -> E
    end.

drift(St, A, B) ->
    case run(St, ["distance", "--a", path(A), "--b", path(B)]) of
        {ok, KV} -> number(dist, KV);
        {error, _} = E -> E
    end.

release(_St, none) -> ok;
release(_St, Dir) -> cleanup(Dir), ok.

%% ---------------------------------------------------------------- running a worker

run(#{cmd := Cmd, args := Fixed, op_timeout := TO}, Args) ->
    Port = open_port({spawn_executable, Cmd},
                     [{args, Fixed ++ Args}, exit_status, binary, stderr_to_stdout, hide, {line, 65536}]),
    Deadline = erlang:monotonic_time(millisecond) + TO,
    loop(Port, Deadline, []).

loop(Port, Deadline, Lines) ->
    Left = max(0, Deadline - erlang:monotonic_time(millisecond)),
    receive
        {Port, {data, {_, Line}}} -> loop(Port, Deadline, [Line | lists:sublist(Lines, 50)]);
        {Port, {exit_status, 0}} -> result(Lines);
        {Port, {exit_status, N}} -> {error, {exit, N, lists:reverse(lists:sublist(Lines, 5))}}
    after Left ->
        catch port_close(Port),
        {error, timeout}
    end.

%% Lines are newest first: the first `RESULT ` line found is the last one the worker printed.
result([]) -> {error, no_result};
result([<<"RESULT ", Rest/binary>> | _]) ->
    {ok, [kv(T) || T <- binary:split(Rest, <<" ">>, [global, trim_all])]};
result([_ | T]) -> result(T).

kv(Tok) ->
    case binary:split(Tok, <<"=">>) of
        [K, V] -> {K, V};
        [K] -> {K, <<>>}
    end.

number(Key, KV) ->
    case lists:keyfind(atom_to_binary(Key), 1, KV) of
        {_, V} -> to_float(V);
        false -> {error, {missing, Key}}
    end.

%% "2", "2.5", "1e-3" and "1.5E+2" all become floats.
to_float(Bin) ->
    L = string:lowercase(binary_to_list(Bin)),
    Norm = case string:split(L, "e") of
               [M, E] -> fix_mantissa(M) ++ "e" ++ E;
               [M] -> fix_mantissa(M)
           end,
    case catch list_to_float(Norm) of
        F when is_float(F) -> {ok, F};
        _ -> {error, {bad_number, Bin}}
    end.

fix_mantissa(M) ->
    case lists:member($., M) of true -> M; false -> M ++ ".0" end.

%% ---------------------------------------------------------------- files

new_dir(#{root := Root}) ->
    Name = "v" ++ integer_to_list(erlang:system_time(microsecond)) ++ "_" ++ integer_to_list(erlang:unique_integer([positive])),
    Dir = filename:join(Root, Name),
    ok = filelib:ensure_path(Dir),
    Dir.

cleanup(Dir) -> catch file:del_dir_r(Dir), ok.

path(none) -> "none";
path(Dir) when is_binary(Dir) -> binary_to_list(Dir).

bin(L) -> unicode:characters_to_binary(L).

sample(S) when is_binary(S); is_list(S) ->
    B = unicode:characters_to_binary(S),
    case binary:match(B, [<<"\n">>, <<"\r">>]) of
        nomatch -> B;
        _ -> error({newline_in_sample, B})
    end.

num(N) when is_integer(N) -> integer_to_list(N);
num(F) when is_float(F) -> float_to_list(F, [short]).
