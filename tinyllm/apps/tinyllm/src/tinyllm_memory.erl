%% Conversation memory, kept in Mnesia (Erlang's built-in transactional database).
%%
%% One record per conversation: {tinyllm_conv, Id, Title, Created, Updated, Turns}
%% where Turns is [{Label, Content}] and Label is <<"User">>, <<"Assistant">> or <<"Qwen">>.
%% Every write is a transaction, so two requests on the same conversation never lose each other's turns
%% and a turn pair (question + answer) is stored together or not at all.
%%
%% Config (application env `tinyllm`):
%%   memory         true | false           default true
%%   memory_storage disc | ram             default disc (survives restarts); ram is for tests
%%   memory_dir     "data/mnesia"          where the disc tables live
%%
%% A disc schema is tied to the node name: always start the app the same way (a plain `rebar3 shell`
%% is `nonode@nohost`), or point memory_dir somewhere else.
-module(tinyllm_memory).
-export([start/0, stop/0, forget/0, enabled/0, new_id/0, valid_id/1,
         append/2, get/1, list/0, list/1, delete/1, transcript/1]).

-record(tinyllm_conv, {id, title = <<>>, created = 0, updated = 0, turns = []}).
-define(TABLE, tinyllm_conv).
-define(MAX_TURNS, 2000).
-define(TITLE_BYTES, 60).

%% ---------------------------------------------------------------- lifecycle

%% -> ok | disabled | {error, Reason}. Never crashes the caller: memory is optional.
start() ->
    case application:get_env(tinyllm, memory, true) of
        false -> disabled;
        true ->
            try start_mnesia(), persistent_term:put(?MODULE, true), ok
            catch Class:Why ->
                logger:warning("tinyllm: conversation memory disabled (~p:~p). Running as node ~p; if ~s was "
                               "created under another node name, start with scripts/run.sh or move that directory aside.",
                               [Class, Why, node(), application:get_env(tinyllm, memory_dir, "data/mnesia")]),
                {error, Why}
            end
    end.

start_mnesia() ->
    Storage = application:get_env(tinyllm, memory_storage, disc),
    case mnesia:system_info(is_running) of
        yes -> ok;
        _ ->
            Storage =:= disc andalso begin
                Dir = filename:absname(application:get_env(tinyllm, memory_dir, "data/mnesia")),
                ok = filelib:ensure_path(Dir),
                ok = application:set_env(mnesia, dir, Dir),
                case mnesia:create_schema([node()]) of
                    ok -> ok;
                    {error, {_, {already_exists, _}}} -> ok
                end
            end,
            ok = mnesia:start()
    end,
    Copies = case Storage of disc -> disc_copies; ram -> ram_copies end,
    case mnesia:create_table(?TABLE, [{attributes, record_info(fields, tinyllm_conv)}, {Copies, [node()]}]) of
        {atomic, ok} -> ok;
        {aborted, {already_exists, ?TABLE}} -> ok
    end,
    ok = mnesia:wait_for_tables([?TABLE], 10000).

stop() ->
    persistent_term:erase(?MODULE),
    _ = mnesia:stop(),
    ok.

%% Mark memory unavailable without touching Mnesia (safe to call from an application stop callback).
forget() -> persistent_term:erase(?MODULE), ok.

enabled() -> persistent_term:get(?MODULE, false).

%% ---------------------------------------------------------------- ids

new_id() -> list_to_binary(string:lowercase(binary_to_list(binary:encode_hex(crypto:strong_rand_bytes(8))))).

valid_id(Id) when is_binary(Id), byte_size(Id) =:= 16 -> lists:all(fun hex/1, binary_to_list(Id));
valid_id(_) -> false.

hex(C) -> (C >= $0 andalso C =< $9) orelse (C >= $a andalso C =< $f).

%% ---------------------------------------------------------------- operations

%% Add turns ([{Label, Content}]) to a conversation, creating it on first use. The title is the
%% first user message.
append(Id, Turns) when is_list(Turns) ->
    case guard(Id) of
        ok -> tx(fun() ->
                     Now = erlang:system_time(second),
                     C = case mnesia:read(?TABLE, Id, write) of
                             [R] -> R;
                             [] -> #tinyllm_conv{id = Id, created = Now}
                         end,
                     All = C#tinyllm_conv.turns ++ Turns,
                     Kept = lists:nthtail(max(0, length(All) - ?MAX_TURNS), All),
                     mnesia:write(C#tinyllm_conv{title = title(C#tinyllm_conv.title, Kept), updated = Now, turns = Kept})
                 end, true);
        Err -> Err
    end.

title(<<>>, Turns) ->
    case [C || {<<"User">>, C} <- Turns] of
        [First | _] -> cut(First);
        [] -> <<>>
    end;
title(T, _) -> T.

cut(B) when byte_size(B) > ?TITLE_BYTES ->
    %% never split a UTF-8 character
    case unicode:characters_to_binary(binary:part(B, 0, ?TITLE_BYTES)) of
        S when is_binary(S) -> S;
        {incomplete, S, _} -> S;
        _ -> <<>>
    end;
cut(B) -> B.

get(Id) ->
    case guard(Id) of
        ok ->
            case tx(fun() -> mnesia:read(?TABLE, Id) end) of
                {ok, [R]} -> {ok, to_map(R)};
                {ok, []} -> not_found;
                Err -> Err
            end;
        Err -> Err
    end.

%% Newest first, without the turns: [#{id, title, created, updated, turns => Count}].
list() -> list(50).

list(Limit) ->
    case enabled() of
        false -> {error, disabled};
        true ->
            case tx(fun() -> mnesia:foldl(fun(R, Acc) -> [summary(R) | Acc] end, [], ?TABLE) end) of
                {ok, L} ->
                    Sorted = lists:sort(fun(#{updated := A}, #{updated := B}) -> A >= B end, L),
                    {ok, lists:sublist(Sorted, Limit)};
                Err -> Err
            end
    end.

delete(Id) ->
    case guard(Id) of
        ok -> tx(fun() -> mnesia:delete({?TABLE, Id}) end, true);
        Err -> Err
    end.

%% The conversation as "User: ...\nAssistant: ...\n" lines (the format the chat model and page use).
transcript(Id) ->
    case ?MODULE:get(Id) of
        {ok, #{turns := Turns}} -> {ok, iolist_to_binary([[L, ": ", C, "\n"] || #{role := L, content := C} <- Turns])};
        Other -> Other
    end.

%% ---------------------------------------------------------------- helpers

guard(Id) ->
    case {enabled(), valid_id(Id)} of
        {false, _} -> {error, disabled};
        {_, false} -> {error, bad_id};
        _ -> ok
    end.

tx(F) -> tx(F, false).

%% Writes are synced to the transaction log before returning, so a saved turn survives a crash or kill -9.
tx(F, Sync) ->
    case mnesia:transaction(F) of
        {atomic, Res} ->
            case Sync of true -> _ = mnesia:sync_log(); false -> ok end,   %% no log (error) in ram mode
            case Res of ok -> ok; _ -> {ok, Res} end;
        {aborted, Why} -> {error, Why}
    end.

to_map(#tinyllm_conv{id = I, title = T, created = C, updated = U, turns = Ts}) ->
    #{id => I, title => T, created => C, updated => U,
      turns => [#{role => L, content => X} || {L, X} <- Ts]}.

summary(#tinyllm_conv{id = I, title = T, created = C, updated = U, turns = Ts}) ->
    #{id => I, title => T, created => C, updated => U, turns => length(Ts)}.
