%% Tool definitions and the parallel, isolated tool runner.
%% A tool is #{name, description, input_schema, handler}; the handler takes
%% the decoded input map and returns a binary (or any JSON-encodable term).
-module(agents_tools).
-export([calculate/0, get_time/0, to_map/1, run_all/3]).

calculate() ->
    #{name => <<"calculate">>,
      description => <<"Evaluate an arithmetic expression with + - * / and parentheses.">>,
      input_schema => #{<<"type">> => <<"object">>,
                        <<"properties">> =>
                            #{<<"expression">> => #{<<"type">> => <<"string">>}},
                        <<"required">> => [<<"expression">>],
                        <<"additionalProperties">> => false},
      handler => fun(#{<<"expression">> := E}) ->
                     case agents_calc:eval(E) of
                         {ok, V} -> format_number(V);
                         {error, R} -> error({bad_expression, R})
                     end
                 end}.

get_time() ->
    #{name => <<"get_time">>,
      description => <<"Current UTC time in ISO 8601.">>,
      input_schema => #{<<"type">> => <<"object">>,
                        <<"properties">> => #{},
                        <<"additionalProperties">> => false},
      handler => fun(_) ->
                     list_to_binary(calendar:system_time_to_rfc3339(
                                      erlang:system_time(second), [{offset, "Z"}]))
                 end}.

format_number(V) when is_integer(V) -> integer_to_binary(V);
format_number(V) when V == trunc(V), abs(V) < 1.0e15 -> integer_to_binary(trunc(V));
format_number(V) -> float_to_binary(V, [{decimals, 10}, compact]).

to_map(Tools) -> maps:from_list([{maps:get(name, T), T} || T <- Tools]).

%% Run every tool_use block concurrently, each in its own process with a
%% deadline. Returns tool_result blocks in call order. A tool that errors,
%% crashes or times out becomes an is_error result; it never takes down the caller.
run_all(Calls, Tools, Timeout) ->
    Started = [begin
                   {Pid, MRef} = spawn_monitor(fun() -> exit({tool_result, execute(C, Tools)}) end),
                   {C, Pid, MRef}
               end || C <- Calls],
    Deadline = erlang:monotonic_time(millisecond) + Timeout,
    [collect(C, Pid, MRef, Deadline) || {C, Pid, MRef} <- Started].

collect(Call, Pid, MRef, Deadline) ->
    Left = max(0, Deadline - erlang:monotonic_time(millisecond)),
    receive
        {'DOWN', MRef, process, Pid, {tool_result, R}} -> block(Call, R);
        {'DOWN', MRef, process, Pid, Reason} ->
            block(Call, {error, fmt("tool crashed: ~p", [Reason])})
    after Left ->
        exit(Pid, kill),
        erlang:demonitor(MRef, [flush]),
        block(Call, {error, <<"tool timed out">>})
    end.

execute(#{<<"name">> := Name} = Call, Tools) ->
    case maps:find(Name, Tools) of
        error -> {error, <<"unknown tool: ", Name/binary>>};
        {ok, #{handler := H}} ->
            try {ok, H(maps:get(<<"input">>, Call, #{}))}
            catch Class:Reason -> {error, fmt("tool failed: ~p:~p", [Class, Reason])}
            end
    end.

block(#{<<"id">> := Id}, {ok, Result}) ->
    #{type => <<"tool_result">>, tool_use_id => Id, content => to_text(Result)};
block(#{<<"id">> := Id}, {error, Msg}) ->
    #{type => <<"tool_result">>, tool_use_id => Id, content => to_text(Msg), is_error => true}.

to_text(B) when is_binary(B) -> B;
to_text(T) ->
    try iolist_to_binary(json:encode(T))
    catch error:_ -> fmt("~p", [T]) end.

fmt(F, A) -> iolist_to_binary(io_lib:format(F, A)).
