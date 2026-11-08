-module(agents_tests).
-include_lib("eunit/include/eunit.hrl").

-define(M, agents_llm_mock).

%% --- helpers ---------------------------------------------------------------

setup() -> {ok, _} = application:ensure_all_started(agents), ok.
cleanup(_) -> application:stop(agents).

agent(Handler, Extra) ->
    {ok, A} = agents:start_agent(maps:merge(
        #{llm => ?M, llm_opts => #{handler => Handler}, tools => [agents_tools:calculate()]}, Extra)),
    A.

last_user_content(#{messages := Msgs}) -> maps:get(content, lists:last(Msgs)).

all_test_() ->
    {setup, fun setup/0, fun cleanup/1,
     [fun simple_reply/0, fun tool_loop/0, fun parallel_tools_one_message/0,
      fun failing_tool_is_error_result/0, fun crashing_tool_is_error_result/0,
      fun tool_timeout/0, fun unknown_tool/0, fun max_turns/0, fun refusal/0,
      fun llm_error_keeps_history/0, fun llm_crash_is_contained/0,
      fun history_accumulates/0, fun run_all_is_parallel_and_isolated/0,
      fun calc/0, fun claude_body/0, fun claude_http_roundtrip/0,
      fun claude_api_error/0, fun missing_key/0]}.

%% --- agent loop ------------------------------------------------------------

simple_reply() ->
    A = agent(fun(_) -> ?M:text(<<"hi">>) end, #{}),
    ?assertEqual({ok, <<"hi">>}, agents:ask(A, <<"hello">>)).

tool_loop() ->
    H = fun(Req) ->
            case last_user_content(Req) of
                <<"2 plus">> -> ?M:tool_use([{<<"t1">>, <<"calculate">>, #{<<"expression">> => <<"2+3*4">>}}]);
                [#{content := Result, tool_use_id := <<"t1">>}] -> ?M:text(<<"answer ", Result/binary>>)
            end
        end,
    A = agent(H, #{}),
    ?assertEqual({ok, <<"answer 14">>}, agents:ask(A, <<"2 plus">>)).

parallel_tools_one_message() ->
    Self = self(),
    H = fun(Req) ->
            case last_user_content(Req) of
                <<"go">> -> ?M:tool_use([{<<"a">>, <<"calculate">>, #{<<"expression">> => <<"1+1">>}},
                                         {<<"b">>, <<"calculate">>, #{<<"expression">> => <<"3*3">>}}]);
                Results -> Self ! {results, Results}, ?M:text(<<"done">>)
            end
        end,
    A = agent(H, #{}),
    ?assertEqual({ok, <<"done">>}, agents:ask(A, <<"go">>)),
    receive {results, R} ->
        ?assertMatch([#{tool_use_id := <<"a">>, content := <<"2">>},
                      #{tool_use_id := <<"b">>, content := <<"9">>}], R)
    after 1000 -> ?assert(false) end.

tool_result_flow(Tool, Input) ->
    Self = self(),
    H = fun(Req) ->
            case last_user_content(Req) of
                <<"go">> -> ?M:tool_use([{<<"x">>, maps:get(name, Tool), Input}]);
                [R] -> Self ! {result, R}, ?M:text(<<"recovered">>)
            end
        end,
    A = agent(H, #{tools => [Tool], tool_timeout => 200}),
    ?assertEqual({ok, <<"recovered">>}, agents:ask(A, <<"go">>)),
    ?assert(is_process_alive(A)),
    receive {result, R} -> R after 1000 -> ?assert(false) end.

failing_tool_is_error_result() ->
    R = tool_result_flow(agents_tools:calculate(), #{<<"expression">> => <<"1/0">>}),
    ?assertMatch(#{is_error := true}, R).

crashing_tool_is_error_result() ->
    T = #{name => <<"boom">>, description => <<"d">>, input_schema => #{},
          handler => fun(_) -> exit(self(), kill), timer:sleep(1000) end},
    ?assertMatch(#{is_error := true, content := <<"tool crashed", _/binary>>},
                 tool_result_flow(T, #{})).

tool_timeout() ->
    T = #{name => <<"slow">>, description => <<"d">>, input_schema => #{},
          handler => fun(_) -> timer:sleep(5000) end},
    ?assertMatch(#{is_error := true, content := <<"tool timed out">>}, tool_result_flow(T, #{})).

unknown_tool() ->
    R = tool_result_flow(agents_tools:calculate(), #{}) ,
    ?assertMatch(#{tool_use_id := <<"x">>}, R),
    T = #{name => <<"ghost">>, description => <<"d">>, input_schema => #{}, handler => fun(_) -> ok end},
    Self = self(),
    H = fun(Req) ->
            case last_user_content(Req) of
                <<"go">> -> ?M:tool_use([{<<"x">>, <<"nope">>, #{}}]);
                [Res] -> Self ! {unknown, Res}, ?M:text(<<"ok">>)
            end
        end,
    A = agent(H, #{tools => [T]}),
    {ok, _} = agents:ask(A, <<"go">>),
    receive {unknown, U} -> ?assertMatch(#{is_error := true, content := <<"unknown tool: nope">>}, U)
    after 1000 -> ?assert(false) end.

max_turns() ->
    H = fun(_) -> ?M:tool_use([{<<"x">>, <<"calculate">>, #{<<"expression">> => <<"1">>}}]) end,
    A = agent(H, #{max_turns => 3}),
    ?assertEqual({error, max_turns}, agents:ask(A, <<"loop">>)).

refusal() ->
    A = agent(fun(_) -> #{<<"stop_reason">> => <<"refusal">>, <<"content">> => [],
                          <<"stop_details">> => #{<<"category">> => <<"cyber">>}} end, #{}),
    ?assertEqual({error, {refusal, <<"cyber">>}}, agents:ask(A, <<"x">>)).

llm_error_keeps_history() ->
    Ctr = counters:new(1, []),
    H = fun(_) ->
            counters:add(Ctr, 1, 1),
            case counters:get(Ctr, 1) of
                1 -> {error, overloaded};
                _ -> ?M:text(<<"ok">>)
            end
        end,
    A = agent(H, #{}),
    ?assertEqual({error, overloaded}, agents:ask(A, <<"one">>)),
    ?assertEqual([], agents:history(A)),
    ?assertEqual({ok, <<"ok">>}, agents:ask(A, <<"two">>)),
    ?assertEqual(2, length(agents:history(A))).

llm_crash_is_contained() ->
    A = agent(fun(_) -> error(kaboom) end, #{}),
    ?assertMatch({error, {llm_crashed, error, kaboom}}, agents:ask(A, <<"x">>)),
    ?assert(is_process_alive(A)).

history_accumulates() ->
    A = agent(fun(Req) -> ?M:text(integer_to_binary(length(maps:get(messages, Req)))) end, #{}),
    ?assertEqual({ok, <<"1">>}, agents:ask(A, <<"a">>)),
    ?assertEqual({ok, <<"3">>}, agents:ask(A, <<"b">>)).

run_all_is_parallel_and_isolated() ->
    Slow = fun(_) -> timer:sleep(200), ?M:text(<<"ok">>) end,
    Opts = fun(H) -> #{llm => ?M, llm_opts => #{handler => H}} end,
    Jobs = [{N, Opts(Slow), <<"p">>} || N <- lists:seq(1, 30)]
           ++ [{bad, Opts(fun(_) -> error(oops) end), <<"p">>}],
    {Micros, Results} = timer:tc(fun() -> agents:run_all(Jobs) end),
    ?assert(Micros < 1500000),   %% 30 x 200ms would be 6s if serial
    ?assertEqual(31, length(Results)),
    ?assertEqual({ok, <<"ok">>}, proplists:get_value(1, Results)),
    ?assertMatch({error, {llm_crashed, _, _}}, proplists:get_value(bad, Results)),
    ?assertEqual(30, length([ok || {_, {ok, _}} <- Results])).

%% --- calculator -------------------------------------------------------------

calc() ->
    ?assertEqual({ok, 14}, agents_calc:eval(<<"2+3*4">>)),
    ?assertEqual({ok, 20}, agents_calc:eval(<<"(2+3)*4">>)),
    ?assertEqual({ok, -3}, agents_calc:eval(<<"-(1+2)">>)),
    ?assertEqual({ok, 2.5}, agents_calc:eval(<<"5/2">>)),
    ?assertEqual({error, division_by_zero}, agents_calc:eval(<<"1/0">>)),
    ?assertMatch({error, _}, agents_calc:eval(<<"1+">>)),
    ?assertMatch({error, _}, agents_calc:eval(<<"os:cmd(\"ls\")">>)),
    ?assertMatch({error, _}, agents_calc:eval(<<"(1+2">>)),
    ?assertMatch({error, _}, agents_calc:eval(<<"1 2">>)).

%% --- Claude client -----------------------------------------------------------

claude_body() ->
    Tool = (agents_tools:calculate())#{handler => fun(_) -> ok end},
    B = agents_llm_claude:build_body(#{system => <<"s">>, messages => [#{role => <<"user">>, content => <<"hi">>}],
                                       tools => [Tool]}, #{}),
    ?assertEqual(<<"claude-opus-5-5">>, maps:get(model, B)),
    ?assertEqual(<<"default">>, maps:get(fallbacks, B)),
    ?assertEqual([name, description, input_schema], maps:keys(hd(maps:get(tools, B))) -- [] -- []) ,
    [?assertNot(maps:is_key(K, B)) || K <- [temperature, top_p, top_k, tool_choice, thinking]],
    ?assertNot(maps:is_key(fallbacks, agents_llm_claude:build_body(#{messages => []}, #{fallbacks => off}))),
    ?assertNot(maps:is_key(handler, hd(maps:get(tools, B)))).

%% A tiny HTTP server that answers one request with a canned reply and
%% forwards the raw request to the test.
fake_server(Status, Body) ->
    {ok, L} = gen_tcp:listen(0, [binary, {active, false}, {reuseaddr, true}]),
    {ok, Port} = inet:port(L),
    Parent = self(),
    spawn_link(fun() ->
        {ok, S} = gen_tcp:accept(L),
        Raw = read_request(S, <<>>),
        Parent ! {request, Raw},
        Resp = iolist_to_binary(["HTTP/1.1 ", Status, "\r\ncontent-type: application/json\r\ncontent-length: ",
                                 integer_to_list(byte_size(Body)), "\r\nconnection: close\r\n\r\n", Body]),
        gen_tcp:send(S, Resp), gen_tcp:close(S), gen_tcp:close(L)
    end),
    "http://127.0.0.1:" ++ integer_to_list(Port).

read_request(S, Acc) ->
    case binary:split(Acc, <<"\r\n\r\n">>) of
        [Head, Body] ->
            [_, CL] = binary:split(string:lowercase(Head), <<"content-length: ">>),
            [Len | _] = binary:split(CL, <<"\r\n">>),
            case byte_size(Body) >= binary_to_integer(Len) of
                true -> Acc;
                false -> more(S, Acc)
            end;
        _ -> more(S, Acc)
    end.
more(S, Acc) -> {ok, D} = gen_tcp:recv(S, 0, 2000), read_request(S, <<Acc/binary, D/binary>>).

claude_http_roundtrip() ->
    Url = fake_server("200 OK", <<"{\"stop_reason\":\"end_turn\",\"content\":[{\"type\":\"text\",\"text\":\"pong\"}]}">>),
    R = agents_llm_claude:chat(#{messages => [#{role => <<"user">>, content => <<"ping">>}]},
                               #{api_key => "sk-test-key", base_url => Url}),
    ?assertMatch({ok, #{<<"stop_reason">> := <<"end_turn">>}}, R),
    receive {request, Raw} ->
        Lower = string:lowercase(Raw),
        ?assertMatch({_, _}, binary:match(Raw, <<"POST /v1/messages">>)),
        ?assertMatch({_, _}, binary:match(Lower, <<"x-api-key: sk-test-key">>)),
        ?assertMatch({_, _}, binary:match(Lower, <<"anthropic-version: 2026-10-21">>)),
        ?assertMatch({_, _}, binary:match(Lower, <<"anthropic-beta: server-side-fallback-2026-07-01">>)),
        [_, Body] = binary:split(Raw, <<"\r\n\r\n">>),
        ?assertMatch(#{<<"model">> := <<"claude-opus-5-5">>, <<"max_tokens">> := 16000}, json:decode(Body))
    after 1000 -> ?assert(false) end.

claude_api_error() ->
    Url = fake_server("401 Unauthorized",
        <<"{\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}">>),
    ?assertEqual({error, {api_error, 401, <<"authentication_error">>, <<"invalid x-api-key">>}},
                 agents_llm_claude:chat(#{messages => []}, #{api_key => "bad", base_url => Url})),
    ?assertEqual({error, {api_error, 400, undefined, undefined}}, agents_llm_claude:parse_response(400, <<"not json">>)).

missing_key() ->
    Old = os:getenv("ANTHROPIC_API_KEY"),
    os:unsetenv("ANTHROPIC_API_KEY"),
    try ?assertEqual({error, missing_api_key}, agents_llm_claude:chat(#{messages => []}, #{}))
    after case Old of false -> ok; V -> os:putenv("ANTHROPIC_API_KEY", V) end end.
