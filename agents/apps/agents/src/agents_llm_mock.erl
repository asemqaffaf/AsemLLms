%% Offline backend for tests and demos: `#{handler => fun((Request) -> Response)}`.
%% The handler returns {ok, Response} | {error, Reason}, or a bare Response map.
-module(agents_llm_mock).
-behaviour(agents_llm).
-export([chat/2, text/1, tool_use/1]).

chat(Req, #{handler := H}) ->
    case H(Req) of
        {ok, _} = Ok -> Ok;
        {error, _} = E -> E;
        Resp when is_map(Resp) -> {ok, Resp}
    end.

text(Text) ->
    #{<<"stop_reason">> => <<"end_turn">>,
      <<"content">> => [#{<<"type">> => <<"text">>, <<"text">> => Text}]}.

%% Calls :: [{Id, Name, Input}]
tool_use(Calls) ->
    #{<<"stop_reason">> => <<"tool_use">>,
      <<"content">> => [#{<<"type">> => <<"tool_use">>, <<"id">> => Id,
                          <<"name">> => Name, <<"input">> => Input}
                        || {Id, Name, Input} <- Calls]}.
