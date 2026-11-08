%% Local backend: LM Studio's OpenAI-compatible server (default http://127.0.0.1:1234).
%% Nothing leaves your machine. The agent loop speaks Anthropic-shaped
%% messages internally; this module translates to/from /v1/chat/completions.
-module(agents_llm_lmstudio).
-behaviour(agents_llm).

-export([chat/2, build_body/2, parse_response/1, pick_model/1]).

-define(BASE_URL, "http://127.0.0.1:1234").

%% Opts: model (default: first non-embedding model LM Studio reports), base_url,
%%       max_tokens (4096), temperature, timeout (ms, 600000).
chat(Req, Opts) ->
    Base = maps:get(base_url, Opts, ?BASE_URL),
    {ok, _} = application:ensure_all_started(inets),
    case model(Base, Opts) of
        {error, _} = E -> E;
        {ok, Model} ->
            Body = iolist_to_binary(json:encode(build_body(Req, Opts#{model => Model}))),
            HttpOpts = [{timeout, maps:get(timeout, Opts, 600000)}, {connect_timeout, 5000}],
            post(Base ++ "/v1/chat/completions", Body, HttpOpts, 3)
    end.

%% Transient connection errors (e.g. a keep-alive socket the server closed) are retried.
post(Url, Body, HttpOpts, Attempts) ->
    case httpc:request(post, {Url, [], "application/json", Body}, HttpOpts, [{body_format, binary}]) of
        {ok, {{_, 200, _}, _, RB}} -> parse_response(RB);
        {ok, {{_, S, _}, _, RB}} -> {error, {api_error, S, error_message(RB)}};
        {error, R} when Attempts > 1 ->
            _ = R, timer:sleep(1000), post(Url, Body, HttpOpts, Attempts - 1);
        {error, {failed_connect, _}} -> {error, lmstudio_unreachable};
        {error, R} -> {error, {http_error, R}}
    end.

model(_, #{model := M}) -> {ok, M};
model(Base, _) ->
    case httpc:request(get, {Base ++ "/v1/models", []}, [{timeout, 5000}, {connect_timeout, 3000}],
                       [{body_format, binary}]) of
        {ok, {{_, 200, _}, _, B}} ->
            try pick_model(json:decode(B)) catch error:_ -> {error, bad_models_response} end;
        {error, {failed_connect, _}} -> {error, lmstudio_unreachable};
        _ -> {error, models_request_failed}
    end.

%% First model that is not an embedding model.
pick_model(#{<<"data">> := Data}) ->
    case [Id || #{<<"id">> := Id} <- Data,
                string:find(string:lowercase(Id), "embed") =:= nomatch] of
        [Id | _] -> {ok, Id};
        [] -> {error, no_chat_model_loaded}
    end.

%% ------------------------------------------------------------ request side

build_body(Req, Opts) ->
    Sys = case maps:get(system, Req, <<>>) of
              <<>> -> [];
              S -> [#{role => <<"system">>, content => S}]
          end,
    B0 = #{model => maps:get(model, Opts), stream => false,
           max_tokens => maps:get(max_tokens, Opts, 4096),
           messages => Sys ++ convert(maps:get(messages, Req))},
    B1 = case maps:get(temperature, Opts, undefined) of
             undefined -> B0;
             T -> B0#{temperature => T}
         end,
    case maps:get(tools, Req, []) of
        [] -> B1;
        Tools -> B1#{tools => [#{type => <<"function">>,
                                 function => #{name => maps:get(name, T),
                                               description => maps:get(description, T),
                                               parameters => maps:get(input_schema, T)}}
                               || T <- Tools]}
    end.

convert([]) -> [];
convert([#{role := <<"user">>, content := C} | T]) when is_binary(C) ->
    [#{role => <<"user">>, content => C} | convert(T)];
convert([#{role := <<"user">>, content := Blocks} | T]) when is_list(Blocks) ->
    %% tool results become one `tool` message each
    [#{role => <<"tool">>, tool_call_id => Id, content => result_text(B)}
     || #{type := <<"tool_result">>, tool_use_id := Id} = B <- Blocks] ++ convert(T);
convert([#{role := <<"assistant">>, content := Blocks} | T]) ->
    Text = iolist_to_binary([X || #{<<"type">> := <<"text">>, <<"text">> := X} <- Blocks]),
    Calls = [#{id => Id, type => <<"function">>,
               function => #{name => N, arguments => iolist_to_binary(json:encode(In))}}
             || #{<<"type">> := <<"tool_use">>, <<"id">> := Id, <<"name">> := N, <<"input">> := In} <- Blocks],
    Msg = #{role => <<"assistant">>, content => Text},
    [case Calls of [] -> Msg; _ -> Msg#{tool_calls => Calls} end | convert(T)].

result_text(#{content := C, is_error := true}) -> <<"Error: ", C/binary>>;
result_text(#{content := C}) -> C.

%% ----------------------------------------------------------- response side

parse_response(Body) ->
    try json:decode(Body) of
        #{<<"choices">> := [#{<<"message">> := Msg} = Choice | _]} ->
            Text = strip_think(text(maps:get(<<"content">>, Msg, null))),
            Calls = [call_block(C) || C <- list(maps:get(<<"tool_calls">>, Msg, null))],
            Blocks = [#{<<"type">> => <<"text">>, <<"text">> => Text} || Text =/= <<>>] ++ Calls,
            {ok, #{<<"content">> => Blocks,
                   <<"stop_reason">> => stop_reason(Calls, maps:get(<<"finish_reason">>, Choice, null))}};
        _ -> {error, bad_response}
    catch error:_ -> {error, bad_response} end.

text(null) -> <<>>;
text(B) when is_binary(B) -> B.

list(L) when is_list(L) -> L;
list(_) -> [].

call_block(#{<<"id">> := Id, <<"function">> := #{<<"name">> := Name} = F}) ->
    Input = try json:decode(maps:get(<<"arguments">>, F, <<"{}">>))
            catch error:_ -> #{<<"_invalid_arguments">> => maps:get(<<"arguments">>, F, <<>>)} end,
    #{<<"type">> => <<"tool_use">>, <<"id">> => Id, <<"name">> => Name,
      <<"input">> => case is_map(Input) of true -> Input; false -> #{<<"_invalid_arguments">> => Input} end}.

stop_reason([_ | _], _) -> <<"tool_use">>;
stop_reason([], <<"length">>) -> <<"max_tokens">>;
stop_reason([], _) -> <<"end_turn">>.

%% Reasoning models (Qwen3 etc.) may wrap their thinking in <think>...</think>.
strip_think(Text) ->
    Clean = re:replace(Text, "<think>.*?</think>", "", [global, dotall, unicode, {return, binary}]),
    string:trim(Clean).

error_message(Body) ->
    try json:decode(Body) of
        #{<<"error">> := #{<<"message">> := M}} -> M;
        #{<<"error">> := M} when is_binary(M) -> M;
        _ -> undefined
    catch error:_ -> undefined end.
