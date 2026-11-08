%% Claude Messages API over raw HTTPS (OTP httpc; no extra dependencies).
%% The API key is read from `api_key` in opts or $ANTHROPIC_API_KEY and is
%% only ever sent in the x-api-key header - never logged or put in errors.
-module(agents_llm_claude).
-behaviour(agents_llm).

-export([chat/2, build_body/2, headers/2, parse_response/2]).

-define(DEFAULT_MODEL, <<"claude-opus-5-5">>).
-define(API_VERSION, "2026-10-21").
-define(BASE_URL, "https://api.anthropic.com").
-define(MAX_ATTEMPTS, 3).

chat(Req, Opts) ->
    case api_key(Opts) of
        undefined -> {error, missing_api_key};
        Key -> post(Key, Req, Opts, 1)
    end.

api_key(#{api_key := K}) -> K;
api_key(_) ->
    case os:getenv("ANTHROPIC_API_KEY") of
        false -> undefined;
        "" -> undefined;
        K -> K
    end.

post(Key, Req, Opts, Attempt) ->
    {ok, _} = application:ensure_all_started(inets),
    {ok, _} = application:ensure_all_started(ssl),
    Url = maps:get(base_url, Opts, ?BASE_URL) ++ "/v1/messages",
    Body = iolist_to_binary(json:encode(build_body(Req, Opts))),
    HttpOpts = [{timeout, maps:get(timeout, Opts, 300000)}, {connect_timeout, 10000}]
               ++ ssl_opts(Url),
    case httpc:request(post, {Url, headers(Key, Opts), "application/json", Body},
                       HttpOpts, [{body_format, binary}]) of
        {ok, {{_, Status, _}, RespHeaders, RespBody}} ->
            case retryable(Status) andalso Attempt < ?MAX_ATTEMPTS of
                true ->
                    timer:sleep(backoff(Attempt, RespHeaders)),
                    post(Key, Req, Opts, Attempt + 1);
                false -> parse_response(Status, RespBody)
            end;
        {error, _} when Attempt < ?MAX_ATTEMPTS ->
            timer:sleep(backoff(Attempt, [])),
            post(Key, Req, Opts, Attempt + 1);
        {error, Reason} ->
            {error, {http_error, Reason}}
    end.

retryable(S) -> S =:= 429 orelse S >= 500.

backoff(Attempt, Headers) ->
    case lists:keyfind("retry-after", 1, Headers) of
        {_, V} ->
            try min(list_to_integer(V) * 1000, 30000) catch error:_ -> expo(Attempt) end;
        false -> expo(Attempt)
    end.

expo(Attempt) -> (1 bsl (Attempt - 1)) * 1000 + rand:uniform(250).

ssl_opts("https://" ++ _) ->
    [{ssl, [{verify, verify_peer},
            {cacerts, public_key:cacerts_get()},
            {customize_hostname_check,
             [{match_fun, public_key:pkix_verify_hostname_match_fun(https)}]}]}];
ssl_opts(_) -> [].

headers(Key, Opts) ->
    Base = [{"x-api-key", Key}, {"anthropic-version", ?API_VERSION}],
    case maps:get(fallbacks, Opts, default) of
        default -> Base ++ [{"anthropic-beta", "server-side-fallback-2026-07-01"}];
        off -> Base
    end.

%% No temperature/top_p/top_k (rejected by current models), no prefill, no
%% forced tool_choice. Thinking and effort are left at the model defaults.
build_body(Req, Opts) ->
    B0 = #{model => maps:get(model, Opts, ?DEFAULT_MODEL),
           max_tokens => maps:get(max_tokens, Opts, 16000),
           messages => maps:get(messages, Req)},
    B1 = case maps:get(system, Req, <<>>) of
             <<>> -> B0;
             Sys -> B0#{system => Sys}
         end,
    B2 = case maps:get(tools, Req, []) of
             [] -> B1;
             Tools -> B1#{tools => [maps:with([name, description, input_schema], T)
                                    || T <- Tools]}
         end,
    case maps:get(fallbacks, Opts, default) of
        default -> B2#{fallbacks => <<"default">>};
        off -> B2
    end.

parse_response(200, Body) ->
    try json:decode(Body) of
        #{<<"content">> := _, <<"stop_reason">> := _} = Resp -> {ok, Resp};
        _ -> {error, {bad_response, 200}}
    catch error:_ -> {error, {bad_response, 200}} end;
parse_response(Status, Body) ->
    try json:decode(Body) of
        #{<<"error">> := #{<<"type">> := Type, <<"message">> := Msg}} ->
            {error, {api_error, Status, Type, Msg}};
        _ -> {error, {api_error, Status, undefined, undefined}}
    catch error:_ -> {error, {api_error, Status, undefined, undefined}} end.
