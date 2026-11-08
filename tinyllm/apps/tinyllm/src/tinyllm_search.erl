%% Web search through APIs meant for programs - no scraping, no keys.
%%   wikipedia (default)  Wikipedia's search API: real results, Wikipedia only
%%   searxng              your own SearXNG instance (JSON format enabled): the whole web,
%%                        including Google, queried by SearXNG itself
%% Config (config/sys.config): {search_provider, wikipedia | searxng}, {searxng_url, "http://127.0.0.1:8888"}.
%% Only the query text is sent to the provider.
-module(tinyllm_search).
-export([search/1, search/2, provider/0, provider_label/0, google_url/1,
         parse_wikipedia/1, parse_searxng/1]).

provider() -> application:get_env(tinyllm, search_provider, wikipedia).

provider_label() ->
    case provider() of wikipedia -> "Wikipedia"; searxng -> "SearXNG" end.

%% A link that opens the query in Google in the user's own browser.
google_url(Query) -> "https://www.google.com/search?q=" ++ uri_string:quote(binary_to_list(Query)).

%% -> {ok, [#{title, url, snippet}]} | {error, rate_limited | unreachable | {http_status, S} | {request_failed, _} | bad_response}
search(Query) -> search(Query, #{}).

search(Query, Opts) when is_binary(Query) ->
    Max = maps:get(max_results, Opts, 5),
    case maps:get(provider, Opts, provider()) of
        wikipedia ->
            Base = maps:get(base_url, Opts, "https://en.wikipedia.org/w/api.php"),
            %% generator=search + intro extracts: a few real sentences per hit, not a snippet fragment
            Url = Base ++ "?action=query&generator=search&prop=extracts&exintro=1&explaintext=1&exsentences=3"
                  ++ "&format=json&utf8=1&gsrlimit=" ++ integer_to_list(Max) ++ "&exlimit=" ++ integer_to_list(Max)
                  ++ "&gsrsearch=" ++ uri_string:quote(binary_to_list(Query)),
            with_json(Url, fun parse_wikipedia/1);
        searxng ->
            Base = maps:get(base_url, Opts, application:get_env(tinyllm, searxng_url, "http://127.0.0.1:8888")),
            Url = Base ++ "/search?format=json&q=" ++ uri_string:quote(binary_to_list(Query)),
            with_json(Url, fun(J) -> lists:sublist(parse_searxng(J), Max) end)
    end.

with_json(Url, Parse) ->
    {ok, _} = application:ensure_all_started(inets),
    {ok, _} = application:ensure_all_started(ssl),
    Headers = [{"user-agent", "tinyllm/0.1 (local research tool)"}, {"accept", "application/json"}],
    HttpOpts = [{timeout, 10000}, {connect_timeout, 5000}] ++ ssl_opts(Url),
    case httpc:request(get, {Url, Headers}, HttpOpts, [{body_format, binary}]) of
        {ok, {{_, 200, _}, _, Body}} ->
            try {ok, Parse(json:decode(Body))}
            catch error:_ -> {error, bad_response} end;
        {ok, {{_, 429, _}, _, _}} -> {error, rate_limited};
        {ok, {{_, Status, _}, _, _}} -> {error, {http_status, Status}};
        {error, {failed_connect, _}} -> {error, unreachable};
        {error, Reason} -> {error, {request_failed, element(1, tuple_for(Reason))}}
    end.

tuple_for(R) when is_tuple(R) -> R;
tuple_for(R) -> {R}.

ssl_opts("https://" ++ _) ->
    [{ssl, [{verify, verify_peer}, {cacerts, public_key:cacerts_get()},
            {customize_hostname_check,
             [{match_fun, public_key:pkix_verify_hostname_match_fun(https)}]}]}];
ssl_opts(_) -> [].

%% ---------------------------------------------------------------- parsers

parse_wikipedia(#{<<"query">> := #{<<"pages">> := Pages}}) ->
    Ordered = lists:sort(fun(A, B) -> maps:get(<<"index">>, A, 0) =< maps:get(<<"index">>, B, 0) end,
                         maps:values(Pages)),
    [#{title => T, url => wiki_url(T), snippet => text(maps:get(<<"extract">>, P, <<>>))}
     || #{<<"title">> := T} = P <- Ordered];
parse_wikipedia(_) -> [].

wiki_url(Title) ->
    Path = uri_string:quote(binary_to_list(binary:replace(Title, <<" ">>, <<"_">>, [global])), "_"),
    list_to_binary("https://en.wikipedia.org/wiki/" ++ Path).

parse_searxng(#{<<"results">> := Rs}) ->
    [#{title => text(T), url => U, snippet => text(maps:get(<<"content">>, R, <<>>))}
     || #{<<"title">> := T, <<"url">> := U} = R <- Rs, http_url(U)];
parse_searxng(_) -> [].

http_url(<<"http://", _/binary>>) -> true;
http_url(<<"https://", _/binary>>) -> true;
http_url(_) -> false.

%% strip tags, decode the few entities APIs emit, collapse whitespace
text(Html) when is_binary(Html) ->
    NoTags = re:replace(Html, "<[^>]*>", "", [global, {return, binary}]),
    Plain = lists:foldl(fun({F, T}, A) -> binary:replace(A, F, T, [global]) end, NoTags,
                        [{<<"&quot;">>, <<"\"">>}, {<<"&#039;">>, <<"'">>}, {<<"&#x27;">>, <<"'">>},
                         {<<"&lt;">>, <<"<">>}, {<<"&gt;">>, <<">">>}, {<<"&amp;">>, <<"&">>}]),
    list_to_binary(string:join(string:tokens(binary_to_list(Plain), " \t\r\n"), " "));
text(_) -> <<>>.
