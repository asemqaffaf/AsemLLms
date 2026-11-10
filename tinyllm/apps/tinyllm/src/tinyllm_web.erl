%% Web interface, all in Erlang: the page is rendered server-side as plain HTML
%% (a form that POSTs back) - no JavaScript. Listens on 127.0.0.1 only.
%%   GET  /              page
%%   POST /              form submit -> page with the generated text
%%   GET  /api/info      model facts (JSON)
%%   POST /api/generate  {prompt, length, temperature, seed?} -> {text, ms} (JSON)
%%   GET  /chat          chat page (no JavaScript); /chat?c=<id> reopens a saved conversation
%%   POST /chat/delete   delete a saved conversation
%%   GET  /api/conversations, GET|DELETE /api/conversations/<id>   saved chats (JSON, Mnesia)
%%   GET  /v1/models, POST /v1/chat/completions   OpenAI-compatible, so other tools
%%                       (and the `agents` workers) can use AsemLLM as an LLM
-module(tinyllm_web).
-behaviour(cowboy_handler).
-export([start/0, start/1, stop/0, init/2, execute/2]).

-define(MAX_PROMPT, 1000).
-define(MAX_LEN, 2000).
-define(NO_MODEL, <<"No model loaded. Train one with: escript scripts/train.escript">>).

start() -> start(application:get_env(tinyllm, port, 8080)).

start(Port) ->
    load_model(),
    Dispatch = cowboy_router:compile([{'_', [
        {"/", ?MODULE, page},
        {"/api/info", ?MODULE, info},
        {"/api/generate", ?MODULE, generate},
        {"/chat", ?MODULE, chat},
        {"/chat/delete", ?MODULE, chat_delete},
        {"/api/conversations/[:id]", ?MODULE, conversations},
        {"/api/search", ?MODULE, search},
        {"/v1/models", ?MODULE, models},
        {"/v1/chat/completions", ?MODULE, completions}]}]),
    {ok, _} = cowboy:start_clear(tinyllm_http, [{port, Port}, {ip, {127, 0, 0, 1}}],
                                 #{env => #{dispatch => Dispatch},
                                   middlewares => [?MODULE, cowboy_router, cowboy_handler]}),
    logger:notice("AsemLLM UI on http://localhost:~p", [Port]),
    ok.

stop() ->
    _ = (catch cowboy:stop_listener(tinyllm_http)),
    ok.

%% Cowboy middleware: refuse requests whose Host (DNS rebinding) or Origin (cross-site form/fetch)
%% is not this machine. The listener is bound to 127.0.0.1, but browsers can still be tricked into
%% reaching it.
execute(Req, Env) ->
    Local = [<<"localhost">>, <<"127.0.0.1">>],
    HostOk = lists:member(cowboy_req:host(Req), Local),
    OriginOk = case cowboy_req:header(<<"origin">>, Req) of
                   undefined -> true;
                   Origin ->
                       case uri_string:parse(Origin) of
                           #{host := H} -> lists:member(H, Local);
                           _ -> false
                       end
               end,
    case HostOk andalso OriginOk of
        true -> {ok, Req, Env};
        false -> {stop, cowboy_req:reply(403, #{<<"content-type">> => <<"text/plain">>},
                                         <<"forbidden: localhost only">>, Req)}
    end.

load_model() ->
    Path = application:get_env(tinyllm, model_file, "model.bin"),
    case filelib:is_regular(Path) of
        true -> persistent_term:put(tinyllm_model, tinyllm:load(Path));
        false -> logger:warning("tinyllm: no model at ~s (train one: escript scripts/train.escript)", [Path])
    end.

model() -> persistent_term:get(tinyllm_model, undefined).

%% ----------------------------------------------------------------- handlers

init(Req, page) ->
    case cowboy_req:method(Req) of
        <<"GET">> -> {ok, html(200, render(#{}), Req), page};
        <<"POST">> -> {ok, submit(Req), page};
        _ -> {ok, html(405, render(#{error => <<"method not allowed">>}), Req), page}
    end;
init(Req, info) ->
    case model() of
        undefined -> {ok, json(503, #{error => ?NO_MODEL}, Req), info};
        M -> {ok, json(200, info(M), Req), info}
    end;
init(Req, models) ->
    {ok, json(200, #{object => <<"list">>,
                     data => [#{id => <<"AsemLLM">>, object => <<"model">>, owned_by => <<"local">>}]}, Req), models};
init(Req, completions) ->
    case cowboy_req:method(Req) of
        <<"POST">> -> {ok, completions(Req), completions};
        _ -> {ok, json(405, #{error => #{message => <<"POST only">>}}, Req), completions}
    end;
init(Req, search) ->
    #{q := Q} = cowboy_req:match_qs([{q, [], <<>>}], Req),
    case search_query(Q) of
        {error, Msg} -> {ok, json(400, #{error => Msg}, Req), search};
        {ok, Query} ->
            case tinyllm_search:search(Query, search_opts()) of
                {ok, Results} -> {ok, json(200, #{query => Query, results => Results}, Req), search};
                {error, Why} -> {ok, json(502, #{error => search_error(Why)}, Req), search}
            end
    end;
init(Req, chat) ->
    case cowboy_req:method(Req) of
        <<"GET">> ->
            #{c := C} = cowboy_req:match_qs([{c, [], <<>>}], Req),
            {Cid, T} = case tinyllm_memory:transcript(C) of
                           {ok, Saved} -> {C, Saved};
                           _ -> {<<>>, <<>>}
                       end,
            {ok, html(200, render_chat(T, <<>>, undefined, false, undefined, Cid), Req), chat};
        <<"POST">> -> {ok, chat_submit(Req), chat};
        _ -> {ok, html(405, render_chat(<<>>, <<>>, <<"method not allowed">>, false, undefined, <<>>), Req), chat}
    end;
init(Req, chat_delete) ->
    case cowboy_req:method(Req) of
        <<"POST">> ->
            {ok, Form, Req1} = cowboy_req:read_urlencoded_body(Req, #{length => 1024}),
            _ = tinyllm_memory:delete(proplists:get_value(<<"cid">>, Form, <<>>)),
            {ok, cowboy_req:reply(303, #{<<"location">> => <<"/chat">>}, Req1), chat_delete};
        _ -> {ok, html(405, <<"POST only">>, Req), chat_delete}
    end;
init(Req, conversations) ->
    {ok, conversations_api(cowboy_req:method(Req), cowboy_req:binding(id, Req), Req), conversations};
init(Req, generate) ->
    case cowboy_req:method(Req) of
        <<"POST">> -> {ok, api_generate(Req), generate};
        _ -> {ok, json(405, #{error => <<"POST only">>}, Req), generate}
    end.

submit(Req0) ->
    {ok, Form, Req} = cowboy_req:read_urlencoded_body(Req0, #{length => 8192}),
    F = maps:from_list(Form),
    Fields = #{prompt => maps:get(<<"prompt">>, F, <<>>),
               length => maps:get(<<"length">>, F, <<"300">>),
               temperature => maps:get(<<"temperature">>, F, <<"0.7">>),
               seed => maps:get(<<"seed">>, F, <<>>)},
    try
        M = model(),
        M =/= undefined orelse throw({bad, ?NO_MODEL}),
        {Prompt, Len, Temp, Opts} = validate(from_form(F)),
        {Text, Ms} = run(M, Prompt, Len, Temp, Opts),
        html(200, render(Fields#{result => {Prompt, Text, Ms}}), Req)
    catch throw:{bad, Msg} -> html(400, render(Fields#{error => Msg}), Req) end.

api_generate(Req0) ->
    case model() of
        undefined -> json(503, #{error => ?NO_MODEL}, Req0);
        M ->
            case cowboy_req:read_body(Req0, #{length => 8192}) of
                {ok, Data, Req} ->
                    try
                        J = try json:decode(Data) catch error:_ -> throw({bad, <<"invalid JSON">>}) end,
                        is_map(J) orelse throw({bad, <<"expected a JSON object">>}),
                        {Prompt, Len, Temp, Opts} = validate(J),
                        {Text, Ms} = run(M, Prompt, Len, Temp, Opts),
                        json(200, #{text => Text, ms => Ms}, Req)
                    catch throw:{bad, Msg} -> json(400, #{error => Msg}, Req) end;
                {more, _, Req} -> json(413, #{error => <<"body too large">>}, Req)
            end
    end.

%% ---- OpenAI-compatible chat completions ------------------------------------

completions(Req0) ->
    case model() of
        undefined -> json(503, oai_error(?NO_MODEL), Req0);
        M ->
            case cowboy_req:read_body(Req0, #{length => 65536}) of
                {ok, Data, Req} ->
                    try
                        J = try json:decode(Data) catch error:_ -> throw({bad, <<"invalid JSON">>}) end,
                        is_map(J) orelse throw({bad, <<"expected a JSON object">>}),
                        maps:get(<<"stream">>, J, false) =:= true andalso throw({bad, <<"streaming is not supported">>}),
                        Msgs = messages(maps:get(<<"messages">>, J, undefined)),
                        MaxTok = clamp_int(maps:get(<<"max_tokens">>, J, 200), 1, ?MAX_LEN, <<"max_tokens">>),
                        Temp = clamp_temp(maps:get(<<"temperature">>, J, 0.7)),
                        Opts = case maps:get(<<"seed">>, J, null) of
                                   S when is_integer(S) -> #{seed => S};
                                   _ -> #{}
                               end,
                        {Reply, Why} = tinyllm:chat(M, Msgs, Opts#{temperature => Temp, max_chars => MaxTok}),
                        PromptChars = lists:sum([byte_size(C) || #{<<"content">> := C} <- Msgs]),
                        json(200, #{id => <<"chatcmpl-asemllm">>, object => <<"chat.completion">>,
                                    created => erlang:system_time(second), model => <<"AsemLLM">>,
                                    choices => [#{index => 0,
                                                  message => #{role => <<"assistant">>, content => printable(Reply)},
                                                  finish_reason => case Why of stop -> <<"stop">>; length -> <<"length">> end}],
                                    usage => #{prompt_tokens => PromptChars, completion_tokens => byte_size(Reply),
                                               total_tokens => PromptChars + byte_size(Reply)}}, Req)
                    catch throw:{bad, Msg} -> json(400, oai_error(Msg), Req) end;
                {more, _, Req} -> json(413, oai_error(<<"body too large">>), Req)
            end
    end.

oai_error(Msg) -> #{error => #{message => Msg, type => <<"invalid_request_error">>}}.

messages(L) when is_list(L), L =/= [] ->
    [case M of
         #{<<"role">> := R, <<"content">> := C} when is_binary(R), is_binary(C) ->
             byte_size(C) =< 4000 orelse throw({bad, <<"message too long (max 4000 bytes)">>}),
             #{<<"role">> => R, <<"content">> => C};
         _ -> throw({bad, <<"each message needs a string role and string content">>})
     end || M <- L];
messages(_) -> throw({bad, <<"messages must be a non-empty list">>}).

clamp_int(V, Lo, Hi, _) when is_integer(V), V >= Lo -> min(V, Hi);
clamp_int(_, _, _, Name) -> throw({bad, <<Name/binary, " must be a positive integer">>}).

clamp_temp(T) when is_number(T) -> float(max(0.05, min(T, 2.0)));
clamp_temp(_) -> throw({bad, <<"temperature must be a number">>}).

%% ---- web search ---------------------------------------------------------------

search_opts() -> maps:from_list([{base_url, U} || U <- [application:get_env(tinyllm, search_url, undefined)], U =/= undefined]).

search_query(Q) ->
    T = string:trim(Q),
    case {byte_size(T), T} of
        {0, _} -> {error, <<"empty query">>};
        {N, _} when N > 200 -> {error, <<"query too long (max 200 bytes)">>};
        _ -> {ok, T}
    end.

search_error(rate_limited) -> <<"Search failed: the provider is rate-limiting requests; try again shortly">>;
search_error(unreachable) ->
    case tinyllm_search:provider() of
        searxng -> <<"Search failed: cannot reach SearXNG - is it running at the configured searxng_url?">>;
        _ -> <<"Search failed: cannot reach the search provider (offline?)">>
    end;
search_error({http_status, S}) -> iolist_to_binary(["Search failed: HTTP ", integer_to_list(S)]);
search_error(bad_response) -> <<"Search failed: unexpected response from the provider">>;
search_error(_) -> <<"Search failed: could not reach the search provider">>.

render_results(undefined) -> "";
render_results({error, Msg}) -> ["<p class=err>", esc(Msg), "</p>"];
render_results({ok, Query, []}) ->
    ["<p class=meta>No results for ", esc(Query), " in ", tinyllm_search:provider_label(), " &middot; <a href=\"",
     esc(tinyllm_search:google_url(Query)), "\" target=_blank rel=\"noopener noreferrer\">open in Google</a></p>"];
render_results({ok, Query, Results}) ->
    ["<div class=web><p class=meta>Web results for &ldquo;", esc(Query), "&rdquo; (", tinyllm_search:provider_label(),
     ") &middot; <a href=\"", esc(tinyllm_search:google_url(Query)), "\" target=_blank rel=\"noopener noreferrer\">open in Google</a></p>",
     [["<p><a href=\"", esc(U), "\" target=_blank rel=\"noopener noreferrer nofollow\">", esc(T), "</a><br><span class=meta>",
       esc(S), "</span></p>"] || #{title := T, url := U, snippet := S} <- Results],
     "</div>"].

%% ---- chat page --------------------------------------------------------------

chat_submit(Req0) ->
    {ok, Form, Req} = cowboy_req:read_urlencoded_body(Req0, #{length => 16384}),
    F = maps:from_list(Form),
    Web = case {maps:get(<<"qwen">>, F, <<>>), maps:get(<<"web">>, F, <<>>)} of
              {<<"1">>, _} -> qwen;
              {_, <<"1">>} -> search;
              _ -> false
          end,
    %% a saved conversation (Mnesia) is the source of truth; the hidden field is the fallback when memory is off
    Cid0 = maps:get(<<"cid">>, F, <<>>),
    {Cid, Stored} = case tinyllm_memory:transcript(Cid0) of
                        {ok, Saved} -> {Cid0, Saved};
                        _ -> {tinyllm_memory:new_id(), maps:get(<<"transcript">>, F, <<>>)}
                    end,
    Transcript = trim_transcript(Stored),
    Msg = string:trim(maps:get(<<"message">>, F, <<>>)),
    case {model(), Msg} of
        {undefined, _} -> html(503, render_chat(Transcript, <<>>, ?NO_MODEL, Web, undefined, keep(Cid, Stored)), Req);
        {_, <<>>} -> html(200, render_chat(Transcript, <<>>, undefined, Web, undefined, keep(Cid, Stored)), Req);
        {_, _} when byte_size(Msg) > 500 ->
            html(400, render_chat(Transcript, Msg, <<"message too long (max 500 bytes)">>, Web, undefined, keep(Cid, Stored)), Req);
        {M, _} ->
            Clean = binary:replace(Msg, <<"\n">>, <<" ">>, [global]),
            Found = case Web of
                        false -> undefined;
                        _ ->
                            case search_query(Clean) of
                                {error, Msg2} -> {error, Msg2};
                                {ok, Query} ->
                                    case tinyllm_search:search(Query, search_opts()) of
                                        {ok, Rs} -> {ok, Query, Rs};
                                        {error, Why} -> {error, search_error(Why)}
                                    end
                            end
                    end,
            %% Qwen (LM Studio) answers from the results when asked and possible; otherwise AsemLLM replies.
            {Label, Reply, Err} =
                case {Web, Found} of
                    {qwen, {ok, Query2, Rs2}} when Rs2 =/= [] ->
                        case tinyllm_lmstudio:answer(Query2, Rs2) of
                            {ok, Text} -> {<<"Qwen">>, one_line(Text, 1200), undefined};
                            {error, Why2} -> {<<"Assistant">>, tinyllm_reply(M, Transcript, Clean), qwen_error(Why2)}
                        end;
                    {qwen, {ok, _, []}} ->
                        {<<"Assistant">>, tinyllm_reply(M, Transcript, Clean), <<"No search results to answer from.">>};
                    _ ->
                        {<<"Assistant">>, tinyllm_reply(M, Transcript, Clean), undefined}
                end,
            New = <<Transcript/binary, "User: ", Clean/binary, "\n", Label/binary, ": ", Reply/binary, "\n">>,
            {Shown, Err2} = save_turns(Cid, [{<<"User">>, Clean}, {Label, Reply}], trim_transcript(New), Err),
            html(200, render_chat(Shown, <<>>, Err2, Web, Found, Cid), Req)
    end.

tinyllm_reply(M, Transcript, Clean) ->
    Turns = parse_transcript(Transcript) ++ [#{<<"role">> => <<"user">>, <<"content">> => Clean}],
    {Reply, _} = tinyllm:chat(M, Turns, #{temperature => 0.7, max_chars => 200}),
    printable(Reply).

one_line(Text, Max) ->
    Flat = binary:replace(printable(Text), [<<"\r">>, <<"\n">>], <<" ">>, [global]),
    case byte_size(Flat) > Max of
        true -> printable(binary:part(Flat, 0, Max));
        false -> Flat
    end.

qwen_error(unreachable) -> <<"Qwen is unavailable: cannot reach LM Studio (is its server running on the configured lmstudio_url?). Showing AsemLLM's reply instead.">>;
qwen_error(no_model) -> <<"Qwen is unavailable: LM Studio has no chat model loaded. Showing AsemLLM's reply instead.">>;
qwen_error({http_status, S}) ->
    iolist_to_binary(["Qwen is unavailable: LM Studio rejected the request (HTTP ", integer_to_list(S),
                      "). Check that lmstudio_model in config/sys.config is a loaded chat model. Showing AsemLLM's reply instead."]);
qwen_error(_) -> <<"Qwen is unavailable: LM Studio returned an unexpected response. Showing AsemLLM's reply instead.">>.

%% Keep the transcript bounded; the model only looks at the last few characters anyway.
trim_transcript(T) when byte_size(T) > 4000 ->
    Cut = binary:part(T, byte_size(T) - 4000, 4000),
    case binary:split(Cut, <<"\nUser: ">>) of
        [_, Rest] -> <<"User: ", Rest/binary>>;
        _ -> Cut
    end;
trim_transcript(T) -> T.

parse_transcript(T) ->
    lists:append([case Line of
                      <<"User: ", C/binary>> -> [#{<<"role">> => <<"user">>, <<"content">> => C}];
                      <<"Assistant: ", C/binary>> -> [#{<<"role">> => <<"assistant">>, <<"content">> => C}];
                      <<"Qwen: ", C/binary>> -> [#{<<"role">> => <<"assistant">>, <<"content">> => C}];
                      _ -> []
                  end || Line <- binary:split(T, <<"\n">>, [global])]).

render_chat(Transcript, Draft, Error, Web, Found, Cid) ->
    Bubbles = [case Line of
                   <<"User: ", C/binary>> -> ["<div class=\"m u\"><b>You</b> ", esc(C), "</div>"];
                   <<"Assistant: ", C/binary>> -> ["<div class=\"m a\"><b>AsemLLM</b> ", esc(C), "</div>"];
                   <<"Qwen: ", C/binary>> -> ["<div class=\"m a\"><b>Qwen</b> ", esc(C), "</div>"];
                   _ -> ""
               end || Line <- binary:split(Transcript, <<"\n">>, [global])],
    ["<!doctype html><html lang=en><head><meta charset=utf-8>",
     "<meta name=viewport content=\"width=device-width,initial-scale=1\">",
     "<title>AsemLLM chat</title><style>", css(),
     ".m{padding:.5rem .8rem;margin:.5rem 0;border-radius:.6rem;border:1px solid var(--b)}"
     ".u{background:#4f46e51a}.m b{display:inline-block;min-width:4.5rem;color:var(--a)}"
     "nav a{margin-right:1rem;color:var(--a)}"
     ".web{border-top:1px solid var(--b);margin-top:1rem}.web a{color:var(--a)}"
     ".chk{display:flex;gap:.5rem;align-items:center}.chk input{width:auto}"
     ".hist{font-size:.9rem;margin:0 0 1rem}.hist a{margin-right:.8rem;color:var(--a)}.hist form{display:inline}"
     ".hist button{width:auto;padding:0 .4rem;margin:0}", "</style></head><body>",
     "<nav><a href=\"/\">Generate</a><a href=\"/chat\">Chat</a></nav>",
     "<h1>AsemLLM chat</h1>",
     render_history(Cid),
     "<p class=sub>A toy model: it sees only the last few characters, so replies sound like the training data ",
     "but may not answer you.</p>",
     info_box(model()),
     Bubbles,
     render_results(Found),
     case Error of undefined -> ""; E -> ["<p class=err>", esc(E), "</p>"] end,
     "<form method=post action=\"/chat\">",
     "<input type=hidden name=cid value=\"", esc(Cid), "\">",
     "<input type=hidden name=transcript value=\"", esc(trim_transcript(Transcript)), "\">",
     "<label for=message>Message</label>",
     "<input id=message name=message maxlength=500 autofocus autocomplete=off value=\"", esc(Draft), "\">",
     "<label class=chk><input type=checkbox name=web value=1", case Web of false -> ""; _ -> " checked" end,
     "> Also search (sends your message to ", tinyllm_search:provider_label(), ")</label>",
     "<label class=chk><input type=checkbox name=qwen value=1", case Web of qwen -> " checked"; _ -> "" end,
     "> Let Qwen (LM Studio, local) answer from the search results</label>",
     "<button>Send</button></form></body></html>"].

%% ---- saved conversations (Mnesia) ---------------------------------------------

%% The conversation id the form should carry: only ids that are really saved, else none.
keep(Cid, Stored) -> case {tinyllm_memory:valid_id(Cid), Stored} of {true, <<_, _/binary>>} -> Cid; _ -> <<>> end.

%% Persist a question/answer pair. On success show the saved conversation (what a reload will show);
%% on failure keep chatting from the trimmed transcript and say so.
save_turns(Cid, Turns, Fallback, Err) ->
    case tinyllm_memory:enabled() of
        false -> {Fallback, Err};
        true ->
            case tinyllm_memory:append(Cid, Turns) of
                ok -> {ok, T} = tinyllm_memory:transcript(Cid), {T, Err};
                {error, Why} ->
                    logger:warning("tinyllm: could not save conversation ~s: ~p", [Cid, Why]),
                    {Fallback, <<"This conversation could not be saved.">>}
            end
    end.

render_history(Cid) ->
    case tinyllm_memory:list(15) of
        {ok, []} -> "";
        {ok, L} ->
            ["<p class=hist><a href=\"/chat\"><b>+ New chat</b></a>",
             [["<span><a href=\"/chat?c=", Id, "\">", esc(history_title(T)), "</a>",
               case Id =:= Cid of
                   true -> ["<form method=post action=\"/chat/delete\"><input type=hidden name=cid value=\"", Id,
                            "\"><button title=\"delete this conversation\">x</button></form>"];
                   false -> ""
               end, "</span> "] || #{id := Id, title := T} <- L],
             "</p>"];
        _ -> ""
    end.

history_title(<<>>) -> <<"(empty)">>;
history_title(T) -> T.

conversations_api(<<"GET">>, undefined, Req) ->
    case tinyllm_memory:list(100) of
        {ok, L} -> json(200, #{conversations => L}, Req);
        {error, disabled} -> json(503, #{error => <<"conversation memory is disabled">>}, Req);
        {error, _} -> json(500, #{error => <<"could not read conversations">>}, Req)
    end;
conversations_api(<<"GET">>, Id, Req) ->
    case tinyllm_memory:get(Id) of
        {ok, C} -> json(200, C, Req);
        not_found -> json(404, #{error => <<"no such conversation">>}, Req);
        {error, bad_id} -> json(400, #{error => <<"bad conversation id">>}, Req);
        {error, disabled} -> json(503, #{error => <<"conversation memory is disabled">>}, Req);
        {error, _} -> json(500, #{error => <<"could not read the conversation">>}, Req)
    end;
conversations_api(<<"DELETE">>, Id, Req) when Id =/= undefined ->
    case tinyllm_memory:delete(Id) of
        ok -> json(200, #{deleted => Id}, Req);
        {error, bad_id} -> json(400, #{error => <<"bad conversation id">>}, Req);
        {error, disabled} -> json(503, #{error => <<"conversation memory is disabled">>}, Req);
        {error, _} -> json(500, #{error => <<"could not delete the conversation">>}, Req)
    end;
conversations_api(_, _, Req) -> json(405, #{error => <<"method not allowed">>}, Req).

run(M, Prompt, Len, Temp, Opts) ->
    T0 = erlang:monotonic_time(millisecond),
    Out = tinyllm:generate(M, Prompt, Len, Opts#{temperature => Temp}),
    Text = binary:part(Out, byte_size(Prompt), byte_size(Out) - byte_size(Prompt)),
    {printable(Text), erlang:monotonic_time(millisecond) - T0}.

%% Form values arrive as strings; turn them into the same shape the JSON API uses.
from_form(F) ->
    #{<<"prompt">> => maps:get(<<"prompt">>, F, <<>>),
      <<"length">> => num(maps:get(<<"length">>, F, <<"300">>)),
      <<"temperature">> => num(maps:get(<<"temperature">>, F, <<"0.7">>)),
      <<"seed">> => case maps:get(<<"seed">>, F, <<>>) of
                        <<>> -> null;
                        S -> num(S)
                    end}.

num(Bin) ->
    T = string:trim(Bin),
    try binary_to_integer(T)
    catch error:badarg ->
        try binary_to_float(T) catch error:badarg -> throw({bad, <<"not a number: ", T/binary>>}) end
    end.

validate(J) ->
    Prompt = maps:get(<<"prompt">>, J, <<>>),
    (is_binary(Prompt) andalso byte_size(Prompt) =< ?MAX_PROMPT)
        orelse throw({bad, <<"prompt must be text of at most 1000 bytes">>}),
    Len = maps:get(<<"length">>, J, 300),
    (is_integer(Len) andalso Len >= 1 andalso Len =< ?MAX_LEN)
        orelse throw({bad, <<"length must be a whole number from 1 to 2000">>}),
    Temp = maps:get(<<"temperature">>, J, 0.7),
    (is_number(Temp) andalso Temp >= 0.05 andalso Temp =< 2.0)
        orelse throw({bad, <<"temperature must be between 0.05 and 2.0">>}),
    Opts = case maps:get(<<"seed">>, J, null) of
               null -> #{};
               S when is_integer(S) -> #{seed => S};
               _ -> throw({bad, <<"seed must be a whole number">>})
           end,
    {Prompt, Len, float(Temp), Opts}.

info(#{cfg := Cfg} = M) ->
    Stats = maps:get(stats, M, #{}),
    Arch = maps:get(arch, Cfg, mlp),
    Base = #{arch => Arch, vocab => maps:get(vocab, Cfg), context => maps:get(context, Cfg),
             params => tinyllm:param_count(M),
             val_loss => maps:get(val_loss, Stats, null), bigram_loss => maps:get(bigram, Stats, null),
             unigram_loss => maps:get(unigram, Stats, null), steps => maps:get(steps, Stats, null),
             train_seconds => maps:get(seconds, Stats, null)},
    case Arch of
        mlp -> Base#{embed => maps:get(embed, Cfg), hidden => maps:get(hidden, Cfg)};
        transformer -> Base#{dim => maps:get(dim, Cfg), heads => maps:get(heads, Cfg),
                             layers => maps:get(layers, Cfg), ff => maps:get(ff, Cfg)}
    end.

%% The model works on bytes, so it can emit a partial UTF-8 sequence; make it valid text.
printable(Bin) ->
    case unicode:characters_to_binary(Bin) of
        B when is_binary(B) -> B;
        _ -> << <<(case C < 128 of true -> C; false -> $? end)>> || <<C>> <= Bin >>
    end.

html(Status, Body, Req) ->
    cowboy_req:reply(Status, #{<<"content-type">> => <<"text/html; charset=utf-8">>}, Body, Req).

json(Status, Term, Req) ->
    cowboy_req:reply(Status, #{<<"content-type">> => <<"application/json">>}, json:encode(Term), Req).

%% -------------------------------------------------------------- rendering

render(Fields) ->
    Prompt = maps:get(prompt, Fields, <<"-module(">>),
    Len = maps:get(length, Fields, <<"300">>),
    Temp = maps:get(temperature, Fields, <<"0.7">>),
    Seed = maps:get(seed, Fields, <<>>),
    ["<!doctype html><html lang=en><head><meta charset=utf-8>",
     "<meta name=viewport content=\"width=device-width,initial-scale=1\">",
     "<title>AsemLLM</title><style>", css(), "</style></head><body>",
     "<nav style=\"margin-bottom:1rem\"><a href=\"/\" style=\"margin-right:1rem;color:#4f46e5\">Generate</a>",
     "<a href=\"/chat\" style=\"color:#4f46e5\">Chat</a></nav>",
     "<h1>AsemLLM</h1><p class=sub>A tiny neural language model, trained and run in pure Erlang.</p>",
     info_box(model()),
     "<form method=post action=\"/\">",
     "<label for=prompt>Prompt</label>",
     "<textarea id=prompt name=prompt rows=3 maxlength=1000>", esc(Prompt), "</textarea>",
     "<div class=row>",
     field("length", "Length (characters)", "number", Len, "min=1 max=2000"),
     field("temperature", "Temperature (0.05 - 2)", "number", Temp, "min=0.05 max=2 step=0.05"),
     field("seed", "Seed (optional)", "number", Seed, "placeholder=random step=1"),
     "</div><button>Generate</button></form>",
     case maps:get(error, Fields, undefined) of
         undefined -> "";
         E -> ["<p class=err>", esc(E), "</p>"]
     end,
     case maps:get(result, Fields, undefined) of
         undefined -> "";
         {P, Text, Ms} ->
             ["<pre id=out><span class=p>", esc(P), "</span>", esc(Text), "</pre>",
              "<p class=meta>", integer_to_list(string:length(Text)), " characters in ",
              integer_to_list(Ms), " ms</p>"]
     end,
     "</body></html>"].

field(Name, Label, Type, Value, Extra) ->
    ["<div><label for=", Name, ">", Label, "</label><input id=", Name, " name=", Name, " type=", Type, " ",
     Extra, " value=\"", esc(Value), "\"></div>"].

info_box(undefined) -> ["<div class=info>", esc(?NO_MODEL), "</div>"];
info_box(M) ->
    #{params := P, vocab := V, context := N, arch := Arch} = I = info(M),
    Shape = case Arch of
                transformer -> [" &middot; transformer, ", integer_to_list(maps:get(layers, I)), " layers x ",
                                integer_to_list(maps:get(heads, I)), " heads, dim ", integer_to_list(maps:get(dim, I)),
                                ", window ", integer_to_list(N)];
                mlp -> [" &middot; context ", integer_to_list(N), " &middot; hidden ", integer_to_list(maps:get(hidden, I))]
            end,
    ["<div class=info>", integer_to_list(P), " parameters &middot; ", integer_to_list(V), " tokens", Shape,
     case I of
         #{val_loss := VL, bigram_loss := BL} when is_float(VL), is_float(BL) ->
             [" &middot; val loss ", float_to_list(VL, [{decimals, 3}]),
              " (bigram baseline ", float_to_list(BL, [{decimals, 3}]), ")"];
         _ -> ""
     end, "</div>"].

esc(B) when is_binary(B) -> esc(binary_to_list(unicode:characters_to_binary(B)), []);
esc(L) when is_list(L) -> esc(L, []).
esc([], Acc) -> lists:reverse(Acc);
esc([$& | T], Acc) -> esc(T, lists:reverse("&amp;", Acc));
esc([$< | T], Acc) -> esc(T, lists:reverse("&lt;", Acc));
esc([$> | T], Acc) -> esc(T, lists:reverse("&gt;", Acc));
esc([$" | T], Acc) -> esc(T, lists:reverse("&quot;", Acc));
esc([C | T], Acc) -> esc(T, [C | Acc]).

css() ->
    ":root{color-scheme:light dark;--b:#8884;--a:#4f46e5}"
    "body{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}"
    "h1{margin:0 0 .25rem}.sub{color:#888;margin:0 0 1.2rem}"
    ".info{font-size:.85rem;color:#888;border:1px solid var(--b);border-radius:.5rem;padding:.6rem .8rem;margin-bottom:1rem}"
    "label{display:block;font-size:.85rem;margin:.8rem 0 .2rem}"
    "textarea,input{width:100%;box-sizing:border-box;padding:.5rem;border:1px solid var(--b);border-radius:.5rem;"
    "background:transparent;color:inherit;font:14px ui-monospace,Menlo,monospace}"
    ".row{display:flex;gap:1rem;flex-wrap:wrap}.row>div{flex:1;min-width:9rem}"
    "button{margin-top:1rem;padding:.55rem 1.2rem;border:0;border-radius:.5rem;background:var(--a);color:#fff;font-size:1rem;cursor:pointer}"
    "pre{white-space:pre-wrap;word-break:break-word;font:14px/1.45 ui-monospace,Menlo,monospace;margin-top:1.2rem;"
    "padding:.8rem;border:1px solid var(--b);border-radius:.5rem}"
    ".p{color:var(--a);font-weight:600}.meta{font-size:.8rem;color:#888}.err{color:#d33}".
