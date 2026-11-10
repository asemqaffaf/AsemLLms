-module(tinyllm_tests).
-include_lib("eunit/include/eunit.hrl").

setup() ->
    application:set_env(tinyllm, memory_storage, ram),   %% tests must not touch data/mnesia
    {ok, _} = application:ensure_all_started(tinyllm), ok.
cleanup(_) -> application:stop(tinyllm).

all_test_() ->
    {setup, fun setup/0, fun cleanup/1,
     {timeout, 300,
      [fun gradient_check/0, fun softmax_sums_to_one/0, fun learns_a_cycle/0,
       fun uses_context_beyond_bigram/0, fun survives_worker_crash/0,
       fun save_load_roundtrip/0, fun generate_stays_in_vocab/0, fun progress_logging/0, fun load_in_fresh_vm/0, fun web_ui_and_api/0, fun web_rejects_foreign_host_and_origin/0, fun chat_function/0, {timeout, 300, fun tf_gradient_check/0}, {timeout, 300, fun tf_learns_a_cycle/0}, {timeout, 300, fun tf_uses_long_range_context/0}, {timeout, 300, fun tf_save_load_generate/0}, {timeout, 300, fun tf_web_info/0}, fun search_parsers/0, fun search_client/0, fun search_in_web_ui/0, fun qwen_answers_from_search/0, fun tokenizer_roundtrip/0, fun word_vocab/0, fun word_model_learns/0, fun legacy_model_loads/0, fun chat_page_and_openai_api/0, fun conversation_memory/0, fun chat_remembers_conversations/0, fun shards_cover_batch/0]}}.

%% ---- backprop is correct: analytic gradient == numeric gradient --------------

gradient_check() ->
    Cfg = #{vocab => 5, context => 3, embed => 2, hidden => 4},
    P = tinyllm_nn:init(Cfg, 7),
    Ex = [{[0, 1, 2], 3}, {[4, 4, 1], 0}, {[2, 0, 3], 1}, {[1, 2, 2], 4}],
    {_, G} = tinyllm_nn:loss_grad(tinyllm_nn:prepare(P), Cfg, Ex),
    Loss = fun(Pm) -> tinyllm_nn:loss_only(tinyllm_nn:prepare(Pm), Ex) end,
    Eps = 1.0e-5,
    Checks = [{e, 0, 1}, {e, 4, 0}, {w1, 0, 0}, {w1, 3, 5}, {b1, 2, none},
              {w2, 0, 1}, {w2, 4, 3}, {b2, 1, none}, {b2, 3, none}],
    [begin
         Num = (Loss(bump(P, K, R, C, Eps)) - Loss(bump(P, K, R, C, -Eps))) / (2 * Eps),
         Ana = get(G, K, R, C),
         ?assert(abs(Num - Ana) < 1.0e-6 + 1.0e-4 * abs(Num))
     end || {K, R, C} <- Checks].

get(G, K, R, none) -> lists:nth(R + 1, maps:get(K, G));
get(G, K, R, C) -> lists:nth(C + 1, lists:nth(R + 1, maps:get(K, G))).

bump(P, K, R, none, D) -> P#{K := set(maps:get(K, P), R, fun(X) -> X + D end)};
bump(P, K, R, C, D) ->
    P#{K := set(maps:get(K, P), R, fun(Row) -> set(Row, C, fun(X) -> X + D end) end)}.

set(L, I, F) -> {A, [X | B]} = lists:split(I, L), A ++ [F(X) | B].

softmax_sums_to_one() ->
    P = tinyllm_nn:softmax([1.0, 2.0, 3.0, -50.0, 800.0], 1.0),
    ?assert(abs(lists:sum(P) - 1.0) < 1.0e-9),
    ?assert(abs(lists:sum(tinyllm_nn:softmax([1.0, 2.0], 0.1)) - 1.0) < 1.0e-9).

shards_cover_batch() ->
    %% every training example is processed exactly once across workers
    Ids = << <<X:16>> || X <- [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3] >>,
    Ex = tinyllm_nn:examples(Ids, 2, [2, 3, 4, 5, 6]),
    ?assertEqual([{[0, 1], 2}, {[1, 2], 3}, {[2, 3], 0}, {[3, 0], 1}, {[0, 1], 2}], Ex).

%% ---- training actually learns ------------------------------------------------

toy_opts() ->
    #{context => 4, embed => 4, hidden => 16, workers => 2, batch => 64,
      steps => 250, lr => 0.02, eval_every => 1000, val_examples => 100,
      verbose => false}.

learns_a_cycle() ->
    Corpus = binary:copy(<<"abcdefgh">>, 400),
    {ok, M} = tinyllm:train(Corpus, toy_opts()),
    #{stats := #{val_loss := V, unigram := U}} = M,
    ?assert(V < 0.1),
    ?assert(U > 1.5),
    ?assertEqual(<<"abcdefghabcdefghabcd">>,
                 tinyllm:generate(M, <<"abcd">>, 16, #{temperature => 0.05, seed => 1})).

uses_context_beyond_bigram() ->
    %% "aab" repeated: a bigram cannot tell which 'a' it is on; a 4-char context can.
    Corpus = binary:copy(<<"aab">>, 1200),
    {ok, M} = tinyllm:train(Corpus, (toy_opts())#{steps => 300}),
    #{stats := #{val_loss := V, bigram := B}} = M,
    ?assert(B > 0.4),
    ?assert(V < 0.1),
    ?assert(V < B - 0.25).

%% ---- fault tolerance ---------------------------------------------------------

survives_worker_crash() ->
    Self = self(),
    Corpus = binary:copy(<<"abcdefgh">>, 400),
    Killer = spawn_link(fun() ->
                 timer:sleep(150),
                 Old = whereis(tinyllm_worker_1),
                 exit(Old, kill),
                 wait_restart(Old),
                 Self ! restarted
             end),
    {ok, M} = tinyllm:train(Corpus, (toy_opts())#{steps => 700, batch => 128}),
    receive restarted -> ok after 5000 -> ?assert(false) end,
    _ = Killer,
    ?assert(maps:get(val_loss, maps:get(stats, M)) < 0.1).

wait_restart(Old) ->
    case whereis(tinyllm_worker_1) of
        Pid when is_pid(Pid), Pid =/= Old -> ok;
        _ -> timer:sleep(20), wait_restart(Old)
    end.

%% ---- persistence & generation ------------------------------------------------

save_load_roundtrip() ->
    {ok, M} = tinyllm:train(binary:copy(<<"abcdefgh">>, 200), (toy_opts())#{steps => 20}),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_test_model.bin"),
    ok = tinyllm:save(M, Path),
    M2 = tinyllm:load(Path),
    file:delete(Path),
    ?assertEqual(M, M2),
    ?assertEqual(tinyllm:generate(M, <<"abcd">>, 30, #{seed => 5}),
                 tinyllm:generate(M2, <<"abcd">>, 30, #{seed => 5})).

generate_stays_in_vocab() ->
    {ok, M} = tinyllm:train(<<"hello world, hello erlang! ", (binary:copy(<<"hello world, hello erlang! ">>, 100))/binary>>,
                            (toy_opts())#{steps => 30}),
    Out = tinyllm:generate(M, <<"zzz unknown prompt">>, 200, #{temperature => 1.5}),
    Vocab = maps:get(vocab, M),
    Generated = binary:part(Out, byte_size(<<"zzz unknown prompt">>), 200),
    ?assert(lists:all(fun(B) -> lists:member(<<B>>, Vocab) end, binary_to_list(Generated))).

progress_logging() ->
    %% the verbose path prints progress lines; make sure its format strings are valid
    {ok, _} = tinyllm:train(binary:copy(<<"abcdefgh">>, 200),
                            (toy_opts())#{steps => 20, eval_every => 10, verbose => true}).

%% A saved model must load in a brand-new VM (where none of the model's atoms exist yet).
load_in_fresh_vm() ->
    {ok, M} = tinyllm:train(binary:copy(<<"abcdefgh">>, 200), (toy_opts())#{steps => 10}),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_fresh_vm_model.bin"),
    ok = tinyllm:save(M, Path),
    Pa = lists:append([["-pa ", P, " "] || P <- code:get_path(), filelib:is_dir(P),
                       string:find(P, "tinyllm") =/= nomatch]),
    Cmd = lists:flatten(["erl -noshell ", Pa, "-eval '",
                         "M = tinyllm:load(\"", Path, "\"), io:format(\"~p\", [maps:is_key(params, M)]), halt().'"]),
    Out = os:cmd(Cmd),
    file:delete(Path),
    ?assertEqual("true", Out).

%% ---- web UI (server-rendered HTML) and JSON API -------------------------------

http(Method, Path, Body, Type) ->
    Url = "http://127.0.0.1:18765" ++ Path,
    Req = case Method of
              get -> {Url, []};
              post -> {Url, [], Type, Body}
          end,
    {ok, {{_, Status, _}, _, RespBody}} =
        httpc:request(Method, Req, [], [{body_format, binary}]),
    {Status, RespBody}.

web_rejects_foreign_host_and_origin() ->
    {ok, _} = application:ensure_all_started(inets),
    ok = tinyllm_web:start(18765),
    try
        Url = "http://127.0.0.1:18765/api/info",
        Status = fun(Headers, HttpOpts) ->
                     {ok, {{_, S, _}, _, _}} = httpc:request(get, {Url, Headers}, HttpOpts, []),
                     S
                 end,
        ?assertNotEqual(403, Status([], [])),
        ?assertEqual(403, Status([{"origin", "https://evil.example"}], [])),
        ?assertNotEqual(403, Status([{"origin", "http://localhost:8080"}], [])),
        %% DNS rebinding: a foreign Host header
        ?assertEqual(403, Status([{"host", "evil.example"}], []))
    after
        tinyllm_web:stop()
    end.

web_ui_and_api() ->
    {ok, _} = application:ensure_all_started(inets),
    {ok, M} = tinyllm:train(binary:copy(<<"abcdefgh">>, 200), (toy_opts())#{steps => 30}),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_web_test_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    ok = tinyllm_web:start(18765),
    try
        {200, Page} = http(get, "/", <<>>, ""),
        ?assertMatch({_, _}, binary:match(iolist_to_binary(Page), <<"<form method=post">>)),
        ?assertEqual(nomatch, binary:match(iolist_to_binary(Page), <<"<script">>)),
        %% form submit renders the result page
        {200, R1} = http(post, "/", <<"prompt=abcd&length=25&temperature=0.5&seed=3">>,
                         "application/x-www-form-urlencoded"),
        ?assertMatch({_, _}, binary:match(R1, <<"25 characters">>)),
        %% user text is escaped, never injected as markup
        {200, R2} = http(post, "/", <<"prompt=%3Cscript%3Ealert(1)%3C%2Fscript%3E&length=5&temperature=0.5">>,
                         "application/x-www-form-urlencoded"),
        ?assertEqual(nomatch, binary:match(R2, <<"<script>alert">>)),
        ?assertMatch({_, _}, binary:match(R2, <<"&lt;script&gt;">>)),
        %% bad input -> 400 and a readable message on the page
        {400, R3} = http(post, "/", <<"prompt=x&length=abc&temperature=0.5">>, "application/x-www-form-urlencoded"),
        ?assertMatch({_, _}, binary:match(R3, <<"not a number: abc">>)),
        {400, _} = http(post, "/", <<"prompt=x&length=99999&temperature=0.5">>, "application/x-www-form-urlencoded"),
        {400, _} = http(post, "/", <<"prompt=x&length=5&temperature=9">>, "application/x-www-form-urlencoded"),
        %% JSON API
        {200, I} = http(get, "/api/info", <<>>, ""),
        ?assertMatch(#{<<"params">> := _, <<"vocab">> := 8}, json:decode(I)),
        {200, G} = http(post, "/api/generate", <<"{\"prompt\":\"abcd\",\"length\":16,\"temperature\":0.1,\"seed\":1}">>,
                        "application/json"),
        ?assertEqual(<<"efghabcdefghabcd">>, maps:get(<<"text">>, json:decode(G))),
        {400, _} = http(post, "/api/generate", <<"not json">>, "application/json")
    after
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        file:delete(Path)
    end.

%% ---- chat: stop sequences, chat page, OpenAI-compatible endpoint ----------------

dialogue_model() ->
    {ok, M} = tinyllm:train(binary:copy(<<"User: hi\nAssistant: yo\n">>, 300),
                            (toy_opts())#{context => 12, steps => 200}),
    M.

chat_function() ->
    M = dialogue_model(),
    ?assertEqual({<<"yo">>, stop},
                 tinyllm:chat(M, [#{role => user, content => <<"hi">>}], #{temperature => 0.05, seed => 1})),
    %% atom or binary keys both work; system messages are ignored
    ?assertEqual({<<"yo">>, stop},
                 tinyllm:chat(M, [#{<<"role">> => <<"system">>, <<"content">> => <<"be nice">>},
                                  #{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}],
                              #{temperature => 0.05, seed => 1})),
    %% stop strings are cut off the output; without one it runs to the length limit
    ?assertEqual({<<" yo">>, stop},
                 tinyllm:complete(M, <<"User: hi\nAssistant:">>, 50, #{temperature => 0.05, seed => 1, stop => <<"\n">>})),
    ?assertMatch({_, length}, tinyllm:complete(M, <<"User: hi\nAssistant:">>, 10, #{temperature => 0.05, seed => 1})).

chat_page_and_openai_api() ->
    {ok, _} = application:ensure_all_started(inets),
    M = dialogue_model(),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_chat_test_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    ok = tinyllm_web:start(18765),
    Form = "application/x-www-form-urlencoded",
    try
        %% OpenAI-compatible
        {200, Models} = http(get, "/v1/models", <<>>, ""),
        ?assertMatch(#{<<"data">> := [#{<<"id">> := <<"AsemLLM">>}]}, json:decode(Models)),
        Body = <<"{\"model\":\"AsemLLM\",\"temperature\":0.05,\"seed\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}">>,
        {200, C} = http(post, "/v1/chat/completions", Body, "application/json"),
        ?assertMatch(#{<<"object">> := <<"chat.completion">>,
                       <<"choices">> := [#{<<"message">> := #{<<"role">> := <<"assistant">>, <<"content">> := <<"yo">>},
                                           <<"finish_reason">> := <<"stop">>}]},
                     json:decode(C)),
        {400, _} = http(post, "/v1/chat/completions", <<"{\"messages\":[]}">>, "application/json"),
        {400, _} = http(post, "/v1/chat/completions", <<"{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}">>, "application/json"),
        {400, _} = http(post, "/v1/chat/completions", <<"{\"messages\":[{\"role\":\"user\"}]}">>, "application/json"),
        %% chat page: a turn appends to the transcript, user text is escaped
        {200, P1} = http(post, "/chat", <<"transcript=&message=hi">>, Form),
        ?assertMatch({_, _}, binary:match(P1, <<"<b>You</b> hi">>)),
        ?assertMatch({_, _}, binary:match(P1, <<"<b>AsemLLM</b>">>)),
        {200, P2} = http(post, "/chat", <<"transcript=&message=%3Cb%3Ebold%3C%2Fb%3E">>, Form),
        ?assertEqual(nomatch, binary:match(P2, <<"<b>bold</b>">>)),
        {400, _} = http(post, "/chat", iolist_to_binary(["transcript=&message=", lists:duplicate(600, $a)]), Form)
    after
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        file:delete(Path)
    end.

%% ---- conversation memory (Mnesia) ------------------------------------------------

clear_conversations() ->
    {ok, All} = tinyllm_memory:list(1000),
    [ok = tinyllm_memory:delete(Id) || #{id := Id} <- All],
    ok.

conversation_memory() ->
    ?assert(tinyllm_memory:enabled()),
    clear_conversations(),
    Id = tinyllm_memory:new_id(),
    ?assert(tinyllm_memory:valid_id(Id)),
    ?assertEqual(not_found, tinyllm_memory:get(Id)),
    ok = tinyllm_memory:append(Id, [{<<"User">>, <<"hello there">>}, {<<"Assistant">>, <<"hi">>}]),
    ok = tinyllm_memory:append(Id, [{<<"User">>, <<"second">>}, {<<"Qwen">>, <<"answer">>}]),
    {ok, #{title := <<"hello there">>, turns := Turns}} = tinyllm_memory:get(Id),
    ?assertEqual([<<"User">>, <<"Assistant">>, <<"User">>, <<"Qwen">>], [R || #{role := R} <- Turns]),
    ?assertEqual({ok, <<"User: hello there\nAssistant: hi\nUser: second\nQwen: answer\n">>}, tinyllm_memory:transcript(Id)),
    {ok, [#{id := Id, turns := 4}]} = tinyllm_memory:list(),
    %% bad ids never reach the database; unknown ones are simply absent
    ?assertEqual({error, bad_id}, tinyllm_memory:get(<<"../etc/passwd">>)),
    ?assertEqual({error, bad_id}, tinyllm_memory:append(<<"x">>, [{<<"User">>, <<"a">>}])),
    %% concurrent writers do not lose turns (transactions)
    Self = self(),
    [spawn(fun() -> ok = tinyllm_memory:append(Id, [{<<"User">>, integer_to_binary(N)}]), Self ! done end) || N <- lists:seq(1, 20)],
    [receive done -> ok after 5000 -> error(timeout) end || _ <- lists:seq(1, 20)],
    {ok, #{turns := All}} = tinyllm_memory:get(Id),
    ?assertEqual(24, length(All)),
    ok = tinyllm_memory:delete(Id),
    ?assertEqual(not_found, tinyllm_memory:get(Id)),
    {ok, []} = tinyllm_memory:list().

chat_remembers_conversations() ->
    clear_conversations(),
    {ok, _} = application:ensure_all_started(inets),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_memory_test_model.bin"),
    ok = tinyllm:save(dialogue_model(), Path),
    application:set_env(tinyllm, model_file, Path),
    ok = tinyllm_web:start(18765),
    Form = "application/x-www-form-urlencoded",
    CidOf = fun(Page) ->
                {match, [Cid]} = re:run(Page, <<"name=cid value=\"([0-9a-f]{16})\"">>, [{capture, all_but_first, binary}]),
                Cid
            end,
    try
        {200, P1} = http(post, "/chat", <<"cid=&message=hi">>, Form),
        Cid = CidOf(P1),
        %% the second turn only carries the id: the history comes from the database
        {200, P2} = http(post, "/chat", <<"cid=", Cid/binary, "&transcript=&message=%3Cb%3Ex">>, Form),
        ?assertEqual(Cid, CidOf(P2)),
        ?assertMatch({_, _}, binary:match(P2, <<"<b>You</b> hi">>)),
        ?assertEqual(nomatch, binary:match(P2, <<"<b>x">>)),
        %% reopening it later, as after a restart, shows both turns
        {200, P3} = http(get, "/chat?c=" ++ binary_to_list(Cid), <<>>, ""),
        ?assertMatch({_, _}, binary:match(P3, <<"<b>You</b> hi">>)),
        ?assertMatch({_, _}, binary:match(P3, <<"&lt;b&gt;x">>)),
        {200, L} = http(get, "/api/conversations", <<>>, ""),
        ?assertMatch(#{<<"conversations">> := [#{<<"id">> := Cid, <<"turns">> := 4, <<"title">> := <<"hi">>}]}, json:decode(L)),
        {200, One} = http(get, "/api/conversations/" ++ binary_to_list(Cid), <<>>, ""),
        ?assertMatch(#{<<"turns">> := [#{<<"role">> := <<"User">>, <<"content">> := <<"hi">>} | _]}, json:decode(One)),
        {404, _} = http(get, "/api/conversations/0123456789abcdef", <<>>, ""),
        {400, _} = http(get, "/api/conversations/nope", <<>>, ""),
        %% an unknown id in the form starts a fresh conversation instead of failing
        {200, P4} = http(post, "/chat", <<"cid=0123456789abcdef&message=yo">>, Form),
        ?assertNotEqual(Cid, CidOf(P4)),
        %% delete (303 back to /chat; httpc follows it)
        {200, _} = http(post, "/chat/delete", <<"cid=", Cid/binary>>, Form),
        {404, _} = http(get, "/api/conversations/" ++ binary_to_list(Cid), <<>>, "")
    after
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        file:delete(Path),
        {ok, Rest} = tinyllm_memory:list(),
        [tinyllm_memory:delete(Id) || #{id := Id} <- Rest]
    end.

%% ---- word-level tokenizer and model ---------------------------------------------

tokenizer_roundtrip() ->
    Text = <<"User: How do I cook pasta?\nAssistant: Boil it, then drain. It's 9 minutes!\n\n  two  spaces\tand tab \xc3\xa9">>,
    Tokens = tinyllm_tok:tokenize(word, Text),
    ?assertEqual(Text, tinyllm_tok:detokenize(Tokens)),
    ?assertEqual([<<"User">>, <<":">>, <<" How">>, <<" do">>, <<" I">>, <<" cook">>, <<" pasta">>, <<"?">>, <<"\n">>],
                 lists:sublist(Tokens, 9)),
    ?assertEqual(Text, tinyllm_tok:detokenize(tinyllm_tok:tokenize(char, Text))).

word_vocab() ->
    Tokens = [<<"a">>, <<"b">>, <<"a">>, <<"c">>, <<"a">>, <<"b">>],
    ?assertEqual([<<"<unk>">>, <<"a">>, <<"b">>],
                 tinyllm_tok:build_vocab(word, Tokens, #{max_vocab => 10, min_count => 2})),
    ?assertEqual([<<"<unk>">>, <<"a">>],
                 tinyllm_tok:build_vocab(word, Tokens, #{max_vocab => 2, min_count => 1})).

word_model_learns() ->
    Corpus = binary:copy(<<"the cat sat on the mat. ">>, 300),
    {ok, M} = tinyllm:train(Corpus, (toy_opts())#{tokenizer => word, context => 3, steps => 300}),
    #{stats := #{val_loss := V}, vocab := Vocab, tokenizer := word} = M,
    ?assert(V < 0.1),
    ?assertEqual(<<"<unk>">>, hd(Vocab)),
    %% prompt at least as long as the context window (shorter prompts are padded with
    %% a token the model never saw in that position)
    Out = tinyllm:generate(M, <<"the cat sat on the">>, 9, #{temperature => 0.05, seed => 1}),
    ?assertEqual(<<"the cat sat on the mat. the cat sat on the mat.">>, Out),
    %% <unk> is never produced, even at high temperature
    Wild = tinyllm:generate(M, <<"the">>, 100, #{temperature => 2.0, seed => 2}),
    ?assertEqual(nomatch, binary:match(Wild, <<"<unk>">>)).

%% Models saved before word-level support (vocab = list of byte values) still load.
legacy_model_loads() ->
    {ok, M} = tinyllm:train(binary:copy(<<"abcdefgh">>, 200), (toy_opts())#{steps => 20}),
    Legacy = maps:remove(tokenizer, M#{vocab := [B || <<B>> <- maps:get(vocab, M)]}),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_legacy_model.bin"),
    ok = file:write_file(Path, term_to_binary(Legacy)),
    M2 = tinyllm:load(Path),
    file:delete(Path),
    ?assertEqual(M, M2),
    ?assertEqual(tinyllm:generate(M, <<"abcd">>, 20, #{seed => 3}), tinyllm:generate(M2, <<"abcd">>, 20, #{seed => 3})).

%% ---- web search (keyless APIs; offline tests use a saved real Wikipedia response) ------

wiki() ->
    {ok, B} = file:read_file(filename:join([filename:dirname(code:which(?MODULE)), "..", "test", "wikipedia_fixture.json"])),
    B.

search_parsers() ->
    [First | _] = Rs = tinyllm_search:parse_wikipedia(json:decode(wiki())),
    ?assertEqual(3, length(Rs)),
    ?assertMatch(#{title := <<"Erlang (programming language)">>,
                   url := <<"https://en.wikipedia.org/wiki/Erlang_%28programming_language%29">>}, First),
    ?assertEqual(nomatch, binary:match(maps:get(snippet, First), <<"<span">>)),   %% HTML stripped
    ?assert(byte_size(maps:get(snippet, First)) > 100),                           %% a real paragraph, not a fragment
    ?assertEqual([], tinyllm_search:parse_wikipedia(#{<<"nothing">> => 1})),
    %% SearXNG: http(s) results only, tags stripped
    Sx = #{<<"results">> => [#{<<"title">> => <<"A <b>page</b>">>, <<"url">> => <<"https://example.org/a">>, <<"content">> => <<"x &amp; y">>},
                             #{<<"title">> => <<"evil">>, <<"url">> => <<"javascript:alert(1)">>, <<"content">> => <<"no">>}]},
    ?assertEqual([#{title => <<"A page">>, url => <<"https://example.org/a">>, snippet => <<"x & y">>}],
                 tinyllm_search:parse_searxng(Sx)),
    ?assertEqual("https://www.google.com/search?q=a%20b%26c", tinyllm_search:google_url(<<"a b&c">>)).

search_client() ->
    Url = fake_server("200 OK", wiki()),
    {ok, Rs} = tinyllm_search:search(<<"erlang language">>, #{provider => wikipedia, base_url => Url, max_results => 2}),
    ?assertEqual(3, length(Rs)),   %% fixture has 3; server decides the count
    receive {request, Raw} ->
        ?assertMatch({_, _}, binary:match(Raw, <<"gsrsearch=erlang%20language">>)),
        ?assertMatch({_, _}, binary:match(Raw, <<"gsrlimit=2">>))
    after 1000 -> ?assert(false) end,
    SxBody = iolist_to_binary(json:encode(#{results => [#{title => <<"T">>, url => <<"https://e.org">>, content => <<"c">>}]})),
    Url2 = fake_server("200 OK", SxBody),
    ?assertMatch({ok, [#{url := <<"https://e.org">>}]}, tinyllm_search:search(<<"q">>, #{provider => searxng, base_url => Url2})),
    receive {request, Raw2} -> ?assertMatch({_, _}, binary:match(Raw2, <<"format=json">>)) after 1000 -> ?assert(false) end,
    Url3 = fake_server("429 Too Many Requests", <<"slow down">>),
    ?assertEqual({error, rate_limited}, tinyllm_search:search(<<"x">>, #{provider => wikipedia, base_url => Url3})),
    Url4 = fake_server("200 OK", <<"not json">>),
    ?assertEqual({error, bad_response}, tinyllm_search:search(<<"x">>, #{provider => wikipedia, base_url => Url4})),
    ?assertEqual({error, unreachable}, tinyllm_search:search(<<"x">>, #{provider => searxng, base_url => "http://127.0.0.1:9"})).

search_in_web_ui() ->
    {ok, _} = application:ensure_all_started(inets),
    M = dialogue_model(),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_search_test_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    application:set_env(tinyllm, search_provider, wikipedia),
    ok = tinyllm_web:start(18765),
    Form = "application/x-www-form-urlencoded",
    try
        application:set_env(tinyllm, search_url, fake_server("200 OK", wiki())),
        {200, J} = http(get, "/api/search?q=erlang", <<>>, ""),
        ?assertMatch(#{<<"results">> := [#{<<"title">> := <<"Erlang (programming language)">>} | _]}, json:decode(J)),
        {400, _} = http(get, "/api/search?q=", <<>>, ""),
        application:set_env(tinyllm, search_url, fake_server("200 OK", wiki())),
        {200, P} = http(post, "/chat", <<"transcript=&message=erlang+language&web=1">>, Form),
        ?assertMatch({_, _}, binary:match(P, <<"Web results for">>)),
        ?assertMatch({_, _}, binary:match(P, <<"https://en.wikipedia.org/wiki/Erlang_%28programming_language%29">>)),
        ?assertMatch({_, _}, binary:match(P, <<"open in Google">>)),
        ?assertMatch({_, _}, binary:match(P, <<"https://www.google.com/search?q=erlang%20language">>)),
        ?assertMatch({_, _}, binary:match(P, <<"rel=\"noopener noreferrer nofollow\"">>)),
        ?assertMatch({_, _}, binary:match(P, <<"checked">>)),
        %% a failing search degrades gracefully: the chat reply is still shown
        application:set_env(tinyllm, search_url, fake_server("429 Too Many Requests", <<"x">>)),
        {200, B} = http(post, "/chat", <<"transcript=&message=hi&web=1">>, Form),
        ?assertMatch({_, _}, binary:match(B, <<"rate-limiting">>)),
        ?assertMatch({_, _}, binary:match(B, <<"<b>AsemLLM</b>">>)),
        %% without the checkbox no search is made
        {200, N} = http(post, "/chat", <<"transcript=&message=hi">>, Form),
        ?assertEqual(nomatch, binary:match(N, <<"Web results">>))
    after
        application:unset_env(tinyllm, search_url),
        application:unset_env(tinyllm, search_provider),
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        file:delete(Path)
    end.

%% ---- Qwen (LM Studio) answers from the search results ---------------------------------

collect_requests(0) -> [];
collect_requests(N) -> receive {request, R} -> [R | collect_requests(N - 1)] after 3000 -> [] end.

flush_requests() ->
    receive {request, _} -> flush_requests() after 0 -> ok end.

qwen_answers_from_search() ->
    flush_requests(),   %% earlier tests may have left unread requests in this process's mailbox
    {ok, _} = application:ensure_all_started(inets),
    M = dialogue_model(),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_qwen_test_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    application:set_env(tinyllm, search_provider, wikipedia),
    application:set_env(tinyllm, lmstudio_model, <<"test-model">>),
    ok = tinyllm_web:start(18765),
    Form = "application/x-www-form-urlencoded",
    Reply = fun(Content) ->
                iolist_to_binary(json:encode(#{choices => [#{message => #{role => <<"assistant">>, content => Content}}]}))
            end,
    try
        application:set_env(tinyllm, search_url, fake_server("200 OK", wiki())),
        application:set_env(tinyllm, lmstudio_url, fake_server("200 OK", Reply(<<"<think>hmm</think>Erlang is a language [1]. <img src=x onerror=alert(1)>">>))),
        {200, P} = http(post, "/chat", <<"transcript=&message=what+is+erlang&web=1&qwen=1">>, Form),
        ?assertMatch({_, _}, binary:match(P, <<"<b>Qwen</b> Erlang is a language [1].">>)),
        ?assertEqual(nomatch, binary:match(P, <<"<think>">>)),
        ?assertEqual(nomatch, binary:match(P, <<"<img src=x">>)),            %% model output is escaped
        ?assertMatch({_, _}, binary:match(P, <<"&lt;img src=x">>)),
        ?assertMatch({_, _}, binary:match(P, <<"Web results for">>)),
        [_SearchReq, LmReq] = collect_requests(2),
        [_, LmBody] = binary:split(LmReq, <<"\r\n\r\n">>),
        Sent = json:decode(LmBody),
        ?assertMatch(#{<<"model">> := <<"test-model">>, <<"stream">> := false, <<"reasoning_effort">> := <<"none">>}, Sent),
        [Sys, User] = maps:get(<<"messages">>, Sent),
        ?assertMatch({_, _}, binary:match(maps:get(<<"content">>, Sys), <<"untrusted data">>)),
        UserText = maps:get(<<"content">>, User),
        ?assertMatch({_, _}, binary:match(UserText, <<"Question: what is erlang">>)),
        ?assertMatch({_, _}, binary:match(UserText, <<"[1] Erlang (programming language) - ">>)),
        ?assertMatch({_, _}, binary:match(UserText, <<"https://en.wikipedia.org/wiki/Erlang_%28programming_language%29">>)),
        %% LM Studio down -> readable notice, tinyllm still replies
        application:set_env(tinyllm, search_url, fake_server("200 OK", wiki())),
        application:set_env(tinyllm, lmstudio_url, "http://127.0.0.1:9"),
        {200, D} = http(post, "/chat", <<"transcript=&message=hi&web=1&qwen=1">>, Form),
        ?assertMatch({_, _}, binary:match(D, <<"Qwen is unavailable">>)),
        ?assertMatch({_, _}, binary:match(D, <<"<b>AsemLLM</b>">>)),
        %% Qwen box without any search result -> no Qwen call, tinyllm replies
        application:set_env(tinyllm, search_url, fake_server("200 OK", <<"{\"batchcomplete\":\"\"}">>)),
        {200, E} = http(post, "/chat", <<"transcript=&message=zzzz&web=1&qwen=1">>, Form),
        ?assertMatch({_, _}, binary:match(E, <<"No search results to answer from">>)),
        ?assertMatch({_, _}, binary:match(E, <<"<b>AsemLLM</b>">>))
    after
        [application:unset_env(tinyllm, K) || K <- [search_url, lmstudio_url, lmstudio_model, search_provider]],
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        file:delete(Path)
    end.

%% ---- transformer ------------------------------------------------------------------

tf_gradient_check() ->
    Cfg = #{vocab => 7, context => 4, dim => 4, heads => 2, layers => 2, ff => 6},
    P0 = tinyllm_tf:init(Cfg, 5),
    Seqs = [{[0, 1, 2, 3], [1, 2, 3, 4]}, {[5, 5, 1, 0], [5, 1, 0, 6]}, {[2, 6, 6, 3], [6, 6, 3, 2]}],
    {Loss, G} = tinyllm_tf:loss_grad(tinyllm_tf:prepare(P0), Cfg, Seqs),
    ?assert(abs(Loss - tinyllm_tf:loss_only(tinyllm_tf:prepare(P0), Cfg, Seqs)) < 1.0e-9),
    LossFn = fun(P) -> tinyllm_tf:loss_only(tinyllm_tf:prepare(P), Cfg, Seqs) end,
    Eps = 1.0e-5,
    %% every tensor (attention, layer norms, MLP, both embeddings, output): analytic == numeric
    Bad = [{K, Pt} || K <- maps:keys(P0), Pt <- tf_points(maps:get(K, P0)),
                      begin
                          Num = (LossFn(tf_bump(P0, K, Pt, Eps)) - LossFn(tf_bump(P0, K, Pt, -Eps))) / (2 * Eps),
                          Ana = tf_get(maps:get(K, G), Pt),
                          abs(Num - Ana) > 1.0e-7 + 1.0e-4 * (abs(Num) + abs(Ana))
                      end],
    ?assertEqual([], Bad),
    ?assertEqual(30, maps:size(P0)).

tf_points([R | _] = M) when is_list(R) -> [{I, J} || I <- tf_idx(length(M)), J <- tf_idx(length(R))];
tf_points(V) -> [{I} || I <- tf_idx(length(V))].
tf_idx(N) -> lists:usort([0, N div 2, N - 1]).
tf_get(M, {I}) -> lists:nth(I + 1, M);
tf_get(M, {I, J}) -> lists:nth(J + 1, lists:nth(I + 1, M)).
tf_bump(P, K, Pt, D) -> P#{K := tf_upd(maps:get(K, P), Pt, D)}.
tf_upd(V, {I}, D) -> set(V, I, fun(X) -> X + D end);
tf_upd(M, {I, J}, D) -> set(M, I, fun(R) -> set(R, J, fun(X) -> X + D end) end).

tf_opts() ->
    #{arch => transformer, tokenizer => word, context => 8, dim => 16, heads => 2, layers => 1, ff => 32,
      workers => 2, batch => 16, steps => 150, lr => 0.01, eval_every => 1000, val_examples => 50,
      verbose => false, min_count => 1}.

tf_learns_a_cycle() ->
    Corpus = binary:copy(<<"the cat sat on the mat. ">>, 300),
    {ok, M} = tinyllm:train(Corpus, tf_opts()),
    #{stats := #{val_loss := V}, cfg := #{arch := transformer}} = M,
    ?assert(V < 0.1),
    %% a prompt shorter than the window works (no padding), and so does one longer than it
    ?assertEqual(<<"the cat sat on the mat. the cat sat">>,
                 tinyllm:generate(M, <<"the cat">>, 8, #{temperature => 0.05, seed => 1})),
    Long = binary:copy(<<"the cat sat on the mat. ">>, 5),
    ?assertMatch(<<"the cat", _/binary>>, tinyllm:generate(M, <<Long/binary, "the cat">>, 3, #{temperature => 0.05, seed => 1})).

%% "a q q q q q q A  b q q q q q q B ...": the closing letter depends on the letter 7 tokens back,
%% which a model with a 4-token window cannot see but attention can.
tf_uses_long_range_context() ->
    rand:seed(exsss, {7, 8, 9}),
    Corpus = iolist_to_binary([case rand:uniform(2) of 1 -> "a q q q q q q A "; 2 -> "b q q q q q q B " end
                               || _ <- lists:seq(1, 500)]),
    Base = #{tokenizer => word, workers => 2, batch => 32, steps => 400, lr => 0.01, eval_every => 1000,
             val_examples => 150, verbose => false, min_count => 1},
    {ok, Tf} = tinyllm:train(Corpus, maps:merge(Base, #{arch => transformer, context => 12, dim => 16, heads => 2,
                                                         layers => 1, ff => 32})),
    {ok, Mlp} = tinyllm:train(Corpus, maps:merge(Base, #{arch => mlp, context => 4, embed => 8, hidden => 32,
                                                          batch => 128, steps => 600})),
    TfLoss = maps:get(val_loss, maps:get(stats, Tf)),
    MlpLoss = maps:get(val_loss, maps:get(stats, Mlp)),
    %% the opening letter is a coin flip for both; only attention can also predict the closing
    %% letter (measured: transformer ~0.25, 4-token-window MLP ~0.41)
    ?assert(TfLoss < MlpLoss - 0.08),
    ?assert(TfLoss < 0.32).

tf_save_load_generate() ->
    {ok, M} = tinyllm:train(binary:copy(<<"the cat sat on the mat. ">>, 200), (tf_opts())#{steps => 10}),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_tf_model.bin"),
    ok = tinyllm:save(M, Path),
    M2 = tinyllm:load(Path),
    file:delete(Path),
    ?assertEqual(M, M2),
    ?assertEqual(tinyllm:generate(M, <<"the cat">>, 12, #{seed => 3}), tinyllm:generate(M2, <<"the cat">>, 12, #{seed => 3})),
    ?assert(tinyllm:param_count(M) > 1000),
    %% a model file loads in a fresh VM: all keys are atoms/tuples already known
    Pa = lists:append([["-pa ", P, " "] || P <- code:get_path(), filelib:is_dir(P), string:find(P, "tinyllm") =/= nomatch]),
    Path2 = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_tf_model2.bin"),
    ok = tinyllm:save(M, Path2),
    Out = os:cmd(lists:flatten(["erl -noshell ", Pa, "-eval 'M = tinyllm:load(\"", Path2,
                                "\"), io:format(\"~p\", [tinyllm:param_count(M)]), halt().'"])),
    file:delete(Path2),
    ?assertEqual(integer_to_list(tinyllm:param_count(M)), Out).

tf_web_info() ->
    {ok, _} = application:ensure_all_started(inets),
    {ok, M} = tinyllm:train(binary:copy(<<"the cat sat on the mat. ">>, 200), (tf_opts())#{steps => 10}),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_tf_web_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    ok = tinyllm_web:start(18765),
    try
        {200, I} = http(get, "/api/info", <<>>, ""),
        ?assertMatch(#{<<"arch">> := <<"transformer">>, <<"layers">> := 1, <<"heads">> := 2, <<"params">> := _}, json:decode(I)),
        {200, Page} = http(get, "/chat", <<>>, ""),
        ?assertMatch({_, _}, binary:match(Page, <<"transformer, 1 layers x 2 heads">>)),
        {200, C} = http(post, "/v1/chat/completions", <<"{\"messages\":[{\"role\":\"user\",\"content\":\"the cat\"}],\"max_tokens\":5}">>, "application/json"),
        ?assertMatch(#{<<"choices">> := [#{<<"message">> := #{<<"content">> := _}}]}, json:decode(C))
    after
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        file:delete(Path)
    end.

%% A tiny HTTP server that answers ONE request with a canned reply and forwards the
%% raw request to the test process. Returns its base URL.
fake_server(Status, Body) ->
    {ok, L} = gen_tcp:listen(0, [binary, {active, false}, {reuseaddr, true}, {ip, {127, 0, 0, 1}}]),
    {ok, Port} = inet:port(L),
    Parent = self(),
    spawn_link(fun() ->
        {ok, S} = gen_tcp:accept(L),
        Raw = read_request(S, <<>>),
        Parent ! {request, Raw},
        Resp = iolist_to_binary(["HTTP/1.1 ", Status, "\r\ncontent-type: text/html\r\ncontent-length: ",
                                 integer_to_list(byte_size(Body)), "\r\nconnection: close\r\n\r\n", Body]),
        gen_tcp:send(S, Resp), gen_tcp:close(S), gen_tcp:close(L)
    end),
    "http://127.0.0.1:" ++ integer_to_list(Port).

%% read headers, then as many body bytes as content-length announces
read_request(S, Acc) ->
    case binary:split(Acc, <<"\r\n\r\n">>) of
        [Head, Body] ->
            case binary:split(string:lowercase(Head), <<"content-length: ">>) of
                [_, CL] ->
                    [Len | _] = binary:split(CL, <<"\r\n">>),
                    case byte_size(Body) >= binary_to_integer(Len) of
                        true -> Acc;
                        false -> more_request(S, Acc)
                    end;
                _ -> Acc
            end;
        _ -> more_request(S, Acc)
    end.

more_request(S, Acc) ->
    {ok, D} = gen_tcp:recv(S, 0, 5000),
    read_request(S, <<Acc/binary, D/binary>>).
