%% KV-cache sessions: same logits and tokens as the stateless path, only new tokens computed, supervised.
-module(tinyllm_session_tests).
-include_lib("eunit/include/eunit.hrl").

setup() ->
    application:set_env(tinyllm, memory_storage, ram),
    {ok, _} = application:ensure_all_started(tinyllm), ok.
cleanup(_) -> application:stop(tinyllm).

session_test_() ->
    {setup, fun setup/0, fun cleanup/1,
     {timeout, 300,
      [fun cache_logits_equal_full_forward/0, fun session_equals_stateless/0, fun only_new_tokens_are_computed/0,
       fun chat_equals_stateless_over_turns/0, fun mlp_session_equals_stateless/0, fun evict_mode_bounds_work/0,
       fun sessions_are_supervised/0, fun web_chat_uses_sessions/0, fun openai_api_session_id/0]}}.

tf_opts(Tok, Ctx) ->
    #{arch => transformer, tokenizer => Tok, context => Ctx, dim => 16, heads => 2, layers => 2, ff => 32,
      workers => 2, batch => 8, steps => 20, lr => 0.01, eval_every => 1000, val_examples => 20,
      verbose => false, min_count => 1}.

corpus() -> binary:copy(<<"User: hi there\nAssistant: hello, how are you?\nUser: fine thanks\nAssistant: good to hear.\n">>, 40).

tf_model(Tok, Ctx) ->
    {ok, M} = tinyllm:train(corpus(), tf_opts(Tok, Ctx)),
    M.

%% ---- the cache changes nothing: every position's logits are exactly those of the full forward pass

cache_logits_equal_full_forward() ->
    Cfg = #{vocab => 11, context => 8, dim => 16, heads => 4, layers => 3, ff => 32},
    P = tinyllm_tf:prepare(tinyllm_tf:init(Cfg, 5)),
    Ids = [3, 0, 10, 7, 7, 2, 9, 1],
    lists:foldl(fun(I, {Cache, Seen}) ->
                    {Logits, Cache1} = tinyllm_tf:step(P, Cfg, Cache, I),
                    Prefix = Seen ++ [I],
                    ?assertEqual(tinyllm_tf:last_logits(P, Cfg, Prefix), Logits),
                    {Cache1, Prefix}
                end, {tinyllm_tf:cache_new(Cfg), []}, Ids),
    %% advance/4 (cache-only, skips the last layer's tail and the head) leaves the cache a later step can use
    CA = lists:foldl(fun(I, C) -> tinyllm_tf:advance(P, Cfg, C, I) end, tinyllm_tf:cache_new(Cfg), lists:droplast(Ids)),
    {LA, _} = tinyllm_tf:step(P, Cfg, CA, lists:last(Ids)),
    ?assertEqual(tinyllm_tf:last_logits(P, Cfg, Ids), LA),
    %% truncating the cache and continuing from a different token matches a fresh pass
    {_, C8} = lists:foldl(fun(I, {_, C}) -> tinyllm_tf:step(P, Cfg, C, I) end, {x, tinyllm_tf:cache_new(Cfg)}, Ids),
    ?assertEqual(8, tinyllm_tf:cache_len(C8)),
    C5 = tinyllm_tf:cache_truncate(C8, 5),
    {L, _} = tinyllm_tf:step(P, Cfg, C5, 4),
    ?assertEqual(tinyllm_tf:last_logits(P, Cfg, [3, 0, 10, 7, 7, 4]), L).

%% ---- feed + complete in a session gives the stateless text, token for token

session_equals_stateless() ->
    [begin
         M = tf_model(Tok, Ctx),
         [begin
              Opts = #{temperature => 0.9, seed => Seed},
              Expected = tinyllm:complete(M, Prompt, Len, Opts),
              {ok, Pid} = tinyllm_session:start(M),
              {ok, _} = tinyllm_session:feed(Pid, Prompt),
              ?assertEqual(Expected, tinyllm_session:complete(Pid, Len, Opts)),
              tinyllm_session:stop(Pid)
          end || {Prompt, Len, Seed} <- [{<<"User: hi">>, 30, 1},                       %% window fills and slides
                                         {<<>>, 20, 2},                                  %% empty prompt
                                         {<<"User: hi there\nAssistant: hello, how are you?\nUser:">>, 12, 3}]]  %% longer than the window
     end || {Tok, Ctx} <- [{word, 12}, {char, 16}]],
    ok.

%% ---- turns extend the context: only the new tokens go through the network

only_new_tokens_are_computed() ->
    M = tf_model(word, 64),
    {ok, Pid} = tinyllm_session:start(M),
    {ok, #{computed := 0}} = tinyllm_session:feed(Pid, <<>>),
    {ok, #{window := W1, computed := C1}} = tinyllm_session:feed(Pid, <<"User: hi there\n">>),
    ?assert(W1 > 3),
    ?assertEqual(W1, C1),
    {ok, #{window := W2, computed := C2, reused := R2}} = tinyllm_session:feed(Pid, <<"Assistant: hello">>),
    ?assert(W2 > W1),
    ?assertEqual(W2 - W1, C2 - C1),          %% the first prompt is not recomputed ...
    ?assertEqual(W1, R2),                    %% ... its positions come from the cache
    %% set_context with the same text plus more reuses everything already cached
    {ok, #{window := W3, computed := C3}} = tinyllm_session:set_context(Pid, <<"User: hi there\nAssistant: hello, how">>),
    ?assertEqual(W3 - W2, C3 - C2),
    %% a different history recomputes from the first differing token
    {ok, #{computed := C4}} = tinyllm_session:set_context(Pid, <<"User: hi here\n">>),
    ?assert(C4 - C3 >= 2),
    ok = tinyllm_session:reset(Pid),
    ?assertMatch(#{window := 0, cached := 0}, tinyllm_session:info(Pid)),
    tinyllm_session:stop(Pid).

%% ---- tinyllm_session:chat/3 == tinyllm:chat/3 over a whole conversation

chat_equals_stateless_over_turns() ->
    M = tf_model(word, 64),
    {ok, Pid} = tinyllm_session:start(M),
    Msgs0 = [#{role => user, content => <<"hi there">>}],
    Final = lists:foldl(
              fun(Q, Msgs) ->
                  Opts = #{temperature => 0.8, seed => 11, max_chars => 8},
                  Expected = tinyllm:chat(M, Msgs, Opts),
                  ?assertEqual(Expected, tinyllm_session:chat(Pid, Msgs, Opts)),
                  {Reply, _} = Expected,
                  Msgs ++ [#{role => assistant, content => Reply}, #{role => user, content => Q}]
              end, Msgs0, [<<"fine thanks">>, <<"and you">>, <<"bye">>]),
    ?assertEqual(7, length(Final)),
    %% 64-token window, short turns: later turns must have been served partly from the cache
    #{reused := Reused, computed := Computed} = tinyllm_session:info(Pid),
    ?assert(Reused > 0),
    ?assert(Computed > 0),
    tinyllm_session:stop(Pid).

mlp_session_equals_stateless() ->
    {ok, M} = tinyllm:train(corpus(), #{arch => mlp, tokenizer => word, context => 4, embed => 8, hidden => 16,
                                        workers => 2, batch => 16, steps => 20, lr => 0.01, eval_every => 1000,
                                        val_examples => 20, verbose => false, min_count => 1}),
    Opts = #{temperature => 0.9, seed => 4},
    {ok, Pid} = tinyllm_session:start(M),
    {ok, _} = tinyllm_session:feed(Pid, <<"User: hi">>),
    ?assertEqual(tinyllm:complete(M, <<"User: hi">>, 25, Opts), tinyllm_session:complete(Pid, 25, Opts)),
    tinyllm_session:stop(Pid).

%% ---- evict > 1 is approximate but bounded and cheaper

evict_mode_bounds_work() ->
    M = tf_model(word, 16),
    Opts = #{temperature => 0.9, seed => 2},
    Run = fun(Evict) ->
              {ok, Pid} = tinyllm_session:start(M, #{evict => Evict}),
              {ok, _} = tinyllm_session:feed(Pid, <<"User: hi there\nAssistant:">>),
              {Text, _} = tinyllm_session:complete(Pid, 60, Opts),
              Info = tinyllm_session:info(Pid),
              tinyllm_session:stop(Pid),
              {Text, Info}
          end,
    {Exact, #{computed := CExact}} = Run(1),
    {_, #{computed := CEvict, window := W}} = Run(8),
    ?assertEqual(element(1, tinyllm:complete(M, <<"User: hi there\nAssistant:">>, 60, Opts)), Exact),
    ?assert(W =< 16),
    ?assert(CEvict < CExact div 2).

%% ---- sessions live under a supervisor and fail alone

sessions_are_supervised() ->
    M = tf_model(word, 12),
    {ok, A} = tinyllm_session:get_or_start(<<"conv-a">>, M, #{}),
    {ok, B} = tinyllm_session:get_or_start(<<"conv-b">>, M, #{}),
    ?assertEqual({ok, A}, tinyllm_session:get_or_start(<<"conv-a">>, M, #{})),
    ?assertEqual(A, tinyllm_session:whereis(<<"conv-a">>)),
    ?assert(lists:member(A, [P || {_, P, _, _} <- supervisor:which_children(tinyllm_session_sup)])),
    {ok, _} = tinyllm_session:feed(A, <<"User: hi">>),
    Ref = monitor(process, A),
    exit(A, kill),
    receive {'DOWN', Ref, _, _, _} -> ok after 2000 -> error(no_down) end,
    ?assert(is_process_alive(B)),
    ?assert(is_process_alive(whereis(tinyllm_session_sup))),
    %% the supervisor does not resurrect it (state is a rebuildable cache); the next turn starts a fresh one
    timer:sleep(50),
    ?assertEqual(undefined, tinyllm_session:whereis(<<"conv-a">>)),
    {ok, A2} = tinyllm_session:get_or_start(<<"conv-a">>, M, #{}),
    ?assertNotEqual(A, A2),
    ?assertMatch(#{window := 0}, tinyllm_session:info(A2)),
    %% idle sessions stop on their own
    {ok, Idle} = tinyllm_session:start(M, #{idle_ms => 50}),
    IRef = monitor(process, Idle),
    receive {'DOWN', IRef, _, _, normal} -> ok after 2000 -> error(not_idle_stopped) end,
    %% bad input is an error reply, not a crash
    ?assertMatch({error, _}, tinyllm_session:feed(B, an_atom)),
    ?assert(is_process_alive(B)),
    %% cap on open sessions
    application:set_env(tinyllm, max_sessions, 1),
    ?assertEqual({error, too_many_sessions}, tinyllm_session:start(M)),
    application:unset_env(tinyllm, max_sessions),
    tinyllm_session:close(<<"conv-a">>), tinyllm_session:close(<<"conv-b">>),
    ?assertEqual(undefined, tinyllm_session:whereis(<<"conv-b">>)).

%% ---- web: saved chats use a session, with the same replies as without

web_chat_uses_sessions() ->
    {ok, _} = application:ensure_all_started(inets),
    M = tf_model(word, 64),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_session_web_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    ok = tinyllm_web:start(18766),
    try
        CidOf = fun(Page) ->
                    {match, [C]} = re:run(Page, <<"name=cid value=\"([0-9a-f]{16})\"">>, [{capture, all_but_first, binary}]),
                    C
                end,
        Post = fun(Msg, Cid) ->
                   Body = iolist_to_binary(["message=", uri_string:quote(Msg), "&cid=", Cid]),
                   {ok, {{_, 200, _}, _, Page}} =
                       httpc:request(post, {"http://127.0.0.1:18766/chat", [], "application/x-www-form-urlencoded", Body},
                                     [], [{body_format, binary}]),
                   Page
               end,
        Cid = CidOf(Post(<<"hi there">>, <<>>)),
        Pid = tinyllm_session:whereis(Cid),
        ?assert(is_pid(Pid)),
        #{computed := C1} = tinyllm_session:info(Pid),
        ?assert(C1 > 0),
        ?assertEqual(Cid, CidOf(Post(<<"fine thanks">>, Cid))),
        #{reused := R2} = tinyllm_session:info(Pid),
        ?assert(R2 > 0),                                   %% turn 2 reused turn 1's cache
        %% deleting the conversation ends its session
        {ok, {{_, 303, _}, _, _}} =
            httpc:request(post, {"http://127.0.0.1:18766/chat/delete", [], "application/x-www-form-urlencoded",
                                 <<"cid=", Cid/binary>>}, [{autoredirect, false}], [{body_format, binary}]),
        timer:sleep(50),
        ?assertEqual(undefined, tinyllm_session:whereis(Cid))
    after
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        tinyllm_session:unpublish(model),
        file:delete(Path)
    end.

openai_api_session_id() ->
    {ok, _} = application:ensure_all_started(inets),
    M = tf_model(word, 64),
    Path = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_session_api_model.bin"),
    ok = tinyllm:save(M, Path),
    application:set_env(tinyllm, model_file, Path),
    ok = tinyllm_web:start(18767),
    try
        Req = fun(Fields) ->
                  Body = iolist_to_binary(json:encode(maps:merge(#{seed => 5, max_tokens => 6, temperature => 0.8}, Fields))),
                  {ok, {{_, Code, _}, _, Resp}} =
                      httpc:request(post, {"http://127.0.0.1:18767/v1/chat/completions", [], "application/json", Body},
                                    [], [{body_format, binary}]),
                  {Code, json:decode(Resp)}
              end,
        Msgs1 = [#{role => user, content => <<"hi there">>}],
        {200, Plain} = Req(#{messages => Msgs1}),
        ?assertMatch(#{<<"tinyllm_session">> := null}, Plain),
        {200, S1} = Req(#{messages => Msgs1, session_id => <<"abc-1">>}),
        Reply = fun(#{<<"choices">> := [#{<<"message">> := #{<<"content">> := C}}]}) -> C end,
        ?assertEqual(Reply(Plain), Reply(S1)),               %% a session changes the speed, not the answer
        Msgs2 = Msgs1 ++ [#{role => assistant, content => Reply(S1)}, #{role => user, content => <<"fine thanks">>}],
        {200, S2} = Req(#{messages => Msgs2, session_id => <<"abc-1">>}),
        {200, Plain2} = Req(#{messages => Msgs2}),
        ?assertEqual(Reply(Plain2), Reply(S2)),
        ?assertMatch(#{<<"tinyllm_session">> := #{<<"reused">> := R}} when R > 0, S2),
        {400, _} = Req(#{messages => Msgs1, session_id => <<"bad id!">>}),
        tinyllm_session:close(<<"api:abc-1">>)
    after
        tinyllm_web:stop(),
        persistent_term:erase(tinyllm_model),
        tinyllm_session:unpublish(model),
        file:delete(Path)
    end.
