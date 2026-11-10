-module(agents_membench_tests).
-include_lib("eunit/include/eunit.hrl").

%% One hand-built case: 40 exchanges. Turn 0 plants "Paris", turn 25 updates to "Berlin",
%% turn 10 is a distractor about a sister in Oslo; everything else is filler.
case_json() ->
    Filler = fun(I) -> {iolist_to_binary(io_lib:format("Can you explain topic number ~b in detail please?", [I])),
                        <<"Sure, here is a short explanation of that topic.">>} end,
    Pair = fun(0) -> {<<"I live in Paris.">>, <<"Noted.">>};
              (10) -> {<<"My sister Maya lives in Oslo.">>, <<"Good to know.">>};
              (25) -> {<<"I moved to Berlin last month.">>, <<"Thanks for the update.">>};
              (I) -> Filler(I) end,
    Turns = lists:append([[#{<<"role">> => <<"user">>, <<"text">> => U}, #{<<"role">> => <<"assistant">>, <<"text">> => A}]
                          || I <- lists:seq(0, 39), {U, A} <- [Pair(I)]]),
    Q = fun(Type, Question, Exp, Stale, Dis, Dist) ->
                #{<<"type">> => Type, <<"relation">> => <<"lives_in">>, <<"question">> => Question,
                  <<"expected">> => Exp, <<"stale">> => Stale, <<"distractor">> => Dis, <<"distance">> => Dist}
        end,
    json:encode(#{<<"id">> => <<"t1">>, <<"exchanges">> => 40, <<"turns">> => Turns,
                  <<"questions">> => [Q(<<"updated">>, <<"Which city do I live in?">>, <<"Berlin">>, <<"Paris">>, <<"Oslo">>, 15),
                                      Q(<<"absent">>, <<"What is my dog's name?">>, null, null, null, null)]}).

fixture() ->
    Dir = filename:join(os:getenv("TMPDIR", "/tmp"), "agents_membench_test"),
    ok = filelib:ensure_dir(filename:join(Dir, "x")),
    P = filename:join(Dir, "cases.jsonl"),
    ok = file:write_file(P, [case_json(), $\n]),
    P.

%% A fake model that "reads" the memory: it answers with a city mentioned in the memory section
%% of the system prompt, preferring the one stated last (a perfect reader of whatever it is given).
reader(Req) ->
    Sys = maps:get(system, Req),
    Q = maps:get(content, hd(maps:get(messages, Req))),
    Mem = hd(tl(binary:split(Sys, <<"MEMORY:\n">>))),
    case Q of
        <<"Which city", _/binary>> ->
            Hits = lists:sort([{Pos, C} || C <- [<<"Berlin">>, <<"Paris">>], {Pos, _} <- [binary:match(Mem, <<"I live in ", C/binary>>)] ++
                                              [binary:match(Mem, <<"moved to ", C/binary>>)], is_integer(Pos)]),
            case Hits of [] -> agents_llm_mock:text(<<"unknown">>); _ -> agents_llm_mock:text(element(2, lists:last(Hits))) end;
        _ -> agents_llm_mock:text(<<"Biscuit">>)   % invents a dog name
    end.

opts(Extra) -> maps:merge(#{llm => agents_llm_mock, llm_opts => #{handler => fun reader/1}, verbose => false}, Extra).

by_setup(Rows, S, Type) -> hd([R || R <- Rows, maps:get(setup, R) =:= S, maps:get(type, R) =:= Type]).

scoring_test() ->
    Q = #{<<"expected">> => <<"Berlin">>, <<"stale">> => <<"Paris">>, <<"distractor">> => <<"Oslo">>},
    ?assertEqual(correct, agents_membench:score(Q, <<"Berlin.">>)),
    ?assertEqual(correct, agents_membench:score(Q, <<"You live in berlin">>)),
    ?assertEqual(stale, agents_membench:score(Q, <<"Paris">>)),
    ?assertEqual(distractor, agents_membench:score(Q, <<"Oslo">>)),
    ?assertEqual(wrong, agents_membench:score(Q, <<"Paris or Berlin">>)),
    ?assertEqual(unknown, agents_membench:score(Q, <<"I don't know">>)),
    ?assertEqual(wrong, agents_membench:score(Q, <<"Lisbon">>)),
    A = #{<<"expected">> => null},
    ?assertEqual(correct, agents_membench:score(A, <<"unknown">>)),
    ?assertEqual(hallucinated, agents_membench:score(A, <<"Biscuit">>)),
    %% whole-word matching: "Berlinger" is not "Berlin"
    ?assertEqual(wrong, agents_membench:score(Q, <<"Berlinger">>)).

truncate_keeps_the_most_recent_test() ->
    Exs = [{I, binary:copy(<<"x">>, 40)} || I <- lists:seq(0, 9)],      % 10 tokens each
    ?assertEqual([7, 8, 9], [I || {I, _} <- agents_membench:truncate(Exs, 35)]),
    ?assertEqual([], agents_membench:truncate(Exs, 5)),
    ?assertEqual(10, length(agents_membench:truncate(Exs, 1000))).

bm25_prefers_matching_documents_test() ->
    Docs = [<<"We talked about the weather today">>, <<"I live in Paris near the river">>, <<"Recipe for soup">>],
    [S0, S1, S2] = agents_membench:bm25_scores(<<"Which city do I live in?">>, Docs),
    ?assert(S1 > S0), ?assert(S1 > S2), ?assertEqual(0.0, S2).

retrieval_is_chronological_and_bounded_test() ->
    [Case] = agents_membench:load_cases(fixture()),
    Exs = agents_membench:exchanges(Case),
    Got = agents_membench:retrieve(<<"Which city do I live in?">>, Exs, lexical, 3),
    Idx = [I || {I, _} <- Got],
    ?assert(length(Idx) =< 3),
    ?assertEqual(lists:sort(Idx), Idx),
    ?assert(lists:member(0, Idx)).

setups_differ_as_expected_test() ->
    Rows = agents_membench:run(fixture(), opts(#{window => 100, k => 6})),
    ?assertEqual(6, length(Rows)),
    %% no memory: cannot know the city, and invents the dog
    ?assertEqual(unknown, maps:get(outcome, by_setup(Rows, none, <<"updated">>))),
    ?assertEqual(hallucinated, maps:get(outcome, by_setup(Rows, none, <<"absent">>))),
    %% a 100-token window only holds the last few exchanges: the city is gone
    ?assertEqual(unknown, maps:get(outcome, by_setup(Rows, truncated, <<"updated">>))),
    %% BM25 finds "I live in Paris" (shares "live") but not "I moved to Berlin" (shares nothing
    %% with the question), so the model is shown only the stale fact: a known weakness of lexical
    %% retrieval, which is why embeddings are the fair retrieval baseline.
    ?assertEqual(stale, maps:get(outcome, by_setup(Rows, retrieval, <<"updated">>))),
    ?assertNot(lists:any(fun(R) -> maps:is_key(error, R) end, Rows)).

big_window_sees_everything_test() ->
    Rows = agents_membench:run(fixture(), opts(#{window => 100000, setups => [truncated]})),
    ?assertEqual(correct, maps:get(outcome, by_setup(Rows, truncated, <<"updated">>))).

embedding_retrieval_is_pluggable_test() ->
    %% fake embeddings: bag of two features, "mentions a city statement" vs not
    Emb = fun(Texts) -> [case binary:match(T, [<<"live in">>, <<"moved to">>, <<"city">>]) of nomatch -> [0.0, 1.0]; _ -> [1.0, 0.0] end
                         || T <- Texts] end,
    Rows = agents_membench:run(fixture(), opts(#{setups => [retrieval], retriever => {embedding, Emb}, k => 4})),
    ?assertEqual(correct, maps:get(outcome, by_setup(Rows, retrieval, <<"updated">>))).

llm_errors_are_reported_not_counted_test() ->
    Bad = fun(_) -> {error, boom} end,
    Rows = agents_membench:run(fixture(), opts(#{llm_opts => #{handler => Bad}, setups => [none]})),
    [S] = agents_membench:summarize(Rows),
    ?assertEqual(2, maps:get(errors, S)),
    ?assertEqual(0, maps:get(n, S)),
    ?assertEqual(undefined, maps:get(accuracy, S)).

summary_and_output_file_test() ->
    Out = filename:join(filename:dirname(fixture()), "rows.jsonl"),
    Rows = agents_membench:run(fixture(), opts(#{window => 100000, out => Out, concurrency => 3})),
    {ok, Bin} = file:read_file(Out),
    ?assertEqual(6, length(binary:split(Bin, <<"\n">>, [global, trim_all]))),
    Sums = agents_membench:summarize(Rows),
    ?assertEqual([none, retrieval, truncated], [maps:get(setup, S) || S <- Sums]),
    #{accuracy := Acc} = hd([S || S <- Sums, maps:get(setup, S) =:= truncated]),
    ?assertEqual(0.5, Acc).                                 % Berlin right, the dog name invented

empty_answer_is_an_error_not_a_wrong_answer_test() ->
    Empty = fun(_) -> #{<<"stop_reason">> => <<"max_tokens">>, <<"content">> => []} end,
    Rows = agents_membench:run(fixture(), opts(#{llm_opts => #{handler => Empty}, setups => [none]})),
    ?assertEqual([<<"empty_answer">>, <<"empty_answer">>], [maps:get(error, R) || R <- Rows]),
    [S] = agents_membench:summarize(Rows),
    ?assertEqual(2, maps:get(errors, S)).
