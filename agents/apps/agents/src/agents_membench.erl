%% Memory benchmark runner. Reads the JSONL written by lisp/bin/membench-gen.lisp, asks an
%% LLM backend (LM Studio by default) each question under a memory setup, and scores the
%% answers against the generator's ground truth.
%%
%% Setups (all share the same system prompt; only the memory text differs):
%%   none       no memory at all
%%   truncated  the whole transcript cut to the most recent `window` tokens
%%   retrieval  the `k` exchanges most similar to the question (BM25, or embeddings), oldest first
%%
%% The `event` annotations in the cases are ground truth and are never shown to the model.
-module(agents_membench).

-export([load_cases/1, exchanges/1, run/2, run_case/2, memory_for/5, truncate/2,
         retrieve/4, bm25_scores/2, system_prompt/1, est_tokens/1, score/2,
         summarize/1, print_summary/1, lmstudio_embed/1]).

-define(STOP, [<<"the">>, <<"a">>, <<"an">>, <<"i">>, <<"my">>, <<"me">>, <<"do">>, <<"does">>,
               <<"is">>, <<"am">>, <<"what">>, <<"which">>, <<"of">>, <<"in">>, <<"to">>, <<"for">>,
               <<"you">>, <<"your">>, <<"and">>, <<"it">>, <<"that">>, <<"this">>, <<"are">>, <<"was">>]).

%% ------------------------------------------------------------------ cases

load_cases(Path) ->
    {ok, Bin} = file:read_file(Path),
    [json:decode(L) || L <- binary:split(Bin, <<"\n">>, [global, trim_all])].

%% [{Index, Text}] with Index 0-based, one entry per user/assistant exchange.
exchanges(Case) ->
    Pairs = pair(maps:get(<<"turns">>, Case)),
    [{I, <<"User: ", U/binary, "\nAssistant: ", A/binary>>} || {I, {U, A}} <- lists:zip(lists:seq(0, length(Pairs) - 1), Pairs)].

pair([#{<<"role">> := <<"user">>, <<"text">> := U}, #{<<"role">> := <<"assistant">>, <<"text">> := A} | T]) ->
    [{U, A} | pair(T)];
pair([]) -> [].

est_tokens(Bin) -> (byte_size(Bin) + 3) div 4.

%% ------------------------------------------------------------------ memory setups

memory_for(none, _Q, _Exs, _Ctx, _Opts) -> [];
memory_for(truncated, _Q, Exs, _Ctx, Opts) -> truncate(Exs, maps:get(window, Opts, 2048));
memory_for(retrieval, Q, Exs, Ctx, Opts) -> retrieve(Q, Exs, Ctx, maps:get(k, Opts, 6)).

%% Most recent exchanges that fit in Window estimated tokens, oldest first.
truncate(Exs, Window) -> take_recent(lists:reverse(Exs), Window, []).

take_recent([], _, Acc) -> Acc;
take_recent([{_, T} = E | R], Left, Acc) ->
    C = est_tokens(T),
    if C =< Left -> take_recent(R, Left - C, [E | Acc]);
       true -> Acc
    end.

%% Ctx is `lexical` or `{embedding, EmbedFun, Vecs}` (Vecs aligned with Exs).
retrieve(Q, Exs, Ctx, K) ->
    Scores = case Ctx of
                 lexical -> bm25_scores(Q, [T || {_, T} <- Exs]);
                 {embedding, EmbedFun, Vecs} ->
                     [QV] = EmbedFun([Q]),
                     [cosine(QV, V) || V <- Vecs]
             end,
    Ranked = lists:sort(fun({S1, I1, _}, {S2, I2, _}) -> {S1, I1} >= {S2, I2} end,
                        [{S, I, E} || {S, {I, _} = E} <- lists:zip(Scores, Exs), S > 0]),
    Top = lists:sublist(Ranked, K),
    lists:sort(fun({I1, _}, {I2, _}) -> I1 =< I2 end, [E || {_, _, E} <- Top]).

cosine(A, B) ->
    Dot = lists:sum(lists:zipwith(fun(X, Y) -> X * Y end, A, B)),
    NA = math:sqrt(lists:sum([X * X || X <- A])),
    NB = math:sqrt(lists:sum([X * X || X <- B])),
    D = NA * NB,
    if D == 0 -> 0.0; true -> Dot / D end.

tokens(Bin) -> [W || W <- binary:split(norm(Bin), <<" ">>, [global]), W =/= <<>>, not lists:member(W, ?STOP)].

%% Okapi BM25 of Query against each document.
bm25_scores(Query, Docs) ->
    Toks = [tokens(D) || D <- Docs],
    N = length(Docs),
    AvgDl = max(1.0, lists:sum([length(T) || T <- Toks]) / max(1, N)),
    QTerms = lists:usort(tokens(Query)),
    Df = maps:from_list([{Term, length([1 || T <- Toks, lists:member(Term, T)])} || Term <- QTerms]),
    [begin
         Dl = length(T),
         lists:sum([begin
                        Tf = length([1 || X <- T, X =:= Term]),
                        Idf = math:log(1 + (N - maps:get(Term, Df) + 0.5) / (maps:get(Term, Df) + 0.5)),
                        case Tf of
                            0 -> 0.0;
                            _ -> Idf * Tf * 2.2 / (Tf + 1.2 * (0.25 + 0.75 * Dl / AvgDl))
                        end
                    end || Term <- QTerms])
     end || T <- Toks].

%% ------------------------------------------------------------------ prompting

system_prompt(Exs) ->
    Memory = case Exs of
                 [] -> <<"(no memory available)">>;
                 _ -> iolist_to_binary(lists:join(<<"\n\n">>,
                          [[<<"[turn ">>, integer_to_binary(I), <<"]\n">>, T] || {I, T} <- Exs]))
             end,
    iolist_to_binary(
      ["You answer questions about the user, using only the memory notes below. ",
       "The notes are excerpts of earlier conversations, oldest first. ",
       "If the notes disagree about the user, the later turn is correct. ",
       "Something said about another person is not a fact about the user. ",
       "If the notes do not say, answer exactly: unknown. ",
       "Reply with only the answer, a few words, no explanation.\n\nMEMORY:\n", Memory]).

%% ------------------------------------------------------------------ scoring

norm(Bin) ->
    L = string:lowercase(Bin),
    R = re:replace(L, "[^a-z0-9]+", " ", [global, {return, binary}]),
    string:trim(R).

has(Answer, Needle) ->
    binary:match(<<" ", (norm(Answer))/binary, " ">>, <<" ", (norm(Needle))/binary, " ">>) =/= nomatch.

abstains(Answer) ->
    lists:any(fun(P) -> has(Answer, P) end,
              [<<"unknown">>, <<"don't know">>, <<"do not know">>, <<"not mentioned">>, <<"no information">>,
               <<"not specified">>, <<"not provided">>, <<"cannot">>, <<"can't">>, <<"no record">>]).

%% Outcome: correct | stale | distractor | unknown | hallucinated | wrong.
score(Q, Answer) ->
    Exp = maps:get(<<"expected">>, Q),
    case Exp of
        null ->
            case abstains(Answer) of true -> correct; false -> hallucinated end;
        _ ->
            Stale = maps:get(<<"stale">>, Q, null),
            Dis = maps:get(<<"distractor">>, Q, null),
            HasE = has(Answer, Exp),
            HasS = Stale =/= null andalso has(Answer, Stale),
            HasD = Dis =/= null andalso has(Answer, Dis),
            if HasE andalso not HasS andalso not HasD -> correct;
               HasS andalso not HasE -> stale;
               HasD andalso not HasE -> distractor;
               HasE -> wrong;                      % hedged between right and wrong values
               true -> case abstains(Answer) of true -> unknown; false -> wrong end
            end
    end.

%% ------------------------------------------------------------------ running

%% Opts: setups ([none, truncated, retrieval]), window (2048), k (6), retriever (lexical |
%% {embedding, EmbedFun}), llm (agents_llm_lmstudio), llm_opts (#{temperature => 0.0, max_tokens => 4096}),
%% concurrency (1), limit (cases), out (JSONL path for the per-question rows), verbose (true).
run(Path, Opts) ->
    Cases = case maps:get(limit, Opts, all) of
                all -> load_cases(Path);
                N -> lists:sublist(load_cases(Path), N)
            end,
    ok = case maps:get(out, Opts, undefined) of
             undefined -> ok;
             Out -> file:write_file(Out, <<>>)
         end,
    lists:append([run_case(C, Opts) || C <- Cases]).

run_case(Case, Opts) ->
    Exs = exchanges(Case),
    Setups = maps:get(setups, Opts, [none, truncated, retrieval]),
    Ctx = case {lists:member(retrieval, Setups), maps:get(retriever, Opts, lexical)} of
              {true, {embedding, F}} -> {embedding, F, F([T || {_, T} <- Exs])};
              _ -> lexical
          end,
    Jobs = [{S, Q} || Q <- maps:get(<<"questions">>, Case), S <- Setups],
    Rows = pmap(fun({S, Q}) -> ask(Case, S, Q, Exs, Ctx, Opts) end, Jobs, maps:get(concurrency, Opts, 1)),
    case maps:get(out, Opts, undefined) of
        undefined -> ok;
        Out -> file:write_file(Out, [[json:encode(R), $\n] || R <- Rows], [append])
    end,
    case maps:get(verbose, Opts, true) of
        true -> io:format("~s: ~b questions x ~b setups done~n",
                          [maps:get(<<"id">>, Case), length(maps:get(<<"questions">>, Case)), length(Setups)]);
        false -> ok
    end,
    Rows.

ask(Case, Setup, Q, Exs, Ctx, Opts) ->
    Llm = maps:get(llm, Opts, agents_llm_lmstudio),
    LlmOpts = maps:merge(#{temperature => 0.0, max_tokens => 4096}, maps:get(llm_opts, Opts, #{})),
    QText = maps:get(<<"question">>, Q),
    Sys = system_prompt(memory_for(Setup, QText, Exs, Ctx, Opts)),
    T0 = erlang:monotonic_time(millisecond),
    Res = Llm:chat(#{system => Sys, messages => [#{role => <<"user">>, content => QText}]}, LlmOpts),
    Lat = erlang:monotonic_time(millisecond) - T0,
    Base = #{case_id => maps:get(<<"id">>, Case), exchanges => maps:get(<<"exchanges">>, Case),
             setup => Setup, type => maps:get(<<"type">>, Q), relation => maps:get(<<"relation">>, Q),
             distance => maps:get(<<"distance">>, Q), latency_ms => Lat, prompt_tokens => est_tokens(Sys)},
    case Res of
        {ok, #{<<"content">> := Blocks}} ->
            Answer = iolist_to_binary([T || #{<<"type">> := <<"text">>, <<"text">> := T} <- Blocks]),
            case Answer of
                <<>> ->
                    %% reasoning models can spend the whole token budget thinking and return nothing
                    Base#{answer => <<>>, outcome => error, correct => false, error => <<"empty_answer">>};
                _ ->
                    Out = score(Q, Answer),
                    Base#{answer => Answer, outcome => Out, correct => (Out =:= correct)}
            end;
        {error, Reason} ->
            Base#{answer => <<>>, outcome => error, correct => false,
                  error => iolist_to_binary(io_lib:format("~p", [Reason]))}
    end.

pmap(F, List, N) when N =< 1 -> [F(X) || X <- List];
pmap(F, List, N) ->
    Parent = self(),
    {Chunk, Rest} = case length(List) > N of true -> lists:split(N, List); false -> {List, []} end,
    Refs = [begin
                Ref = make_ref(),
                spawn_link(fun() -> Parent ! {Ref, F(X)} end),
                Ref
            end || X <- Chunk],
    [receive {Ref, R} -> R end || Ref <- Refs] ++ case Rest of [] -> []; _ -> pmap(F, Rest, N) end.

%% ------------------------------------------------------------------ summary

%% One map per {setup, exchanges}: accuracy over answered questions, accuracy by question type,
%% failure shares, mean latency and mean estimated prompt tokens.
summarize(Rows) ->
    Groups = lists:foldl(fun(R, Acc) ->
                                 maps:update_with({maps:get(setup, R), maps:get(exchanges, R)},
                                                  fun(L) -> [R | L] end, [R], Acc)
                         end, #{}, Rows),
    [summary({S, E}, lists:reverse(G)) || {{S, E}, G} <- lists:sort(maps:to_list(Groups))].

summary({S, E}, G) ->
    Ok = [R || R <- G, maps:get(outcome, R) =/= error],
    Types = lists:usort([maps:get(type, R) || R <- Ok]),
    #{setup => S, exchanges => E, n => length(Ok), errors => length(G) - length(Ok),
      accuracy => rate(Ok, fun(R) -> maps:get(correct, R) end),
      by_type => maps:from_list([{T, rate([R || R <- Ok, maps:get(type, R) =:= T], fun(R) -> maps:get(correct, R) end)}
                                 || T <- Types]),
      stale_rate => rate([R || R <- Ok, maps:get(type, R) =:= <<"updated">>], fun(R) -> maps:get(outcome, R) =:= stale end),
      hallucination_rate => rate([R || R <- Ok, maps:get(type, R) =:= <<"absent">>], fun(R) -> maps:get(outcome, R) =:= hallucinated end),
      latency_ms => mean([maps:get(latency_ms, R) || R <- G]),
      prompt_tokens => mean([maps:get(prompt_tokens, R) || R <- G])}.

rate([], _) -> undefined;
rate(L, P) -> length([1 || X <- L, P(X)]) / length(L).

mean([]) -> undefined;
mean(L) -> lists:sum(L) / length(L).

print_summary(Rows) ->
    io:format("~-10s ~6s ~5s ~5s | ~6s ~6s ~7s ~7s | ~6s ~6s | ~8s ~7s~n",
              ["setup", "turns", "n", "err", "acc", "static", "updated", "absent", "stale", "halluc", "lat_ms", "prompt"]),
    [io:format("~-10s ~6b ~5b ~5b | ~6s ~6s ~7s ~7s | ~6s ~6s | ~8s ~7s~n",
               [maps:get(setup, S), maps:get(exchanges, S), maps:get(n, S), maps:get(errors, S),
                pct(maps:get(accuracy, S)), pct(by(S, <<"static">>)), pct(by(S, <<"updated">>)), pct(by(S, <<"absent">>)),
                pct(maps:get(stale_rate, S)), pct(maps:get(hallucination_rate, S)),
                num(maps:get(latency_ms, S)), num(maps:get(prompt_tokens, S))])
     || S <- summarize(Rows)],
    ok.

by(S, T) -> maps:get(T, maps:get(by_type, S), undefined).
pct(undefined) -> "-";
pct(X) -> io_lib:format("~.1f%", [X * 100]).
num(undefined) -> "-";
num(X) -> integer_to_list(round(X)).

%% ------------------------------------------------------------------ embeddings (LM Studio)

%% Opts: model (required), base_url ("http://127.0.0.1:1234"). Returns fun([binary()]) -> [[float()]].
lmstudio_embed(Opts) ->
    Base = maps:get(base_url, Opts, "http://127.0.0.1:1234"),
    Model = maps:get(model, Opts),
    fun(Texts) -> lists:append([embed_batch(Base, Model, B) || B <- batches(Texts, 32)]) end.

batches([], _) -> [];
batches(L, N) when length(L) =< N -> [L];
batches(L, N) -> {A, B} = lists:split(N, L), [A | batches(B, N)].

embed_batch(Base, Model, Texts) ->
    {ok, _} = application:ensure_all_started(inets),
    Body = iolist_to_binary(json:encode(#{model => Model, input => Texts})),
    case httpc:request(post, {Base ++ "/v1/embeddings", [], "application/json", Body},
                       [{timeout, 120000}, {connect_timeout, 5000}], [{body_format, binary}]) of
        {ok, {{_, 200, _}, _, RB}} ->
            #{<<"data">> := Data} = json:decode(RB),
            [E || #{<<"embedding">> := E} <- lists:sort(fun(#{<<"index">> := A}, #{<<"index">> := B}) -> A =< B end, Data)];
        Other -> erlang:error({embedding_failed, Other})
    end.
