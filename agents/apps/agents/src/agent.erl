%% One agent = one process: it owns a conversation and runs the tool-use loop.
%% Different agents run in parallel; a failure in one never reaches another.
-module(agent).
-behaviour(gen_server).

-export([start_link/1, ask/2, ask/3, history/1, stop/1]).
-export([init/1, handle_call/3, handle_cast/2]).

-record(s, {llm, llm_opts, system, tools, max_turns, tool_timeout, allow_truncated = false, history = []}).

start_link(Opts) -> gen_server:start_link(?MODULE, Opts, []).

ask(Pid, Prompt) -> ask(Pid, Prompt, 300000).
ask(Pid, Prompt, Timeout) -> gen_server:call(Pid, {ask, Prompt}, Timeout).
history(Pid) -> gen_server:call(Pid, history).
stop(Pid) -> gen_server:stop(Pid).

init(Opts) ->
    {ok, #s{llm = maps:get(llm, Opts, agents_llm_lmstudio),
            llm_opts = maps:get(llm_opts, Opts, #{}),
            system = maps:get(system, Opts, <<>>),
            tools = maps:get(tools, Opts, []),
            max_turns = maps:get(max_turns, Opts, 10),
            tool_timeout = maps:get(tool_timeout, Opts, 30000),
            allow_truncated = maps:get(allow_truncated, Opts, false)}}.

handle_call({ask, Prompt}, _From, S) ->
    Msgs = S#s.history ++ [#{role => <<"user">>, content => Prompt}],
    case loop(S, Msgs, 1) of
        {ok, Text, NewHistory} -> {reply, {ok, Text}, S#s{history = NewHistory}};
        {error, _} = E -> {reply, E, S}   %% history unchanged on failure
    end;
handle_call(history, _From, S) -> {reply, S#s.history, S}.

handle_cast(_, S) -> {noreply, S}.

loop(#s{max_turns = Max}, _Msgs, Turn) when Turn > Max -> {error, max_turns};
loop(#s{llm = Llm, llm_opts = LO, system = Sys, tools = Tools} = S, Msgs, Turn) ->
    Req = #{system => Sys, messages => Msgs, tools => Tools},
    case safe_chat(Llm, Req, LO) of
        {ok, #{<<"stop_reason">> := <<"tool_use">>, <<"content">> := Content}} ->
            Calls = [B || #{<<"type">> := <<"tool_use">>} = B <- Content],
            Results = agents_tools:run_all(Calls, agents_tools:to_map(Tools), S#s.tool_timeout),
            %% content is replayed verbatim (thinking blocks must stay intact);
            %% all tool_results go back in ONE user message.
            loop(S, Msgs ++ [#{role => <<"assistant">>, content => Content},
                             #{role => <<"user">>, content => Results}], Turn + 1);
        {ok, #{<<"stop_reason">> := <<"refusal">>} = R} ->
            {error, {refusal, category(R)}};
        {ok, #{<<"stop_reason">> := <<"max_tokens">>, <<"content">> := Content}}
          when S#s.allow_truncated =:= true ->
            %% opt-in: keep a reply that hit the token limit (e.g. bulk text generation)
            {ok, text(Content), Msgs ++ [#{role => <<"assistant">>, content => Content}]};
        {ok, #{<<"stop_reason">> := <<"max_tokens">>}} ->
            {error, max_tokens};
        {ok, #{<<"stop_reason">> := Reason, <<"content">> := Content}}
          when Reason =:= <<"end_turn">>; Reason =:= <<"stop_sequence">> ->
            {ok, text(Content), Msgs ++ [#{role => <<"assistant">>, content => Content}]};
        {ok, Other} ->
            {error, {unexpected_response, maps:get(<<"stop_reason">>, Other, undefined)}};
        {error, _} = E -> E
    end.

safe_chat(Llm, Req, Opts) ->
    try Llm:chat(Req, Opts)
    catch Class:Reason -> {error, {llm_crashed, Class, Reason}} end.

text(Content) ->
    iolist_to_binary([T || #{<<"type">> := <<"text">>, <<"text">> := T} <- Content]).

category(#{<<"stop_details">> := #{<<"category">> := C}}) -> C;
category(_) -> null.
