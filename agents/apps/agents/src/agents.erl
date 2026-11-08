%% Public API.
%%   {ok, A} = agents:start_agent(#{system => <<"...">>, tools => [agents_tools:calculate()]}),
%%   {ok, Text} = agents:ask(A, <<"What is 17 * 23?">>).
-module(agents).
-export([start_agent/1, ask/2, ask/3, history/1, stop/1, run_all/1, run_all/2]).

start_agent(Opts) -> agent_sup:start_agent(Opts).
ask(Pid, Prompt) -> agent:ask(Pid, Prompt).
ask(Pid, Prompt, Timeout) -> agent:ask(Pid, Prompt, Timeout).
history(Pid) -> agent:history(Pid).
stop(Pid) -> agent:stop(Pid).

%% Run many one-shot agents concurrently: Jobs = [{Name, AgentOpts, Prompt}].
%% Returns [{Name, {ok, Text} | {error, Reason}}] in job order; one failing
%% job never affects the others.
run_all(Jobs) -> run_all(Jobs, 300000).

run_all(Jobs, Timeout) ->
    Started = [begin
                   {Pid, MRef} = spawn_monitor(fun() -> exit({job_result, run_job(Opts, Prompt, Timeout)}) end),
                   {Name, Pid, MRef}
               end || {Name, Opts, Prompt} <- Jobs],
    [{Name, collect(Pid, MRef, Timeout + 5000)} || {Name, Pid, MRef} <- Started].

run_job(Opts, Prompt, Timeout) ->
    case start_agent(Opts) of
        {ok, A} ->
            try ask(A, Prompt, Timeout)
            catch exit:Reason -> {error, {agent_down, Reason}}
            after catch stop(A) end;
        {error, Reason} -> {error, {start_failed, Reason}}
    end.

collect(Pid, MRef, Timeout) ->
    receive
        {'DOWN', MRef, process, Pid, {job_result, R}} -> R;
        {'DOWN', MRef, process, Pid, Reason} -> {error, {job_crashed, Reason}}
    after Timeout ->
        exit(Pid, kill), erlang:demonitor(MRef, [flush]), {error, timeout}
    end.
