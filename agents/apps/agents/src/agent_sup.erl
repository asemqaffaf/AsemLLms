%% Dynamic supervisor for agent workers. Agents are `temporary`: a crashed
%% agent is not restarted (its conversation lives in its process), and its
%% crash never touches other agents.
-module(agent_sup).
-behaviour(supervisor).
-export([start_link/0, start_agent/1, init/1]).

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

start_agent(Opts) -> supervisor:start_child(?MODULE, [Opts]).

init([]) ->
    {ok, {#{strategy => simple_one_for_one, intensity => 5, period => 10},
          [#{id => agent, start => {agent, start_link, []},
             restart => temporary, shutdown => 5000}]}}.
