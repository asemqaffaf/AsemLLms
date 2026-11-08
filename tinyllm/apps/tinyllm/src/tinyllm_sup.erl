-module(tinyllm_sup).
-behaviour(supervisor).
-export([start_link/0, init/1]).

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    {ok, {#{strategy => one_for_one, intensity => 5, period => 10},
          [#{id => tinyllm_worker_sup, start => {tinyllm_worker_sup, start_link, []},
             type => supervisor}]}}.
