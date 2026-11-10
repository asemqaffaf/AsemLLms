%% Supervises the chat sessions (tinyllm_session), one per conversation. Sessions are `temporary`:
%% their state is a cache that can be rebuilt from the transcript, so a crashed session is not
%% restarted; the next turn simply starts a fresh one (and the other sessions are unaffected).
-module(tinyllm_session_sup).
-behaviour(supervisor).
-export([start_link/0, init/1]).

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    {ok, {#{strategy => simple_one_for_one, intensity => 20, period => 10},
          [#{id => tinyllm_session, start => {tinyllm_session, start_link, []},
             restart => temporary, shutdown => 5000, type => worker}]}}.
