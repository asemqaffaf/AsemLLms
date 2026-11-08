%% Supervises the gradient workers. Workers are `permanent`: if one crashes
%% mid-training it is restarted at once (same registered name) and the
%% coordinator retries its shard.
-module(tinyllm_worker_sup).
-behaviour(supervisor).
-export([start_link/0, start_workers/1, stop_workers/1, name/1, init/1]).

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

name(N) -> list_to_atom("tinyllm_worker_" ++ integer_to_list(N)).

start_workers(Count) ->
    [{ok, _} = supervisor:start_child(?MODULE,
        #{id => N, start => {tinyllm_worker, start_link, [N]}, restart => permanent})
     || N <- lists:seq(1, Count)],
    [name(N) || N <- lists:seq(1, Count)].

stop_workers(Count) ->
    [begin supervisor:terminate_child(?MODULE, N), supervisor:delete_child(?MODULE, N) end
     || N <- lists:seq(1, Count)],
    ok.

init([]) -> {ok, {#{strategy => one_for_one, intensity => 20, period => 10}, []}}.
