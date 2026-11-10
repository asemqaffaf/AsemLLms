-module(tinyllm_app).
-behaviour(application).
-export([start/2, stop/1]).
start(_, _) ->
    Res = tinyllm_sup:start_link(),
    _ = tinyllm_memory:start(),
    case application:get_env(tinyllm, web, false) of
        true -> ok = tinyllm_web:start();
        false -> ok
    end,
    Res.
%% Mnesia is deliberately left running: stopping an application from inside another application's
%% stop callback calls the (busy) application controller and deadlocks. It flushes on VM shutdown.
stop(_) -> tinyllm_web:stop(), tinyllm_memory:forget().
