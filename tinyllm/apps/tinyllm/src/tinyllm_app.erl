-module(tinyllm_app).
-behaviour(application).
-export([start/2, stop/1]).
start(_, _) ->
    Res = tinyllm_sup:start_link(),
    case application:get_env(tinyllm, web, false) of
        true -> ok = tinyllm_web:start();
        false -> ok
    end,
    Res.
stop(_) -> tinyllm_web:stop().
