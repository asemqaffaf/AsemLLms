%% The pure Erlang backend: delegates to the original implementations in tinyllm_tf.
-module(tinyllm_backend_erl).
-behaviour(tinyllm_backend).
-export([cache_len/1, cache_truncate/2, advance/4, step/4, last_logits/3]).

cache_len(Cache) -> tinyllm_tf:cache_len(Cache).
cache_truncate(Cache, N) -> tinyllm_tf:cache_truncate_erl(Cache, N).
advance(P, Cfg, Cache, Id) -> tinyllm_tf:advance_erl(P, Cfg, Cache, Id).
step(P, Cfg, Cache, Id) -> tinyllm_tf:step_erl(P, Cfg, Cache, Id).
last_logits(P, Cfg, Ids) -> tinyllm_tf:last_logits_erl(P, Cfg, Ids).
