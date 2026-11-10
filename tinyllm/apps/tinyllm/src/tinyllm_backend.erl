%% The tensor-math backend behind tinyllm_tf's inference path (the part tinyllm_session uses).
%%
%% Two implementations:
%%   tinyllm_backend_erl  the original pure Erlang code (nested float lists); always available, used by the
%%                        EUnit logits-parity tests and for small models;
%%   tinyllm_backend_nif  C code over an mmap'd TLM1 file; weights and the KV cache are opaque resources, no
%%                        floats are boxed on the Erlang heap.
%% Both produce the same logits bit for bit (tinyllm_native_tests checks that with assertEqual).
%%
%% Which one runs is decided by the data, not by a flag: a model whose params are #{nif := Resource} (made
%% with tinyllm_backend_nif:to_native/2 or open/2) uses the NIF, everything else the Erlang code. Caches are
%% either the Erlang map of nested lists or #{n := N, res := Resource}; both answer to cache_len/1.
-module(tinyllm_backend).
-export([for_params/1, for_cache/1]).

-callback cache_len(Cache :: map()) -> non_neg_integer().
-callback cache_truncate(Cache :: map(), N :: non_neg_integer()) -> map().
-callback advance(Params :: map(), Cfg :: map(), Cache :: map(), TokenId :: non_neg_integer()) -> map().
-callback step(Params :: map(), Cfg :: map(), Cache :: map(), TokenId :: non_neg_integer()) ->
    {Logits :: [float()], map()}.
-callback last_logits(Params :: map(), Cfg :: map(), Ids :: [non_neg_integer()]) -> [float()].

for_params(#{nif := _}) -> tinyllm_backend_nif;
for_params(_) -> tinyllm_backend_erl.

for_cache(#{res := _}) -> tinyllm_backend_nif;
for_cache(_) -> tinyllm_backend_erl.
