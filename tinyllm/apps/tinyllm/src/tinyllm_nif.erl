%% NIF stubs for native/tinyllm_nif.c (built by `make -C native install`). If the library is not there the
%% module still loads and every call raises nif_library_not_loaded; tinyllm_backend_nif:available/0 says which.
-module(tinyllm_nif).
-on_load(init/0).
-export([abi/0, load/1, cache_new/1, step/5, step_dirty/5, last_logits/2, last_logits_dirty/2]).

init() ->
    Base = case os:getenv("TINYLLM_NIF") of
               false -> filename:join(priv_dir(), "tinyllm_nif");
               Path -> Path
           end,
    _ = erlang:load_nif(Base, 0),
    ok.

priv_dir() ->
    case code:priv_dir(tinyllm) of
        {error, _} -> filename:join([filename:dirname(code:which(?MODULE)), "..", "priv"]);
        Dir -> Dir
    end.

%% {FormatVersion, BlasBuild}
abi() -> erlang:nif_error(nif_library_not_loaded).
%% {ok, ModelResource, InfoMap} | {error, Reason}: mmaps a TLM1 file read-only.
load(_Path) -> erlang:nif_error(nif_library_not_loaded).
cache_new(_Model) -> erlang:nif_error(nif_library_not_loaded).
%% step(Model, Cache, N, TokenId, full | kv_only) -> {Cache1, Logits | undefined}
step(_Model, _Cache, _N, _Id, _Mode) -> erlang:nif_error(nif_library_not_loaded).
step_dirty(_Model, _Cache, _N, _Id, _Mode) -> erlang:nif_error(nif_library_not_loaded).
last_logits(_Model, _Ids) -> erlang:nif_error(nif_library_not_loaded).
last_logits_dirty(_Model, _Ids) -> erlang:nif_error(nif_library_not_loaded).
