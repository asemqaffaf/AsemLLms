%% The training engine behind tinyllm_tuner.
%%
%% The tuner (Erlang) decides WHAT to learn and WHETHER to keep it; an engine does the numerical work. Engines:
%%   tinyllm_engine_fake  deterministic 1-D quadratic, no model at all; the tuner's tests run on it everywhere;
%%   tinyllm_engine_port  runs an external worker (MLX LoRA on a Mac, llama.cpp LoRA, ...) one OS process per
%%                        operation and reads a `RESULT k=v ...` line back.
%%
%% A handle is an opaque term naming one set of trainable weights (an adapter directory, a head binary, a
%% number). The engine never mutates a handle: train/3 returns a NEW handle, so keeping the old one is a
%% rollback and dropping it is release/2. The tuner owns that lifecycle; engines only create and delete.
%%
%% Engine state (the term returned by init/1) is shared by many processes at once because the tuner trains
%% candidates in parallel, so it must be plain read-only configuration.
-module(tinyllm_engine).

-export_type([handle/0, job/0, metrics/0, eval_set/0]).

-type handle() :: term().
-type eval_set() :: heldout | probe.
-type job() :: #{lr := number(), steps := pos_integer(), rank := pos_integer(), batch := [term()]}.
-type metrics() :: #{train_loss => number(), atom() => term()}.

%% Start the engine and return the handle of the untouched base (version 0). It is never released.
-callback init(Opts :: map()) -> {ok, State :: term(), Base :: handle()} | {error, term()}.

%% Train from From (not modified) on Job; the result is a new handle.
-callback train(State :: term(), From :: handle(), job()) -> {ok, handle(), metrics()} | {error, term()}.

%% Mean loss of a handle on a fixed set. heldout: data the tuner learns toward; probe: reference data the
%% tuner must not make worse (the forgetting check).
-callback eval(State :: term(), handle(), eval_set()) -> {ok, number()} | {error, term()}.

%% Non-negative distance between two handles (how far the weights moved).
-callback drift(State :: term(), From :: handle(), To :: handle()) -> {ok, number()} | {error, term()}.

%% Free what a handle holds. Called for rejected candidates and pruned versions; must not fail the caller.
-callback release(State :: term(), handle()) -> ok.
