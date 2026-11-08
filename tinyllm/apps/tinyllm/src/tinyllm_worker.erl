%% A gradient worker. It holds the (shared, reference-counted) corpus and,
%% given the current parameters and some positions, returns the summed loss
%% and gradient for those examples.
-module(tinyllm_worker).
-behaviour(gen_server).
-export([start_link/1, grad/3, loss/3]).
-export([init/1, handle_call/3, handle_cast/2]).

start_link(N) -> gen_server:start_link({local, tinyllm_worker_sup:name(N)}, ?MODULE, [], []).

grad(Worker, Params, Positions) -> gen_server:call(Worker, {grad, Params, Positions}, 300000).
loss(Worker, Params, Positions) -> gen_server:call(Worker, {loss, Params, Positions}, 300000).

init([]) ->
    #{ids := Ids, cfg := Cfg} = persistent_term:get(tinyllm_corpus),
    {ok, {Ids, Cfg}}.

handle_call({grad, Params, Positions}, _From, {Ids, Cfg} = S) ->
    case maps:get(arch, Cfg, mlp) of
        mlp ->
            Examples = tinyllm_nn:examples(Ids, maps:get(context, Cfg), Positions),
            {Loss, Grad} = tinyllm_nn:loss_grad(tinyllm_nn:prepare(Params), Cfg, Examples),
            {reply, {Loss, length(Examples), Grad}, S};
        transformer ->
            T = maps:get(context, Cfg),
            Seqs = tinyllm_tf:sequences(Ids, T, Positions),
            {Loss, Grad} = tinyllm_tf:loss_grad(tinyllm_tf:prepare(Params), Cfg, Seqs),
            {reply, {Loss, length(Seqs) * T, Grad}, S}   %% every position of a sequence is a target
    end;
handle_call({loss, Params, Positions}, _From, {Ids, Cfg} = S) ->
    case maps:get(arch, Cfg, mlp) of
        mlp ->
            Examples = tinyllm_nn:examples(Ids, maps:get(context, Cfg), Positions),
            {reply, {tinyllm_nn:loss_only(tinyllm_nn:prepare(Params), Examples), length(Examples)}, S};
        transformer ->
            T = maps:get(context, Cfg),
            Seqs = tinyllm_tf:sequences(Ids, T, Positions),
            {reply, {tinyllm_tf:loss_only(tinyllm_tf:prepare(Params), Cfg, Seqs), length(Seqs) * T}, S}
    end.

handle_cast(_, S) -> {noreply, S}.
