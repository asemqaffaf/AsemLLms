%% Public API.
%%   {ok, M} = tinyllm:train(tinyllm:erlang_corpus(), #{steps => 1500}),
%%   tinyllm:generate(M, <<"-module(">>, 300, #{temperature => 0.7}).
-module(tinyllm).
-export([train/2, generate/4, complete/4, chat/3, save/2, load/1, erlang_corpus/0, param_count/1,
         normalize/1, chat_prompt/1, sample_loop/8]).

train(Corpus, Opts) -> tinyllm_train:train(Corpus, Opts).

%% A ready-made local corpus: OTP's own stdlib source files.
erlang_corpus() ->
    Files = lists:sort(filelib:wildcard(filename:join([code:lib_dir(stdlib), "src", "*.erl"]))),
    iolist_to_binary([begin {ok, B} = file:read_file(F), B end || F <- Files]).

save(Model, Path) -> file:write_file(Path, term_to_binary(Model, [compressed])).
load(Path) ->
    %% `safe` only accepts atoms that already exist; the model's atom keys live in
    %% these modules, so make sure they are loaded before decoding in a fresh VM.
    _ = [code:ensure_loaded(M) || M <- [tinyllm_train, tinyllm_nn, tinyllm_tf]],
    {ok, B} = file:read_file(Path),
    normalize(binary_to_term(B, [safe])).

%% Models saved before word-level support stored the vocab as a list of byte values.
normalize(#{vocab := [I | _] = V} = M) when is_integer(I) ->
    M#{vocab := [<<X>> || X <- V], tokenizer => char};
normalize(#{cfg := Cfg} = M) -> maps:merge(#{tokenizer => char}, M#{cfg := maps:merge(#{arch => mlp}, Cfg)});
normalize(M) -> maps:merge(#{tokenizer => char}, M).

%% Number of trainable parameters.
param_count(#{params := P}) -> tinyllm_tf:param_count(P).

%% Extend Prompt by Len characters. Opts: temperature (default 0.8), seed, stop.
generate(Model, Prompt, Len, Opts) ->
    {Text, _} = complete(Model, Prompt, Len, Opts),
    <<Prompt/binary, Text/binary>>.

%% Like generate/4 but returns only the continuation plus why it ended:
%% `stop` (a stop string was produced; it is not included) or `length`.
%% Len counts tokens (characters for char-level models).
%% Opts: temperature, seed, stop => binary() | [binary()].
complete(Model0, Prompt, Len, Opts) ->
    #{cfg := #{context := N} = Cfg, vocab := Vocab, params := Params, tokenizer := Tok} = normalize(Model0),
    case maps:find(seed, Opts) of
        {ok, S} -> rand:seed(exsss, {S, S, S});
        error -> ok
    end,
    Temp = maps:get(temperature, Opts, 0.8),
    Stops = case maps:get(stop, Opts, []) of B when is_binary(B) -> [B]; L -> L end,
    VMap = maps:from_list(lists:zip(Vocab, lists:seq(0, length(Vocab) - 1))),
    Inv = list_to_tuple(Vocab),
    %% unknown prompt tokens: <unk> for word models, newline for char models
    Pad = case Tok of word -> 0; char -> maps:get(<<"\n">>, VMap, 0) end,
    PadCtx = maps:get(<<"\n">>, VMap, 0),
    Seed = [maps:get(T, VMap, Pad) || T <- tinyllm_tok:tokenize(Tok, Prompt)],
    {LogitsFun, Ctx} =
        case maps:get(arch, Cfg) of
            mlp ->
                PP = tinyllm_nn:prepare(Params),
                {fun(C) -> tinyllm_nn:logits(PP, C) end,
                 last_n(lists:duplicate(max(0, N - length(Seed)), PadCtx) ++ Seed, N)};
            transformer ->
                %% no padding: attention handles any length up to the context size
                PP = tinyllm_tf:prepare(Params),
                {fun(C) -> tinyllm_tf:last_logits(PP, Cfg, C) end,
                 last_n_or_all(case Seed of [] -> [PadCtx]; _ -> Seed end, N)}
        end,
    {Text, Why, _} = sample_loop(fun(C) -> LogitsFun(C) end,
                                 fun(C, I) -> last_n_or_all(C ++ [I], N) end,
                                 Ctx, Len, Temp, Inv, Tok, Stops),
    {Text, Why}.

%% The sampling loop, shared by the stateless path above and by tinyllm_session (KV-cache).
%%   LogitsFun(State) -> logits for the next token;  Push(State, TokenId) -> State with the token appended.
%% Draws one random number per token, in the same order, whatever the state is, so equal logits
%% and an equal seed give equal text. Returns {Text, stop | length, State}.
%% Stops are matched only against the tail of the text (a stop string may span tokens), so the cost
%% per token does not grow with the length of the reply.
sample_loop(LogitsFun, Push, State, Len, Temp, Inv, Tok, Stops) ->
    MaxStop = lists:max([0 | [byte_size(S) || S <- Stops]]),
    gen(LogitsFun, Push, State, Len, Temp, Inv, Tok, Stops, MaxStop, <<>>).

gen(_, _, State, 0, _, _, _, _, _, Text) -> {Text, length, State};
gen(LogitsFun, Push, State, Len, Temp, Inv, Tok, Stops, MaxStop, Text) ->
    Logits0 = LogitsFun(State),
    Logits = case Tok of word -> [-1.0e30 | tl(Logits0)]; char -> Logits0 end,   %% never emit <unk>
    I = tinyllm_nn:sample(tinyllm_nn:softmax(Logits, Temp), ok),
    Piece = element(I + 1, Inv),
    Text1 = <<Text/binary, Piece/binary>>,
    %% a stop string ending in the new piece starts at most MaxStop - 1 bytes before it
    From = max(0, byte_size(Text) - MaxStop + 1),
    case Stops =/= [] andalso binary:match(Text1, Stops, [{scope, {From, byte_size(Text1) - From}}]) of
        {Pos, _} -> {binary:part(Text1, 0, Pos), stop, Push(State, I)};
        _ -> gen(LogitsFun, Push, Push(State, I), Len - 1, Temp, Inv, Tok, Stops, MaxStop, Text1)
    end.

last_n_or_all(L, N) when length(L) > N -> lists:nthtail(length(L) - N, L);
last_n_or_all(L, _) -> L.

%% Chat with a model trained on "User: ...\nAssistant: ...\n" text (see the dataset
%% generator). Messages: [#{role := user | assistant | system, content := binary}]
%% (atom or binary keys; roles as atoms or binaries; system messages are ignored -
%% this model has never seen them). Returns {Reply, stop | length}.
chat(Model, Messages, Opts) ->
    Prompt = chat_prompt(Messages),
    {Reply, Why} = complete(Model, Prompt, maps:get(max_chars, Opts, 200),
                            Opts#{stop => [<<"\n">>, <<"User:">>]}),
    {string:trim(Reply), Why}.

%% The text the chat model is prompted with for a message list.
chat_prompt(Messages) ->
    Turns = [turn(M) || M <- Messages],
    iolist_to_binary([[L, ": ", C, "\n"] || {L, C} <- Turns, L =/= skip] ++ ["Assistant:"]).

turn(M) ->
    Role = field(role, M), Content = field(content, M),
    case Role of
        R when R =:= user; R =:= <<"user">> -> {"User", Content};
        R when R =:= assistant; R =:= <<"assistant">> -> {"Assistant", Content};
        _ -> {skip, Content}
    end.

field(K, M) ->
    case M of
        #{K := V} -> V;
        _ -> maps:get(atom_to_binary(K), M)
    end.

last_n(L, N) -> lists:nthtail(length(L) - N, L).
