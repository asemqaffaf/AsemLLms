%% The TLM1 file format and the native backend. The format tests always run. The NIF tests run only when
%% native/tinyllm_nif.so has been built (`make -C native install`); without it they are skipped, so a checkout
%% with no C compiler still passes `rebar3 eunit`.
%%
%% Every comparison is assertEqual on lists of floats: the native code is meant to be bit-identical to the
%% pure Erlang code, not merely close.
-module(tinyllm_native_tests).
-include_lib("eunit/include/eunit.hrl").

cfgs() ->
    [#{vocab => 23, context => 12, dim => 16, heads => 4, layers => 3, ff => 32},
     #{vocab => 31, context => 9, dim => 24, heads => 3, layers => 2, ff => 40},
     #{vocab => 7, context => 5, dim => 8, heads => 1, layers => 1, ff => 8}].

tmp(Name) ->
    Dir = filename:join(os:getenv("TMPDIR", "/tmp"), "tinyllm_native_tests"),
    ok = filelib:ensure_dir(filename:join(Dir, "x")),
    filename:join(Dir, integer_to_list(erlang:unique_integer([positive])) ++ "_" ++ Name).

ids(Cfg, Len) ->
    V = maps:get(vocab, Cfg),
    [(I * 7 + 3) rem V || I <- lists:seq(1, Len)].

native(Cfg, Params, Name) ->
    {ok, #{params := PN}} = tinyllm_backend_nif:to_native(#{cfg => Cfg, params => Params}, tmp(Name)),
    PN.

%% ------------------------------------------------------------------ format

format_roundtrip_test_() ->
    [fun() ->
         Params = tinyllm_tf:init(Cfg, 7),
         Path = tmp("roundtrip.tlm"),
         ok = tinyllm_tlm:write(Path, Params, Cfg),
         {ok, Bin} = file:read_file(Path),
         ?assertEqual(0, byte_size(Bin) rem 4096),                    %% whole pages: mmap-friendly
         {ok, #{params := Back, cfg := Cfg2, norm := layer, biases := true, data_off := Off}} = tinyllm_tlm:read(Path),
         ?assertEqual(0, Off rem 4096),
         ?assertEqual(Params, Back),
         ?assertEqual(Cfg, Cfg2)
     end || Cfg <- cfgs()].

format_rejects_bad_shapes_test() ->
    Cfg = hd(cfgs()),
    Params = (tinyllm_tf:init(Cfg, 1))#{bout => [0.0, 0.0]},
    ?assertError({bad_shape, "out.b"}, tinyllm_tlm:write(tmp("bad.tlm"), Params, Cfg)).

%% ------------------------------------------------------------------ native backend

native_test_() ->
    case tinyllm_backend_nif:available() of
        false -> [];
        true -> {timeout, 300, [fun last_logits_bit_identical/0, fun step_and_advance_bit_identical/0,
                                fun dirty_scheduler_path/0, fun truncate_and_copy_on_write/0,
                                fun errors_are_reported/0, fun bad_files_are_rejected/0,
                                fun session_equals_pure_erlang/0, fun info_and_param_count/0]}
    end.

last_logits_bit_identical() ->
    [begin
         Params = tinyllm_tf:init(Cfg, 5),
         PE = tinyllm_tf:prepare(Params),
         PN = native(Cfg, Params, "ll.tlm"),
         [?assertEqual(tinyllm_tf:last_logits(PE, Cfg, ids(Cfg, Len)), tinyllm_tf:last_logits(PN, Cfg, ids(Cfg, Len)))
          || Len <- lists:seq(1, maps:get(context, Cfg))]
     end || Cfg <- cfgs()].

step_and_advance_bit_identical() ->
    [begin
         Params = tinyllm_tf:init(Cfg, 11),
         PE = tinyllm_tf:prepare(Params),
         PN = native(Cfg, Params, "st.tlm"),
         Ids = ids(Cfg, maps:get(context, Cfg)),
         %% step at every position against the Erlang cache, and against the full forward pass
         lists:foldl(fun(I, {CE, CN, Seen}) ->
                             {LE, CE1} = tinyllm_tf:step(PE, Cfg, CE, I),
                             {LN, CN1} = tinyllm_tf:step(PN, Cfg, CN, I),
                             ?assertEqual(LE, LN),
                             ?assertEqual(tinyllm_tf:last_logits(PE, Cfg, Seen ++ [I]), LN),
                             ?assertEqual(tinyllm_tf:cache_len(CE1), tinyllm_tf:cache_len(CN1)),
                             {CE1, CN1, Seen ++ [I]}
                     end, {tinyllm_tf:cache_new(Cfg), tinyllm_tf:cache_new(Cfg), []}, Ids),
         %% prefill with advance/4, then step the last token
         Pre = lists:droplast(Ids),
         CA = lists:foldl(fun(I, C) -> tinyllm_tf:advance(PN, Cfg, C, I) end, tinyllm_tf:cache_new(Cfg), Pre),
         ?assertEqual(length(Pre), tinyllm_tf:cache_len(CA)),
         {LA, _} = tinyllm_tf:step(PN, Cfg, CA, lists:last(Ids)),
         ?assertEqual(tinyllm_tf:last_logits(PE, Cfg, Ids), LA)
     end || Cfg <- cfgs()].

dirty_scheduler_path() ->
    Cfg = hd(cfgs()),
    Params = tinyllm_tf:init(Cfg, 3),
    PE = tinyllm_tf:prepare(Params),
    #{dirty := false} = PN = native(Cfg, Params, "dirty.tlm"),
    PD = PN#{dirty := true},
    Ids = ids(Cfg, 8),
    ?assertEqual(tinyllm_tf:last_logits(PE, Cfg, Ids), tinyllm_tf:last_logits(PD, Cfg, Ids)),
    {L, _} = lists:foldl(fun(I, {_, C}) -> tinyllm_tf:step(PD, Cfg, C, I) end, {x, tinyllm_tf:cache_new(Cfg)}, Ids),
    ?assertEqual(tinyllm_tf:last_logits(PE, Cfg, Ids), L).

%% The same properties the pure Erlang cache has, although the native buffer is mutable.
truncate_and_copy_on_write() ->
    Cfg = hd(cfgs()),
    Params = tinyllm_tf:init(Cfg, 5),
    PE = tinyllm_tf:prepare(Params),
    PN = native(Cfg, Params, "cow.tlm"),
    Ref = fun(Ids) -> tinyllm_tf:last_logits(PE, Cfg, Ids) end,
    Ids8 = [3, 0, 10, 7, 7, 4, 9, 1],
    {_, C8} = lists:foldl(fun(I, {_, C}) -> tinyllm_tf:step(PN, Cfg, C, I) end, {x, tinyllm_tf:cache_new(Cfg)}, Ids8),
    ?assertEqual(8, tinyllm_tf:cache_len(C8)),
    %% truncate and continue: same as the full forward pass over the shorter prefix
    C5 = tinyllm_tf:cache_truncate(C8, 5),
    ?assertEqual(5, tinyllm_tf:cache_len(C5)),
    {LA, C6a} = tinyllm_tf:step(PN, Cfg, C5, 4),
    ?assertEqual(Ref([3, 0, 10, 7, 7, 4]), LA),
    %% the longer value is untouched by what happened to its truncation ...
    {LB, _} = tinyllm_tf:step(PN, Cfg, C8, 12),
    ?assertEqual(Ref(Ids8 ++ [12]), LB),
    %% ... and so is the truncation, stepped again with a different token after the first branch advanced
    {LC, _} = tinyllm_tf:step(PN, Cfg, C5, 2),
    ?assertEqual(Ref([3, 0, 10, 7, 7, 2]), LC),
    {LD, _} = tinyllm_tf:step(PN, Cfg, C6a, 6),
    ?assertEqual(Ref([3, 0, 10, 7, 7, 4, 6]), LD),
    %% truncating to a longer length is a no-op, to zero restarts
    ?assertEqual(C8, tinyllm_tf:cache_truncate(C8, 20)),
    {LE, _} = tinyllm_tf:step(PN, Cfg, tinyllm_tf:cache_truncate(C8, 0), 5),
    ?assertEqual(Ref([5]), LE).

errors_are_reported() ->
    Cfg = hd(cfgs()),
    Params = tinyllm_tf:init(Cfg, 5),
    PN = native(Cfg, Params, "err.tlm"),
    T = maps:get(context, Cfg),
    Full = lists:foldl(fun(I, C) -> tinyllm_tf:advance(PN, Cfg, C, I) end, tinyllm_tf:cache_new(Cfg), ids(Cfg, T)),
    ?assertError(context_full, tinyllm_tf:step(PN, Cfg, Full, 1)),
    ?assertError(context_full, tinyllm_tf:last_logits(PN, Cfg, ids(Cfg, T + 1))),
    ?assertError(bad_token, tinyllm_tf:step(PN, Cfg, tinyllm_tf:cache_new(Cfg), maps:get(vocab, Cfg))),
    ?assertError(badarg, tinyllm_tf:last_logits(PN, Cfg, [])),
    %% a cache value that claims more positions than the buffer holds
    C2 = tinyllm_tf:advance(PN, Cfg, tinyllm_tf:cache_new(Cfg), 1),
    ?assertError(badarg, tinyllm_tf:step(PN, Cfg, C2#{n := 5}, 1)),
    %% a native cache needs the model it was made with
    PN2 = native(Cfg, Params, "err2.tlm"),
    ?assertError(badarg, tinyllm_tf:step(PN2, Cfg, C2, 1)),
    %% an Erlang cache with content cannot be continued natively
    {_, CE} = tinyllm_tf:step(tinyllm_tf:prepare(Params), Cfg, tinyllm_tf:cache_new(Cfg), 1),
    ?assertError(cache_backend_mismatch, tinyllm_tf:step(PN, Cfg, CE, 1)).

bad_files_are_rejected() ->
    Cfg = hd(cfgs()),
    Params = tinyllm_tf:init(Cfg, 5),
    Path = tmp("good.tlm"),
    ok = tinyllm_tlm:write(Path, Params, Cfg),
    Meta = #{cfg => Cfg},
    ?assertMatch({ok, _}, tinyllm_backend_nif:open(Path, Meta)),
    ?assertMatch({error, _}, tinyllm_backend_nif:open(tmp("missing.tlm"), Meta)),
    ?assertMatch({error, {cfg_mismatch, [dim]}}, tinyllm_backend_nif:open(Path, #{cfg => Cfg#{dim := 24}})),
    {ok, Bin} = file:read_file(Path),
    Bad = fun(Name, B) -> P = tmp(Name), ok = file:write_file(P, B), tinyllm_backend_nif:open(P, Meta) end,
    ?assertMatch({error, _}, Bad("short.tlm", binary:part(Bin, 0, 100))),
    ?assertMatch({error, _}, Bad("cut.tlm", binary:part(Bin, 0, byte_size(Bin) - 4096))),
    ?assertMatch({error, _}, Bad("magic.tlm", <<"XXXX", (binary:part(Bin, 4, byte_size(Bin) - 4))/binary>>)),
    %% a tensor whose offset points past the end of the file
    <<Head:(128 + 32)/binary, _Off:64/little, Rest/binary>> = Bin,
    ?assertMatch({error, _}, Bad("offset.tlm", <<Head/binary, (byte_size(Bin) * 2):64/little, Rest/binary>>)).

%% tinyllm_session on a native model: same text, same bookkeeping, also across window slides (the cache is
%% truncated to 0 and rebuilt every token once the context is full).
session_equals_pure_erlang() ->
    Cfg = hd(cfgs()),
    Params = tinyllm_tf:init(Cfg, 9),
    Vocab = [<<C>> || C <- lists:seq($a, $a + maps:get(vocab, Cfg) - 1)],
    Model = #{cfg => Cfg#{arch => transformer}, vocab => Vocab, params => Params, tokenizer => char},
    {ok, Native} = tinyllm_backend_nif:to_native(Model, tmp("session.tlm")),
    Run = fun(M, Opts) ->
                  {ok, Pid} = tinyllm_session:start_link(M, Opts),
                  {ok, _} = tinyllm_session:feed(Pid, <<"abcde">>),
                  R1 = tinyllm_session:complete(Pid, 40, #{temperature => 0.9, seed => 42}),
                  {ok, _} = tinyllm_session:feed(Pid, <<"cab">>),
                  R2 = tinyllm_session:complete(Pid, 10, #{temperature => 0.5, seed => 7}),
                  Logits = tinyllm_session:logits(Pid),
                  Info = tinyllm_session:info(Pid),
                  tinyllm_session:stop(Pid),
                  {R1, R2, Logits, Info}
          end,
    [?assertEqual(Run(Model, O), Run(Native, O)) || O <- [#{}, #{evict => 4}]],
    %% and the stateless path in tinyllm:complete/4
    ?assertEqual(tinyllm:complete(Model, <<"abc">>, 30, #{seed => 1}), tinyllm:complete(Native, <<"abc">>, 30, #{seed => 1})).

info_and_param_count() ->
    Cfg = hd(cfgs()),
    Params = tinyllm_tf:init(Cfg, 5),
    {ok, #{params := PN}} = tinyllm_backend_nif:to_native(#{cfg => Cfg, params => Params}, tmp("info.tlm")),
    ?assertEqual(tinyllm_tf:param_count(Params), tinyllm_tf:param_count(PN)),
    #{norm := layer, biases := true, layers := 3} = tinyllm_backend_nif:info(PN).
