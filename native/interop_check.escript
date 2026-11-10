#!/usr/bin/env escript
%% Used by interop_test.sh. Args: EbinDir DataDir
main([Ebin, Dir]) ->
    true = code:add_pathz(Ebin),
    {ok, Res, #{norm := rms, biases := false, vocab := 13, context := 9}} = tinyllm_nif:load(list_to_binary(Dir ++ "/lisp.tlm")),
    {ok, Bin} = file:read_file(Dir ++ "/logits.bin"),
    Expected = [X || <<X:64/float-little>> <= Bin],
    {Got, 0} = lists:foldl(fun(Len, {Acc, _}) ->
                                   Ids = [(3 + 7 * I) rem 13 || I <- lists:seq(1, Len)],
                                   {Acc ++ tinyllm_nif:last_logits(Res, Ids), 0}
                           end, {[], 0}, lists:seq(1, 9)),
    case Got =:= Expected of
        true -> io:format("interop ok: ~p logits identical (Lisp-written RMSNorm model, Erlang NIF)~n", [length(Got)]);
        false -> io:format("interop FAILED~n"), halt(1)
    end.
