#!/usr/bin/env escript
%% Generate a dialogue dataset with the local LM Studio model, using parallel
%% Erlang agent workers. Appends to the output file after every wave, so you can
%% stop any time and keep what's there.
%%   rebar3 compile && escript scripts/make_dataset.escript [N] [Concurrency] [OutFile] [StartIndex]
main(Args) ->
    N = arg(Args, 1, 250, fun list_to_integer/1),
    C = arg(Args, 2, 3, fun list_to_integer/1),
    Out = arg(Args, 3, "../dataset/chat.txt", fun(X) -> X end),
    Start = arg(Args, 4, 1, fun list_to_integer/1),
    true = code:add_pathz("_build/default/lib/agents/ebin"),
    {ok, _} = application:ensure_all_started(agents),
    ok = filelib:ensure_dir(Out),
    Topics = topics(),
    Jobs = [job(I, lists:nth((I - 1) rem length(Topics) + 1, Topics)) || I <- lists:seq(Start, Start + N - 1)],
    T0 = erlang:monotonic_time(second),
    Total = waves(Jobs, C, Out, 0, 0, T0),
    io:format("~ndone: ~p conversations, ~p bytes in ~p s -> ~s~n",
              [Total, filelib:file_size(Out), erlang:monotonic_time(second) - T0, Out]).

arg(Args, I, Default, F) -> case length(Args) >= I of true -> F(lists:nth(I, Args)); false -> Default end.

job(I, Topic) ->
    Prompt = iolist_to_binary(
        ["Write 6 different short conversations between a User and an Assistant, all related to ", Topic,
         ". Each conversation has 3 or 4 exchanges, and each message is 1-2 plain sentences. ",
         "Separate conversations with one blank line. Start every conversation with a line beginning ",
         "'User:'. Write only the conversations, nothing else.\n\nExample of the format:\n",
         "User: How do I boil an egg?\nAssistant: Place it in boiling water for about nine minutes, then cool it in cold water.\n",
         "User: Can I use older eggs?\nAssistant: Yes, older eggs are actually easier to peel.\n\n",
         "User: What is a good snack?\nAssistant: An apple with peanut butter is quick and filling."]),
    {I, #{llm_opts => #{max_tokens => 1800, temperature => 0.9},
          system => <<"/no_think">>, max_turns => 1, allow_truncated => true}, Prompt}.

waves([], _, _, _Done, Kept, _T0) -> Kept;
waves(Jobs, C, Out, Done, Kept, T0) ->
    {Wave, Rest} = lists:split(min(C, length(Jobs)), Jobs),
    Results = agents:run_all(Wave, 600000),
    [io:format("  job ~p failed: ~p~n", [Id, Why]) || {Id, {error, Why}} <- Results],
    Good = [Clean || {_, {ok, Text}} <- Results, Clean <- [clean(Text)], Clean =/= invalid],
    ok = file:write_file(Out, [[G, "\n\n"] || G <- Good], [append]),
    Done1 = Done + length(Wave), Kept1 = Kept + length(Good),
    io:format("~p/~p jobs, ~p kept, ~p bytes, ~p s~n",
              [Done1, Done1 + length(Rest), Kept1, filelib:file_size(Out),
               erlang:monotonic_time(second) - T0]),
    waves(Rest, C, Out, Done1, Kept1, T0).

%% Split the reply into blank-line-separated conversations; keep the well-formed ones.
%% Returns a single binary of kept conversations, or `invalid` if none.
clean(Text) ->
    Blocks = re:split(Text, "\\n\\s*\\n", [{return, list}]),
    Good = [B || Block <- Blocks, B <- [conversation(Block)], B =/= invalid],
    case Good of
        [] -> invalid;
        _ -> unicode:characters_to_binary(string:join(Good, "\n\n"))
    end.

conversation(Block) ->
    Lines0 = [string:trim(L) || L <- string:split(Block, "\n", all)],
    Chat0 = [L || L <- Lines0, lists:prefix("User:", L) orelse lists:prefix("Assistant:", L),
                  string:find(L, "...") =:= nomatch],
    %% a reply cut off by the token limit ends mid-sentence: drop the unfinished line,
    %% and end on an Assistant line so every kept turn is complete
    Chat1 = drop_incomplete(Chat0),
    Chat = drop_trailing_user(Chat1),
    Users = length([L || L <- Chat, lists:prefix("User:", L)]),
    Asst = length([L || L <- Chat, lists:prefix("Assistant:", L)]),
    case Chat =/= [] andalso Users >= 1 andalso Asst >= 1 andalso lists:prefix("User:", hd(Chat)) of
        true -> string:join(Chat, "\n");
        false -> invalid
    end.

drop_incomplete([]) -> [];
drop_incomplete(Lines) ->
    case lists:member(lists:last(lists:last(Lines)), ".!?\")'") of
        true -> Lines;
        false -> lists:droplast(Lines)
    end.

drop_trailing_user(Lines) ->
    case Lines =/= [] andalso lists:prefix("User:", lists:last(Lines)) of
        true -> lists:droplast(Lines);
        false -> Lines
    end.

topics() ->
    ["gardening", "cooking pasta", "learning to code", "travel planning", "personal budgeting",
     "fitness routines", "choosing a laptop", "pets and dogs", "cats", "weather", "healthy sleep",
     "studying for exams", "job interviews", "writing emails", "home repairs", "cars and maintenance",
     "photography", "music practice", "learning a language", "movies", "books to read", "coffee and tea",
     "baking bread", "public transport", "moving to a new city", "making friends", "time management",
     "meditation", "hiking", "cycling", "football", "chess", "video games", "board games", "space and planets",
     "the ocean", "dinosaurs", "the history of Rome", "how computers work", "the internet", "smartphones",
     "online privacy", "recycling", "solar energy", "electric cars", "birthday gifts", "wedding planning",
     "cooking rice", "vegetarian meals", "street food", "a rainy day", "city parks", "libraries", "museums",
     "painting", "drawing for beginners", "knitting", "woodworking", "growing tomatoes", "bees", "birds",
     "the moon", "mountains", "rivers", "trains", "airports", "hotels", "restaurants", "cooking breakfast",
     "making a budget for a trip", "starting a small business", "customer service", "doctors appointments",
     "healthy snacks", "drinking water", "running a first 5k", "yoga", "swimming", "learning guitar",
     "learning piano", "podcasts", "news and media", "math homework", "science experiments", "chemistry",
     "volcanoes", "earthquakes", "the human body", "vitamins", "dental care", "cleaning the house",
     "laundry", "organizing a closet", "decorating a room", "buying furniture", "renting an apartment"].
