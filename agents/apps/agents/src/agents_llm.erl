%% An LLM backend: takes an Anthropic-shaped request, returns an
%% Anthropic-shaped response (`stop_reason` + `content` blocks).
-module(agents_llm).

-type request() :: #{system => binary(), messages := [map()], tools => [map()]}.
-type response() :: #{binary() => term()}.
-export_type([request/0, response/0]).

-callback chat(request(), Opts :: map()) -> {ok, response()} | {error, term()}.
