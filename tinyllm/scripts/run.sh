#!/bin/sh
# Start the web UI with a fixed node name. The Mnesia disc schema (data/mnesia) belongs to the node
# name, so always launch through this script. 127.0.0.1 keeps the name stable when the hostname changes.
set -e
cd "$(dirname "$0")/.."
exec rebar3 shell --name tinyllm@127.0.0.1 --setcookie tinyllm_cookie "$@"
