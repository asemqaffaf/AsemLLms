#!/bin/sh
# Cross-runtime check of the shared file: Lisp writes a TLM1 model (RMSNorm, no biases) and the logits it
# computes itself; the Erlang NIF mmaps the same file and must reproduce those logits bit for bit.
# Needs sbcl, erl, a built native/ (make -C native).
set -e
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
sbcl --noinform --non-interactive --load native/interop_write.lisp --eval "(asem-tf::interop-write \"$OUT\")" >/dev/null
ebin=$(mktemp -d)
erlc -o "$ebin" tinyllm/apps/tinyllm/src/tinyllm_nif.erl
TINYLLM_NIF="$PWD/native/build/tinyllm_nif" escript native/interop_check.escript "$ebin" "$OUT"
rm -rf "$OUT" "$ebin"
