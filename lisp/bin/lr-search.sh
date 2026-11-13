#!/bin/sh
# Learning-rate search with one fixed budget per setting, selected on validation loss only (the test split is never read).
#   lisp/bin/lr-search.sh DATA OUT "1m 3m 10m" "0.001 0.003 0.01" [steps=800] [batch=32] [workers=8] [wait-for=PATTERN]
# For every size and rate: train OUT/SIZE-lrLR/ from the same seed, then evaluate best.ckpt on the full validation split.
# One line per setting is appended to OUT/results.jsonl: {"size","lr","steps","train":<the "done" event>,"eval":<evaluate.lisp output>}.
# Settings run one at a time so throughput and memory numbers are clean. wait-for: wait until no process matches PATTERN.
set -e
DATA=$1; OUT=$2; SIZES=$3; LRS=$4; STEPS=${5:-800}; BATCH=${6:-32}; WORKERS=${7:-8}; WAIT=$8
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
if [ -n "$WAIT" ]; then while pgrep -f "$WAIT" >/dev/null; do sleep 20; done; fi
for size in $SIZES; do
  for lr in $LRS; do
    run="$OUT/$size-lr$lr"
    [ -f "$run/best.ckpt" ] && grep -q "\"size\":\"$size\",\"lr\":\"$lr\"" "$OUT/results.jsonl" 2>/dev/null && continue
    rm -rf "$run"
    TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script "$HERE/pretrain.lisp" data="$DATA" out="$run" size=$size steps=$STEPS batch=$BATCH \
        workers=$WORKERS lr=$lr warmup=100 eval-every=$STEPS val-windows=64 seed=1 eval=/nonexistent > "$run.log" 2>&1
    done_line=$(grep '"event":"done"' "$run/log.jsonl")
    eval_line=$(TLM_BLAS=1 sbcl --dynamic-space-size 4GB --script "$HERE/evaluate.lisp" data="$DATA" ckpt="$run/best.ckpt" test=0 eval=/nonexistent | tail -1)
    printf '{"size":"%s","lr":"%s","steps":%s,"train":%s,"eval":%s}\n' "$size" "$lr" "$STEPS" "$done_line" "$eval_line" >> "$OUT/results.jsonl"
    echo "$size lr=$lr: $(echo "$eval_line" | sed 's/.*"val_loss":\([0-9.]*\).*/val loss \1/')"
  done
done
