#!/bin/sh
# The controlled three-model comparison: same corpus, tokenizer, splits, seed, schedule shape and token budget; only size and the
# validation-selected learning rate differ. For each SIZE:LR pair, in order:
#   0. stability probe: 600 steps at this rate (warmup 200, so it covers the peak-rate phase), loss and gradient norm every 10 steps,
#      then check-stability.lisp; an unstable rate stops the script for that size instead of spending hours on it
#   1. train OUT/SIZE/ (log every 50 steps, validation every 500, a checkpoint every 1000 steps, best.ckpt chosen on validation loss)
#   2. evaluate.lisp on best.ckpt: loss over the ENTIRE validation split, the test split (read once, here), the 90 questions
#   3. diagnose.lisp on best.ckpt: is the model using the question?
# Results go to OUT/SIZE/eval.json and OUT/SIZE/diagnosis.json; one summary line per size is appended to OUT/summary.jsonl.
#   lisp/bin/final-comparison.sh DATA OUT "1m:0.003 3m:0.003 10m:0.001" [steps=7300] [batch=32] [workers=8]
set -e
DATA=$1; OUT=$2; PAIRS=$3; STEPS=${4:-7300}; BATCH=${5:-32}; WORKERS=${6:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for pair in $PAIRS; do
  size=${pair%%:*}; lr=${pair##*:}; run="$OUT/$size"
  [ -f "$run/diagnosis.json" ] && continue
  rm -rf "$run"
  probe="$OUT/$size-probe"; rm -rf "$probe"
  TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script "$HERE/pretrain.lisp" data="$DATA" out="$probe" size=$size steps=600 batch=$BATCH \
      workers=$WORKERS lr=$lr warmup=200 eval-every=600 val-windows=16 seed=1 log-every=10 eval=/nonexistent > "$probe.log" 2>&1
  if ! sbcl --script "$HERE/check-stability.lisp" "$probe/log.jsonl" > "$probe/stability.json"; then
    echo "$size lr=$lr is NOT stable: $(cat "$probe/stability.json" | tr -d '\n ') -- stopping"; exit 3
  fi
  echo "$size lr=$lr stable"
  TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script "$HERE/pretrain.lisp" data="$DATA" out="$run" size=$size steps=$STEPS batch=$BATCH \
      workers=$WORKERS lr=$lr warmup=200 eval-every=500 val-windows=128 seed=1 log-every=50 save-every=1000 > "$run.log" 2>&1
  TLM_BLAS=1 sbcl --dynamic-space-size 4GB --script "$HERE/evaluate.lisp" data="$DATA" ckpt="$run/best.ckpt" | tail -1 > "$run/eval.json"
  TLM_BLAS=1 sbcl --dynamic-space-size 4GB --script "$HERE/diagnose.lisp" data="$DATA" ckpt="$run/best.ckpt" > "$run/diagnosis.json"
  done_line=$(grep '"event":"done"' "$run/log.jsonl")
  printf '{"size":"%s","lr":"%s","steps":%s,"train":%s,"eval":%s}\n' "$size" "$lr" "$STEPS" "$done_line" "$(cat "$run/eval.json")" >> "$OUT/summary.jsonl"
  echo "$size lr=$lr done"
done
