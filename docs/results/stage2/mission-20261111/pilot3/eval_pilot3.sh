#!/bin/sh
# Evaluate Pilot 3 checkpoints (steps 250 and 500 of both arms) using unmodified evaluators only.
# Run from repository root.
set -e

P3=docs/results/stage2/mission-20261111/pilot3
DIAG=docs/results/stage2/diagnostics-20261113
RUN=data/runs/retrieval/p3-learnability
DATA=data/fineweb/stage1

eval_qa() {
  CKPT=$1
  ITEMS=$2
  OUT=$3
  if [ -e "$OUT" ]; then
    echo "exists: $OUT"
    return 0
  fi
  TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/qa-analysis.lisp \
    data="$DATA" ckpt="$CKPT" items="$ITEMS" out="$OUT" workers=4 controls=0 paired=0 2>&1 | grep -v "^QA analysis"
}

eval_sel() {
  CKPT=$1
  OUT=$2
  if [ -e "$OUT" ]; then
    echo "exists: $OUT"
    return 0
  fi
  TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/mission-eval.lisp \
    dataset=selection ckpt="$CKPT" out="$OUT" 2>&1 | grep -v "^QA analysis"
}

for ARM in 5e-4 1e-3; do
  for STEP in 250 500; do
    TAG="arm-${ARM}.step${STEP}"
    CKPT="$RUN/arm-${ARM}/step-$(printf '%05d' "$STEP").ckpt"
    echo "=== Evaluating $CKPT ==="
    eval_qa "$CKPT" "$P3/probe-fresh-500.jsonl" "$P3/probe-fresh-500.${TAG}.json"
    eval_qa "$CKPT" "$P3/probe-seen-500.jsonl" "$P3/probe-seen-500.${TAG}.json"
    eval_qa "$CKPT" "$DIAG/d1-rev-pf.jsonl" "$P3/d1-rev-pf.${TAG}.json"
    eval_qa "$CKPT" "$DIAG/d1-rev-vf.jsonl" "$P3/d1-rev-vf.${TAG}.json"
    eval_qa "$CKPT" "$DIAG/d1-ext-pf.jsonl" "$P3/d1-ext-pf.${TAG}.json"
    eval_sel "$CKPT" "$P3/selection.${TAG}.json"
  done
done
