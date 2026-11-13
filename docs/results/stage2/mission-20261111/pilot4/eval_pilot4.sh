#!/bin/sh
set -e

P4=docs/results/stage2/mission-20261111/pilot4
P3=docs/results/stage2/mission-20261111/pilot3
DIAG=docs/results/stage2/diagnostics-20261113
RUN=data/runs/retrieval/p4-retention
DATA=data/fineweb/stage1

export TLM_BLAS=1
export VECLIB_MAXIMUM_THREADS=1

for ARM in a b c d; do
  TAG="arm-${ARM}"
  CKPT="${RUN}/arm-${ARM}/last.ckpt"
  OUT="${P4}/eval_arm_${ARM}.json"
  if [ ! -e "$OUT" ]; then
    echo "=== 7-Probe Suite: $TAG ($CKPT) ==="
    sbcl --dynamic-space-size 6GB --script "${P4}/eval_probes.lisp" ckpt="$CKPT" tag="arm_${ARM}" out="$OUT"
  fi
  for CELL in d1-rev-pf d1-rev-vf d1-ext-pf; do
    DOUT="${P4}/${CELL}.${TAG}.json"
    if [ ! -e "$DOUT" ]; then
      echo "=== Frozen D1: ${CELL} ${TAG} ==="
      sbcl --dynamic-space-size 4GB --script lisp/bin/qa-analysis.lisp \
        data="$DATA" ckpt="$CKPT" items="${DIAG}/${CELL}.jsonl" out="$DOUT" workers=4 controls=0 paired=0 2>&1 | grep -v "^QA analysis"
    fi
  done
  SOUT="${P4}/selection.${TAG}.json"
  if [ ! -e "$SOUT" ]; then
    echo "=== Selection Retention Diagnostic: ${TAG} ==="
    sbcl --dynamic-space-size 4GB --script lisp/bin/mission-eval.lisp \
      dataset=selection ckpt="$CKPT" out="$SOUT" 2>&1 | grep -v "^QA analysis"
  fi
done

for ARM in b c d; do
  CKPT250="${RUN}/arm-${ARM}/step-00250.ckpt"
  OUT250="${P4}/eval_arm_${ARM}_250.json"
  if [ -e "$CKPT250" ] && [ ! -e "$OUT250" ]; then
    echo "=== 7-Probe Suite (step 250 trajectory): arm_${ARM}_250 ==="
    sbcl --dynamic-space-size 6GB --script "${P4}/eval_probes.lisp" ckpt="$CKPT250" tag="arm_${ARM}_250" out="$OUT250"
  fi
done
