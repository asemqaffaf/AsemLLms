#!/bin/sh
# Score item files with the unmodified evaluator. Run from the repository root.
#   docs/results/stage2/diagnostics-20261113/score.sh CKPT TAG file1 file2 ...   (file names without .jsonl; output file.TAG.json)
set -e
D=docs/results/stage2/diagnostics-20261113
CKPT=$1; TAG=$2; shift 2
for f in "$@"; do
  [ -e "$D/$f.$TAG.json" ] && { echo "exists: $f.$TAG.json"; continue; }
  TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/qa-analysis.lisp data=data/fineweb/stage1 \
    ckpt=$CKPT items=$D/$f.jsonl out=$D/$f.$TAG.json workers=4 controls=0 paired=0 2>&1 | grep -v "^QA analysis"
done
