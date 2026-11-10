#!/bin/sh
# Runs every Lisp test file; non-zero exit if any fails. Needs SBCL.
cd "$(dirname "$0")"
status=0
for t in tests/test-*.lisp; do
  echo "== $t"
  sbcl --noinform --non-interactive --load "$t" || status=1
done
exit $status
