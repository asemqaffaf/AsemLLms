#!/bin/sh
# Run development tests without opening or hashing the locked QA benchmark.
set -u
cd "$(dirname "$0")/../.." || exit 1
asem_test_tmp=$(mktemp -d /tmp/asem-unlocked-tests.XXXXXX) || exit 1
asem_test_runner="$asem_test_tmp/qagen.lisp"
trap 'rm -f "$asem_test_runner"; rmdir "$asem_test_tmp"' EXIT HUP INT TERM
cat > "$asem_test_runner" <<'LISP'
(let* ((path (truename "lisp/tests/test-qagen.lisp")) (*load-truename* path))
  (with-open-file (in path)
    (loop for form = (read in nil :eof) until (eq form :eof)
          unless (and (consp form) (string= (symbol-name (car form)) "DEFTEST")
                      (string= (symbol-name (second form)) "LOCKED-TEST-FILE-MATCHES-ITS-LOCK"))
            do (eval form))))
LISP
asem_test_exit=0
for asem_test_path in lisp/tests/test-*.lisp; do
  echo "== $asem_test_path"
  if [ "$asem_test_path" = lisp/tests/test-qagen.lisp ]; then
    sbcl --noinform --non-interactive --load "$asem_test_runner" || asem_test_exit=1
  else
    sbcl --noinform --non-interactive --load "$asem_test_path" || asem_test_exit=1
  fi
done
exit "$asem_test_exit"
