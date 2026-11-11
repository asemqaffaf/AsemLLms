#!/bin/sh
# All Lisp tests without parsing locked QA answer keys. Verify locked bytes against the committed hash separately.
set -u
cd "$(dirname "$0")/../.." || exit 1
asem_temp=$(mktemp -d /tmp/asem-dev-tests.XXXXXX) || exit 1
asem_runner="$asem_temp/qagen.lisp"
trap 'rm -f "$asem_runner"; rmdir "$asem_temp"' EXIT HUP INT TERM
cat > "$asem_runner" <<'LISP'
(let* ((path (truename "lisp/tests/test-qagen.lisp")) (*load-truename* path))
  (with-open-file (in path)
    (loop for form = (read in nil :eof) until (eq form :eof)
          unless (and (consp form) (string= (symbol-name (car form)) "DEFTEST")
                      (string= (symbol-name (second form)) "LOCKED-TEST-FILE-MATCHES-ITS-LOCK"))
            do (eval form))))
LISP
asem_exit=0
for asem_test in lisp/tests/test-*.lisp; do
  echo "== $asem_test"
  if [ "$asem_test" = lisp/tests/test-qagen.lisp ]; then
    sbcl --noinform --non-interactive --load "$asem_runner" || asem_exit=1
  else
    sbcl --noinform --non-interactive --load "$asem_test" || asem_exit=1
  fi
done
asem_expected=$(tr -d ' \r\n' < lisp/eval/qa/test.lock)
asem_actual=$(shasum -a 256 lisp/eval/qa/test.jsonl | cut -d ' ' -f 1)
if [ "$asem_actual" = "$asem_expected" ]; then
  echo "Locked QA bytes match test.lock; answer keys were not parsed."
else
  echo "Locked QA hash mismatch" >&2
  asem_exit=1
fi
exit "$asem_exit"
