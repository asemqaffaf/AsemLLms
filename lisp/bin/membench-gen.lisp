;;; sbcl --script lisp/bin/membench-gen.lisp --out cases.jsonl [--cases 20] [--lengths 50,200,800] [--facts 6] [--seed 1]
(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/membench.lisp" here)))
(asem-membench:main (cdr sb-ext:*posix-argv*))
