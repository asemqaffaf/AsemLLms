;;; sbcl --script lisp/bin/clean-chat.lisp dataset/chat.txt out.txt
(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/clean.lisp" here)))
(asem-clean:main (cdr sb-ext:*posix-argv*))
