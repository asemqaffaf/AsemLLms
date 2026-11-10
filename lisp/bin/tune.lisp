;;; sbcl --script lisp/bin/tune.lisp [--erlang]
;;; Generates, compiles, verifies and times kernel variants; writes the winners (see src/tune.lisp).
;;;   (no flag)  Lisp kernels -> src/tuned.sexp
;;;   --erlang   also tune tinyllm's dot/axpy -> tinyllm/apps/tinyllm/src/tinyllm_kernels.hrl (needs escript, rebar3)
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "tf" "tune")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*here* here))
(in-package :asem-tune)
(setf *read-default-float-format* 'double-float)
(let ((root (merge-pathnames "../../" cl-user::*here*)))
  (tune-lisp (merge-pathnames "../src/tuned.sexp" cl-user::*here*))
  (when (member "--erlang" sb-ext:*posix-argv* :test #'string=)
    (format t "~%Erlang kernels~%")
    (tune-erlang (merge-pathnames "tinyllm/" root) (merge-pathnames "../tune/" cl-user::*here*))))
