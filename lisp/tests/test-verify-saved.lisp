(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "verify-saved")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here))
  (defparameter cl-user::*verifier-fixtures* (merge-pathnames "fixtures/verify-saved.json" here)))
(in-package :harness)
(deftest verifier-loads-no-evaluator-or-model-package
  (check (not (find-package :asem-qa-analysis)))
  (check (not (find-package :asem-binding)))
  (check (not (find-package :asem-tf32))))
(deftest verifier-valid-and-corrupted-reference-fixtures
  (check (asem-verify-saved:run-self-checks))
  (check-equal (asem-verify-saved:g (asem-verify-saved:verify-fixtures cl-user::*verifier-fixtures*) "passed") 11))
(deftest verifier-rejects-nonbinary-outcomes
  (check (handler-case (progn (asem-verify-saved:paired-binary '(t :false) '(nil t)) nil) (error () t))))
(deftest verifier-exact-mcnemar-asymmetric
  (let ((r (asem-verify-saved:paired-binary '(t t t nil) '(nil nil nil t))))
    (check-close (asem-verify-saved:g r "mcnemar_exact_two_sided_p") 0.625d0 1d-12)
    (check-close (asem-verify-saved:g r "diff") -0.5d0 1d-12)))
(run-all)
