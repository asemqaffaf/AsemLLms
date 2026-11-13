(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "json" "bpe" "qagen" "retrieval")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  ;; Definitions needed by this module's PREPARE function, not by these semantic fixtures.
  (unless (find-package :asem-mission-data) (make-package :asem-mission-data :use '(:cl)))
  (export (intern "TRACED-SEQUENCES" :asem-mission-data) :asem-mission-data)
  (load (merge-pathnames "../experimental/inverse-contrastive/data.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(deftest opposite-role-follows-directed-chain
  (let ((f '(:edges (("Alice" "Bob") ("Bob" "Carol") ("Carol" "Dana")))))
    (check-equal (asem-inverse-data:negative-for :rel "Who helped Bob?" "Alice" '("Alice" "Bob" "Carol" "Dana") f) "Carol")
    (check-equal (asem-inverse-data:negative-for :rel "Who was helped by Bob?" "Carol" '("Alice" "Bob" "Carol" "Dana") f) "Alice")
    (check (null (asem-inverse-data:negative-for :rel "Who helped Dana?" "Carol" '("Alice" "Bob" "Carol" "Dana") f)))
    (check (handler-case (progn (asem-inverse-data:negative-for :rel "Who helped Bob?" "Dana" '("Dana") f) nil) (error () t)))))
(deftest reverse-negative-is-another-bound-person
  (let ((f '(:bindings (("Alice" "Osaka") ("Bob" "Tokyo")))))
    (check-equal (asem-inverse-data:negative-for :attr "Who lives in Osaka?" "Alice" '("Alice" "Bob" "Carol" "Dana") f) "Bob")
    (check (null (asem-inverse-data:negative-for :attr "Where does Alice live?" "Osaka" '("Osaka" "Tokyo") f)))
    (check (handler-case (progn (asem-inverse-data:negative-for :attr "Who lives in Osaka?" "Bob" '("Alice" "Bob") f) nil) (error () t)))))
(deftest trace-distinguishes-value-from-incidental-template-word
  (let ((f '(:bindings (("Dabel" "northern") ("Kriso" "best"))
             :queries ((:text "Who for now likes northern best?" :role :reverse :anchor "northern")))))
    (check-equal (asem-inverse-data:negative-for :attr "Who for now likes northern best?" "Dabel" '("Dabel" "Kriso") f) "Kriso")))
(run-all)
