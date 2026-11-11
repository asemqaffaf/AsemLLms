(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/candidate-decode.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(deftest shared-prefix-trie-and-order-independent-ties
  (let ((c '(#(1 2) #(1 3) #(4 5) #(4 6))))
    (check-equal (asem-candidate-decode:allowed-next c nil) '(1 4))
    (check-equal (asem-candidate-decode:allowed-next c '(1)) '(2 3))
    (check-equal (asem-candidate-decode:argmax #(0 5 5 0) '(2 1)) 1)
    (multiple-value-bind (tokens index) (asem-candidate-decode:greedy-candidate c (lambda (p) (if p #(0 0 1 5 0 0 0) #(9 8 0 0 7 0 0))))
      (check-equal tokens '(1 3)) (check-equal index 1))))
(deftest constrained-greedy-is-not-global-sequence-ranking
  ;; First token favors candidate 0, but its full sequence is less probable than candidate 1.
  (let* ((c '(#(1 2) #(3 4) #(5 6) #(7 8)))
         (root #(0 .6 0 .4 0 .01 0 .01 0))
         (after-one #(0 0 .01 0 0 0 0 0 0)))
    (multiple-value-bind (tokens index) (asem-candidate-decode:greedy-candidate c (lambda (p) (if p after-one root)))
      (check-equal index 0) (check-equal tokens '(1 2))
      (check (< (* .6 .01) (* .4 .99))))))
(deftest invalid-candidate-boundaries-are-rejected
  (dolist (c '((#(1) #(1 2) #(3) #(4)) (#(1) #(1) #(2) #(3)) (#() #() #() #())))
    (check (handler-case (progn (asem-candidate-decode:greedy-candidate c (constantly #(0 1 2 3 4))) nil) (error () t)))))
(deftest unrestricted-trace-is-not-forced-into-candidates
  (check-equal (asem-candidate-decode:greedy-unrestricted 3 (lambda (p) (declare (ignore p)) #(9 8 7))) '(0 0 0)))
(run-all)
