;;;; Load only the independent verifier's definitions, not its CLI entry form.
(let* ((here (directory-namestring *load-truename*))
       (cli (truename (merge-pathnames "../bin/mission-verify.lisp" here)))
       (*load-truename* cli))
  (with-open-file (s cli)
    (eval (read s))
    (let ((forms (loop for f = (read s nil :eof) until (eq f :eof) collect f)))
      (dolist (f (butlast forms)) (eval f))))
  (load (merge-pathnames "harness.lisp" here)))
(defun saved-fixture () (asem-verify-saved:read-json "docs/results/stage2/mission-20261111/r7-selection.json"))
(defun set-json (object key value) (setf (cdr (assoc key (cdr object) :test #'equal)) value))
(defparameter *fixture-bpe* (asem-bpe:load-bpe "data/fineweb/stage1/tokenizer.bpe"))
(defun rejects (x) (handler-case (progn (independently-validate x *fixture-bpe*) nil) (error () t)))
(harness:deftest valid-saved-records-recompute
  (harness:check (= 500 (length (independently-validate (saved-fixture) *fixture-bpe*)))))
(harness:deftest corrupted-ranking-is-rejected
  (let* ((x (saved-fixture)) (r (first (items x))))
    (set-json r "summed_option_prediction" (mod (1+ (g r "summed_option_prediction")) 4))
    (harness:check (rejects x))))
(harness:deftest corrupted-prefix-boolean-is-rejected
  (let* ((x (saved-fixture)) (r (first (items x))))
    (set-json r "gold_length_greedy_prefix_exact" (if (eq :true (g r "gold_length_greedy_prefix_exact")) :false :true))
    (harness:check (rejects x))))
(harness:deftest corrupted-prefix-text-is-rejected
  (let* ((x (saved-fixture)) (r (first (items x))))
    (set-json r "generated_text" "corrupt") (harness:check (rejects x))))
(harness:deftest duplicate-record-is-rejected
  (let* ((x (saved-fixture)) (r (items x)))
    (set-json (second r) "id" (g (first r) "id")) (harness:check (rejects x))))
(harness:deftest corrupted-summary-is-rejected
  (let* ((x (saved-fixture)) (r (g (g (g (g x "conditions") "real") "summary") "summed_option_likelihood")))
    (set-json r "correct" (1+ (g r "correct"))) (harness:check (rejects x))))
(harness:run-all)
