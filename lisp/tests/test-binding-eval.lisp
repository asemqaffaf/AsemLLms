;;;; Model-free semantic and independent-verifier checks. No checkpoint or locked QA access.
(let ((here (directory-namestring *load-truename*)))
  (defparameter cl-user::*binding-here* here)
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "../experimental/phase2/frames.lisp" here))
  (load (merge-pathnames "../src/binding.lisp" here))
  (load (merge-pathnames "../src/verify-saved.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(defparameter *bpe* (asem-bpe:load-bpe (merge-pathnames "fixtures/stage1-tokenizer.bpe" cl-user::*binding-here*)))
(defparameter *pairs* (asem-binding:generate-pairs *bpe*))
(defparameter *json-pairs* (asem-binding:fixture-json *pairs* *bpe*))
(defparameter *families* (mapcar #'asem-binding:family-json (asem-binding:families)))
(defun g (x k) (asem-verify-saved:g x k))
(defun setj (x key v) (setf (cdr (assoc key (cdr x) :test #'equal)) v) x)
(defun clone (x) (asem-json:parse-json (with-output-to-string (s) (asem-json:write-json x s))))
(defun family-of (p) (find (g p "family") *families* :key (lambda (f) (g f "id")) :test #'equal))
(defun raises (thunk) (handler-case (progn (funcall thunk) nil) (error () t)))
(deftest stable-generator-does-not-install-phase2
  (check (not (fboundp 'asem-retrieval::pick-predicate-frame)))
  (check (not (fboundp 'asem-retrieval::%make-fact-set))))
(deftest all-400-pairs-pass-two-independent-semantic-checkers
  (check-equal (length *pairs*) 400)
  (loop for pair in *pairs* for jp across *json-pairs* do
    (check (asem-binding:validate-pair pair *bpe*))
    (check (asem-verify-saved::validate-counterfactual jp (family-of jp))))
  (dolist (cat '("relation" "extract" "reverse" "compare" "temporal"))
    (check-equal (count cat *pairs* :key (lambda (p) (getf p :category)) :test #'equal) (if (equal cat "relation") 200 50))))
(deftest role-queries-and-gold-positions-are-balanced
  (let* ((ps (remove-if-not (lambda (p) (equal (getf p :category) "relation")) *pairs*))
         (worlds (mapcar (lambda (p) (first (getf p :worlds))) ps)))
    (check-equal (count :rel-subj worlds :key (lambda (w) (first (getf w :query)))) 100)
    (dotimes (key 4) (check-equal (count key worlds :key (lambda (w) (getf w :answer))) 50))))
(deftest all-families-have-unambiguous-manual-agent-patient-examples
  ;; Literal examples independently check passive/cleft role interpretation.
  (dolist (case '((2 "Bren received instruction from Alba.") (5 "The person who hired Bren was Alba.")
                  (9 "Bren received a telephone call placed by Alba.") (13 "Bren was the person receiving Alba's thanks.")
                  (15 "Bren received a visit made by Alba.") (18 "Assistance for Bren came from Alba.")))
    (let* ((f (nth (first case) (asem-binding:families)))
           (w (asem-binding::relation-world f '("Alba" "Bren") '(("Alba" "Bren")) '(:rel-subj "Bren"))))
      (check-equal (getf w :passage) (second case))
      (check-equal (asem-binding:solve-world f w) "Alba"))))
(deftest independent-verifier-rejects-corrupted-counterfactuals
  (dolist (edit (list
                 (lambda (p) (let ((w (aref (g p "worlds") 1))) (setj w "key" (g (aref (g p "worlds") 0) "key"))))
                 (lambda (p) (let ((w (aref (g p "worlds") 0))) (setj w "facts" (g (aref (g p "worlds") 1) "facts"))))
                 (lambda (p) (setf (aref (g p "worlds") 1) (clone (aref (g p "worlds") 0))))
                 (lambda (p) (setj (aref (g p "worlds") 1) "question" "A different question?"))
                 (lambda (p) (setj (aref (g p "worlds") 1) "passage" "Unrelated text."))))
    (let ((p (clone (aref *json-pairs* 0)))) (funcall edit p) (check (raises (lambda () (asem-verify-saved::validate-counterfactual p (family-of p))))))))
(deftest constant-answer-cannot-pass-joint-metric
  (dolist (p *pairs*)
    (let ((keys (mapcar (lambda (w) (getf w :answer)) (getf p :worlds))))
      (dotimes (i 4) (check (not (asem-binding:joint-hit-p (list i i) keys)))))))
(deftest independent-statistics-handle-zero-discordance-and-family-clusters
  (check-close (asem-verify-saved::binomial-lower 200 200 .025d0) (expt .025d0 (/ 1d0 200)) 1d-10)
  (check-close (asem-verify-saved::paired-lower-bound 0 0 200) (- (expt .025d0 (/ 1d0 200)) 1d0) 1d-10)
  (let ((r (asem-verify-saved::family-bootstrap '(1d0 1d0 -1d0 -1d0) '("a" "a" "b" "b"))))
    (check-equal (g r "families") 2) (check-close (g r "ci95_low") -1d0 1d-12) (check-close (g r "ci95_high") 1d0 1d-12)))
(deftest first-index-ties-and-constant-gold-switching
  (check-equal (asem-verify-saved::maximum-index #(0d0 0d0 -1d0 -1d0)) 0)
  (check (not (asem-verify-saved::joint-outcome (list (asem-verify-saved:obj "key" 0 "summed_option_prediction" 0)
                                                    (asem-verify-saved:obj "key" 1 "summed_option_prediction" 0)) "summed_option_prediction"))))
(defun synthetic-perfect-row (w)
  (let* ((key (g w "key")) (lengths (map 'vector #'length (g w "option_token_ids")))
         (scores (make-array 4 :initial-element -12d0))
         (r (asem-verify-saved:obj "id" (g w "id") "category" (g w "category") "subtype" (g w "subtype") "question" (g w "question") "key" key
                                  "gold_text" (aref (g w "options") key) "option_token_lengths" lengths "summed_option_logprob" scores
                                  "summed_option_prediction" key "mean_token_option_prediction" key "gold_length_greedy_prefix_exact" :true
                                  "generated_token_ids" (aref (g w "option_token_ids") key) "prompt_tokens_without_eot" (1- (length (g w "prompt_token_ids")))
                                  "bounded_generated_text" (aref (g w "options") key) "generation_termination_reason" "token_limit"
                                  "bounded_free_generation_exact" :true "bounded_free_generation_terminated_exact" :false)))
    (setf (aref scores key) -1d0)
    (append r (list (cons "mean_token_option_logprob" (map 'vector #'/ scores lengths))))))
(deftest independent-prediction-validation-rejects-score-prefix-and-mean-corruption
  (let* ((w (aref (g (aref *json-pairs* 0) "worlds") 0)) (r (synthetic-perfect-row w)))
    (check (asem-verify-saved::validate-prediction r w))
    (dolist (edit (list (lambda (r) (setj r "summed_option_prediction" (mod (1+ (g r "key")) 4)))
                       (lambda (r) (setj r "gold_length_greedy_prefix_exact" :false))
                       (lambda (r) (setf (aref (g r "mean_token_option_logprob") 0) 999d0))
                       (lambda (r) (setj r "bounded_free_generation_exact" :false))))
      (let ((bad (clone r))) (funcall edit bad) (check (raises (lambda () (asem-verify-saved::validate-prediction bad w))))))))
(run-all)
