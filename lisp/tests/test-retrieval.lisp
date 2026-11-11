;;;; Entity-retrieval curriculum (src/retrieval.lisp): answers come from the passage, probes use unseen entities,
;;;; counterfactual questions change the answer, dense training targets are exactly the answer tokens.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf" "train" "pretrain" "sft"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(defun words (s) (asem-util:split-on (asem-data::normalize s) #\Space))

(deftest probe-answers-come-from-the-passage
  (dolist (level asem-retrieval:+levels+)
    (dolist (ent '(:dev-real :new))
      (let ((items (asem-retrieval:probe-items level ent :seen 60 (+ level 10))))
        (check-equal (length items) 60)
        (check (every (lambda (it) (search (getf it :answer) (getf it :passage))) items) (format nil "L~d ~a: answer in passage" level ent))
        (when (> level 1)
          (check (every (lambda (it) (and (getf it :cf-answer) (string/= (getf it :cf-answer) (getf it :answer)))) items)
                 (format nil "L~d ~a: counterfactual answer differs" level ent)))))))

(deftest probe-entities-never-occur-in-training
  (let* ((b (asem-bpe:train-bpe (list "a b c") 300))
         (train-names (asem-qagen:synthetic-pool 'asem-qagen:*names*))
         (train-real (asem-qagen:split-pool asem-qagen:*names* :train))
         (new (asem-retrieval::new-pool :names)) (dev (asem-qagen:split-pool asem-qagen:*names* :dev))
         (test (asem-qagen:split-pool asem-qagen:*names* :test)))
    (declare (ignore b))
    (check (null (intersection new train-names :test #'string-equal)) "new names vs generated training names")
    (check (null (intersection new (append train-real dev test) :test #'string-equal)) "new names vs every real pool")
    (check (null (intersection dev train-names :test #'string-equal)) "dev names vs generated training names")
    ;; and no probe ever uses a QA test-pool entity
    (let ((test-words (make-hash-table :test 'equal)))
      (dolist (pool (list asem-qagen:*names* asem-qagen::*towns* asem-qagen::*jobs* asem-qagen::*pets* asem-qagen::*colors* asem-qagen::*instruments*))
        (dolist (x (asem-qagen:split-pool pool :test)) (setf (gethash (string-downcase x) test-words) t)))
      (dolist (ent '(:dev-real :new))
        (check (notany (lambda (it) (some (lambda (w) (gethash w test-words)) (words (getf it :passage))))
                       (asem-retrieval:probe-items 4 ent :unseen 100 3))
               (format nil "~a probes contain no QA test entity" ent))))))

(deftest training-targets-are-exactly-the-answers
  (let* ((b (asem-bpe:train-bpe (mapcar #'asem-qagen:item-prompt (asem-qagen:generate-split :train 60 11)) 1500))
         (seqs (asem-retrieval:train-sequences 50 '(1 2 3 4) 4 b 256)))
    (check (= 50 (length seqs)))
    (dolist (s seqs)
      (destructuring-bind (ids rows targets) s
        (check (every (lambda (r tk) (= (nth (1+ r) ids) tk)) rows targets) "target = the token after each row")
        (check (search "The answer is" (asem-bpe:bpe-decode b (subseq ids 0 (1+ (first rows))))) "first target follows \"The answer is\"")))))

(deftest multi-examples-train-and-score
  (let* ((b (asem-bpe:train-bpe (mapcar #'asem-qagen:item-prompt (asem-qagen:generate-split :train 60 11)) 1500))
         (seqs (asem-retrieval:train-sequences 8 '(1) 5 b 256))
         (exs (coerce (loop for (ids rows targets) in seqs for i from 0 collect (asem-sft:multi-ex (format nil "m~d" i) ids rows targets)) 'vector))
         (m (asem-tf:init-model (asem-tf:make-config :vocab (asem-bpe:bpe-vocab-size b) :ctx 256 :dim 32 :heads 2 :layers 2 :ff 64) 3))
         (dir (format nil "/tmp/asem-test-retrieval-~d" (sb-posix:getpid)))
         (st (asem-sft:sft dir exs exs '(1 2 3 4) :init-model m :steps 200 :batch 8 :workers 2 :lr 3d-3 :warmup 10 :eval-every 100
                           :log-every 0 :train-probe 8 :target-acc 0.99d0)))
    (check (>= (getf st :best-acc) 0.99l0) (format nil "8 memorised retrieval sequences, exact-match accuracy ~a" (getf st :best-acc)))
    (sb-ext:delete-directory dir :recursive t)))

(deftest augmentation-keeps-answers-and-entities-intact
  (let ((asem-retrieval:*augment* t) (rng (sb-ext:seed-random-state 3)))
    (dotimes (k 200)
      (multiple-value-bind (passage qa) (asem-retrieval:make-fact-set (1+ (mod k 4)) :train :seen rng)
        (dolist (x qa)
          (check (search (cdr x) passage) "answer still in the passage"))))
    ;; probes are never augmented
    (let ((asem-retrieval:*augment* nil))
      (check (equal (asem-retrieval:probe-items 2 :new :seen 20 5) (asem-retrieval:probe-items 2 :new :seen 20 5))))))

(deftest reverse-lookups-ask-for-the-person
  (let ((asem-retrieval:*reverse* t) (persons 0) (n 0))
    (dolist (it (asem-retrieval:probe-items 3 :dev-real :seen 200 21 :options t))
      (incf n)
      (check (member (getf it :answer) (getf it :options) :test #'string=) "answer among the options")
      (when (find (getf it :answer) (asem-qagen:split-pool asem-qagen:*names* :dev) :test #'string=)
        (incf persons)
        (when (equal (getf it :kind) "attr")
          (check (every (lambda (o) (find o (asem-qagen:split-pool asem-qagen:*names* :dev) :test #'string=)) (getf it :options))
                 "a reverse lookup offers people as options"))))
    (check (< (* 0.2 n) persons) (format nil "person answers ~d of ~d" persons n))))

(run-all)
