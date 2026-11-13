;;;; 2x2 wording cells change ONLY wording: same facts, entities, sentence order, question meaning, options, option order and keys.
;;;; Development items only (generated :dev items and lisp/eval/qa/dev.jsonl); the locked test file is never opened.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis" "wording"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here))
  (defparameter cl-user::*wording-here* here))
(in-package :harness)

(defparameter *generated* (asem-qagen:generate-split :dev 400 61117))
(defparameter *dev* (asem-qa-analysis:read-development (merge-pathnames "../eval/qa/dev.jsonl" cl-user::*wording-here*)
                                                       (merge-pathnames "../eval/qa/test.lock" cl-user::*wording-here*)))
(defparameter *stage1-bpe* (asem-bpe:load-bpe (merge-pathnames "fixtures/stage1-tokenizer.bpe" cl-user::*wording-here*)))
(defparameter *probes*
  (let ((asem-retrieval:*reverse* t) (asem-retrieval:*reasoning* t)
        (fits (lambda (prompt answer)
                (<= (+ (length (asem-bpe:bpe-encode *stage1-bpe* prompt))
                       (length (asem-bpe:bpe-encode *stage1-bpe* (format nil " ~a" answer)))) 128))))
    (mapcar #'asem-wording:probe-to-qa-item (asem-retrieval:probe-items 3 :new :unseen 200 19032 :fits fits :options t))))
(defun raises (thunk) (handler-case (progn (funcall thunk) nil) (error () t)))
(defun cell (it p q &optional (policy :seeded)) (asem-wording:cell-item it (list p q) :policy policy))

(deftest every-development-sentence-parses-losslessly-to-one-meaning
  (dolist (it (append *generated* *dev*))
    (let ((records (asem-wording:parse-passage (getf it :passage) '(:dev))))
      (check (plusp (length records)))
      ;; refilling each matched frame with its bindings reproduces the original sentence exactly
      (loop for (entry . b) in records for s in (asem-qagen::sentences-of (getf it :passage))
            do (check-equal (apply #'asem-qagen::fill-frame (getf entry :frame)
                                   (loop for (k v) on b by #'cddr append (if (eq k :n) (list :n v :v v) (list k v))))
                            s))
      ;; facts recomputed from the original text equal the generator's stored facts
      (let ((derived (asem-wording:derived-facts it records)) (stored (getf it :facts)))
        (if (member (getf it :category) '("extract" "reverse" "compare") :test #'string=)
            (check-equal (sort (mapcar #'list (getf derived :people) (or (getf derived :values) (getf derived :numbers))) #'string< :key #'first)
                         (sort (mapcar #'list (getf stored :people) (or (getf stored :values) (getf stored :numbers))) #'string< :key #'first))
            (check-equal (if (getf derived :pairs) (sort (copy-list (getf derived :pairs)) #'string< :key #'first) derived)
                         (if (getf stored :pairs) (sort (copy-list (getf stored :pairs)) #'string< :key #'first) stored))))
      ;; original questions parse to the stored structured queries
      (check-equal (asem-wording::question-query (asem-wording:parse-question (getf it :question) '(:dev))) (getf it :query))
      (check-equal (asem-wording::question-query (asem-wording:parse-question (getf it :cf-question) '(:dev))) (getf it :cf-query)))))

(deftest all-four-cells-preserve-meaning-and-answers-under-both-policies
  (dolist (policy '(:seeded :first))
    (dolist (it (append *generated* *dev* *probes*))
      (dolist (c asem-wording:+cells+)
        (let* ((v (asem-wording:cell-item it c :policy policy)) (problems (asem-wording:check-cell-item it v)))
          (check (null problems) (format nil "~a ~a ~a: ~{~a~^; ~}" (getf it :id) c policy problems)))))))

(deftest all-four-cells-fit-context-window-without-truncation
  (dolist (policy '(:seeded :first))
    (dolist (cell (asem-wording:cell-items (append *dev* *probes*) :policy policy))
      (let ((real (asem-sft32:encode-items (cdr cell) *stage1-bpe* 128 :mode :text))
            (cf (asem-sft32:encode-items (mapcar (lambda (it) (asem-qa-analysis:condition-item (cdr cell) it :counterfactual)) (cdr cell))
                                         *stage1-bpe* 128 :mode :text)))
        (check-equal (length real) (length (cdr cell)))
        (check-equal (length cf) (length (cdr cell)))))))

(deftest unfamiliar-cell-is-the-original-item-verbatim
  (dolist (it *dev*)
    (let ((v (cell it :u :u)))
      (dolist (k '(:passage :question :cf-question :options :answer :cf-answer :qframe)) (check-equal (getf v k) (getf it k))))))

(deftest familiar-wording-is-training-wording-and-differs-in-every-category
  (dolist (c asem-qagen:+categories+)
    (let ((items (remove-if-not (lambda (it) (equal (getf it :category) c)) *dev*)))
      (check (= 100 (length items)))
      (dolist (it items)
        (let ((ff (cell it :f :f)))
          (check (string/= (getf ff :passage) (getf it :passage)))
          (check (string/= (getf ff :question) (getf it :question)))
          (check (member (getf ff :qframe) (asem-qagen:all-frames :train) :test #'equal))
          (dolist (s (asem-qagen::sentences-of (getf ff :passage)))
            (check (member (getf (asem-wording:parse-sentence s '(:train)) :frame) (asem-qagen:all-frames :train) :test #'equal))
            (check (raises (lambda () (asem-wording:parse-sentence s '(:dev))))))
          ;; mixed cells take passage and question independently from the two pure cells
          (let ((fu (cell it :f :u)) (uf (cell it :u :f)))
            (check-equal (getf fu :passage) (getf ff :passage)) (check-equal (getf fu :question) (getf it :question))
            (check-equal (getf uf :passage) (getf it :passage)) (check-equal (getf uf :question) (getf ff :question))
            (check-equal (getf uf :cf-question) (getf ff :cf-question))))))))

(deftest first-policy-question-equals-the-historical-familiar-question
  (let ((historical (asem-qa-analysis:familiar-items *dev*)))
    (loop for it in *dev* for h in historical
          do (let ((uf (cell it :u :f :first)))
               (check-equal (getf uf :question) (getf h :question))
               (check-equal (getf uf :cf-question) (getf h :cf-question))))))

(deftest frame-choice-is-deterministic-varied-and-independent-of-the-answer
  (let ((seen (make-hash-table :test 'equal)))
    (dolist (it *dev*)
      (let* ((a (cell it :f :f)) (b (cell it :f :f))
             (perm (copy-list it)) (order '(3 1 0 2)))
        (check-equal (getf a :passage) (getf b :passage))
        (check-equal (getf a :question) (getf b :question))
        ;; permute options and remap keys: wording must not move
        (setf (getf perm :options) (mapcar (lambda (i) (nth i (getf it :options))) order)
              (getf perm :answer) (position (getf it :answer) order) (getf perm :cf-answer) (position (getf it :cf-answer) order))
        (let ((p (cell perm :f :f)))
          (check-equal (getf p :passage) (getf a :passage))
          (check-equal (getf p :question) (getf a :question))
          (check-equal (getf p :cf-question) (getf a :cf-question)))
        (dolist (s (asem-qagen::sentences-of (getf a :passage)))
          (setf (gethash (getf (asem-wording:parse-sentence s '(:train)) :frame) seen) t))))
    ;; seeded choice uses many training passage frames, not one per type
    (check (> (hash-table-count seen) 40))))

(deftest the-checker-detects-meaning-changes
  (let* ((cmp (find "compare" *dev* :key (lambda (it) (getf it :category)) :test #'equal))
         (tmp (find "temporal" *dev* :key (lambda (it) (getf it :category)) :test #'equal))
         (ext (find "extract" *dev* :key (lambda (it) (getf it :category)) :test #'equal))
         (rel (find "relation" *dev* :key (lambda (it) (getf it :category)) :test #'equal)))
    (flet ((flagged (orig v) (check (asem-wording:check-cell-item orig v) (format nil "undetected change in ~a" (getf orig :id)))))
      ;; a number changes: the max/min answer may change
      (let ((v (cell cmp :f :f)) (n (first (getf (getf cmp :facts) :numbers))))
        (setf (getf v :passage) (asem-qagen::replace-all (getf v :passage) (princ-to-string n) (princ-to-string (+ n 1000))))
        (flagged cmp v))
      ;; option order changes
      (let ((v (cell ext :f :f))) (setf (getf v :options) (reverse (getf v :options))) (flagged ext v))
      ;; temporal order changes: swap two sentences
      (let* ((v (cell tmp :f :f)) (ss (asem-qagen::sentences-of (getf v :passage))) (n (length ss)))
        (rotatef (nth (- n 2) ss) (nth (- n 3) ss))
        (setf (getf v :passage) (asem-qagen::join-sentences ss))
        (flagged tmp v))
      ;; relation direction flips: agent and patient swap in one sentence
      (let* ((v (cell rel :f :f)) (recs (asem-wording:parse-passage (getf v :passage) '(:train)))
             (r (find :rel recs :key (lambda (x) (getf (car x) :kind)))))
        (setf (getf v :passage) (asem-qagen::replace-all (getf v :passage) (asem-qagen::fill-frame (getf (car r) :frame) :a (getf (cdr r) :a) :b (getf (cdr r) :b))
                                                         (asem-qagen::fill-frame (getf (car r) :frame) :a (getf (cdr r) :b) :b (getf (cdr r) :a))))
        (flagged rel v))
      ;; the question asks about another entity
      (let ((v (cell ext :u :f))) (setf (getf v :question) (getf v :cf-question)) (flagged ext v))
      ;; a passage labelled familiar that kept development wording
      (let ((v (cell ext :u :u))) (setf (getf v :cell) "p-fam/q-unf") (flagged ext v))
      ;; a question labelled familiar that kept development wording
      (let ((v (cell ext :u :u))) (setf (getf v :cell) "p-unf/q-fam") (flagged ext v)))))

(deftest refuses-non-development-items
  (let ((it (copy-list (first *dev*)))) (setf (getf it :split) "test") (check (raises (lambda () (cell it :f :f))))))

(deftest paired-statistics-and-factorial-contrasts-match-exact-formulas
  (let* ((a (list (list :id "1" :category "compare" :subtype "age-max" :key 0 :gold-text "Ada" :sum-pred 0 :prefix-exact t)
                  (list :id "2" :category "compare" :subtype "age-min" :key 1 :gold-text "Ben" :sum-pred 0 :prefix-exact nil)
                  (list :id "3" :category "temporal" :subtype "first" :key 2 :gold-text "gym" :sum-pred 0 :prefix-exact nil)
                  (list :id "4" :category "temporal" :subtype "after" :key 3 :gold-text "zoo" :sum-pred 3 :prefix-exact nil)))
         (b (list (list :id "1" :category "compare" :subtype "age-max" :key 0 :gold-text "Ada" :sum-pred 0 :prefix-exact t)
                  (list :id "2" :category "compare" :subtype "age-min" :key 1 :gold-text "Ben" :sum-pred 1 :prefix-exact t)
                  (list :id "3" :category "temporal" :subtype "first" :key 2 :gold-text "gym" :sum-pred 2 :prefix-exact t)
                  (list :id "4" :category "temporal" :subtype "after" :key 3 :gold-text "zoo" :sum-pred 0 :prefix-exact nil)))
         (d (asem-wording:paired-diff a b :sum-pred))
         (f (asem-wording:factorial-effects a a a b :sum-pred)))
    (check-equal (asem-json:json-get d "first_correct") 2)
    (check-equal (asem-json:json-get d "second_correct") 3)
    (check-equal (asem-json:json-get d "both_correct") 1)
    (check-equal (asem-json:json-get d "both_wrong") 0)
    (check-equal (asem-json:json-get d "gained") 2)
    (check-equal (asem-json:json-get d "lost") 1)
    (check-close (asem-json:json-get d "diff") 0.25d0 1d-9)
    (check-close (asem-json:json-get d "mcnemar_exact_two_sided_p") 1d0 1d-9)
    (check-close (asem-json:json-get (asem-json:json-get f "passage_main_effect") "effect") 0.125d0 1d-9)
    (check-close (asem-json:json-get (asem-json:json-get f "question_main_effect") "effect") 0.125d0 1d-9)
    (check-close (asem-json:json-get (asem-json:json-get f "passage_x_question_interaction") "effect") 0.25d0 1d-9)))

(run-all)
