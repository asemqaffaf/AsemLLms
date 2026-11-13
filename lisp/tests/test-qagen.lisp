;;;; Stage 2 QA generator (src/qagen.lisp): determinism, balance, disjoint splits, validator, and the locked test file.
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "qagen")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here))
  (defparameter cl-user::*eval-dir* (merge-pathnames "../eval/qa/" here)))
(in-package :harness)

(defvar *dev* (asem-qagen:generate-split :dev 200 42))

(deftest same-seed-same-items
  (check (equalp (mapcar #'asem-qagen:item-json-line (asem-qagen:generate-split :dev 40 9))
                 (mapcar #'asem-qagen:item-json-line (asem-qagen:generate-split :dev 40 9))))
  (check (not (equalp (mapcar #'asem-qagen:item-json-line (asem-qagen:generate-split :dev 40 9))
                      (mapcar #'asem-qagen:item-json-line (asem-qagen:generate-split :dev 40 10))))))

(deftest answer-positions-balanced-within-each-category
  (dolist (c asem-qagen:+categories+)
    (let ((its (remove-if-not (lambda (it) (equal (getf it :category) c)) *dev*)))
      (check-equal (length its) 40)
      (dotimes (p 4) (check-equal (count p its :key (lambda (it) (getf it :answer))) 10)))))

(deftest every-item-validates-and-the-counterfactual-moves
  (check (every (lambda (it) (null (asem-qagen:validate-item it))) *dev*))
  (check (every (lambda (it) (/= (getf it :answer) (getf it :cf-answer))) *dev*)))

(deftest the-validator-catches-broken-items
  (let ((it (copy-list (first *dev*))))
    (setf (getf it :answer) (mod (1+ (getf it :answer)) 4))
    (check (asem-qagen:validate-item it) "wrong key"))
  (let ((it (copy-list (first *dev*))))
    (setf (getf it :options) (list* (second (getf it :options)) (rest (getf it :options))))
    (check (asem-qagen:validate-item it) "duplicate option"))
  (let ((it (copy-list (first *dev*))))
    (setf (getf it :options) (substitute "Zorblax" (nth (mod (1+ (getf it :answer)) 4) (getf it :options)) (getf it :options) :test #'equal))
    (check (asem-qagen:validate-item it) "option missing from the passage")))

(deftest entity-pools-are-disjoint
  (dolist (pool (list asem-qagen:*names* asem-qagen::*towns* asem-qagen::*jobs* asem-qagen::*pets* asem-qagen::*colors*
                      asem-qagen::*instruments* asem-qagen::*places*))
    (let ((tr (asem-qagen:split-pool pool :train)) (dv (asem-qagen:split-pool pool :dev)) (te (asem-qagen:split-pool pool :test)))
      (check (null (intersection tr te :test #'equal)))
      (check (null (intersection dv te :test #'equal)))
      (check (null (intersection tr dv :test #'equal)))
      (check (>= (length te) 4)))))

(deftest no-frame-is-shared-between-splits
  (let ((tr (asem-qagen:all-frames :train)) (dv (asem-qagen:all-frames :dev)) (te (asem-qagen:all-frames :test)))
    (check (null (intersection tr te :test #'string-equal)))
    (check (null (intersection dv te :test #'string-equal)))
    (check (null (intersection tr dv :test #'string-equal)))))

(deftest test-entities-never-appear-in-training-items
  (let ((train (asem-qagen:generate-split :train 500 3)) (test-names (asem-qagen:split-pool asem-qagen:*names* :test)))
    (check (notany (lambda (it) (some (lambda (n) (plusp (asem-qagen::word-boundary-count n (getf it :passage)))) test-names)) train))))

(deftest locked-test-file-matches-its-lock
  (let* ((f (merge-pathnames "test.jsonl" cl-user::*eval-dir*)) (lock (merge-pathnames "test.lock" cl-user::*eval-dir*)))
    (when (probe-file lock)
      (check-equal (asem-sha256:sha256-file-hex f) (with-open-file (in lock) (string-trim " " (read-line in))))
      (let ((items (asem-qagen:read-items f)))
        (check-equal (length items) 1000)
        (check (every (lambda (it) (equal (getf it :split) "test")) items))
        (dotimes (p 4) (check-equal (count p items :key (lambda (it) (getf it :answer))) 250))))))

(run-all)
