(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/json.lisp" here))
  (load (merge-pathnames "../src/membench.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(defun gen (&rest args) (apply #'asem-membench:make-case "t" 7 args))
(defun user-texts (c) (loop for tn in (asem-membench::cc-turns c) when (string= (getf tn :role) "user") collect (getf tn :text)))
(defun events (c) (loop for tn in (asem-membench::cc-turns c) when (getf tn :event) collect (getf tn :event)))

(deftest deterministic
  (check-equal (asem-membench:case->json (gen :exchanges 60)) (asem-membench:case->json (gen :exchanges 60)))
  (check (string/= (asem-membench:case->json (gen :exchanges 60))
                   (asem-membench:case->json (asem-membench:make-case "t" 8 :exchanges 60)))))

(deftest shape
  (let ((c (gen :exchanges 80 :facts 6)))
    (check-equal (length (asem-membench::cc-turns c)) 160)
    (check-equal (length (asem-membench::cc-questions c)) 8)   ; 6 facts + 2 absent
    (check-equal (length (remove "absent" (asem-membench::cc-questions c) :key (lambda (q) (getf q :type)) :test-not #'string=)) 2)))

(deftest ground-truth-matches-the-transcript
  ;; replaying the events with "later statement wins" must give the expected answers
  (dolist (len '(40 120 400))
    (let ((c (gen :exchanges len)) (store (make-hash-table :test 'equal)))
      (dolist (ev (events c))
        (when (member (getf ev :kind) '(:plant :update))
          (setf (gethash (getf ev :relation) store) (getf ev :value))))
      (dolist (q (asem-membench::cc-questions c))
        (check-equal (gethash (getf q :relation) store) (getf q :expected))))))

(deftest updates-and-distractors
  (let* ((c (gen :exchanges 200 :facts 6 :update-fraction 0.5d0)) (qs (asem-membench::cc-questions c)))
    (check-equal (count "updated" qs :key (lambda (q) (getf q :type)) :test #'string=) 3)
    (dolist (q qs)
      (when (string= (getf q :type) "updated")
        (check (string/= (getf q :stale) (getf q :expected)))
        (check (< (getf q :planted-at) (getf q :updated-at))))
      (when (getf q :distractor)
        (check (string/= (getf q :distractor) (getf q :expected)))
        (check (not (equal (getf q :distractor) (getf q :stale))))))
    ;; distractors really appear in the transcript, as statements about someone else
    (check (some (lambda (ev) (eq (getf ev :kind) :other)) (events c)))))

(deftest placement
  (let ((c (gen :exchanges 300)))
    (loop for tn in (asem-membench::cc-turns c) for i from 0
          when (and (getf tn :event) (eq (getf (getf tn :event) :kind) :plant))
            do (check (< (floor i 2) 60)))                       ; planted in the first fifth
    (dolist (q (asem-membench::cc-questions c))
      (when (getf q :distance) (check (<= 1 (getf q :distance) 300))))))

(deftest absent-values-never-appear
  (let* ((c (gen :exchanges 100)) (all (asem-util:join " " (user-texts c))))
    (dolist (q (asem-membench::cc-questions c))
      (when (string= (getf q :type) "absent")
        (check (null (find (getf q :relation) (events c) :key (lambda (e) (getf e :relation)) :test #'string=)))))
    (check (plusp (length all)))))

(deftest output-is-valid-one-line-json
  (let ((line (asem-membench:case->json (gen :exchanges 50))))
    (check (not (find #\Newline line)))
    (let ((j (asem-json:parse-json line)))
      (check-equal (asem-json:json-get j "exchanges") 50)
      (check-equal (length (asem-json:json-get j "turns")) 100)
      (check-equal (length (asem-json:json-get j "questions")) 8))))

(run-all)
