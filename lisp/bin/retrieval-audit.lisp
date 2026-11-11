;;; Deterministic, model-free audit of the actual retrieval generator and BPE.
;;; Example (the R7 recipe, including its copy replay count):
;;;   sbcl --script lisp/bin/retrieval-audit.lisp data=data/fineweb/stage1 
;;;     train-n=75000 levels=1,2,3 seed=7 ctx=128 broad=1 augment=1
;;;     options=1 reverse=1 reasoning=1 copy-mix=15000 out=audit.json
;;; No QA dataset is opened. Static held-out entity pools are used only for
;;; exclusion and tokenizer fragmentation statistics, as in the generator.
(defparameter *audit-script* *load-truename*)
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(defpackage :retrieval-audit-cli (:use :cl))
(in-package :retrieval-audit-cli)

(defun opt (name default)
  (let ((arg (find-if (lambda (a) (and (> (length a) (length name))
                                      (string= name a :end2 (length name))
                                      (char= (char a (length name)) #\=)))
                      (cdr sb-ext:*posix-argv*))))
    (if arg (subseq arg (1+ (length name))) default)))

(defun integer-opt (name default &optional (minimum 0))
  (let ((value (parse-integer (opt name default))))
    (unless (>= value minimum) (error "~a must be >= ~d" name minimum)) value))

(when (equal (opt "predicate-paraphrase" "0") "1")
  (error "Phase 2 is isolated; use experimental-phase2-audit.lisp explicitly."))

(defun flag (name default)
  (let ((s (opt name default)))
    (unless (member s '("0" "1") :test #'string=) (error "~a must be 0 or 1" name))
    (string= s "1")))

(defun obj (&rest fields)
  (cons :obj (loop for (k v) on fields by #'cddr collect (cons k v))))
(defun fraction (a b) (/ a (max 1 b) 1d0))
(defun json-bool (x) (if x :true :false))

(defun tokenizer-stats (bpe strings)
  (let ((histogram (make-hash-table)) (tokens 0) (errors 0))
    (dolist (s strings)
      (let* ((text (format nil " ~a" s)) (ids (asem-bpe:bpe-encode bpe text)))
        (incf tokens (length ids)) (incf (gethash (length ids) histogram 0))
        (unless (string= text (asem-bpe:bpe-decode bpe ids)) (incf errors))))
    (obj "strings" (length strings) "answer_tokens" tokens
         "mean_answer_tokens" (fraction tokens (length strings))
         "roundtrip_errors" errors
         "answer_token_length_histogram"
         (coerce (loop for k in (sort (loop for k being the hash-keys of histogram collect k) #'<)
                       collect (obj "tokens" k "strings" (gethash k histogram))) 'vector))))

(defun question-count (rows)
  "One contiguous target-row run per answer, separated by its terminator and next question."
  (let ((previous nil) (count 0))
    (dolist (r rows count)
      (when (or (null previous) (/= r (1+ previous))) (incf count))
      (setf previous r))))

(defun audit-generator (bpe n levels seed ctx options copy-mix)
  (let* ((original (symbol-function 'asem-retrieval:make-fact-set))
         (attempts (make-hash-table)) (accepted (make-hash-table))
         (targets (make-hash-table)) (questions (make-hash-table))
         (passages (make-hash-table :test 'equal))
         (digest (asem-sha256:make-sha256)) (actual 0) (total-targets 0) (total-questions 0))
    ;; Observe the existing generator rather than independently reimplementing it.
    ;; Refuse ambiguous passage-to-kind mappings instead of assigning a guess.
    (setf (symbol-function 'asem-retrieval:make-fact-set)
          (lambda (&rest args)
            (multiple-value-bind (passage qa kind candidates padder) (apply original args)
              (let ((previous (gethash passage passages)))
                (when (and previous (not (eq kind previous)))
                  (error "Identical passage produced different kinds: ~a and ~a" previous kind)))
              (setf (gethash passage passages) kind) (incf (gethash kind attempts 0))
              (values passage qa kind candidates padder))))
    (unwind-protect
         (let ((seqs (asem-retrieval:train-sequences n levels seed bpe ctx :options options)))
           (setf actual (length seqs))
           (dolist (seq seqs)
             (destructuring-bind (ids rows tg) seq
               (let* ((text (asem-bpe:bpe-decode bpe ids))
                      (passage (subseq text (length "Text: ") (position #\Newline text)))
                      (kind (or (gethash passage passages) (error "Cannot recover generated passage kind")))
                      (nq (question-count rows)))
                 (unless (and (<= (length ids) ctx)
                              (= (length rows) (length tg))
                              (every (lambda (r token) (= (nth (1+ r) ids) token)) rows tg))
                   (error "Generated sequence violates context or target alignment"))
                 (incf (gethash kind accepted 0)) (incf (gethash kind targets 0) (length tg))
                 (incf (gethash kind questions 0) nq)
                 (incf total-targets (length tg)) (incf total-questions nq)
                 (asem-sha256:sha256-update digest
                   (asem-sha256:string-octets
                     (format nil "~{~d~^,~}|~{~d~^,~}|~{~d~^,~}~%" ids rows tg)))))))
      (setf (symbol-function 'asem-retrieval:make-fact-set) original))
    (obj "requested_sequences" n "accepted_sequences" actual
         "attempted_fact_sets" (loop for x being the hash-values of attempts sum x)
         "answer_tokens" total-targets "questions" total-questions
         "sequence_digest_serialization" "UTF-8 decimal ids joined by comma, |, rows, |, targets, newline; generated order"
         "sequence_sha256" (asem-sha256:sha256-final-hex digest)
         "copy_replay_sequences" copy-mix
         "copy_replay_sequence_share" (fraction copy-mix (+ actual copy-mix))
         "copy_replay_loss_weight_share" (fraction copy-mix (+ total-targets copy-mix))
         "loss_weight_interpretation" "Existing :multi retrieval gives each answer token weight 1; :copy averages each sequence to weight 1. Shares describe objective coefficients before model-dependent gradients and clipping."
         "by_kind"
         (coerce (loop for kind in '(:attr :num :rel :cmp :tmp)
                       collect (obj "kind" (string-downcase (symbol-name kind))
                                    "attempted" (gethash kind attempts 0)
                                    "accepted" (gethash kind accepted 0)
                                    "questions" (gethash kind questions 0)
                                    "answer_tokens" (gethash kind targets 0)
                                    "retrieval_sequence_share" (fraction (gethash kind accepted 0) actual)
                                    "retrieval_token_weight_share" (fraction (gethash kind targets 0) total-targets)
                                    "including_copy_loss_weight_share" (fraction (gethash kind targets 0) (+ total-targets copy-mix)))) 'vector))))

(let* ((data (or (opt "data" nil) (error "data=DIR is required")))
       (manifest-path (format nil "~a/manifest.json" data))
       (manifest (asem-data:load-manifest data))
       (tokenizer-path (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file")))
       (bpe (asem-bpe:load-bpe tokenizer-path))
       (n (integer-opt "train-n" "75000" 1)) (seed (integer-opt "seed" "7"))
       (ctx (integer-opt "ctx" "128" 1)) (copy-mix (integer-opt "copy-mix" "15000"))
       (levels (mapcar #'parse-integer (asem-util:split-on (opt "levels" "1,2,3") #\,)))
       (broad (flag "broad" "1")) (augment (flag "augment" "1"))
       (options (flag "options" "1")) (reverse (flag "reverse" "1")) (reasoning (flag "reasoning" "1"))
       (asem-retrieval:*broad-train* broad) (asem-retrieval:*augment* augment)
       (asem-retrieval:*reverse* reverse) (asem-retrieval:*reasoning* reasoning)
       (asem-retrieval:*vocab-words* (and broad (asem-retrieval:vocab-words-of bpe)))
       (out (opt "out" nil)))
  (unless (and levels (every (lambda (l) (member l asem-retrieval:+levels+)) levels))
    (error "levels must be a comma-separated nonempty list of 1,2,3,4"))
  (let ((result
          (obj "format" "asem-retrieval-audit-1"
               "recipe" (obj "train_n" n "seed" seed "ctx" ctx "levels" (coerce levels 'vector)
                             "broad" (json-bool broad) "augment" (json-bool augment) "options" (json-bool options)
                             "reverse" (json-bool reverse) "reasoning" (json-bool reasoning) "copy_mix" copy-mix)
               "provenance" (obj "tokenizer_sha256" (asem-sha256:sha256-file-hex tokenizer-path)
                                 "manifest_sha256" (asem-sha256:sha256-file-hex manifest-path)
                                 "sbcl_version" (lisp-implementation-version)
                                 "source_sha256"
                                 (cons :obj
                                       (loop for relative in '("bin/retrieval-audit.lisp" "src/retrieval.lisp" "src/qagen.lisp" "src/bpe.lisp" "src/sft.lisp")
                                             collect (cons relative
                                                           (asem-sha256:sha256-file-hex
                                                             (merge-pathnames (format nil "../~a" relative) cl-user::*audit-script*))))))
               "generator" (audit-generator bpe n levels seed ctx options copy-mix)
               "tokenizer_name_pools"
               (obj "dev_real" (tokenizer-stats bpe (asem-qagen:split-pool asem-qagen:*names* :dev))
                    "new_probe" (tokenizer-stats bpe (asem-retrieval::new-pool :names))
                    "training" (tokenizer-stats bpe (asem-retrieval::pool :train '*names*))))))
    (if out
        (progn
          (ensure-directories-exist out)
          (with-open-file (stream out :direction :output :if-exists (if (flag "overwrite" "0") :supersede :error)
                                      :if-does-not-exist :create :external-format :utf-8)
            (asem-json:write-json result stream) (terpri stream)))
        (progn (asem-json:write-json result *standard-output*) (terpri)))))
