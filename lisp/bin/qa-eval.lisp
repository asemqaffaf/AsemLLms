;;; Evaluate a checkpoint on a QA file: raw letter accuracy with a 95% Wilson interval, per category / subtype / question frame,
;;; and the controls (no question, mismatched question, no passage, counterfactual question on the same passage, options reversed).
;;;   TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script lisp/bin/qa-eval.lisp data=data/fineweb/stage1 ckpt=CKPT items=data/qa/v1/dev.jsonl
;;;        [workers=8] [precision=32|64] [mode=letter|text] [out=FILE.json]
;;; The locked test set (lisp/eval/qa/test.jsonl) is refused unless locked-test=1 is given; its SHA-256 must then match
;;; lisp/eval/qa/test.lock, and every such run is appended to docs/results/stage2/locked-test-runs.jsonl.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*here* here))
(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(defun file-text (p) (with-open-file (in p) (string-trim '(#\Space #\Newline) (read-line in))))
(let* ((p32 (not (equal (opt-arg "precision" "32") "64"))) (here cl-user::*here*))
  (dolist (f (if p32 '("tf32" "train32" "pretrain32" "sft32") '("tf" "train" "pretrain" "sft"))) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (let* ((pkg (if p32 "ASEM-SFT32" "ASEM-SFT")) (ppkg (if p32 "ASEM-PRETRAIN32" "ASEM-PRETRAIN"))
         (fn (lambda (name &optional (p pkg)) (symbol-function (find-symbol name p))))
         (data (or (opt-arg "data" nil) (error "data=DIR is required"))) (ckpt (or (opt-arg "ckpt" nil) (error "ckpt=PATH is required")))
         (items-file (or (opt-arg "items" nil) (error "items=FILE is required")))
         (sha (asem-sha256:sha256-file-hex items-file))
         (lock-file (namestring (merge-pathnames "../eval/qa/test.lock" here)))
         (locked-sha (and (probe-file lock-file) (file-text lock-file)))
         (is-test (or (equal sha locked-sha) (search "test" (file-namestring items-file)))))
    (when is-test
      (unless (equal (opt-arg "locked-test" "0") "1")
        (format *error-output* "~a looks like the locked test set; pass locked-test=1 only for the final, pre-selected checkpoint.~%" items-file)
        (sb-ext:exit :code 2))
      (unless (equal sha locked-sha) (format *error-output* "test file SHA-256 ~a does not match the lock ~a~%" sha locked-sha) (sb-ext:exit :code 3)))
    (let* ((manifest (asem-data:load-manifest data))
           (bpe (asem-bpe:load-bpe (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
           (model (funcall (funcall fn "LOAD-CHECKPOINT" ppkg) ckpt))
           (items (asem-qagen:read-items items-file))
           (t0 (get-internal-real-time))
           (res (funcall (funcall fn "QA-EVALUATE-CONDITIONS") model bpe items :workers (parse-integer (opt-arg "workers" "8")) :mode (intern (string-upcase (opt-arg "mode" "letter")) :keyword)))
           (obj (list* :obj (cons "checkpoint" ckpt) (cons "checkpoint_sha256" (asem-sha256:sha256-file-hex ckpt))
                       (cons "items" items-file) (cons "mode" (opt-arg "mode" "letter")) (cons "items_sha256" sha) (cons "locked_test" (if is-test :true :false))
                       (cons "seconds" (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0))
                       (cdr res)))
           (out (opt-arg "out" nil)))
      (if out
          (with-open-file (o out :direction :output :if-exists :supersede) (asem-json:write-json obj o) (terpri o))
          (progn (asem-json:write-json obj *standard-output*) (terpri)))
      (flet ((g (&rest ks) (let ((o obj)) (dolist (k ks o) (setf o (asem-json:json-get o k))))))
        (format *error-output* "~a: real ~d/~d = ~,3f [~,3f, ~,3f] | no question ~,3f | no passage ~,3f | mismatched q ~,3f | counterfactual q ~,3f | reversed options ~,3f~%"
                (file-namestring items-file) (g "real_prompt" "correct") (g "real_prompt" "n") (g "real_prompt" "accuracy")
                (g "real_prompt" "ci95_low") (g "real_prompt" "ci95_high") (g "no_question" "accuracy") (g "no_passage" "accuracy")
                (g "mismatched_question_from_another_item" "accuracy") (g "counterfactual_question_same_passage" "accuracy")
                (g "options_in_reverse_order" "accuracy")))
      (when is-test
        (let ((log (namestring (merge-pathnames "../../docs/results/stage2/locked-test-runs.jsonl" here))))
          (ensure-directories-exist log)
          (with-open-file (o log :direction :output :if-exists :append :if-does-not-exist :create)
            (format o "{\"checkpoint\":~a,\"checkpoint_sha256\":~a,\"correct\":~d,\"n\":~d,\"accuracy\":~f,\"universal_time\":~d}~%"
                    (asem-util:json-string ckpt) (asem-util:json-string (asem-sha256:sha256-file-hex ckpt))
                    (g "real_prompt" "correct") (g "real_prompt" "n") (g "real_prompt" "accuracy") (get-universal-time))))))))
