;;; Is the model using the question? Deterministic diagnostics for a checkpoint on the multiple-choice set; one JSON object on stdout.
;;;   sbcl --dynamic-space-size 4GB --script lisp/bin/diagnose.lisp data=DIR ckpt=OUT/best.ckpt [eval=lisp/eval/asem_eval.jsonl]
;;;        [precision=32|64] [mismatches=3] [text=1]
;;; Sections: accuracy (with the question, without it, with no input at all, with a mismatched question), length_bias and
;;; frequency_bias (does the score follow answer length or token frequency?), margin (correct minus best wrong), question_sensitivity
;;; (how much the correct answer's log-probability changes when the question changes and the options stay fixed: a model that
;;; conditions on the question gives a clearly positive mean and z), by_category, and conditions_on_the_question (three yes/no flags).
;;; text=1 also prints a short summary to stderr. Training-token frequencies come from DIR's train split.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*here* here))
(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(let* ((p32 (not (equal (opt-arg "precision" "32") "64"))) (here cl-user::*here*))
  (dolist (f (if p32 '("tf32" "train32" "pretrain32") '("tf" "train" "pretrain"))) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (let* ((pkg (if p32 "ASEM-PRETRAIN32" "ASEM-PRETRAIN")) (fn (lambda (name) (symbol-function (find-symbol name pkg))))
         (data (or (opt-arg "data" nil) (error "data=DIR is required"))) (ckpt (or (opt-arg "ckpt" nil) (error "ckpt=PATH is required")))
         (eval-file (opt-arg "eval" (namestring (merge-pathnames "../eval/asem_eval.jsonl" here))))
         (manifest (asem-data:load-manifest data))
         (bpe (asem-bpe:load-bpe (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
         (model (funcall (funcall fn "LOAD-CHECKPOINT") ckpt))
         (vocab (asem-json:json-get (asem-json:json-get manifest "tokenizer") "vocab_size"))
         (logfreq (funcall (funcall fn "TOKEN-LOGFREQ") (funcall (funcall fn "LOAD-SPLIT-TOKENS") data "train") vocab))
         (items (funcall (funcall fn "READ-EVAL-SET") eval-file))
         (result (funcall (funcall fn "DIAGNOSE-MODEL") model bpe items logfreq :mismatches (parse-integer (opt-arg "mismatches" "3")))))
    (asem-json:write-json (list* :obj (cons "checkpoint" ckpt) (cons "eval_file" eval-file) (cdr result)) *standard-output*)
    (terpri)
    (when (equal (opt-arg "text" "0") "1")
      (flet ((g (&rest keys) (let ((o result)) (dolist (k keys o) (setf o (asem-json:json-get o k))))))
        (format *error-output* "accuracy: with question ~,3f | no question ~,3f | no input ~,3f | mismatched question ~,3f (chance 0.25, 95% band to ~,3f)~%"
                (g "accuracy" "with_question_mean_logprob") (g "accuracy" "no_question_context_kept") (g "accuracy" "bare_prior_no_input")
                (g "accuracy" "mismatched_question") (g "chance_95_upper"))
        (format *error-output* "question sensitivity: mean change in the correct answer's log-prob ~,3f nats (z ~,2f), top option unchanged by another question ~,2f~%"
                (g "question_sensitivity" "pmi_sum_mean_nats") (g "question_sensitivity" "pmi_sum_z") (g "question_sensitivity" "top_option_unchanged_by_other_question"))
        (format *error-output* "margin: mean ~,3f, positive in ~,2f of items | length corr ~,2f | frequency corr ~,2f~%"
                (g "margin" "mean_logprob_mean") (g "margin" "mean_logprob_fraction_positive")
                (g "length_bias" "corr_length_vs_mean_logprob") (g "frequency_bias" "corr_option_token_frequency_vs_score"))))))
