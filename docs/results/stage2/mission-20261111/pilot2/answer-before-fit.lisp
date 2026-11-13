;;;; Pilot 2 secondary check: does a checkpoint answer four-person answer-before-entity questions in TRAINING wording with
;;;; TRAINING-pool entities? Inference only, teacher-forced, full-vocabulary argmax (K = 8192 per token, chance ~0%).
;;;;  (a) fresh: 400 new fact sets from the pilot's own generator on a random stream never used for training;
;;;;  (b) trained: the first 400 replaced sequences of data-treatment.sexp (most were seen once in training by the treatment arm).
;;;;   TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 6GB --script docs/results/stage2/mission-20261111/pilot2/answer-before-fit.lisp ckpt=PATH out=NEW.json
(let ((src (merge-pathnames "../../../../../lisp/src/" (directory-namestring *load-truename*))))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis"))
    (load (merge-pathnames (format nil "~a.lisp" f) src))))
;; Take the generator definitions verbatim from the pilot driver (definitions only; its top-level run form is skipped).
(with-open-file (in "lisp/experimental/answer-before/pilot.lisp")
  (loop for form = (read in nil :eof) until (eq form :eof)
        when (and (consp form) (member (car form) '(defpackage in-package defparameter defun))) do (eval form)))
(in-package :answer-before)
(defun score (model ids rows targets cache)
  (let* ((cfg (asem-tf32:model-config model)) (v (asem-tf32:config-vocab cfg))
         (acts (asem-sft32::acts-for cache cfg (length ids) t)))
    (asem-tf32:forward-rows model (asem-sft32::padded-x (asem-sft32::ivec-of ids) (asem-tf32::acts-tlen acts)) acts (asem-sft32::ivec-of rows))
    (loop for token in targets for k from 0
          collect (let ((best 0) (offset (* k v)))
                    (dotimes (j v) (when (> (aref (asem-tf32::acts-logits acts) (+ offset j)) (aref (asem-tf32::acts-logits acts) (+ offset best))) (setf best j)))
                    (= best token)))))
(defun question-groups (rows)
  "Lengths of runs of consecutive answer rows = answer token counts per question."
  (let ((out nil) (n 0) (prev nil))
    (dolist (r rows) (if (and prev (= r (1+ prev))) (incf n) (progn (when (plusp n) (push n out)) (setf n 1))) (setf prev r))
    (when (plusp n) (push n out)) (nreverse out)))
(defun per-question (hits rows)
  (let ((out nil)) (dolist (n (question-groups rows)) (push (list (first hits) (every #'identity (subseq hits 0 n))) out) (setf hits (nthcdr n hits))) (nreverse out)))
(let* ((ckpt (or (opt "ckpt") (error "ckpt= required"))) (out (or (opt "out") (error "out= required")))
       (hash (asem-sha256:sha256-file-hex ckpt)) (bpe (asem-bpe:load-bpe "data/fineweb/stage1/tokenizer.bpe"))
       (model (asem-pretrain32:load-checkpoint ckpt)) (cache (make-hash-table)) (rows-out nil))
  (when (probe-file out) (error "Refusing to overwrite ~a" out))
  (set-recipe bpe)
  ;; (a) fresh
  (dotimes (i 400)
    (let ((rng (rng-for i 0 2)))
      (multiple-value-bind (passage qa cands padder people) (answer-before-fact-set rng)
        (destructuring-bind (ids rows targets texts nq) (build passage qa cands padder rng bpe 128)
          (let ((meta (cdr (reverse texts))))
            (when (and rows (= (length meta) (* 6 nq)))
              (loop for (first-hit all-hit) in (per-question (score model ids rows targets cache) rows) for j from 0
                    do (let ((answer (nth (+ 1 (* 6 j)) meta)))
                         (push (obj "set" "fresh" "id" (format nil "fresh-~d/q~d" i j) "ordinal" j
                                    "task" (if (member answer people :test #'string=) "reverse" "extract")
                                    "question" (nth (* 6 j) meta) "answer" answer
                                    "first_token_correct" (if first-hit :true :false) "prefix_exact" (if all-hit :true :false))
                               rows-out)))))))))
  ;; (b) trained
  (let ((c (read-sexp (format nil "~a/data-control.sexp" +run+))) (tr (read-sexp (format nil "~a/data-treatment.sexp" +run+))) (n 0))
    (loop for a in c for b in tr for i from 0 while (< n 400)
          unless (equal (first a) (first b))
            do (incf n)
               (destructuring-bind (ids rows targets) b
                 (loop for (first-hit all-hit) in (per-question (score model ids rows targets cache) rows) for j from 0
                       do (push (obj "set" "trained" "id" (format nil "r~d/q~d" i j) "ordinal" j "task" "answer-before"
                                     "first_token_correct" (if first-hit :true :false) "prefix_exact" (if all-hit :true :false))
                                rows-out)))))
  (unless (equal hash (asem-sha256:sha256-file-hex ckpt)) (error "Checkpoint changed"))
  (write-json out (obj "schema" "asem-answer-before-fit-1" "checkpoint" ckpt "checkpoint_sha256" hash "training_updates" 0 "items" (coerce (nreverse rows-out) 'vector)))
  (format t "Saved ~a~%" out))
