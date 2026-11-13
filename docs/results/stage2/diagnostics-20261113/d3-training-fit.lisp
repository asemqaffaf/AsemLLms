;;;; D3: per-question fit of a checkpoint on the 256 saved traced R7 training sequences (training entities, training wording).
;;;; Inference only: teacher-forced argmax over the full vocabulary at every answer row, dense context exactly as trained.
;;;; No option restriction (K = 8192 at each token; chance is ~0). Reads only audit-training.sexp; never a locked set.
;;;;   TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script docs/results/stage2/diagnostics-20261113/d3-training-fit.lisp \
;;;;       out=NEW.json [ckpt=PATH]
(let ((src (merge-pathnames "../../../../lisp/src/" (directory-namestring *load-truename*))))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "tf32" "train32" "pretrain32" "sft32" "qa-analysis"))
    (load (merge-pathnames (format nil "~a.lisp" f) src))))
(defun opt (key)
  (let ((p (concatenate 'string key "=")))
    (let ((x (find-if (lambda (x) (and (>= (length x) (length p)) (string= p x :end2 (length p)))) (cdr sb-ext:*posix-argv*)))) (and x (subseq x (length p))))))
(defun obj (&rest x) (apply #'asem-qa-analysis::obj x))
(defun score (model ids targets cache)
  "Same computation as lisp/bin/mission-supervision-audit.lisp SCORE."
  (let* ((n (- (length ids) (length targets))) (cfg (asem-tf32:model-config model)) (v (asem-tf32:config-vocab cfg))
         (acts (asem-sft32::acts-for cache cfg (length ids) t)) (rows (asem-sft32::ivec-of (loop for k below (length targets) collect (+ n k -1)))))
    (asem-tf32:forward-rows model (asem-sft32::padded-x (asem-sft32::ivec-of ids) (asem-tf32::acts-tlen acts)) acts rows)
    (let ((hits nil) (nll nil))
      (loop for token in targets for k from 0 do
        (let ((best 0) (offset (* k v)))
          (dotimes (j v) (when (> (aref (asem-tf32::acts-logits acts) (+ offset j)) (aref (asem-tf32::acts-logits acts) (+ offset best))) (setf best j)))
          (push (= best token) hits)
          (push (- (log (max 1d-300 (coerce (aref (asem-tf32::acts-probs acts) (+ offset token)) 'double-float)))) nll)))
      (list :first-hit (car (last hits)) :all-hit (every #'identity hits) :first-nll (car (last nll)) :sum-nll (reduce #'+ nll)))))
(let* ((out (or (opt "out") (error "out= required")))
       (trace "docs/results/stage2/mission-20261111/audit-training.sexp")
       (ckpt (or (opt "ckpt") "data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt")) (hash (asem-sha256:sha256-file-hex ckpt))
       (model (asem-pretrain32:load-checkpoint ckpt)) (cache (make-hash-table))
       (records (with-open-file (s trace) (let ((*read-eval* nil)) (read s)))) (rows nil))
  (when (probe-file out) (error "Refusing to overwrite ~a" out))
  (loop for r in records for i from 0 do
    (loop for q in (getf r :questions) for j from 0 do
      (let* ((targets (getf q :answer-tokens)) (prefix (subseq (getf r :ids) 0 (getf q :answer-start)))
             (s (score model (append prefix targets) targets cache)))
        (push (obj "id" (format nil "train-~d/q~d" i j) "kind" (string-downcase (symbol-name (getf r :kind))) "ordinal" j
                   "passage" (coerce (getf q :passage) 'simple-string) "question" (coerce (getf q :question) 'simple-string)
                   "answer" (coerce (getf q :answer) 'simple-string)
                   "options" (coerce (mapcar (lambda (o) (coerce o 'simple-string)) (getf q :options)) 'vector)
                   "answer_tokens" (length targets) "first_token_correct" (if (getf s :first-hit) :true :false)
                   "prefix_exact" (if (getf s :all-hit) :true :false) "first_nll" (getf s :first-nll) "sum_nll" (getf s :sum-nll))
              rows))))
  (unless (equal hash (asem-sha256:sha256-file-hex ckpt)) (error "Checkpoint changed"))
  (with-open-file (s out :direction :output :if-exists :error :if-does-not-exist :create :external-format :utf-8)
    (asem-json:write-json (obj "schema" "asem-d3-training-fit-1" "checkpoint" ckpt "checkpoint_sha256" hash "trace" trace
                               "trace_sha256" (asem-sha256:sha256-file-hex trace) "training_updates" 0
                               "items" (coerce (nreverse rows) 'vector)) s)
    (terpri s))
  (format t "Saved ~a~%" out))
