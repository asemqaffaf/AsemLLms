;;; Entity-retrieval curriculum: train on dense retrieval sequences, select on a held-out probe, evaluate on the full probe suite.
;;;   train: TLM_BLAS=1 sbcl --dynamic-space-size 10GB --script lisp/bin/retrieval.lisp train data=DIR init=CKPT out=DIR levels=1[,2,...]
;;;            [train-n=60000] [steps=1500] [batch=32] [workers=8] [lr=5e-4] [warmup=50] [eval-every=100] [target-acc=0.9] [seed=1]
;;;            [probe-n=300]
;;;   eval:  TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script lisp/bin/retrieval.lisp eval data=DIR ckpt=CKPT [n=200] [out=FILE.json]
;;; Training uses only generated names/towns, training attribute pools and training wording. The selection probe (dev during
;;; training) uses newly generated names (never in training) at the highest level being trained, training wording, seed 7001.
;;; The evaluation suite uses fresh seeds: levels 1-4 x entities (QA dev pools / new names) x wording (training / unseen QA-dev
;;; frames), plus controls at every level: question removed, passage removed, counterfactual question (another fact of the same
;;; passage: the model must give that fact's answer). QA test pools and wording are never used.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (dolist (f '("tf32" "train32" "pretrain32" "sft32")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(defpackage :retrieval-cli (:use :cl))
(in-package :retrieval-cli)
(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(defun num (name default) (parse-integer (opt-arg name default)))
(defun flt (name default) (float (let ((*read-default-float-format* 'double-float)) (read-from-string (opt-arg name default))) 1f0))
(defun jobj (&rest plist) (cons :obj (loop for (k v) on plist by #'cddr collect (cons k v))))

(defun bpe-of (data)
  (let ((m (asem-data:load-manifest data)))
    (asem-bpe:load-bpe (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get m "tokenizer") "file")))))

(defun probe-exs (items bpe ctx &key (variant :real))
  "Examples for a probe list. VARIANT :real, :no-question, :no-passage, :counterfactual (only items that have one)."
  (coerce
   (loop for it in items
         for prompt = (ecase variant
                        (:real (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :question)))
                        (:no-question (asem-retrieval:retrieval-prompt (getf it :passage) nil :question-p nil))
                        (:no-passage (asem-retrieval:retrieval-prompt nil (getf it :question) :passage-p nil))
                        (:counterfactual (and (getf it :cf-question) (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :cf-question)))))
         for answer = (if (eq variant :counterfactual) (getf it :cf-answer) (getf it :answer))
         when prompt
           collect (asem-sft32:generation-ex (getf it :id) (format nil "L~d" (getf it :level)) (getf it :kind) prompt answer bpe ctx))
   'vector))

(defun accuracy (model exs)
  (let ((preds (asem-sft32:qa-predict model exs '(316 367 338 443) :workers (num "workers" "8"))))
    (let ((k (count 0 preds :key #'first)) (n (length exs)))
      (destructuring-bind (lo hi) (asem-sft32:wilson k n) (jobj "correct" k "n" n "accuracy" (/ k (max 1 n) 1d0) "ci95_low" lo "ci95_high" hi)))))

(let ((cmd (second sb-ext:*posix-argv*)) (data (or (opt-arg "data" nil) (error "data=DIR is required"))))
  (cond
    ((equal cmd "train")
     (let* ((bpe (bpe-of data)) (init (or (opt-arg "init" nil) (error "init=CKPT is required")))
            (model (asem-pretrain32:load-checkpoint init)) (ctx (asem-tf32:config-ctx (asem-tf32:model-config model)))
            (levels (mapcar #'parse-integer (asem-util:split-on (opt-arg "levels" "1") #\,)))
            (seqs (asem-retrieval:train-sequences (num "train-n" "60000") levels (num "seed" "1") bpe ctx))
            (train (coerce (loop for (ids rows targets) in seqs for i from 0 collect (asem-sft32:multi-ex (format nil "r~d" i) ids rows targets)) 'vector))
            (probe (asem-retrieval:probe-items (reduce #'max levels) :new :seen (num "probe-n" "300") 7001))
            (dev (probe-exs probe bpe ctx)))
       (format t "~d training sequences (~,1f answers each), ~d probe items at level ~d~%" (length train)
               (/ (reduce #'+ (map 'list (lambda (e) (length (asem-sft32::ex-tgt e))) train)) (length train) 1.0) (length dev) (reduce #'max levels))
       (asem-sft32:sft (or (opt-arg "out" nil) (error "out=DIR is required")) train dev '(316 367 338 443) :init-model model
                       :steps (num "steps" "1500") :batch (num "batch" "32") :workers (num "workers" "8") :lr (flt "lr" "5e-4")
                       :warmup (num "warmup" "50") :eval-every (num "eval-every" "100") :log-every (num "log-every" "20")
                       :train-probe 300 :seed (num "seed" "1")
                       :target-acc (let ((s (opt-arg "target-acc" nil))) (and s (flt "target-acc" s)))
                       :header (list :init init :init-sha256 (asem-sha256:sha256-file-hex init) :levels (opt-arg "levels" "1")
                                     :train-n (length train) :argv (format nil "~{~a~^ ~}" (cdr sb-ext:*posix-argv*))))))
    ((equal cmd "eval")
     (let* ((bpe (bpe-of data)) (ckpt (or (opt-arg "ckpt" nil) (error "ckpt=CKPT is required")))
            (model (asem-pretrain32:load-checkpoint ckpt)) (ctx (asem-tf32:config-ctx (asem-tf32:model-config model)))
            (n (num "n" "200")) (results nil))
       (dolist (level asem-retrieval:+levels+)
         (dolist (ent '(:dev-real :new))
           (dolist (w '(:seen :unseen))
             (let* ((items (asem-retrieval:probe-items level ent w n (+ 9000 (* 100 level) (if (eq ent :new) 10 0) (if (eq w :unseen) 1 0))))
                    (key (format nil "L~d/~(~a/~a~)" level ent w))
                    (row (list "real" (accuracy model (probe-exs items bpe ctx)))))
               (when (and (eq ent :dev-real) (eq w :unseen))
                 (setf row (append row (list "no_question" (accuracy model (probe-exs items bpe ctx :variant :no-question))
                                             "no_passage" (accuracy model (probe-exs items bpe ctx :variant :no-passage))))
                       row (if (> level 1) (append row (list "counterfactual_question" (accuracy model (probe-exs items bpe ctx :variant :counterfactual)))) row)))
               (push (cons key (apply #'jobj row)) results)
               (format *error-output* "~a: ~,3f~%" key (asem-json:json-get (second row) "accuracy"))))))
       (let ((obj (list* :obj (cons "checkpoint" ckpt) (cons "checkpoint_sha256" (asem-sha256:sha256-file-hex ckpt)) (nreverse results)))
             (out (opt-arg "out" nil)))
         (if out (with-open-file (o out :direction :output :if-exists :supersede) (asem-json:write-json obj o) (terpri o))
             (progn (asem-json:write-json obj *standard-output*) (terpri))))))
    (t (format *error-output* "usage: retrieval.lisp train|eval ...~%") (sb-ext:exit :code 2))))
