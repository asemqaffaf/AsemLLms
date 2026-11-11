;;; Entity-retrieval curriculum: train on dense retrieval sequences, select on a held-out probe, evaluate on the full probe suite.
;;;   train: TLM_BLAS=1 sbcl --dynamic-space-size 10GB --script lisp/bin/retrieval.lisp train data=DIR init=CKPT out=DIR levels=1[,2,...]
;;;            [train-n=60000] [steps=1500] [batch=32] [workers=8] [lr=5e-4] [warmup=50] [eval-every=100] [target-acc=0.9] [seed=1]
;;;            [probe-n=300] [broad=0|1] [augment=0|1] [options=0|1] [reverse=0|1] [reasoning=0|1] [copy-mix=0]
;;;   eval:  TLM_BLAS=1 sbcl --dynamic-space-size 8GB --script lisp/bin/retrieval.lisp eval data=DIR ckpt=CKPT [n=200] [options=0|1] [out=FILE.json]
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

(defun fits-fn (bpe ctx)
  (lambda (prompt answer) (<= (+ (length (asem-bpe:bpe-encode bpe prompt)) (length (asem-bpe:bpe-encode bpe (format nil " ~a" answer)))) ctx)))

(defun probe-exs (items bpe ctx &key (variant :real))
  "Examples for a probe list. VARIANT :real, :no-question, :no-passage, :counterfactual (only items that have one)."
  (coerce
   (loop for it in items
         for prompt = (ecase variant
                        (:real (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :question) :options (getf it :options)))
                        (:no-question (asem-retrieval:retrieval-prompt (getf it :passage) nil :question-p nil :options (getf it :options)))
                        (:no-passage (asem-retrieval:retrieval-prompt nil (getf it :question) :passage-p nil :options (getf it :options)))
                        (:counterfactual (and (getf it :cf-question) (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :cf-question) :options (getf it :options)))))
         for answer = (if (eq variant :counterfactual) (getf it :cf-answer) (getf it :answer))
         when prompt
           collect (asem-sft32:generation-ex (getf it :id) (format nil "L~d" (getf it :level)) (getf it :kind) prompt answer bpe ctx))
   'vector))

(defun choice-accuracies (model items bpe ctx)
  "Option-likelihood and A-D letter accuracy on option-mode probe ITEMS (real question)."
  (let ((txt nil) (let nil))
    (dolist (it items)
      (multiple-value-bind (a b) (asem-sft32:choice-exs (getf it :id) "c" "" (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :question) :options (getf it :options))
                                                        (getf it :options) (getf it :answer) bpe ctx)
        (push a txt) (push b let)))
    (flet ((acc (exs) (let* ((v (coerce (nreverse exs) 'vector)) (s (asem-sft32:qa-summary v (asem-sft32:qa-predict model v '(316 367 338 443) :workers (num "workers" "8")))))
                        (destructuring-bind (lo hi) (asem-sft32:wilson (getf s :correct) (getf s :n))
                          (jobj "correct" (getf s :correct) "n" (getf s :n) "accuracy" (float (getf s :accuracy) 1d0) "ci95_low" lo "ci95_high" hi
                                "predicted_letter_counts" (format nil "~a" (getf s :pred-dist)))))))
      (let* ((v (coerce (reverse txt) 'vector)) (preds (asem-sft32:qa-predict model v '(316 367 338 443) :workers (num "workers" "8")))
             (mean-hits 0) (short-picks 0) (short-n 0))
        (loop for e across v for (nil nil scores) in preds
              do (let* ((lens (mapcar #'length (asem-sft32::ex-opts e))) (means (mapcar (lambda (s l) (/ s (max 1 l))) scores lens))
                        (pm (position (reduce #'max means) means)) (ps (position (reduce #'max scores) scores)) (mn (reduce #'min lens)))
                   (when (= pm (asem-sft32:ex-answer e)) (incf mean-hits))
                   (when (= 1 (count mn lens)) (incf short-n) (when (= ps (position mn lens)) (incf short-picks)))))
        (list "option_likelihood" (acc txt) "option_likelihood_scoring" "argmax of summed token log-probability of ' <option>' after the prompt"
              "option_likelihood_mean_per_token" (jobj "correct" mean-hits "n" (length v) "accuracy" (/ mean-hits (max 1 (length v)) 1d0))
              "summed_score_picks_unique_shortest_option" (jobj "items" short-n "fraction" (/ short-picks (max 1 short-n) 1d0))
              "letter_A_to_D" (acc let))))))

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
            (seqs (progn (when (equal (opt-arg "augment" "0") "1") (setf asem-retrieval:*augment* t))
                         (when (equal (opt-arg "reverse" "0") "1") (setf asem-retrieval:*reverse* t))
                         (when (equal (opt-arg "reasoning" "0") "1") (setf asem-retrieval:*reasoning* t))
                         (when (equal (opt-arg "broad" "0") "1")
                           (setf asem-retrieval:*broad-train* t asem-retrieval:*vocab-words* (asem-retrieval:vocab-words-of bpe)))
                         (asem-retrieval:train-sequences (num "train-n" "60000") levels (num "seed" "1") bpe ctx :options (equal (opt-arg "options" "0") "1"))))
            (train (concatenate 'vector
                               (loop for (ids rows targets) in seqs for i from 0 collect (asem-sft32:multi-ex (format nil "r~d" i) ids rows targets))
                               (if (plusp (num "copy-mix" "0")) (asem-sft32:make-copy-examples (num "copy-mix" "0") 77 :lo 1 :hi 8192 :min-len 10 :max-len 40) #())))
            (probe (let ((asem-retrieval:*augment* nil))          ; probes are never augmented
                     (asem-retrieval:probe-items (reduce #'max levels) :new :seen (num "probe-n" "300") 7001 :fits (fits-fn bpe ctx) :options (equal (opt-arg "options" "0") "1"))))
            (dev (probe-exs probe bpe ctx)))
       (format t "~d training sequences (~,1f answers each), ~d probe items at level ~d~%" (length train)
               (/ (reduce #'+ (map 'list (lambda (e) (length (asem-sft32::ex-tgt e))) train)) (length train) 1.0) (length dev) (reduce #'max levels))
       (asem-sft32:sft (or (opt-arg "out" nil) (error "out=DIR is required")) train dev '(316 367 338 443) :init-model model
                       :steps (num "steps" "1500") :batch (num "batch" "32") :workers (num "workers" "8") :lr (flt "lr" "5e-4")
                       :warmup (num "warmup" "50") :eval-every (num "eval-every" "100") :log-every (num "log-every" "20")
                       :train-probe 300 :seed (num "seed" "1")
                       :target-acc (let ((s (opt-arg "target-acc" nil))) (and s (flt "target-acc" s)))
                       :header (list :init init :init-sha256 (asem-sha256:sha256-file-hex init) :levels (opt-arg "levels" "1") :broad (opt-arg "broad" "0") :augment (opt-arg "augment" "0") :options (opt-arg "options" "0") :reverse (opt-arg "reverse" "0") :reasoning (opt-arg "reasoning" "0") :copy-mix (opt-arg "copy-mix" "0")
                                     :train-n (length train) :argv (format nil "~{~a~^ ~}" (cdr sb-ext:*posix-argv*))))))
    ((equal cmd "eval")
     (let* ((bpe (bpe-of data)) (ckpt (or (opt-arg "ckpt" nil) (error "ckpt=CKPT is required")))
            (model (asem-pretrain32:load-checkpoint ckpt)) (ctx (asem-tf32:config-ctx (asem-tf32:model-config model)))
            (n (num "n" "200")) (results nil))
       (when (equal (opt-arg "reverse" "0") "1") (setf asem-retrieval:*reverse* t))
       (when (equal (opt-arg "reasoning" "0") "1") (setf asem-retrieval:*reasoning* t))
       (dolist (level asem-retrieval:+levels+)
         (dolist (ent '(:dev-real :new))
           (dolist (w '(:seen :unseen))
             (let* ((items (asem-retrieval:probe-items level ent w n (+ 9000 (* 100 level) (if (eq ent :new) 10 0) (if (eq w :unseen) 1 0)) :fits (fits-fn bpe ctx) :options (equal (opt-arg "options" "0") "1")))
                    (key (format nil "L~d/~(~a/~a~)" level ent w))
                    (row (list "real" (accuracy model (probe-exs items bpe ctx)))))
               (when (equal (opt-arg "options" "0") "1") (setf row (append row (choice-accuracies model items bpe ctx))))
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
