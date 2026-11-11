;;;; Paired Phase 2 pilot: identical initialization (R7 final), optimizer, LR schedule, seeds, tasks, entities, and answers;
;;;; only training-only predicate paraphrase augmentation (*PREDICATE-PARAPHRASE*) for temporal, relationship, and comparison changes.
;;;; Continuation experiment from r7-L123-aug-opts-rev-rsn/last.ckpt (not a reproduction of the 2,500-step R6->R7 trajectory).
;;;;   TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/retrieval-paraphrase-pilot.lisp
;;;;     data=data/fineweb/stage1 init=CKPT out=NEW-DIR [steps=300] [workers=4]
;;;; Selection probe uses fresh L3 :new entities / :unseen wording (seed 18002, n=300); never reads dev.jsonl or the locked test.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))

(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))

(defun save-data (path value)
  (with-open-file (out path :direction :output :if-exists :error :if-does-not-exist :create)
    (with-standard-io-syntax (let ((*print-readably* t)) (write value :stream out)))))

(let* ((data (or (opt-arg "data" nil) (error "data=DIR required")))
       (init (or (opt-arg "init" nil) (error "init=CKPT required")))
       (out (or (opt-arg "out" nil) (error "out=NEW-DIR required")))
       (steps (parse-integer (opt-arg "steps" "300")))
       (workers (parse-integer (opt-arg "workers" "4")))
       (train-n (parse-integer (opt-arg "train-n" "75000")))
       (copy-n (parse-integer (opt-arg "copy-n" "15000")))
       (probe-n (parse-integer (opt-arg "probe-n" "300")))
       (seed (parse-integer (opt-arg "seed" "802")))
       (copy-seed (parse-integer (opt-arg "copy-seed" "77")))
       (probe-seed (parse-integer (opt-arg "probe-seed" "18002")))
       (manifest (asem-data:load-manifest data))
       (bpe (asem-bpe:load-bpe (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
       (model (asem-pretrain32:load-checkpoint init))
       (ctx (asem-tf32:config-ctx (asem-tf32:model-config model)))
       (asem-retrieval:*broad-train* t) (asem-retrieval:*augment* t) (asem-retrieval:*reverse* t) (asem-retrieval:*reasoning* t)
       (asem-retrieval:*vocab-words* (asem-retrieval:vocab-words-of bpe)))
  (when (probe-file out) (error "Refusing to overwrite pilot directory ~a" out))
  (unless (and (plusp steps) (plusp workers) (plusp train-n) (not (minusp copy-n)) (plusp probe-n)) (error "Invalid pilot size"))
  (ensure-directories-exist (format nil "~a/" out))
  (let* ((ctrl-seqs (let ((asem-retrieval:*predicate-paraphrase* nil))
                      (asem-retrieval:train-sequences train-n '(1 2 3) seed bpe ctx :options t)))
         (trt-seqs (let ((asem-retrieval:*predicate-paraphrase* t))
                     (asem-retrieval:train-sequences train-n '(1 2 3) seed bpe ctx :options t)))
         (copies (asem-sft32:make-copy-examples copy-n copy-seed :lo 1 :hi 8192 :min-len 10 :max-len 40))
         (probe (let ((asem-retrieval:*augment* nil) (asem-retrieval:*predicate-paraphrase* nil))
                  (asem-retrieval:probe-items 3 :new :unseen probe-n probe-seed :options t
                    :fits (lambda (prompt answer)
                            (<= (+ (length (asem-bpe:bpe-encode bpe prompt)) (length (asem-bpe:bpe-encode bpe (format nil " ~a" answer)))) ctx)))))
         (dev (map 'vector (lambda (it)
                            (asem-sft32:generation-ex (getf it :id) (getf it :kind) "selection"
                              (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :question) :options (getf it :options))
                              (getf it :answer) bpe ctx)) probe))
         (ctrl-file (format nil "~a/control-data.sexp" out))
         (trt-file (format nil "~a/treatment-data.sexp" out))
         (probe-file (format nil "~a/selection-data.sexp" out)))
    (unless (= (length ctrl-seqs) (length trt-seqs) train-n) (error "Could not generate requested training count"))
    (save-data ctrl-file ctrl-seqs) (save-data trt-file trt-seqs) (save-data probe-file probe)
    (save-data (format nil "~a/recipe.sexp" out)
               (list :experiment "Phase 2 predicate-paraphrase pilot"
                     :continuation-from init
                     :init init :init-sha256 (asem-sha256:sha256-file-hex init)
                     :steps steps :workers workers :seed seed :probe-seed probe-seed :train-n train-n :copy-n copy-n :copy-seed copy-seed :probe-n probe-n
                     :selection-entities :new :selection-wording :unseen :lr 1f-4 :warmup 30 :min-lr-frac 0.1f0 :batch 32 :clip 1.0f0 :wd 0.0f0
                     :multi-normalization :token :aux-lm 0
                     :control-sha256 (asem-sha256:sha256-file-hex ctrl-file)
                     :treatment-sha256 (asem-sha256:sha256-file-hex trt-file)
                     :selection-sha256 (asem-sha256:sha256-file-hex probe-file)
                     :primary-endpoint :fixed-final
                     :note "Continuation experiment from R7 final checkpoint. Control and treatment share identical initialization, optimizer, LR schedule, batch order, copy replay, and seed; only training-only predicate paraphrasing for temporal, relationship, and comparison differs."))
    (loop for (arm-name para seqs seq-file) in (list (list "control" nil ctrl-seqs ctrl-file)
                                                     (list "treatment" t trt-seqs trt-file))
          do (let* ((train (concatenate 'vector
                                        (loop for (ids rows targets) in seqs for i from 0
                                              collect (asem-sft32:multi-ex (format nil "p2-~d" i) ids rows targets :normalization :token))
                                        copies))
                    (arm-out (format nil "~a/~a" out arm-name)))
               (asem-sft32:sft arm-out train dev (asem-sft32:letter-token-ids bpe)
                               :init-model (asem-pretrain32:load-checkpoint init) :steps steps :batch 32 :workers workers
                               :lr 1f-4 :warmup 30 :eval-every 100 :log-every 20 :train-probe 100 :seed seed
                               :header (list :experiment "Phase 2 predicate-paraphrase pilot" :arm arm-name
                                             :predicate-paraphrase (if para 1 0) :normalization "token"
                                             :init init :init-sha256 (asem-sha256:sha256-file-hex init)
                                             :retrieval-data-sha256 (asem-sha256:sha256-file-hex seq-file)
                                             :selection-data-sha256 (asem-sha256:sha256-file-hex probe-file)
                                             :selection-seed probe-seed :selection-wording "unseen" :primary-endpoint "fixed final"))
               (sb-ext:gc :full t)))))
