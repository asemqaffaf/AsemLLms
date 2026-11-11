;;; Paired 2x2 passage x question wording diagnostic on development QA items or held-out L3 retrieval probes.
;;;   QA dev:   TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/wording-eval.lisp \
;;;               data=data/fineweb/stage1 ckpt=CKPT items=lisp/eval/qa/dev.jsonl out=NEW.json [workers=4] [policy=both|seeded|first] [cf=1]
;;;   L3 probe: TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/wording-eval.lisp \
;;;               data=data/fineweb/stage1 ckpt=CKPT probes=600 probe-seed=19032 out=NEW.json [workers=4] [policy=both|seeded|first] [cf=1]
;;; Refuses locked-test inputs and refuses to overwrite existing output files.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis" "wording"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*wording-eval-here* here)
  (defparameter cl-user::*wording-eval-source* *load-truename*))

(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name))
                                     (string= name a :end2 (length name))
                                     (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))

(let* ((data (or (opt-arg "data" nil) (error "data=DIR required")))
       (ckpt (or (opt-arg "ckpt" nil) (error "ckpt=FILE required")))
       (items-path (opt-arg "items" nil))
       (probes-n (parse-integer (opt-arg "probes" "0")))
       (probe-seed (parse-integer (opt-arg "probe-seed" "19032")))
       (out (or (opt-arg "out" nil) (error "out=NEW.json required")))
       (workers (parse-integer (opt-arg "workers" "4")))
       (policy-arg (opt-arg "policy" "both"))
       (cf (equal (opt-arg "cf" "1") "1"))
       (limit (parse-integer (opt-arg "limit" "0")))
       (lock (merge-pathnames "../eval/qa/test.lock" cl-user::*wording-eval-here*))
       (manifest (asem-data:load-manifest data))
       (bpe-file (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file")))
       (bpe (asem-bpe:load-bpe bpe-file))
       (model (asem-pretrain32:load-checkpoint ckpt))
       (ctx (asem-tf32:config-ctx (asem-tf32:model-config model)))
       (items (cond (items-path (asem-qa-analysis:read-development items-path lock))
                    ((plusp probes-n)
                     (let ((asem-retrieval:*reverse* t) (asem-retrieval:*reasoning* t)
                           (fits (lambda (prompt answer)
                                   (<= (+ (length (asem-bpe:bpe-encode bpe prompt))
                                          (length (asem-bpe:bpe-encode bpe (format nil " ~a" answer)))) ctx))))
                       (mapcar #'asem-wording:probe-to-qa-item
                               (asem-retrieval:probe-items 3 :new :unseen probes-n probe-seed :fits fits :options t))))
                    (t (error "Either items=DEV.jsonl or probes=N is required"))))
       (policies (cond ((equal policy-arg "both") '(:seeded :first))
                       ((equal policy-arg "seeded") '(:seeded))
                       ((equal policy-arg "first") '(:first))
                       (t (error "policy must be both, seeded, or first"))))
       (t0 (get-internal-real-time)))
  (when (probe-file out) (error "Refusing to overwrite existing output ~a" out))
  (when (plusp limit) (setf items (subseq items 0 (min limit (length items)))))
  (let* ((runs (loop for pol in policies
                     collect (cons (string-downcase (symbol-name pol))
                                   (asem-wording:evaluate-2x2 model bpe items :workers workers :policy pol :counterfactual cf))))
         (obj (asem-qa-analysis::obj
               "schema" "asem-wording-2x2-1"
               "checkpoint" ckpt "checkpoint_sha256" (asem-sha256:sha256-file-hex ckpt)
               "dataset_kind" (if items-path "qa_dev" "l3_new_probes")
               "items" (or items-path (format nil "L3:new:unseen:n=~d:seed=~d" probes-n probe-seed))
               "items_sha256" (if items-path (asem-sha256:sha256-file-hex items-path) :null)
               "locked_test" :false
               "tokenizer" bpe-file "tokenizer_sha256" (asem-sha256:sha256-file-hex bpe-file)
               "wording_source_sha256" (asem-sha256:sha256-file-hex (merge-pathnames "../src/wording.lisp" cl-user::*wording-eval-here*))
               "eval_source_sha256" (asem-sha256:sha256-file-hex cl-user::*wording-eval-source*)
               "item_count" (length items) "workers" workers
               "seconds" (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0)
               "design_note" "Paired 2x2 factorial evaluation: passage wording (familiar :train vs unfamiliar :dev) x question wording (familiar :train vs unfamiliar :dev). Underlying facts, entities, sentence order, options, option ordering and answer keys are identical across all four conditions and verified by ASEM-WORDING:CHECK-CELL-ITEM before inference."
               "policies" (cons :obj runs))))
    (ensure-directories-exist out)
    (with-open-file (stream out :direction :output :if-exists :error :if-does-not-exist :create)
      (asem-json:write-json obj stream) (terpri stream))
    (format *error-output* "Wrote paired 2x2 wording diagnostics to ~a~%" out)))
