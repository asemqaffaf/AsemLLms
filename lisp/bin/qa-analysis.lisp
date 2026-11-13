;;; QA DEV only: paired familiar-question diagnostics and per-item scoring/control results.
;;; TLM_BLAS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/qa-analysis.lisp 
;;;   data=data/fineweb/stage1 ckpt=CKPT items=lisp/eval/qa/dev.jsonl out=NEW.json [workers=4] [controls=1] [paired=1] [limit=0]
;;; LIMIT is for smoke tests only. No locked-test override exists.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "tf32" "train32" "pretrain32" "sft32" "qa-analysis"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*analysis-here* here))
(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(let* ((data (or (opt-arg "data" nil) (error "data=DIR required")))
       (ckpt (or (opt-arg "ckpt" nil) (error "ckpt=FILE required")))
       (path (or (opt-arg "items" nil) (error "items=DEV required")))
       (out (or (opt-arg "out" nil) (error "out=NEW.json required")))
       (lock (merge-pathnames "../eval/qa/test.lock" cl-user::*analysis-here*))
       (items (asem-qa-analysis:read-development path lock))
       (limit (parse-integer (opt-arg "limit" "0")))
       (workers (parse-integer (opt-arg "workers" "4")))
       (manifest (asem-data:load-manifest data))
       (bpe-file (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file")))
       (bpe (asem-bpe:load-bpe bpe-file))
       (model (asem-pretrain32:load-checkpoint ckpt))
       (t0 (get-internal-real-time)))
  (unless (<= 0 limit (length items)) (error "limit must be between zero and dev size"))
  (when (probe-file out) (error "Refusing to overwrite existing analysis ~a" out))
  (when (plusp limit) (setf items (subseq items 0 limit)))
  (let* ((result (asem-qa-analysis:evaluate model bpe items :workers workers :controls (equal (opt-arg "controls" "1") "1")
                                          :paired (equal (opt-arg "paired" "1") "1")))
         (obj (asem-qa-analysis::obj
               "schema" "asem-qa-dev-analysis-1" "checkpoint" ckpt "checkpoint_sha256" (asem-sha256:sha256-file-hex ckpt)
               "items" path "items_sha256" (asem-sha256:sha256-file-hex path) "locked_test" :false
               "tokenizer" bpe-file "tokenizer_sha256" (asem-sha256:sha256-file-hex bpe-file)
               "item_count" (length items) "limit" limit "workers" workers
               "seconds" (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0)
               "paired_wording_note" "First training question frame for each structured query; identical passage/options/keys. Passage wording remains unfamiliar. Counterfactual uses cf_query."
               "generation_note" "Autoregressive greedy generation stops on EOT, first decoded period/newline, context limit, or eight tokens (independent of gold). Exact compares trimmed generated answer to gold; terminated_exact additionally requires EOT or delimiter. Gold-length prefix is a separate diagnostic and does not score termination."
               "mismatch_note" "Historical same-category question replacement is reproduced. Literal unchanged flags count collisions; compare questions may retain the same max/min operation even when text changes. Use opposite-query counterfactual evidence for conditioning."
               "conditions" result)))
    (ensure-directories-exist out)
    (with-open-file (stream out :direction :output :if-exists :error :if-does-not-exist :create)
      (asem-json:write-json obj stream) (terpri stream))
    (format *error-output* "Wrote QA development diagnostics to ~a~%" out)))
