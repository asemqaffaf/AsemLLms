;;;; Evaluation only. Separate prepare/freeze and inference phases; zero training updates.
;;;; sbcl --script lisp/bin/binding-eval.lisp mode=prepare bundle=NEW-DIR
;;;; TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/binding-eval.lisp mode=eval bundle=DIR label=r7|control|treatment workers=4
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "../experimental/phase2/frames.lisp" here))
  (load (merge-pathnames "../src/binding.lisp" here)))
(defun opt (name default)
  (let ((arg (find-if (lambda (x) (and (> (length x) (length name)) (string= (concatenate 'string name "=") x :end2 (1+ (length name))))) (cdr sb-ext:*posix-argv*))))
    (if arg (subseq arg (1+ (length name))) default)))
(defun jwrite (path x)
  (with-open-file (s path :direction :output :if-exists :error :if-does-not-exist :create :external-format :utf-8) (asem-json:write-json x s) (terpri s)))
(defun jread (path)
  (with-open-file (s path :external-format :utf-8) (let ((buf (make-string (file-length s)))) (asem-json:parse-json (subseq buf 0 (read-sequence buf s))))))
(defun j (x k) (asem-json:json-get x k))
(defun obj (&rest xs) (apply #'asem-qa-analysis::obj xs))
(defparameter *sources* '("lisp/bin/binding-eval.lisp" "lisp/src/binding.lisp" "lisp/experimental/phase2/frames.lisp"
                         "lisp/src/qa-analysis.lisp" "lisp/src/retrieval.lisp" "lisp/src/qagen.lisp" "lisp/src/bpe.lisp"
                         "lisp/src/tf.lisp" "lisp/src/tf32.lisp" "lisp/src/sft.lisp" "lisp/src/sft32.lisp" "lisp/src/derive.lisp"
                         "lisp/src/verify-saved.lisp" "lisp/bin/verify-saved.lisp"))
(defparameter *models*
  '(("r7" "data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt" "33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1")
    ("control" "data/runs/retrieval/p2-paraphrase-pilot/control/last.ckpt" "435b5d7c2eb1d8afecd67e0c9eb86709aecb966b60fbe76bbc40aee5ae959a84")
    ("treatment" "data/runs/retrieval/p2-paraphrase-pilot/treatment/last.ckpt" "bce9f94765e4503bedf2674a40e066c13780709425441e9c7a7de2d7e9d8833c")))
(let* ((mode (opt "mode" "prepare")) (bundle (string-right-trim "/" (or (opt "bundle" nil) (error "bundle=NEW-DIR required"))))
       (tokenizer "data/fineweb/stage1/tokenizer.bpe") (bpe (asem-bpe:load-bpe tokenizer)))
  (unless (equal (asem-sha256:sha256-file-hex tokenizer) "30d876e0d13bf8001ad66e1dd376a4240bc14b44cdf42e397c47133165cc45e4") (error "Wrong tokenizer"))
  (cond
    ((equal mode "prepare")
     (when (probe-file (format nil "~a/" bundle)) (error "Refusing to reuse bundle directory"))
     (multiple-value-bind (pairs rejections pools) (asem-binding:generate-pairs bpe)
       (unless (= 400 (length pairs)) (error "Exactly 400 pairs required"))
       (ensure-directories-exist (format nil "~a/" bundle))
       (with-open-file (s (format nil "~a/pairs.sexp" bundle) :direction :output :if-exists :error :if-does-not-exist :create)
         (with-standard-io-syntax (let ((*print-readably* t)) (write pairs :stream s))))
       (jwrite (format nil "~a/pairs.json" bundle) (asem-binding:fixture-json pairs bpe))
       (jwrite (format nil "~a/families.json" bundle) (coerce (mapcar #'asem-binding:family-json (asem-binding:families)) 'vector))
       (jwrite (format nil "~a/protocol.json" bundle)
         (obj "schema" "asem-binding-protocol-1" "hypothesis" "Phase 2 loses at least 10 percentage points of joint relationship prefix correctness under fresh unfamiliar language."
              "primary" "relationship both-world gold-length greedy prefix exact; Treatment minus Control"
              "support" "delta <= -0.10; exact two-sided McNemar p < .05; paired and family-bootstrap 95% upper bounds < 0"
              "exclude_material_deficit" "conservative one-sided paired lower bound and family-bootstrap one-sided lower bound > -0.10"
              "otherwise" "inconclusive" "bootstrap_seed" 11011 "bootstrap_resamples" 10000 "fixture_seed" 1101101
              "pair_count" 400 "relationship_pairs" 200 "relationship_families" 20 "secondary_pairs_per_category" 50
              "training_updates" 0 "reference" "r7" "promotion" "none"
              "eligibility" "Largest equal-answer-token-length pools; both prompts equal length; all four options fit ctx=128; symmetric rejection before inference."
              "limitations" "One training seed; authored template families, not an untouched benchmark; prefix exact excludes termination. No claim of complete generative reasoning."))
       (jwrite (format nil "~a/manifest.json" bundle)
         (obj "schema" "asem-binding-bundle-1" "pair_count" 400 "locked_test_accessed" :false "training_updates" 0
              "sbcl_version" (lisp-implementation-version) "seed" 1101101 "ctx" 128
              "tokenizer" tokenizer "tokenizer_sha256" (asem-sha256:sha256-file-hex tokenizer)
              "files_sha256" (cons :obj (loop for name in '("pairs.sexp" "pairs.json" "families.json" "protocol.json") collect (cons name (asem-sha256:sha256-file-hex (format nil "~a/~a" bundle name)))))
              "sources_sha256" (cons :obj (mapcar (lambda (p) (cons p (asem-sha256:sha256-file-hex p))) *sources*))
              "models" (cons :obj (loop for (label path hash) in *models* collect (cons label (obj "path" path "sha256" hash))))
              "eligible_pool_sizes" (obj "names" (getf pools :names) "towns" (getf pools :towns) "places" (getf pools :places))
              "rejection_count" (length rejections) "rejections" (coerce (mapcar (lambda (r) (obj "family" (getf r :family) "reason" (getf r :reason))) rejections) 'vector)))
       (format t "Frozen 400 pairs, ~d eligibility rejections, no model inference.~%" (length rejections))))
    ((equal mode "eval")
     (let* ((manifest (jread (format nil "~a/manifest.json" bundle))) (label (opt "label" "r7"))
            (spec (or (assoc label *models* :test #'equal) (error "Unknown model label")))
            (path (second spec)) (expected (third spec)) (out (format nil "~a/~a.json" bundle label))
            (workers (parse-integer (opt "workers" "4"))))
       (when (probe-file out) (error "Refusing to overwrite result ~a" out))
       (unless (and (equal (j manifest "schema") "asem-binding-bundle-1") (= 400 (j manifest "pair_count")) (<= 1 workers 4)) (error "Invalid frozen bundle/workers"))
       (dolist (entry (cdr (j manifest "files_sha256")))
         (unless (equal (cdr entry) (asem-sha256:sha256-file-hex (format nil "~a/~a" bundle (car entry)))) (error "Frozen fixture changed")))
       (dolist (entry (cdr (j manifest "sources_sha256")))
         (unless (equal (cdr entry) (asem-sha256:sha256-file-hex (car entry))) (error "Frozen source changed: ~a" (car entry))))
       (unless (equal expected (asem-sha256:sha256-file-hex path)) (error "Checkpoint changed"))
       (let* ((pairs (with-open-file (s (format nil "~a/pairs.sexp" bundle)) (let ((*read-eval* nil)) (read s))))
              (worlds (asem-binding:flatten-worlds pairs)) (start (get-internal-real-time)))
         (unless (and (= 400 (length pairs)) (= 800 (length worlds))) (error "Wrong fixture size"))
         (dolist (p pairs) (asem-binding:validate-pair p bpe))
         (let* ((model (asem-pretrain32:load-checkpoint path))
                (result (asem-qa-analysis:evaluate model bpe worlds :workers workers :controls nil :paired nil)))
           (jwrite out (obj "schema" "asem-binding-result-1" "label" label "checkpoint" path "checkpoint_sha256" expected
                            "bundle_manifest_sha256" (asem-sha256:sha256-file-hex (format nil "~a/manifest.json" bundle))
                            "pair_count" 400 "world_count" 800 "workers" workers "training_updates" 0 "locked_test_accessed" :false
                            "seconds" (/ (- (get-internal-real-time) start) internal-time-units-per-second 1d0)
                            "conditions" result))
           (unless (equal expected (asem-sha256:sha256-file-hex path)) (error "Checkpoint changed during evaluation"))
           (format t "Saved immutable ~a predictions: 800 worlds.~%" label)))))
    (t (error "mode must be prepare or eval"))))
