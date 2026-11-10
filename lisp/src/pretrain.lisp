;;;; Pretraining on BPE token shards (the output of bin/build-data.lisp): data-parallel SBCL threads, Adam with warmup and
;;;; cosine decay, validation loss on fixed windows, a multiple-choice evaluation scored by answer log-likelihood,
;;;; JSONL logging (loss, tokens/s, peak memory, checkpoint hash) and binary float32 checkpoints.
;;;; This file is the FP64 source; src/pretrain32.lisp derives the single-float version (see derive.lisp).
;;;; Host-side scalars use long-float / 0l0 (always double); everything that touches model tensors uses double-float.
(defpackage :asem-pretrain
  (:use :cl :asem-tf :asem-util)
  (:export #:pretrain #:load-checkpoint #:save-checkpoint #:read-eval-set #:score-eval #:sample-text #:load-split-tokens
           #:preset-config #:rss-mb #:evaluate-checkpoint #:diagnose-model #:token-logfreq))
(in-package :asem-pretrain)

;;; ------------------------------------------------------------------ tokens
(defun load-split-tokens (dir split)
  "All tokens of SPLIT (\"train\" / \"val\" / \"test\") as a (simple-array (unsigned-byte 16) (*)), in shard order."
  (let* ((m (asem-data:load-manifest dir))
         (shards (asem-json:json-get (asem-json:json-get (asem-json:json-get m "splits") split) "shards"))
         (total (reduce #'+ (map 'list (lambda (s) (asem-json:json-get s "tokens")) shards)))
         (out (make-array total :element-type '(unsigned-byte 16))) (n 0))
    (loop for sh across shards
          do (let* ((path (format nil "~a/~a" dir (asem-json:json-get sh "file"))) (k (asem-json:json-get sh "tokens"))
                    (bytes (make-array (* 2 k) :element-type '(unsigned-byte 8))))
               (with-open-file (in path :element-type '(unsigned-byte 8)) (read-sequence bytes in))
               (dotimes (i k) (setf (aref out (+ n i)) (+ (aref bytes (* 2 i)) (ash (aref bytes (1+ (* 2 i))) 8))))
               (incf n k)))
    out))

(defun window-at (tokens s ctx)
  "(values x y): TOKENS[s, s+ctx) and TOKENS[s+1, s+ctx+1) as fixnum vectors."
  (let ((x (make-array ctx :element-type 'fixnum)) (y (make-array ctx :element-type 'fixnum)))
    (dotimes (i ctx) (setf (aref x i) (aref tokens (+ s i)) (aref y i) (aref tokens (+ s i 1))))
    (values x y)))

;;; ------------------------------------------------------------------ model sizes
(defun preset-config (name vocab ctx)
  "Named sizes for an 8k vocabulary: parameters are dominated by the embeddings at this scale."
  (ecase name
    (:1m (make-config :vocab vocab :ctx ctx :dim 64 :heads 4 :layers 2 :ff 256))
    (:3m (make-config :vocab vocab :ctx ctx :dim 128 :heads 4 :layers 4 :ff 512))
    (:10m (make-config :vocab vocab :ctx ctx :dim 256 :heads 8 :layers 8 :ff 1024))
    (:25m (make-config :vocab vocab :ctx ctx :dim 384 :heads 8 :layers 12 :ff 1536))))

;;; ------------------------------------------------------------------ memory
(defun rss-mb ()
  "Resident set size of this process in MB (ps), or 0."
  (handler-case
      (let* ((out (make-string-output-stream))
             (p (sb-ext:run-program "/bin/ps" (list "-o" "rss=" "-p" (princ-to-string (sb-posix:getpid))) :output out :error nil :wait t)))
        (declare (ignorable p))
        (/ (or (parse-integer (get-output-stream-string out) :junk-allowed t) 0) 1024l0))
    (error () 0l0)))

;;; ------------------------------------------------------------------ losses and multiple-choice evaluation
(defun window-loss-mean (model tokens starts)
  "Mean per-token cross-entropy of MODEL over the windows beginning at STARTS (single thread); a long-float."
  (let* ((cfg (model-config model)) (ctx (config-ctx cfg)) (acts (make-acts cfg ctx)) (sum 0l0))
    (dolist (s starts)
      (multiple-value-bind (x y) (window-at tokens s ctx)
        (forward model x acts)
        (incf sum (loss-of-acts acts y))))
    (/ sum (max 1 (length starts)))))

(defun read-eval-set (path)
  "List of plists (:id :category :context :q :options :answer) from a JSONL file."
  (with-open-file (in path :external-format :utf-8)
    (loop for line = (read-line in nil) while line
          when (plusp (length (string-trim " " line)))
            collect (let ((o (asem-json:parse-json line)))
                      (list :id (asem-json:json-get o "id") :category (asem-json:json-get o "category")
                            :context (asem-json:json-get o "context") :q (asem-json:json-get o "q")
                            :options (coerce (asem-json:json-get o "options") 'list) :answer (asem-json:json-get o "answer"))))))

(defun option-logprob (model bpe prompt option)
  "(values total-logprob token-count) of OPTION's tokens given PROMPT, with an end-of-text token in front."
  (let* ((cfg (model-config model)) (ctx (config-ctx cfg)) (v (config-vocab cfg))
         (opt (asem-bpe:bpe-encode bpe option))
         (ids (append (list 0) (asem-bpe:bpe-encode bpe prompt) opt))
         (over (- (length ids) (1+ ctx))))
    (when (plusp over) (setf ids (nthcdr over ids)))       ; keep the end of the prompt and the whole option
    (let* ((n (length ids)) (x (make-array (1- n) :element-type 'fixnum)) (acts (make-acts cfg (1- n))) (sum 0l0))
      (loop for id in ids for i from 0 below (1- n) do (setf (aref x i) id))
      (forward model x acts)
      (let ((probs (acts-probs acts)) (ys (cdr ids)))
        (loop for tgt in ys for i from 0 below (1- n)
              when (>= i (- (1- n) (length opt)))
                do (let ((pr (aref probs (+ (* i v) tgt))))
                     (incf sum (log (if (< pr 1d-30) 1d-30 pr))))))      ; a NaN is not clamped: it poisons the sum
      (values sum (length opt)))))

(defun score-eval (model bpe items)
  "Multiple-choice scores by answer likelihood. Returns a plist: :accuracy (mean token log-prob picks the answer),
:accuracy-sum (total log-prob), :margin (mean over items of correct minus best wrong, mean log-prob), :by-category, :n."
  (let ((hits 0) (hits-sum 0) (margins 0l0) (cats (make-hash-table :test 'equal)))
    (dolist (it items)
      (let* ((prompt (format nil "~@[~a~%~]Question: ~a~%Answer:" (getf it :context) (getf it :q)))
             (means nil) (sums nil))
        (dolist (o (getf it :options))
          (multiple-value-bind (lp n) (option-logprob model bpe prompt (format nil " ~a" o))
            (push (/ lp (max 1 n)) means) (push lp sums)))
        (setf means (nreverse means) sums (nreverse sums))
        (let* ((ans (getf it :answer)) (best (position (reduce #'max means) means)) (best-s (position (reduce #'max sums) sums))
               (wrong (loop for m in means for i from 0 unless (= i ans) maximize m)))
          (when (= best ans) (incf hits))
          (when (= best-s ans) (incf hits-sum))
          (incf margins (- (nth ans means) wrong))
          (let ((c (gethash (getf it :category) cats (list 0 0))))
            (incf (first c) (if (= best ans) 1 0)) (incf (second c))
            (setf (gethash (getf it :category) cats) c)))))
    (let ((n (length items)))
      (list :accuracy (/ hits (max 1 n) 1l0) :accuracy-sum (/ hits-sum (max 1 n) 1l0) :margin (/ margins (max 1 n))
            :by-category (loop for k being the hash-keys of cats using (hash-value c) collect (cons k (/ (first c) (second c) 1l0)))
            :n n))))

;;; ------------------------------------------------------------------ sampling (for eyeballing coherence)
(defun sample-text (model bpe prompt n &key (temperature 0.8l0) (seed 1))
  (let* ((cfg (model-config model)) (ctx (config-ctx cfg)) (v (config-vocab cfg)) (state (sb-ext:seed-random-state seed))
         (ids (asem-bpe:bpe-encode bpe prompt)) (out nil) (probs (new-dvec v)))
    (dotimes (k n)
      (let* ((window (last ids ctx)) (logits (last-logits model (coerce window '(simple-array fixnum (*)))))
             (u (random 1l0 state)) (acc 0l0) (pick (1- v)))
        (softmax-row logits 0 probs 0 v (host-float temperature))
        (dotimes (i v) (incf acc (aref probs i)) (when (> acc u) (setf pick i) (return)))
        (setf ids (append ids (list pick))) (push pick out)))
    (asem-bpe:bpe-decode bpe (nreverse out))))

;;; ------------------------------------------------------------------ checkpoints: JSON header line + float32 LE tensors
(defun put-f32 (bytes pos x)
  (let ((bits (ldb (byte 32 0) (sb-kernel:single-float-bits (coerce x 'single-float)))))
    (dotimes (b 4) (setf (aref bytes (+ pos b)) (ldb (byte 8 (* 8 b)) bits)))))

(defun get-f32 (bytes pos)
  (let ((bits (+ (aref bytes pos) (ash (aref bytes (+ pos 1)) 8) (ash (aref bytes (+ pos 2)) 16) (ash (aref bytes (+ pos 3)) 24))))
    (sb-kernel:make-single-float (if (logbitp 31 bits) (- bits #x100000000) bits))))

(defun save-checkpoint (path model &rest header)
  "Write MODEL as float32; HEADER is a plist of extra JSON fields. Returns the file's SHA-256."
  (let* ((cfg (model-config model)) (params (model-params model))
         (h (append (list :format "asem-ckpt-1" :dtype "f32" :vocab (config-vocab cfg) :ctx (config-ctx cfg) :dim (config-dim cfg)
                          :heads (config-heads cfg) :layers (config-layers cfg) :ff (config-ff cfg) :params (param-count model))
                    header))
         (total (reduce #'+ (map 'list #'length params)))
         (bytes (make-array (* 4 total) :element-type '(unsigned-byte 8))) (pos 0))
    (loop for p across params do (dotimes (i (length p)) (put-f32 bytes pos (aref p i)) (incf pos 4)))
    (with-open-file (o path :direction :output :if-exists :supersede :external-format :utf-8)
      (write-string (header-json h) o) (terpri o))
    (with-open-file (o path :direction :output :if-exists :append :element-type '(unsigned-byte 8)) (write-sequence bytes o))
    (asem-sha256:sha256-file-hex path)))

(defun header-json (plist)
  (let ((*read-default-float-format* 'double-float))
    (with-output-to-string (o)
      (write-char #\{ o)
      (loop for (k v) on plist by #'cddr for first = t then nil
            do (unless first (write-char #\, o))
               (format o "~a:" (asem-util:json-string (substitute #\_ #\- (string-downcase (symbol-name k)))))
               (cond ((stringp v) (write-string (asem-util:json-string v) o))
                     ((integerp v) (format o "~d" v))
                     ((realp v) (format o "~,6f" (float v 1l0)))
                     (t (write-string (asem-util:json-string (princ-to-string v)) o))))
      (write-char #\} o))))

(defun load-checkpoint (path)
  "(values model header-object)."
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let* ((all (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (read-sequence all in)
      (let* ((nl (position 10 all)) (h (asem-json:parse-json (sb-ext:octets-to-string all :end nl :external-format :utf-8)))
             (g (lambda (k) (asem-json:json-get h k)))
             (cfg (make-config :vocab (funcall g "vocab") :ctx (funcall g "ctx") :dim (funcall g "dim") :heads (funcall g "heads")
                               :layers (funcall g "layers") :ff (funcall g "ff")))
             (m (init-model cfg 1)) (pos (1+ nl)))
        (unless (equal (funcall g "format") "asem-ckpt-1") (error "not an asem checkpoint: ~a" path))
        (unless (= (- (length all) pos) (* 4 (param-count m))) (error "checkpoint size does not match its header"))
        (loop for p across (model-params m) do (dotimes (i (length p)) (setf (aref p i) (float (get-f32 all pos) 1d0)) (incf pos 4)))
        (values m h)))))

;;; ------------------------------------------------------------------ training
(defun log-event (stream plist)
  (let ((line (header-json plist)))
    (write-string line stream) (terpri stream) (finish-output stream)
    (write-string line *standard-output*) (terpri *standard-output*) (finish-output *standard-output*)))

(defun lr-at (step steps lr warmup min-frac)
  "Linear warmup, then cosine decay to MIN-FRAC of LR; a long-float."
  (let ((lr (float lr 1l0)) (min-frac (float min-frac 1l0)))
    (if (< step warmup)
        (* lr (/ (1+ step) (max 1 warmup)))
        (let ((progress (/ (- step warmup) (max 1 (- steps warmup)))))
          (* lr (+ min-frac (* (- 1l0 min-frac) 0.5l0 (+ 1l0 (cos (* pi progress))))))))))

(defun pretrain (data-dir out-dir &key (size :1m) (ctx 128) (steps 1000) (batch 16) (workers 1) (lr 0.003d0) (min-lr-frac 0.1d0)
                                       (warmup 100) (clip 1d0) (seed 1) (eval-every 250) (val-windows 64) (eval-file nil) (log-path nil)
                                       (sample-prompt "The") (config nil) (log-every 0) (save-every 0))
  "Train from scratch on the shards in DATA-DIR; checkpoints and a JSONL log go to OUT-DIR. Returns a plist of final numbers.
LOG-EVERY N > 0 adds a \"train\" event every N steps (mean training loss since the last one, gradient norm, learning rate) to watch
for divergence; SAVE-EVERY N > 0 keeps OUT-DIR/step-NNNNN.ckpt every N steps. Neither changes the numerics."
  (ensure-directories-exist (concatenate 'string out-dir "/"))
  (let* ((version (asem-data:verify data-dir))
         (manifest (asem-data:load-manifest data-dir))
         (vocab (asem-json:json-get (asem-json:json-get manifest "tokenizer") "vocab_size"))
         (bpe (asem-bpe:load-bpe (format nil "~a/~a" data-dir (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
         (train (load-split-tokens data-dir "train")) (val (load-split-tokens data-dir "val"))
         (items (and eval-file (read-eval-set eval-file)))
         (cfg (or config (preset-config size vocab ctx)))
         (ctx (config-ctx cfg))
         (model (init-model cfg seed)) (params (model-params model))
         (state (sb-ext:seed-random-state seed))
         (nwin (- (length train) ctx 1))
         (val-starts (let ((span (- (length val) ctx 1))) (loop for i below val-windows collect (floor (* i span) val-windows))))
         (ws (loop repeat workers collect (list (make-acts cfg ctx) (make-grads cfg))))
         (m1 (map 'simple-vector (lambda (p) (new-dvec (length p))) params))
         (m2 (map 'simple-vector (lambda (p) (new-dvec (length p))) params))
         (sum-grads (make-grads cfg))
         (b1 0.9l0) (b2 0.95l0) (cur-lr 0l0) (last-nrm 0l0) (log-loss 0l0) (log-n 0)
         (best-val most-positive-double-float) (peak-mb 0l0) (acc-loss 0l0) (acc-n 0) (t0 (get-internal-real-time)) (t-last t0)
         (tokens-last 0)
         (log (open (or log-path (format nil "~a/log.jsonl" out-dir)) :direction :output :if-exists :supersede :external-format :utf-8))
         (result nil))
    (unless (> nwin 1000) (error "training split too small (~d tokens) for context ~d" (length train) ctx))
    (unwind-protect
         (labels ((evaluate (step tokens train-loss)
                    (let* ((vl (window-loss-mean model val val-starts))
                           (ans (and items (score-eval model bpe items)))
                           (now (get-internal-real-time)) (dt (/ (- now t-last) internal-time-units-per-second 1l0))
                           (rss (rss-mb)) (sample (ignore-errors (sample-text model bpe sample-prompt 24 :seed 7))))
                      (setf peak-mb (max peak-mb rss))
                      (log-event log (append (list :event "eval" :step step :tokens tokens :train-loss (or train-loss 0) :val-loss vl
                                                   :val-ppl (exp (min 30l0 vl)) :tokens-per-sec (if (plusp dt) (/ (- tokens tokens-last) dt) 0)
                                                   :rss-mb rss :peak-rss-mb peak-mb)
                                             (when ans (list :answer-accuracy (getf ans :accuracy) :answer-accuracy-sum (getf ans :accuracy-sum)
                                                             :answer-margin (getf ans :margin) :random-baseline 0.25l0
                                                             :acc-fact (or (cdr (assoc "fact" (getf ans :by-category) :test #'equal)) 0)
                                                             :acc-reading (or (cdr (assoc "reading" (getf ans :by-category) :test #'equal)) 0)
                                                             :acc-reasoning (or (cdr (assoc "reasoning" (getf ans :by-category) :test #'equal)) 0)))
                                             (list :sample (or sample ""))))
                      (setf t-last now tokens-last tokens)
                      (when (< vl best-val)
                        (setf best-val vl)
                        (save-checkpoint (format nil "~a/best.ckpt" out-dir) model :step step :val-loss vl :data-version version))
                      (values vl ans))))
           (log-event log (list :event "start" :size (string-downcase (symbol-name size)) :params (param-count model) :ctx ctx :dim (config-dim cfg)
                                :layers (config-layers cfg) :heads (config-heads cfg) :ff (config-ff cfg) :vocab vocab :steps steps :batch batch
                                :workers workers :lr lr :seed seed :train-tokens (length train) :val-tokens (length val) :data-version version
                                :precision (if (typep (aref params 0) '(simple-array long-float (*))) "f64" "f32")))
           (multiple-value-bind (vl0 ans0) (evaluate 0 0 nil)          ; the untrained baseline
             (declare (ignorable vl0 ans0)))
           (loop for step from 1 to steps
                 do (let* ((starts (loop repeat batch collect (random nwin state)))
                           (results
                             (asem-train::parallel-map
                              (lambda (job)
                                (destructuring-bind (w sts) job
                                  (destructuring-bind (acts grads) w
                                    (zero-grads grads)
                                    (let ((ls 0l0))
                                      (dolist (s sts)
                                        (multiple-value-bind (x y) (window-at train s ctx)
                                          (forward model x acts)
                                          (incf ls (backward model x y acts grads))))
                                      ls))))
                              (loop for w in ws for sts in (asem-train::shards starts workers) when sts collect (list w sts)))))
                      (zero-grads sum-grads)
                      (loop for w in ws do (loop for g across (second w) for sg across sum-grads do (asem-train::add-into sg g)))
                      (incf acc-loss (reduce #'+ results)) (incf acc-n batch)
                      (incf log-loss (reduce #'+ results)) (incf log-n batch)
                      (let ((inv (host-float (/ 1l0 batch))))
                        (loop for sg across sum-grads do (asem-train::scale-into sg inv))
                        (let ((nrm (asem-train::grad-norm sum-grads)))
                          (setf last-nrm nrm)
                          (when (or (/= nrm nrm) (> nrm 1l30)) (error "non-finite gradient norm at step ~d: stopping instead of training on garbage" step))
                          (when (> nrm clip)
                            (loop for sg across sum-grads do (asem-train::scale-into sg (host-float (/ clip nrm)))))))
                      (setf cur-lr (lr-at (1- step) steps lr warmup min-lr-frac))
                      (let* ((lrt (host-float (let ((l cur-lr))
                                                 (* l (/ (sqrt (- 1l0 (expt b2 step))) (- 1l0 (expt b1 step)))))))
                             (b1f (host-float b1)) (b2f (host-float b2)) (c1 (host-float (- 1l0 b1))) (c2 (host-float (- 1l0 b2))))
                        (loop for p across params for g across sum-grads for a across m1 for b across m2
                              do (let ((p p) (g g) (a a) (b b))
                                   (declare (type dvec p g a b) (type double-float lrt b1f b2f c1 c2) (optimize (speed 3) (safety 0)))
                                   (dotimes (i (length p))
                                     (let ((gi (aref g i)))
                                       (setf (aref a i) (+ (* b1f (aref a i)) (* c1 gi)))
                                       (setf (aref b i) (+ (* b2f (aref b i)) (* c2 gi gi)))
                                       (decf (aref p i) (/ (* lrt (aref a i)) (+ (sqrt (aref b i)) 1d-8))))))))
                      (when (and (plusp log-every) (zerop (mod step log-every)))
                        (log-event log (list :event "train" :step step :tokens (* step batch ctx) :train-loss (/ log-loss (max 1 log-n))
                                             :grad-norm last-nrm :lr cur-lr))
                        (setf log-loss 0l0 log-n 0))
                      (when (and (plusp save-every) (zerop (mod step save-every)))
                        (save-checkpoint (format nil "~a/step-~5,'0d.ckpt" out-dir step) model :step step :data-version version))
                      (when (or (zerop (mod step eval-every)) (= step steps))
                        (evaluate step (* step batch ctx) (/ acc-loss acc-n))
                        (setf acc-loss 0l0 acc-n 0))))
           (let* ((secs (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1l0))
                  (sha (save-checkpoint (format nil "~a/last.ckpt" out-dir) model :step steps :data-version version :seed seed)))
             (setf result (list :params (param-count model) :best-val-loss best-val :seconds secs :tokens-seen (* steps batch ctx)
                                :tokens-per-sec (/ (* steps batch ctx) secs) :peak-rss-mb peak-mb :checkpoint-sha256 sha :data-version version))
             (log-event log (append (list :event "done") result))))
      (close log))
    result))

;;; ------------------------------------------------------------------ final evaluation of a checkpoint
(defun split-loss (model tokens)
  "Mean cross-entropy over the whole split: consecutive windows of CTX tokens that do not overlap."
  (let* ((ctx (config-ctx (model-config model))) (starts (loop for s from 0 to (- (length tokens) ctx 1) by ctx collect s)))
    (values (window-loss-mean model tokens starts) (* ctx (length starts)))))

(defun evaluate-checkpoint (data-dir ckpt-path &key eval-file (use-test t))
  "Loss on the full validation and test splits plus the multiple-choice scores. The test split is only ever used here,
and with USE-TEST NIL it is not even read (hyperparameter searches select on validation loss only)."
  (let* ((model (load-checkpoint ckpt-path))
         (manifest (asem-data:load-manifest data-dir))
         (bpe (asem-bpe:load-bpe (format nil "~a/~a" data-dir (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
         (val (load-split-tokens data-dir "val")) (test (and use-test (load-split-tokens data-dir "test")))
         (items (and eval-file (read-eval-set eval-file))) (n (length items))
         (ans (and items (score-eval model bpe items))))
    (multiple-value-bind (vl vn) (split-loss model val)
      (multiple-value-bind (tl tn) (if use-test (split-loss model test) (values nil nil))
        (append (list :checkpoint ckpt-path :params (param-count model) :val-loss vl :val-ppl (exp vl) :val-tokens vn)
                (when use-test (list :test-loss tl :test-ppl (exp tl) :test-tokens tn))
                (when ans
                  (list :answer-accuracy (getf ans :accuracy) :answer-accuracy-sum (getf ans :accuracy-sum) :answer-margin (getf ans :margin)
                        :questions n :random-baseline 0.25l0
                        :chance-95-upper (+ 0.25l0 (* 1.96l0 (sqrt (/ (* 0.25l0 0.75l0) n))))
                        :acc-fact (or (cdr (assoc "fact" (getf ans :by-category) :test #'equal)) 0)
                        :acc-reading (or (cdr (assoc "reading" (getf ans :by-category) :test #'equal)) 0)
                        :acc-reasoning (or (cdr (assoc "reasoning" (getf ans :by-category) :test #'equal)) 0))))))))
;;; ------------------------------------------------------------------ diagnostics: is the model using the question?
;;; Everything here is deterministic (no sampling): forward passes only. Numbers in the result are long-floats (doubles) so the
;;; JSON is identical whichever precision the model runs in. See bin/diagnose.lisp.
(defun prompt-of (it &key (question t) (context t))
  "The prompt SCORE-EVAL uses (question and context), or an ablation of it."
  (format nil "~@[~a~%~]~@[Question: ~a~%~]Answer:" (and context (getf it :context)) (and question (getf it :q))))

(defun option-scores (model bpe prompt options)
  "List of (total-logprob . token-count), one per option."
  (loop for o in options
        collect (multiple-value-bind (lp n) (option-logprob model bpe prompt (format nil " ~a" o)) (cons (float lp 1l0) n))))

(defun token-logfreq (tokens vocab)
  "Vector of smoothed unigram log-frequencies of each token id in TOKENS (add-one), as long-floats."
  (let ((counts (make-array vocab :initial-element 0)) (out (make-array vocab)))
    (loop for tk across tokens do (incf (aref counts tk)))
    (dotimes (i vocab out) (setf (aref out i) (log (/ (+ 1l0 (aref counts i)) (+ (length tokens) vocab)))))))

(defun mean-of (xs) (if xs (/ (reduce #'+ xs) (length xs)) 0l0))
(defun sd-of (xs) (if (< (length xs) 2) 0l0 (let ((m (mean-of xs))) (sqrt (/ (reduce #'+ (mapcar (lambda (x) (expt (- x m) 2)) xs)) (1- (length xs)))))))
(defun median-of (xs) (let* ((v (sort (copy-list xs) #'<)) (n (length v))) (if (zerop n) 0l0 (if (oddp n) (nth (floor n 2) v) (/ (+ (nth (1- (floor n 2)) v) (nth (floor n 2) v)) 2)))))
(defun pearson (xs ys)
  (let* ((mx (mean-of xs)) (my (mean-of ys))
         (sxy (reduce #'+ (mapcar (lambda (x y) (* (- x mx) (- y my))) xs ys)))
         (sxx (reduce #'+ (mapcar (lambda (x) (expt (- x mx) 2)) xs))) (syy (reduce #'+ (mapcar (lambda (y) (expt (- y my) 2)) ys))))
    (if (or (zerop sxx) (zerop syy)) 0l0 (/ sxy (sqrt (* sxx syy))))))
(defun argmax-of (xs) (position (reduce #'max xs) xs))
(defun softmax-of (xs) (let* ((m (reduce #'max xs)) (es (mapcar (lambda (x) (exp (- x m))) xs)) (z (reduce #'+ es))) (mapcar (lambda (e) (/ e z)) es)))
(defun tv-distance (p q) (* 0.5l0 (reduce #'+ (mapcar (lambda (a b) (abs (- a b))) p q))))
(defun frac (pred xs) (if xs (/ (count-if pred xs) (length xs) 1l0) 0l0))

(defun mismatched-items (items it idx k)
  "K other items of the same category, chosen deterministically (cyclically from a position that depends on IDX)."
  (let* ((same (remove-if (lambda (o) (eq o it)) (remove-if-not (lambda (o) (equal (getf o :category) (getf it :category))) items)))
         (len (length same)))
    (when (zerop len) (setf same (remove it items) len (length same)))
    (loop for j below (min k len) collect (nth (mod (+ (* idx 5) j) len) same))))

(defun jobj (&rest plist) (cons :obj (loop for (k v) on plist by #'cddr collect (cons k v))))

(defun diagnose-model (model bpe items logfreq &key (mismatches 3))
  "A JSON-ready object (asem-json :obj) describing whether MODEL's answers depend on the question; see bin/diagnose.lisp."
  (let* ((n (length items)) (recs nil))
    (loop for it in items for idx from 0
          do (let ((opts (getf it :options)))
               (push (list :it it :ans (getf it :answer)
                           :real (option-scores model bpe (prompt-of it) opts)
                           :noq (option-scores model bpe (prompt-of it :question nil) opts)
                           :bare (option-scores model bpe (prompt-of it :question nil :context nil) opts)
                           :mism (mapcar (lambda (o) (option-scores model bpe (prompt-of o) opts)) (mismatched-items items it idx mismatches))
                           :toks (mapcar (lambda (o) (asem-bpe:bpe-encode bpe (format nil " ~a" o))) opts))
                     recs)))
    (setf recs (nreverse recs))
    (labels ((sums (sc) (mapcar #'car sc)) (means (sc) (mapcar (lambda (c) (/ (car c) (max 1 (cdr c)))) sc))
             (r (rec k) (getf rec k))
             (hit (rec key fn) (= (argmax-of (funcall fn (r rec key))) (r rec :ans)))
             (acc (recs key fn) (frac (lambda (rec) (hit rec key fn)) recs))
             (margin (rec fn) (let ((v (funcall fn (r rec :real)))) (- (nth (r rec :ans) v) (loop for x in v for i from 0 unless (= i (r rec :ans)) maximize x))))
             (margin-of (rec key fn) (let ((v (funcall fn (r rec key)))) (- (nth (r rec :ans) v) (loop for x in v for i from 0 unless (= i (r rec :ans)) maximize x))))
             (m-of (v ans) (- (nth ans v) (loop for x in v for i from 0 unless (= i ans) maximize x)))
             (cal (rec) (mapcar #'- (sums (r rec :real)) (sums (r rec :bare))))     ; calibrated: log P(option | prompt) - log P(option | "Answer:")
             (gain (rec fn) (let ((ans (r rec :ans)))                                 ; does the right answer gain on the wrong ones with its own question?
                              (- (m-of (funcall fn (r rec :real)) ans) (mean-of (mapcar (lambda (m) (m-of (funcall fn m) ans)) (r rec :mism))))))
             (z-of (xs) (let ((se (/ (sd-of xs) (sqrt (max 1 (length xs)))))) (if (zerop se) 0l0 (/ (mean-of xs) se))))
             (lens (rec) (mapcar #'length (r rec :toks)))
             (logf (rec) (mapcar (lambda (ts) (mean-of (mapcar (lambda (tk) (aref logfreq tk)) ts))) (r rec :toks)))
             (pmi (rec fn) (let ((ans (r rec :ans))) (- (nth ans (funcall fn (r rec :real))) (mean-of (mapcar (lambda (m) (nth ans (funcall fn m))) (r rec :mism))))))
             (acc-mism (recs fn) (let ((hits 0) (tot 0))
                                   (dolist (rec recs (if (zerop tot) 0l0 (/ hits tot 1l0)))
                                     (dolist (m (r rec :mism)) (incf tot) (when (= (argmax-of (funcall fn m)) (r rec :ans)) (incf hits))))))
             (strict-pick (recs fn extreme) ; among items with a unique shortest/longest option: how often the model picks it
               (let ((hits 0) (tot 0))
                 (dolist (rec recs)
                   (let* ((ls (lens rec)) (target (if (eq extreme :shortest) (reduce #'min ls) (reduce #'max ls))))
                     (when (= 1 (count target ls))
                       (incf tot) (when (= (argmax-of (funcall fn (r rec :real))) (position target ls)) (incf hits)))))
                 (values (if (zerop tot) 0l0 (/ hits tot 1l0)) tot)))
             (section (recs)
               (let* ((pmis (mapcar (lambda (rec) (pmi rec #'sums)) recs)) (se (/ (sd-of pmis) (sqrt (max 1 (length recs))))))
                 (jobj "n" (length recs)
                       "accuracy_mean_logprob" (acc recs :real #'means) "accuracy_sum_logprob" (acc recs :real #'sums)
                       "accuracy_no_question" (acc recs :noq #'means) "accuracy_bare_prior" (acc recs :bare #'means)
                       "accuracy_mismatched_question" (acc-mism recs #'means)
                       "margin_mean" (mean-of (mapcar (lambda (rec) (margin rec #'means)) recs))
                       "accuracy_calibrated" (frac (lambda (rec) (= (argmax-of (cal rec)) (r rec :ans))) recs)
                       "pmi_sum_mean" (mean-of pmis) "pmi_sum_z" (if (zerop se) 0l0 (/ (mean-of pmis) se))
                       "margin_gain_sum_mean" (mean-of (mapcar (lambda (rec) (gain rec #'sums)) recs))
                       "margin_gain_sum_z" (z-of (mapcar (lambda (rec) (gain rec #'sums)) recs)))))
             (cats () (remove-duplicates (mapcar (lambda (rec) (getf (r rec :it) :category)) recs) :test #'equal :from-end t)))
      (let* ((all-len (loop for rec in recs append (mapcar #'float (lens rec)))) (all-len (mapcar (lambda (x) (float x 1l0)) all-len))
             (all-sum (loop for rec in recs append (sums (r rec :real)))) (all-mean (loop for rec in recs append (means (r rec :real))))
             (all-prior (loop for rec in recs append (means (r rec :bare)))) (all-logf (loop for rec in recs append (logf rec)))
             (margins (mapcar (lambda (rec) (margin rec #'means)) recs)) (margins-sum (mapcar (lambda (rec) (margin rec #'sums)) recs))
             (prior-margins (mapcar (lambda (rec) (margin-of rec :bare #'means)) recs))
             (pmis-sum (mapcar (lambda (rec) (pmi rec #'sums)) recs)) (pmis-mean (mapcar (lambda (rec) (pmi rec #'means)) recs))
             (gains-sum (mapcar (lambda (rec) (gain rec #'sums)) recs)) (gains-mean (mapcar (lambda (rec) (gain rec #'means)) recs))
             (se-sum (/ (sd-of pmis-sum) (sqrt (max 1 n)))) (z-sum (if (zerop se-sum) 0l0 (/ (mean-of pmis-sum) se-sum)))
             (tvs (loop for rec in recs append (let ((p (softmax-of (sums (r rec :real))))) (mapcar (lambda (m) (tv-distance p (softmax-of (sums m)))) (r rec :mism)))))
             (stable (loop for rec in recs append (let ((top (argmax-of (sums (r rec :real))))) (mapcar (lambda (m) (if (= top (argmax-of (sums m))) 1l0 0l0)) (r rec :mism)))))
             (acc-real (acc recs :real #'means)) (chance-hi (+ 0.25l0 (* 1.96l0 (sqrt (/ (* 0.25l0 0.75l0) (max 1 n))))))
             (freq-pick (frac (lambda (rec) (= (argmax-of (logf rec)) (r rec :ans))) recs)))
        (multiple-value-bind (short-sum n-short) (strict-pick recs #'sums :shortest)
          (multiple-value-bind (long-sum n-long) (strict-pick recs #'sums :longest)
            (jobj "questions" n "deterministic" :true "random_baseline" 0.25l0 "chance_95_upper" chance-hi
                  "accuracy" (jobj "with_question_mean_logprob" acc-real "with_question_sum_logprob" (acc recs :real #'sums)
                                   "no_question_context_kept" (acc recs :noq #'means) "bare_prior_no_input" (acc recs :bare #'means)
                                   "mismatched_question" (acc-mism recs #'means)
                                   "calibrated_sum_logprob_minus_no_input" (frac (lambda (rec) (= (argmax-of (cal rec)) (r rec :ans))) recs))
                  "length_bias" (jobj "corr_length_vs_sum_logprob" (pearson all-len all-sum) "corr_length_vs_mean_logprob" (pearson all-len all-mean)
                                      "mean_option_tokens_correct" (mean-of (mapcar (lambda (rec) (float (nth (r rec :ans) (lens rec)) 1l0)) recs))
                                      "mean_option_tokens_wrong" (mean-of (mapcar (lambda (rec) (mean-of (loop for l in (lens rec) for i from 0 unless (= i (r rec :ans)) collect (float l 1l0)))) recs))
                                      "items_with_unique_shortest_option" n-short "sum_logprob_picks_the_unique_shortest" short-sum
                                      "items_with_unique_longest_option" n-long "sum_logprob_picks_the_unique_longest" long-sum
                                      "expected_if_unbiased" 0.25l0)
                  "frequency_bias" (jobj "corr_option_token_frequency_vs_score" (pearson all-logf all-mean)
                                         "corr_option_token_frequency_vs_prior_score" (pearson all-logf all-prior)
                                         "accuracy_of_always_picking_the_most_frequent_option" freq-pick
                                         "model_picks_the_most_frequent_option" (frac (lambda (rec) (= (argmax-of (means (r rec :real))) (argmax-of (logf rec)))) recs))
                  "margin" (jobj "mean_logprob_mean" (mean-of margins) "mean_logprob_median" (median-of margins) "mean_logprob_fraction_positive" (frac #'plusp margins)
                                 "sum_logprob_mean" (mean-of margins-sum) "prior_only_mean" (mean-of prior-margins))
                  "question_sensitivity" (jobj "mismatched_questions_per_item" mismatches
                                               "pmi_sum_mean_nats" (mean-of pmis-sum) "pmi_sum_standard_error" se-sum "pmi_sum_z" z-sum
                                               "pmi_sum_fraction_positive" (frac #'plusp pmis-sum) "pmi_mean_logprob_mean" (mean-of pmis-mean)
                                               "mean_total_variation_of_option_distribution" (mean-of tvs)
                                               "top_option_unchanged_by_other_question" (mean-of stable)
                                               "margin_gain_sum_mean" (mean-of gains-sum) "margin_gain_sum_z" (z-of gains-sum)
                                               "margin_gain_mean_logprob_mean" (mean-of gains-mean) "margin_gain_mean_logprob_z" (z-of gains-mean)
                                               "margin_calibrated_mean" (mean-of (mapcar (lambda (rec) (m-of (cal rec) (r rec :ans))) recs)))
                  "by_category" (cons :obj (loop for c in (cats) collect (cons c (section (remove-if-not (lambda (rec) (equal (getf (r rec :it) :category) c)) recs)))))
                  "conditions_on_the_question" (jobj "answer_pmi_z_above_3_weak_question_type_effect_possible" (if (> z-sum 3) :true :false)
                                                     "margin_gain_z_above_3_sum_and_mean" (if (and (> (z-of gains-sum) 3) (> (z-of gains-mean) 3)) :true :false)
                                                     "accuracy_above_chance_band" (if (> acc-real chance-hi) :true :false)
                                                     "accuracy_exceeds_no_question_by_5_points" (if (> acc-real (+ 0.05l0 (acc recs :noq #'means))) :true :false)))))))))
