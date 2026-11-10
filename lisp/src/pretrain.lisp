;;;; Pretraining on BPE token shards (the output of bin/build-data.lisp): data-parallel SBCL threads, Adam with warmup and
;;;; cosine decay, validation loss on fixed windows, a multiple-choice evaluation scored by answer log-likelihood,
;;;; JSONL logging (loss, tokens/s, peak memory, checkpoint hash) and binary float32 checkpoints.
;;;; This file is the FP64 source; src/pretrain32.lisp derives the single-float version (see derive.lisp).
;;;; Host-side scalars use long-float / 0l0 (always double); everything that touches model tensors uses double-float.
(defpackage :asem-pretrain
  (:use :cl :asem-tf :asem-util)
  (:export #:pretrain #:load-checkpoint #:save-checkpoint #:read-eval-set #:score-eval #:sample-text #:load-split-tokens
           #:preset-config #:rss-mb #:evaluate-checkpoint))
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
                                       (sample-prompt "The") (config nil))
  "Train from scratch on the shards in DATA-DIR; checkpoints and a JSONL log go to OUT-DIR. Returns a plist of final numbers."
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
         (b1 0.9l0) (b2 0.95l0)
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
                      (let ((inv (host-float (/ 1l0 batch))))
                        (loop for sg across sum-grads do (asem-train::scale-into sg inv))
                        (let ((nrm (asem-train::grad-norm sum-grads)))
                          (when (or (/= nrm nrm) (> nrm 1l30)) (error "non-finite gradient norm at step ~d: stopping instead of training on garbage" step))
                          (when (> nrm clip)
                            (loop for sg across sum-grads do (asem-train::scale-into sg (host-float (/ clip nrm)))))))
                      (let* ((lrt (host-float (let ((l (lr-at (1- step) steps lr warmup min-lr-frac)))
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

(defun evaluate-checkpoint (data-dir ckpt-path &key eval-file)
  "Loss on the full validation and test splits plus the multiple-choice scores. The test split is only ever used here."
  (let* ((model (load-checkpoint ckpt-path))
         (manifest (asem-data:load-manifest data-dir))
         (bpe (asem-bpe:load-bpe (format nil "~a/~a" data-dir (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
         (val (load-split-tokens data-dir "val")) (test (load-split-tokens data-dir "test"))
         (items (and eval-file (read-eval-set eval-file))) (n (length items))
         (ans (and items (score-eval model bpe items))))
    (multiple-value-bind (vl vn) (split-loss model val)
      (multiple-value-bind (tl tn) (split-loss model test)
        (append (list :checkpoint ckpt-path :params (param-count model) :val-loss vl :val-ppl (exp vl) :val-tokens vn
                      :test-loss tl :test-ppl (exp tl) :test-tokens tn)
                (when ans
                  (list :answer-accuracy (getf ans :accuracy) :answer-accuracy-sum (getf ans :accuracy-sum) :answer-margin (getf ans :margin)
                        :questions n :random-baseline 0.25l0
                        :chance-95-upper (+ 0.25l0 (* 1.96l0 (sqrt (/ (* 0.25l0 0.75l0) n))))
                        :acc-fact (or (cdr (assoc "fact" (getf ans :by-category) :test #'equal)) 0)
                        :acc-reading (or (cdr (assoc "reading" (getf ans :by-category) :test #'equal)) 0)
                        :acc-reasoning (or (cdr (assoc "reasoning" (getf ans :by-category) :test #'equal)) 0))))))))
