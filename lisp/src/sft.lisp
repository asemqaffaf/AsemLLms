;;;; Supervised fine-tuning for four-choice passage QA (Stage 2). FP64 source; src/sft32.lisp derives the FP32 version.
;;;;
;;;; One example = [end-of-text] + prompt tokens (asem-qagen:format-prompt, ending in "Answer:"); the target is the single token of
;;;; the answer letter (" A" .. " D") predicted at the LAST prompt position. Loss = cross-entropy of that one token over the full
;;;; vocabulary, nothing else (answer-only masking: FORWARD-ROWS / BACKWARD-ROWS compute the head only at that row, so prompt
;;;; tokens get no loss and no gradient through the output head). Sequences are right-padded to a multiple of 16 so a worker can
;;;; reuse a few activation buffers; with causal attention the padding cannot influence the answer row (tested in test-tf.lisp).
;;;; Prediction = argmax over the logits of the four letter tokens at the answer row ("raw accuracy": no calibration, no priors).
;;;;
;;;; Training: data-parallel SBCL threads, gradient accumulation, AdamW (decoupled decay on matrices only), warmup + cosine, clipping,
;;;; deterministic data order (a seeded permutation per epoch, so step S always sees the same examples), development-set
;;;; evaluation every EVAL-EVERY steps, best checkpoint by development accuracy (ties: lower development loss), and exact resume
;;;; from OUT/last.ckpt + Adam moments + state.
(defpackage :asem-sft
  (:use :cl :asem-tf :asem-util)
  (:export #:sft #:encode-items #:qa-predict #:qa-summary #:letter-token-ids #:ex-ids #:ex-target #:ex-answer #:ex-category #:ex-id
           #:make-ex #:shuffle-label-targets #:load-ex-model))
(in-package :asem-sft)

(defstruct ex id category subtype qframe ids target answer mode tgt opts)
;;; ids: ivec [eot prompt...]; answer: 0-3; target: the answer letter's token id.
;;; MODE :letter -> TGT = #(letter token): the model must name the letter (prediction: argmax over the four letter logits).
;;; MODE :text   -> TGT = tokens of " <answer text>": the model must copy the answer (prediction: the option whose tokens have
;;;                 the highest summed log-probability after the prompt). OPTS: the four options' continuation tokens.

(defparameter +max-target+ 8)

(defun letter-token-ids (bpe)
  (mapcar (lambda (l) (let ((ids (asem-bpe:bpe-encode bpe (format nil " ~a" l))))
                        (unless (= 1 (length ids)) (error "answer letter ~s is ~d tokens" l (length ids)))
                        (first ids)))
          asem-qagen:+letters+))

(defun ivec-of (list) (coerce list '(simple-array fixnum (*))))

(defun encode-items (items bpe ctx &key (prompt-fn #'asem-qagen:item-prompt) (answer-fn (lambda (it) (getf it :answer))) (mode :letter))
  "Vector of EX. Errors (never truncates) if a prompt plus its longest continuation does not fit CTX, or if a letter or an option
does not tokenize as the same tokens after the prompt as on its own."
  (let ((letters (letter-token-ids bpe)))
    (map 'vector
         (lambda (it)
           (let* ((prompt (funcall prompt-fn it)) (p (asem-bpe:bpe-encode bpe prompt)) (n (1+ (length p))) (ans (funcall answer-fn it))
                  (opts (mapcar (lambda (o) (asem-bpe:bpe-encode bpe (format nil " ~a" o))) (getf it :options))))
             (when (> n ctx) (error "item ~a needs ~d positions, the context is ~d: refusing to truncate" (getf it :id) n ctx))
             (let ((full (asem-bpe:bpe-encode bpe (format nil "~a ~a" prompt (nth ans asem-qagen:+letters+)))))
               (unless (equal full (append p (list (nth ans letters))))
                 (error "item ~a: the answer letter does not tokenize as one token after the prompt" (getf it :id))))
             (when (eq mode :text)
               (loop for o in (getf it :options) for ot in opts
                     do (unless (equal (asem-bpe:bpe-encode bpe (format nil "~a ~a" prompt o)) (append p ot))
                          (error "item ~a: option ~s tokenizes differently after the prompt" (getf it :id) o))
                        (when (> (length ot) +max-target+) (error "item ~a: option ~s is longer than ~d tokens" (getf it :id) o +max-target+))
                        (when (> (+ n (length ot) -1) ctx) (error "item ~a: prompt + option ~s exceed the context: refusing to truncate" (getf it :id) o))))
             (make-ex :id (getf it :id) :category (getf it :category) :subtype (getf it :subtype) :qframe (getf it :qframe)
                      :ids (ivec-of (cons 0 p)) :answer ans :target (nth ans letters) :mode mode
                      :opts (mapcar #'ivec-of opts)
                      :tgt (if (eq mode :text) (ivec-of (nth ans opts)) (ivec-of (list (nth ans letters)))))))
         items)))

(defun shuffle-label-targets (exs letters seed)
  "Control: copies of EXS whose answer is drawn uniformly at random (seeded), independent of the content."
  (let ((rng (sb-ext:seed-random-state seed)))
    (map 'vector (lambda (e) (let ((a (random 4 rng)) (c (copy-ex e)))
                               (setf (ex-answer c) a (ex-target c) (nth a letters)
                                     (ex-tgt c) (if (eq (ex-mode e) :text) (nth a (ex-opts e)) (ivec-of (list (nth a letters)))))
                               c))
         exs)))

;;; ------------------------------------------------------------------ per-worker buffers
(defun bucket-len (n ctx) (min ctx (* 16 (ceiling n 16))))

(defun acts-for (cache cfg n &optional aux)
  "Activation buffers for a padded length (up to +MAX-TARGET+ output rows, or a whole context with AUX), created on first use."
  (let* ((len (bucket-len n (config-ctx cfg))) (key (if aux (- len) len)))
    (or (gethash key cache) (setf (gethash key cache) (make-acts cfg len (if aux (+ len +max-target+) +max-target+))))))

(defun padded-x (ids len &optional tail)
  "IDS followed by TAIL (a continuation, all but its last token are inputs), zero-padded to LEN."
  (let ((x (make-array len :element-type 'fixnum :initial-element 0))) (replace x ids)
    (when tail (replace x tail :start1 (length ids) :end2 (1- (length tail))))
    x))

(defun continuation-rows (e tail)
  (let ((n (length (ex-ids e)))) (ivec-of (loop for k below (length tail) collect (+ n -1 k)))))

(defun letter-logits (acts letters)
  (mapcar (lambda (id) (float (aref (acts-logits acts) id) 1l0)) letters))

(defun rows-logprob (acts tail &optional (v (floor (length (acts-logits acts)) +max-target+)))
  "Summed log-probability of TAIL at the first forward-rows positions."
  (let ((s 0l0))
    (loop for k below (length tail) do (incf s (log (max (float (aref (acts-probs acts) (+ (* k v) (aref tail k))) 1l0) 1l-300))))
    s))

(defun score-continuation (model e tail cache)
  (let* ((cfg (model-config model)) (acts (acts-for cache cfg (+ (length (ex-ids e)) (length tail) -1)))
         (x (padded-x (ex-ids e) (asem-tf::acts-tlen acts) tail)))
    (forward-rows model x acts (continuation-rows e tail))
    (rows-logprob acts tail (config-vocab cfg))))

(defun run-example (model e cache letters &key grads (aux 0))
  "Forward (and backward when GRADS) one example on its training target. Returns (values answer-loss-sum predicted-index scores ntarget).
Letter mode: prediction = argmax of the four letter logits, scores = those logits. Text mode during training: the prediction is the
key when every target token is the argmax (greedy exact match), else -1; at evaluation use SCORE-OPTIONS.
AUX > 0 (training only) adds next-token losses on every prompt position, weighted AUX / (prompt positions), to the answer loss
(weight 1); the answer rows come first, so the returned loss and prediction are the answer's alone."
  (let* ((cfg (model-config model)) (tail (ex-tgt e)) (n (length (ex-ids e))) (auxp (and grads (plusp aux)))
         (acts (acts-for cache cfg (+ n (length tail) -1) auxp))
         (x (padded-x (ex-ids e) (asem-tf::acts-tlen acts) tail)) (arows (continuation-rows e tail)) (v (config-vocab cfg))
         (rows (if auxp (concatenate '(simple-array fixnum (*)) arows (ivec-of (loop for i from 0 below (1- n) collect i))) arows))
         (targets (if auxp (concatenate '(simple-array fixnum (*)) tail (subseq (ex-ids e) 1)) tail))
         (weights (when auxp (let ((w (new-dvec (length rows))) (wa (host-float (/ (float aux 1l0) (max 1 (1- n))))))
                               (dotimes (k (length rows) w) (setf (aref w k) (if (< k (length tail)) (host-float 1l0) wa)))))))
    (forward-rows model x acts rows)
    (let* ((ll (when (eq (ex-mode e) :letter) (letter-logits acts letters)))
           (pred (if ll
                     (position (reduce #'max ll) ll)
                     (if (loop for k below (length tail)
                               always (let ((o (* k v)) (best 0) (bv most-negative-double-float))
                                        (dotimes (j v) (when (> (aref (acts-logits acts) (+ o j)) bv) (setf bv (aref (acts-logits acts) (+ o j)) best j)))
                                        (= best (aref tail k))))
                         (ex-answer e) -1)))
           (answer-loss (- (rows-logprob acts tail v))))
      (when grads (backward-rows model x rows targets acts grads weights))
      (values (float answer-loss 1l0) pred ll (length tail)))))

(defun score-options (model e cache letters)
  "(values pred loss scores) for evaluation. Letter mode: one forward. Text mode: the four options' summed log-probabilities."
  (if (eq (ex-mode e) :letter)
      (multiple-value-bind (loss pred ll) (run-example model e cache letters) (values pred loss ll))
      (let* ((scores (mapcar (lambda (o) (score-continuation model e o cache)) (ex-opts e)))
             (pred (position (reduce #'max scores) scores)))
        (values pred (/ (- (nth (ex-answer e) scores)) (length (nth (ex-answer e) (ex-opts e)))) scores))))

;;; ------------------------------------------------------------------ evaluation
(defun qa-predict (model exs letters &key (workers 1) caches)
  "List of (pred loss scores) per example, in order. Forward passes only."
  (let* ((n (length exs)) (caches (or caches (loop repeat workers collect (make-hash-table))))
         (parts (asem-train::parallel-map
                 (lambda (job)
                   (destructuring-bind (cache idxs) job
                     (mapcar (lambda (i) (multiple-value-bind (pred loss ll) (score-options model (aref exs i) cache letters) (list i pred loss ll))) idxs)))
                 (loop for c in caches for idxs in (asem-train::shards (loop for i below n collect i) (length caches)) when idxs collect (list c idxs))))
         (out (make-array n)))
    (dolist (part parts) (dolist (r part) (setf (aref out (first r)) (rest r))))
    (coerce out 'list)))

(defun mean (xs) (if xs (/ (reduce #'+ xs) (length xs)) 0l0))

(defun qa-summary (exs preds)
  "Accuracy, mean loss, 4-way letter loss, accuracy by category, predicted-letter distribution."
  (let* ((n (length exs)) (hits 0) (cats (make-hash-table :test 'equal)) (dist (make-array 4 :initial-element 0)) (l4 nil))
    (loop for e across exs for (pred loss ll) in preds
          do (let ((ok (= pred (ex-answer e))) (c (gethash (ex-category e) cats (list 0 0))))
               (when ok (incf hits)) (incf (aref dist pred))
               (incf (first c) (if ok 1 0)) (incf (second c)) (setf (gethash (ex-category e) cats) c)
               (let* ((m (reduce #'max ll)) (z (+ m (log (reduce #'+ (mapcar (lambda (x) (exp (- x m))) ll))))))
                 (push (- z (nth (ex-answer e) ll)) l4))))
    (list :n n :correct hits :accuracy (/ hits (max 1 n) 1l0) :loss (mean (mapcar #'second preds)) :letter-loss (mean l4)
          :by-category (sort (loop for k being the hash-keys of cats using (hash-value c) collect (cons k (/ (first c) (second c) 1l0))) #'string< :key #'car)
          :pred-dist (coerce dist 'list))))

;;; ------------------------------------------------------------------ checkpoints of the optimizer state (same file format)
(defun save-moments (path cfg vecs &rest header)
  (apply #'asem-pretrain:save-checkpoint path (asem-tf::make-model :config cfg :params vecs) header))

(defun load-moments (path) (model-params (asem-pretrain:load-checkpoint path)))

(defun load-ex-model (path) (asem-pretrain:load-checkpoint path))

(defun write-state (path plist)
  (with-open-file (o path :direction :output :if-exists :supersede) (with-standard-io-syntax (let ((*print-readably* nil)) (prin1 plist o)))))
(defun read-state (path) (with-open-file (in path) (with-standard-io-syntax (let ((*read-eval* nil)) (read in)))))

;;; ------------------------------------------------------------------ training
(defun epoch-perm (n seed epoch)
  (let ((v (make-array n)) (rng (sb-ext:seed-random-state (+ (* 1000003 seed) epoch))))
    (dotimes (i n) (setf (aref v i) i))
    (loop for i from (1- n) downto 1 do (rotatef (aref v i) (aref v (random (1+ i) rng))))
    v))

(defun step-examples (step per-step n seed perms)
  "Example indices for optimizer STEP (1-based): a contiguous slice of the concatenated epoch permutations."
  (loop for g from (* (1- step) per-step) below (* step per-step)
        collect (multiple-value-bind (ep k) (floor g n)
                  (aref (or (gethash ep perms) (setf (gethash ep perms) (epoch-perm n seed ep))) k))))

(defun matrix-param-p (name) (not (or (search ".g" name) (string= name "gf"))))

(defun cat-plist (by-category prefix)
  (loop for (c . a) in by-category append (list (intern (string-upcase (format nil "~a~a" prefix c)) :keyword) a)))

(defun sft (out-dir train-exs dev-exs letters &key init-model (steps 1000) (batch 32) (accum 1) (workers 8) (lr 3d-4) (min-lr-frac 0.1d0)
                                                (warmup 50) (clip 1d0) (wd 0d0) (seed 1) (eval-every 100) (log-every 10) (resume nil)
                                                (train-probe 500) (save-every 0) (header nil) (stop-at nil) (aux-lm 0))
  "Fine-tune INIT-MODEL on TRAIN-EXS. Writes OUT-DIR/log.jsonl, best.ckpt (best development accuracy), last.ckpt, last-m1.ckpt,
last-m2.ckpt and state.sexp (for RESUME). HEADER: extra plist logged in the start event. STOP-AT N ends the run after step N
(saving everything needed to resume) as if it had been interrupted; the schedule still runs to STEPS. Returns the final state plist."
  (ensure-directories-exist (concatenate 'string out-dir "/"))
  (let* ((state-path (format nil "~a/state.sexp" out-dir))
         (resuming (and resume (probe-file state-path)))
         (st (if resuming (read-state state-path) (list :step 0 :best-acc -1l0 :best-loss 1l30 :best-step 0 :peak-mb 0l0 :seconds 0l0)))
         (model (if resuming (asem-pretrain:load-checkpoint (format nil "~a/last.ckpt" out-dir)) init-model))
         (cfg (model-config model)) (params (model-params model)) (names (coerce (param-names cfg) 'simple-vector))
         (m1 (if resuming (load-moments (format nil "~a/last-m1.ckpt" out-dir)) (map 'simple-vector (lambda (p) (new-dvec (length p))) params)))
         (m2 (if resuming (load-moments (format nil "~a/last-m2.ckpt" out-dir)) (map 'simple-vector (lambda (p) (new-dvec (length p))) params)))
         (ws (loop repeat workers collect (list (make-hash-table) (make-grads cfg))))
         (eval-caches (mapcar #'first ws))
         (sum-grads (make-grads cfg)) (n (length train-exs)) (per-step (* batch accum)) (perms (make-hash-table))
         (probe (subseq train-exs 0 (min train-probe n)))
         (b1 0.9l0) (b2 0.95l0) (step-targets 0) (log-targets 0) (log-loss 0l0) (log-hits 0) (log-n 0) (log-tokens 0)
         (t0 (get-internal-real-time)) (t-last t0) (secs0 (getf st :seconds))
         (log (open (format nil "~a/log.jsonl" out-dir) :direction :output :if-exists (if resuming :append :supersede)
                                                         :if-does-not-exist :create :external-format :utf-8)))
    (flet ((ev (plist) (asem-pretrain::log-event log plist))
           (elapsed () (+ secs0 (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1l0))))
      (unwind-protect
           (labels ((evaluate (step)
                      (let* ((dev (qa-summary dev-exs (qa-predict model dev-exs letters :caches eval-caches)))
                             (tr (qa-summary probe (qa-predict model probe letters :caches eval-caches)))
                             (rss (asem-pretrain:rss-mb)))
                        (setf (getf st :peak-mb) (max (getf st :peak-mb) rss))
                        (ev (append (list :event "eval" :step step :epoch (/ (* step per-step) n 1l0)
                                          :dev-acc (getf dev :accuracy) :dev-correct (getf dev :correct) :dev-n (getf dev :n)
                                          :dev-loss (getf dev :loss) :dev-letter-loss (getf dev :letter-loss)
                                          :train-probe-acc (getf tr :accuracy) :train-probe-loss (getf tr :loss)
                                          :pred-a (first (getf dev :pred-dist)) :pred-b (second (getf dev :pred-dist))
                                          :pred-c (third (getf dev :pred-dist)) :pred-d (fourth (getf dev :pred-dist))
                                          :rss-mb rss :peak-rss-mb (getf st :peak-mb) :seconds (elapsed))
                                    (cat-plist (getf dev :by-category) "dev-")))
                        (when (or (> (getf dev :accuracy) (getf st :best-acc))
                                  (and (= (getf dev :accuracy) (getf st :best-acc)) (< (getf dev :loss) (getf st :best-loss))))
                          (setf (getf st :best-acc) (getf dev :accuracy) (getf st :best-loss) (getf dev :loss) (getf st :best-step) step)
                          (asem-pretrain:save-checkpoint (format nil "~a/best.ckpt" out-dir) model :step step :dev-acc (getf dev :accuracy)
                                                         :dev-loss (getf dev :loss)))
                        dev))
                    (save-last (step)
                      (asem-pretrain:save-checkpoint (format nil "~a/last.ckpt" out-dir) model :step step)
                      (save-moments (format nil "~a/last-m1.ckpt" out-dir) cfg m1 :kind "adam-m1" :step step)
                      (save-moments (format nil "~a/last-m2.ckpt" out-dir) cfg m2 :kind "adam-m2" :step step)
                      (setf (getf st :step) step (getf st :seconds) (elapsed))
                      (write-state state-path st)))
             (ev (append (list :event (if resuming "resume" "start") :from-step (getf st :step) :steps steps :batch batch :accum accum
                               :workers workers :aux-lm aux-lm :lr lr :min-lr-frac min-lr-frac :warmup warmup :clip clip :wd wd :seed seed
                               :train-n n :dev-n (length dev-exs) :params (param-count model)
                               :precision (if (= (elt-bytes) 8) "f64" "f32"))
                         header))
             (when (zerop (getf st :step)) (evaluate 0))
             (loop for step from (1+ (getf st :step)) to (min steps (or stop-at steps))
                   do (zero-grads sum-grads) (setf step-targets 0)
                      (dotimes (a accum)
                        (let* ((idxs (step-examples step per-step n seed perms))
                               (micro (subseq idxs (* a batch) (* (1+ a) batch)))
                               (results
                                 (asem-train::parallel-map
                                  (lambda (job)
                                    (destructuring-bind (w is) job
                                      (destructuring-bind (cache grads) w
                                        (zero-grads grads)
                                        (let ((ls 0l0) (hits 0) (toks 0) (ntg 0))
                                          (dolist (i is)
                                            (let ((e (aref train-exs i)))
                                              (multiple-value-bind (loss pred ll nt) (run-example model e cache letters :grads grads :aux aux-lm)
                                                (declare (ignore ll))
                                                (incf ls loss) (incf toks (length (ex-ids e))) (incf ntg nt)
                                                (when (= pred (ex-answer e)) (incf hits)))))
                                          (list ls hits toks ntg)))))
                                  (loop for w in ws for is in (asem-train::shards micro workers) when is collect (list w is)))))
                          (loop for w in ws do (loop for g across (second w) for sg across sum-grads do (asem-train::add-into sg g)))
                          (dolist (r results) (incf log-loss (first r)) (incf log-hits (second r)) (incf log-tokens (third r)) (incf step-targets (fourth r)) (incf log-targets (fourth r)))
                          (incf log-n batch)))
                      (let ((inv (host-float (/ 1l0 (max 1 step-targets)))) (nrm 0l0))
                        (loop for sg across sum-grads do (asem-train::scale-into sg inv))
                        (setf nrm (asem-train::grad-norm sum-grads))
                        (when (or (/= nrm nrm) (> nrm 1l30)) (error "non-finite gradient norm at step ~d" step))
                        (when (> nrm clip) (loop for sg across sum-grads do (asem-train::scale-into sg (host-float (/ clip nrm)))))
                        (let* ((cur-lr (asem-pretrain::lr-at (1- step) steps lr warmup min-lr-frac))
                               (lrt (host-float (* cur-lr (/ (sqrt (- 1l0 (expt b2 step))) (- 1l0 (expt b1 step))))))
                               (b1f (host-float b1)) (b2f (host-float b2)) (c1 (host-float (- 1l0 b1))) (c2 (host-float (- 1l0 b2))))
                          (loop for p across params for g across sum-grads for a across m1 for b across m2 for name across names
                                do (let ((p p) (g g) (a a) (b b) (decay (host-float (if (matrix-param-p name) (- 1l0 (* cur-lr wd)) 1l0))))
                                     (declare (type dvec p g a b) (type double-float lrt b1f b2f c1 c2 decay) (optimize (speed 3) (safety 0)))
                                     (dotimes (i (length p))
                                       (let ((gi (aref g i)))
                                         (setf (aref a i) (+ (* b1f (aref a i)) (* c1 gi)))
                                         (setf (aref b i) (+ (* b2f (aref b i)) (* c2 gi gi)))
                                         (setf (aref p i) (- (* decay (aref p i)) (/ (* lrt (aref a i)) (+ (sqrt (aref b i)) 1d-8))))))))
                          (when (and (plusp log-every) (zerop (mod step log-every)))
                            (let* ((now (get-internal-real-time)) (dt (/ (- now t-last) internal-time-units-per-second 1l0)))
                              (ev (list :event "train" :step step :epoch (/ (* step per-step) n 1l0) :train-loss (/ log-loss (max 1 log-targets))
                                        :train-acc (/ log-hits (max 1 log-n) 1l0) :grad-norm nrm :lr cur-lr
                                        :tokens-per-sec (if (plusp dt) (/ log-tokens dt) 0l0) :examples-per-sec (if (plusp dt) (/ log-n dt) 0l0)))
                              (setf t-last now log-loss 0l0 log-hits 0 log-n 0 log-tokens 0 log-targets 0)))))
                      (when (and (plusp save-every) (zerop (mod step save-every)))
                        (asem-pretrain:save-checkpoint (format nil "~a/step-~5,'0d.ckpt" out-dir step) model :step step))
                      (when (or (zerop (mod step eval-every)) (= step steps) (eql step stop-at))
                        (evaluate step)
                        (save-last step)))
             (setf (getf st :seconds) (elapsed))
             (write-state state-path st)
             (ev (list :event "done" :steps steps :best-dev-acc (getf st :best-acc) :best-step (getf st :best-step)
                       :best-dev-loss (getf st :best-loss) :seconds (getf st :seconds) :peak-rss-mb (getf st :peak-mb)
                       :best-sha256 (asem-sha256:sha256-file-hex (format nil "~a/best.ckpt" out-dir))
                       :last-sha256 (asem-sha256:sha256-file-hex (format nil "~a/last.ckpt" out-dir))))
             st)
        (close log)))))

;;; ------------------------------------------------------------------ evaluation with controls (bin/qa-eval.lisp)
(defun wilson (k n &optional (z 1.96l0))
  "95% Wilson score interval for K successes in N trials: (lo hi)."
  (if (zerop n) (list 0l0 1l0)
      (let* ((p (/ k n 1l0)) (z2 (* z z)) (den (+ 1l0 (/ z2 n))) (c (/ (+ p (/ z2 (* 2 n))) den))
             (h (/ (* z (sqrt (+ (/ (* p (- 1 p)) n) (/ z2 (* 4 n n))))) den)))
        (list (- c h) (+ c h)))))

(defun mismatched-question (items it idx)
  "The question of another item of the same category (the next one in file order, cyclically): its entities are not in IT's passage."
  (let* ((same (remove-if-not (lambda (o) (equal (getf o :category) (getf it :category))) items)) (len (length same))
         (k (position it same)))
    (declare (ignore idx))
    (getf (nth (mod (+ k 1 (floor len 2)) len) same) :question)))

(defun reverse-options (it)
  (let ((c (copy-list it)))
    (setf (getf c :options) (reverse (getf it :options)) (getf c :answer) (- 3 (getf it :answer))
          (getf c :cf-answer) (- 3 (getf it :cf-answer)))
    c))

(defun summary-obj (exs preds)
  (let* ((s (qa-summary exs preds)) (n (getf s :n)) (k (getf s :correct)) (ci (wilson k n)) (sub (make-hash-table :test 'equal))
         (qf (make-hash-table :test 'equal)))
    (loop for e across exs for (pred) in preds
          do (dolist (pair (list (cons sub (format nil "~a/~a" (ex-category e) (ex-subtype e))) (cons qf (format nil "~a | ~a" (ex-category e) (ex-qframe e)))))
               (let ((c (gethash (cdr pair) (car pair) (list 0 0))))
                 (incf (first c) (if (= pred (ex-answer e)) 1 0)) (incf (second c)) (setf (gethash (cdr pair) (car pair)) c))))
    (flet ((tab (h) (cons :obj (loop for k being the hash-keys of h using (hash-value c)
                                     collect (cons k (asem-pretrain::jobj "accuracy" (/ (first c) (second c) 1l0) "n" (second c)))))))
      (asem-pretrain::jobj "n" n "correct" k "accuracy" (getf s :accuracy) "ci95_low" (first ci) "ci95_high" (second ci)
                           "mean_loss" (getf s :loss) "mean_letter_loss" (getf s :letter-loss)
                           "by_category" (cons :obj (loop for (c . a) in (getf s :by-category)
                                                          collect (cons c (let* ((m (count c exs :key #'ex-category :test #'equal))
                                                                                 (h (round (* a m))) (ci (wilson h m)))
                                                                            (asem-pretrain::jobj "accuracy" a "n" m "correct" h "ci95_low" (first ci) "ci95_high" (second ci))))))
                           "by_subtype" (tab sub) "by_question_frame" (tab qf)
                           "predicted_letter_counts" (asem-pretrain::jobj "A" (first (getf s :pred-dist)) "B" (second (getf s :pred-dist))
                                                                          "C" (third (getf s :pred-dist)) "D" (fourth (getf s :pred-dist)))))))

(defun qa-evaluate-conditions (model bpe items &key (workers 8) (mode :letter))
  "Raw letter accuracy on ITEMS under the real prompt and under controls. JSON-ready :obj."
  (let* ((ctx (config-ctx (model-config model))) (letters (letter-token-ids bpe))
         (run (lambda (its &key (prompt-fn #'asem-qagen:item-prompt) (answer-fn (lambda (it) (getf it :answer))))
                (let ((exs (encode-items its bpe ctx :prompt-fn prompt-fn :answer-fn answer-fn :mode mode)))
                  (values exs (qa-predict model exs letters :workers workers)))))
         (results nil))
    (multiple-value-bind (exs-full p-full) (funcall run items)
      (multiple-value-bind (exs-cf p-cf) (funcall run items :prompt-fn (lambda (it) (asem-qagen:item-prompt it :question-text (getf it :cf-question)))
                                                     :answer-fn (lambda (it) (getf it :cf-answer)))
        (push (cons "real_prompt" (summary-obj exs-full p-full)) results)
        (push (cons "counterfactual_question_same_passage" (summary-obj exs-cf p-cf)) results)
        (let ((both (loop for e across exs-full for c across exs-cf for (pf) in p-full for (pc) in p-cf
                          count (and (= pf (ex-answer e)) (= pc (ex-answer c)))))
              (moved (loop for (pf) in p-full for (pc) in p-cf count (/= pf pc))))
          (push (cons "counterfactual_consistency"
                      (asem-pretrain::jobj "both_correct" both "fraction_both_correct" (/ both (length items) 1l0)
                                           "prediction_changed_with_the_question" (/ moved (length items) 1l0)
                                           "note" "same passage and options, the question asks about another entity; the answer moves to cf_answer"))
                results))
        (multiple-value-bind (e p) (funcall run items :prompt-fn (lambda (it) (asem-qagen:item-prompt it :question nil)))
          (push (cons "no_question" (summary-obj e p)) results))
        (multiple-value-bind (e p) (funcall run items :prompt-fn (lambda (it) (asem-qagen:item-prompt it :passage nil)))
          (push (cons "no_passage" (summary-obj e p)) results))
        (multiple-value-bind (e p) (funcall run items :prompt-fn (lambda (it) (asem-qagen:item-prompt it :question-text (mismatched-question items it 0))))
          (push (cons "mismatched_question_from_another_item" (summary-obj e p)) results))
        (multiple-value-bind (e p) (funcall run (mapcar #'reverse-options items))
          (push (cons "options_in_reverse_order" (summary-obj e p)) results)
          (let ((same (loop for (pf) in p-full for (pr) in p for it in items count (= (- 3 pr) pf))))
            (push (cons "reverse_order_consistency" (asem-pretrain::jobj "same_option_chosen" (/ same (length items) 1l0))) results)))
        (cons :obj (nreverse results))))))

(export '(qa-evaluate-conditions wilson))
