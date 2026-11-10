;;;; Word tokenizer, data-parallel trainer (SBCL threads), Adam, evaluation, generation, save/load.
;;;; Mirrors tinyllm_tok.erl / tinyllm_train.erl / tinyllm.erl.
(defpackage :asem-train
  (:use :cl :asem-tf :asem-util)
  (:export #:tokenize #:build-vocab #:train #:generate #:save-model #:load-model #:encode #:window-loss
           #:lm #:lm-model #:lm-vocab #:lm-stats #:default-options))
(in-package :asem-train)

;;; ------------------------------------------------------------------ tokenizer
(defun alnum-p (c) (or (char<= #\a c #\z) (char<= #\A c #\Z) (char<= #\0 c #\9)))
(defun word-char-p (c) (or (alnum-p c) (char= c #\')))
(defun regex-space-p (c) (member c '(#\Space #\Tab #\Newline #\Return #\Page #\Vt)))

(defun tokenize (text)
  "Words (with their leading space), single punctuation marks (with leading space) and whitespace
characters, so (apply #'concatenate 'string tokens) = text. Same rules as the Erlang regex
\" ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']|\\n|\\s\"."
  (let ((n (length text)) (i 0) (out nil))
    (loop while (< i n)
          do (let* ((c (char text i)) (j (if (char= c #\Space) (1+ i) i)))
               (cond ((and (< j n) (word-char-p (char text j)))
                      (let ((k (or (position-if-not #'word-char-p text :start j) n)))
                        (push (subseq text i k) out) (setf i k)))
                     ((and (< j n) (not (regex-space-p (char text j))) (not (word-char-p (char text j))))
                      (push (subseq text i (1+ j)) out) (setf i (1+ j)))
                     (t (push (string c) out) (incf i)))))
    (nreverse out)))

(defun build-vocab (tokens max-vocab min-count)
  "<unk> first, then the most frequent tokens seen at least MIN-COUNT times (ties: alphabetical)."
  (let ((counts (make-hash-table :test 'equal)))
    (dolist (tk tokens) (incf (gethash tk counts 0)))
    (let ((items (loop for k being the hash-keys of counts using (hash-value v)
                       when (>= v min-count) collect (cons k v))))
      (setf items (sort items (lambda (a b) (or (> (cdr a) (cdr b))
                                                (and (= (cdr a) (cdr b)) (string< (car a) (car b)))))))
      (cons "<unk>" (mapcar #'car (subseq items 0 (min (length items) (1- max-vocab))))))))

(defun vocab-map (vocab)
  (let ((h (make-hash-table :test 'equal)))
    (loop for w in vocab for i from 0 do (setf (gethash w h) i))
    h))

(defun encode (tokens vmap)
  (let ((v (make-array (length tokens) :element-type 'fixnum)))
    (loop for tk in tokens for i from 0 do (setf (aref v i) (gethash tk vmap 0)))
    v))

(defstruct lm model vocab stats)

;;; ------------------------------------------------------------------ options
(defun default-options ()
  (list :ctx 16 :dim 32 :heads 2 :layers 2 :ff 64 :batch 8 :steps 300 :lr 0.003d0 :seed 1
        :eval-every 100 :val-examples 40 :workers 1 :clip 5d0 :max-vocab 2000 :min-count 2
        :verbose t))

(defun opt (opts key) (getf opts key))

;;; ------------------------------------------------------------------ helpers
(defun slice (ids s len) (make-array len :element-type 'fixnum :displaced-to ids :displaced-index-offset s))

(defun window (ids s ctx)
  "(values x y): ids[s..s+ctx) and ids[s+1..s+ctx+1) as fresh simple fixnum vectors."
  (let ((x (make-array ctx :element-type 'fixnum)) (y (make-array ctx :element-type 'fixnum)))
    (dotimes (i ctx) (setf (aref x i) (aref ids (+ s i)) (aref y i) (aref ids (+ s i 1))))
    (values x y)))

(defun window-loss (model ids starts)
  "Mean per-token loss over windows beginning at STARTS (single thread)."
  (let* ((cfg (model-config model)) (ctx (config-ctx cfg)) (acts (make-acts cfg ctx)) (sum 0d0))
    (dolist (s starts)
      (multiple-value-bind (x y) (window ids s ctx)
        (forward model x acts)
        (incf sum (loss-of-acts acts y))))
    (/ sum (max 1 (length starts)))))

(defun shards (list w)
  (let ((out (make-array w :initial-element nil)))
    (loop for x in list for i from 0 do (push x (aref out (mod i w))))
    (map 'list #'nreverse out)))

(defun parallel-map (fn items)
  "Run FN on every item in its own thread (or inline for one item); results in order."
  (if (null (rest items))
      (list (funcall fn (first items)))
      (mapcar #'sb-thread:join-thread
              (mapcar (lambda (it) (sb-thread:make-thread (lambda () (funcall fn it)))) items))))

(defun grad-norm (grads)
  (let ((s 0d0))
    (declare (type double-float s))
    (loop for g across grads
          do (let ((g g))
               (declare (type dvec g) (optimize (speed 3) (safety 0)))
               (dotimes (i (length g)) (incf s (* (aref g i) (aref g i))))))
    (sqrt s)))

(defun add-into (dst src)
  (declare (type dvec dst src) (optimize (speed 3) (safety 0)))
  (dotimes (i (length dst)) (incf (aref dst i) (aref src i))))

(defun scale-into (dst k)
  (declare (type dvec dst) (type double-float k) (optimize (speed 3) (safety 0)))
  (dotimes (i (length dst)) (setf (aref dst i) (* k (aref dst i)))))

;;; ------------------------------------------------------------------ training
(defun baselines (ids train-end val-starts ctx v)
  "Unigram/bigram losses on the same validation targets (add-one / add-0.1 smoothing)."
  (let ((uni (make-hash-table)) (bi (make-hash-table :test 'equal)))
    (loop for i below train-end do (incf (gethash (aref ids i) uni 0))
          when (> i 0) do (incf (gethash (cons (aref ids (1- i)) (aref ids i)) bi 0)))
    (let ((ul 0d0) (bl 0d0) (n 0))
      (dolist (s val-starts)
        (dotimes (i ctx)
          (let ((p (aref ids (+ s i))) (tgt (aref ids (+ s i 1))))
            (incf n)
            (decf ul (log (/ (+ (gethash tgt uni 0) 1d0) (+ train-end v))))
            (decf bl (log (/ (+ (gethash (cons p tgt) bi 0) 0.1d0) (+ (gethash p uni 0) (* 0.1d0 v))))))))
      (list :unigram (/ ul n) :bigram (/ bl n)))))

(defun train (text &rest user-opts)
  "Train on TEXT; returns an LM. Options as in DEFAULT-OPTIONS (plist)."
  (let* ((o (append user-opts (default-options)))
         (ctx (opt o :ctx)) (steps (opt o :steps)) (batch (opt o :batch)) (workers (opt o :workers))
         (state (sb-ext:seed-random-state (opt o :seed)))
         (tokens (tokenize text))
         (vocab (build-vocab tokens (opt o :max-vocab) (opt o :min-count)))
         (vmap (vocab-map vocab)) (v (length vocab))
         (ids (encode tokens vmap)) (size (length ids))
         (train-end (floor (* size 9) 10))
         (cfg (make-config :vocab v :ctx ctx :dim (opt o :dim) :heads (opt o :heads) :layers (opt o :layers)
                           :ff (opt o :ff)))
         (model (init-model cfg (opt o :seed)))
         (params (model-params model))
         (val-lo train-end) (val-hi (- size ctx 1))
         (t0 (get-internal-real-time)))
    (unless (and (> (- train-end ctx 1) 10) (> (- val-hi val-lo) 10)) (error "corpus too small for context ~d" ctx))
    (let* ((val-starts (sort (remove-duplicates
                              (loop repeat (opt o :val-examples) collect (+ val-lo (random (- val-hi val-lo) state))))
                             #'<))
           (base (baselines ids train-end val-starts ctx v))
           (ws (loop repeat workers collect (list (make-acts cfg ctx) (make-grads cfg))))
           (m1 (map 'simple-vector (lambda (p) (new-dvec (length p))) params))
           (m2 (map 'simple-vector (lambda (p) (new-dvec (length p))) params))
           (sum-grads (make-grads cfg))
           (best-val most-positive-double-float) (best-params (copy-model-params model)) (acc-loss 0d0) (acc-n 0)
           (b1 0.9d0) (b2 0.999d0))
      (when (opt o :verbose)
        (format t "~d tokens, vocab ~d, ~d params, ~d workers; baselines val loss: unigram ~,3f, bigram ~,3f~%"
                size v (param-count model) workers (getf base :unigram) (getf base :bigram)))
      (loop for step from 1 to steps
            do (let* ((starts (loop repeat batch collect (random (- train-end ctx 1) state)))
                      (results
                        (parallel-map
                         (lambda (job)
                           (destructuring-bind (w sts) job
                             (destructuring-bind (acts grads) w
                               (zero-grads grads)
                               (let ((ls 0d0))
                                 (dolist (s sts)
                                   (multiple-value-bind (x y) (window ids s ctx)
                                     (forward model x acts)
                                     (incf ls (backward model x y acts grads))))
                                 ls))))
                         (loop for w in ws for sts in (shards starts workers) when sts collect (list w sts)))))
                 (zero-grads sum-grads)
                 (loop for w in ws
                       do (loop for g across (second w) for sg across sum-grads do (add-into sg g)))
                 (incf acc-loss (reduce #'+ results)) (incf acc-n batch)
                 ;; mean gradient, clipped by global norm
                 (let* ((inv (/ 1d0 batch)))
                   (loop for sg across sum-grads do (scale-into sg inv))
                   (let ((nrm (grad-norm sum-grads)) (clip (opt o :clip)))
                     (when (> nrm clip)
                       (loop for sg across sum-grads do (scale-into sg (/ clip nrm))))))
                 ;; Adam with linearly decayed learning rate
                 (let* ((lr (* (opt o :lr) (- 1d0 (* 0.9d0 (/ (1- step) (max 1 (1- steps)))))))
                        (lrt (* lr (/ (sqrt (- 1d0 (expt b2 step))) (- 1d0 (expt b1 step))))))
                   (loop for p across params for g across sum-grads for a across m1 for b across m2
                         do (let ((p p) (g g) (a a) (b b))
                              (declare (type dvec p g a b) (optimize (speed 3) (safety 0)))
                              (dotimes (i (length p))
                                (let ((gi (aref g i)))
                                  (setf (aref a i) (+ (* b1 (aref a i)) (* (- 1d0 b1) gi)))
                                  (setf (aref b i) (+ (* b2 (aref b i)) (* (- 1d0 b2) gi gi)))
                                  (decf (aref p i) (/ (* lrt (aref a i)) (+ (sqrt (aref b i)) 1d-8))))))))
                 (when (or (zerop (mod step (opt o :eval-every))) (= step steps))
                   (let ((val (window-loss model ids val-starts)))
                     (when (< val best-val) (setf best-val val best-params (copy-model-params model)))
                     (when (opt o :verbose)
                       (format t "step ~5d/~d  train ~,3f  val ~,3f  best ~,3f  ~,1f ex/s~%" step steps
                               (/ acc-loss acc-n) val best-val
                               (/ (* step batch) (max 1d-3 (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0)))))
                     (setf acc-loss 0d0 acc-n 0)))))
      (let ((secs (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0)))
        (setf (model-params model) best-params)
        (make-lm :model model :vocab vocab
                 :stats (append base (list :val-loss best-val :steps steps :seconds secs :tokens size
                                           :examples-per-sec (/ (* steps batch) secs)
                                           :tokens-per-sec (/ (* steps batch ctx) secs))))))))

;;; ------------------------------------------------------------------ generation
(defun sample-index (probs state)
  (let ((r (random 1d0 state)) (acc 0d0))
    (dotimes (i (length probs) (1- (length probs)))
      (incf acc (aref probs i))
      (when (< r acc) (return i)))))

(defun generate (lm prompt n &key (temperature 0.8d0) (seed 1))
  "PROMPT plus N sampled tokens (never <unk>)."
  (let* ((model (lm-model lm)) (cfg (model-config model)) (ctx (config-ctx cfg)) (v (config-vocab cfg))
         (vocab (coerce (lm-vocab lm) 'simple-vector)) (vmap (vocab-map (lm-vocab lm)))
         (state (sb-ext:seed-random-state seed))
         (seq (coerce (or (coerce (encode (tokenize prompt) vmap) 'list) (list (gethash (string #\Newline) vmap 0))) 'list))
         (out (make-string-output-stream)))
    (write-string prompt out)
    (dotimes (step n)
      (let* ((window (last seq ctx))
             (x (make-array (length window) :element-type 'fixnum :initial-contents window))
             (logits (last-logits model x)) (probs (new-dvec v)))
        (setf (aref logits 0) -1d30)
        (softmax-row logits 0 probs 0 v (float temperature 1d0))
        (let ((i (sample-index probs state)))
          (setf seq (append seq (list i)))
          (write-string (svref vocab i) out))))
    (get-output-stream-string out)))

;;; ------------------------------------------------------------------ save / load (text format)
(defun save-model (lm path)
  (with-open-file (s path :direction :output :if-exists :supersede :external-format :utf-8)
    (let ((*read-default-float-format* 'double-float) (*print-readably* nil) (*print-pretty* nil)
          (cfg (model-config (lm-model lm))))
      (prin1 (list :format 1 :vocab-size (config-vocab cfg) :ctx (config-ctx cfg) :dim (config-dim cfg)
                   :heads (config-heads cfg) :layers (config-layers cfg) :ff (config-ff cfg)
                   :vocab (lm-vocab lm) :stats (lm-stats lm))
             s)
      (terpri s)
      (loop for p across (model-params (lm-model lm))
            do (prin1 (coerce p 'list) s) (terpri s))))
  path)

(defun load-model (path)
  (with-open-file (s path :external-format :utf-8)
    (let* ((*read-default-float-format* 'double-float) (*read-eval* nil)
           (h (read s))
           (cfg (make-config :vocab (getf h :vocab-size) :ctx (getf h :ctx) :dim (getf h :dim) :heads (getf h :heads)
                             :layers (getf h :layers) :ff (getf h :ff)))
           (m (init-model cfg 1)))
      (loop for p across (model-params m)
            do (let ((lst (read s)))
                 (unless (= (length lst) (length p)) (error "corrupt model file: tensor size mismatch"))
                 (loop for x in lst for i from 0 do (setf (aref p i) (float x 1d0)))))
      (make-lm :model m :vocab (getf h :vocab) :stats (getf h :stats)))))
