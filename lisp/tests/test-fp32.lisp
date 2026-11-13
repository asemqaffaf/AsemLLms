;;;; Single-float (FP32) training, derived from the double-float source (src/derive.lisp). FP64 stays the reference:
;;;; every FP32 result is compared with it under a stated tolerance, never required to be bit-identical.
;;;; Tolerances (single precision has ~1.2e-7 relative resolution): sgemm products rtol 1e-4 / atol 1e-5; a forward and
;;;; backward pass against FP64 within 1e-4 relative L2 per tensor; finite-difference check 2e-2 (eps 3e-3).
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "tf" "train" "tf32" "train32"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(defun rnd32 (n state)
  (let ((v (asem-tf32:new-dvec n))) (dotimes (i n v) (setf (aref v i) (float (- (random 2d0 state) 1d0) 1f0)))))

(defun finite32-p (x) (and (= x x) (< (abs x) most-positive-single-float)))
(defun all-finite32-p (v) (every #'finite32-p v))

(defun close32-p (a b &optional (rtol 1f-4) (atol 1f-5))
  (and (= (length a) (length b))
       (every (lambda (x y) (and (finite32-p x) (<= (abs (- x y)) (+ atol (* rtol (abs y)))))) a b)))

(defun worst32 (a b) (loop for x across a for y across b maximize (abs (- x y))))

(defun rel-l2 (a b)
  "||a - b|| / ||b|| with the sums in double."
  (let ((num 0d0) (den 0d0))
    (loop for x across a for y across b do (incf num (expt (- (float x 1d0) (float y 1d0)) 2)) (incf den (expt (float y 1d0) 2)))
    (sqrt (/ num (max den 1d-300)))))

(defun skip-unless-sgemm ()
  (unless (asem-tf32:blas-init) (format t "  no CBLAS library: skipped~%") t))

(defparameter *cfg* (asem-tf:make-config :vocab 11 :ctx 8 :dim 16 :heads 2 :layers 2 :ff 32))
(defparameter *cfg32* (asem-tf32::make-config :vocab 11 :ctx 8 :dim 16 :heads 2 :layers 2 :ff 32))
(defparameter *x* (make-array 8 :element-type 'fixnum :initial-contents '(1 2 3 4 5 6 7 8)))
(defparameter *y* (make-array 8 :element-type 'fixnum :initial-contents '(2 3 4 5 6 7 8 9)))

(defun fp32-copy-of (m64)
  "An FP32 model with the (rounded) weights of the FP64 model M64."
  (asem-tf32::make-model
   :config (let ((c (asem-tf:model-config m64)))
             (asem-tf32::make-config :vocab (asem-tf:config-vocab c) :ctx (asem-tf:config-ctx c) :dim (asem-tf:config-dim c)
                                    :heads (asem-tf:config-heads c) :layers (asem-tf:config-layers c) :ff (asem-tf:config-ff c)))
   :params (map 'simple-vector (lambda (p) (let ((q (asem-tf32:new-dvec (length p)))) (dotimes (i (length p) q) (setf (aref q i) (float (aref p i) 1f0)))))
                (asem-tf:model-params m64))))

(defun step64 (m x y)
  (let* ((cfg (asem-tf:model-config m)) (acts (asem-tf:make-acts cfg (length x))) (g (asem-tf:make-grads cfg)))
    (asem-tf:forward m x acts)
    (values (asem-tf:backward m x y acts g) g (copy-seq (asem-tf:acts-probs acts)))))

(defun step32 (m x y)
  (let* ((cfg (asem-tf32:model-config m)) (acts (asem-tf32:make-acts cfg (length x))) (g (asem-tf32:make-grads cfg)))
    (asem-tf32:forward m x acts)
    (values (asem-tf32:backward m x y acts g) g (copy-seq (asem-tf32:acts-probs acts)))))

(deftest fp32-uses-single-float-everywhere
  (let* ((m (asem-tf32:init-model *cfg32* 3)) (acts (asem-tf32:make-acts *cfg32* 8)) (g (asem-tf32:make-grads *cfg32*)))
    (check (every (lambda (p) (typep p '(simple-array single-float (*)))) (asem-tf32:model-params m)) "weights are not single-float")
    (check (every (lambda (p) (typep p '(simple-array single-float (*)))) g) "gradients are not single-float")
    (check (typep (asem-tf32:acts-probs acts) '(simple-array single-float (*))) "activations are not single-float")
    (check (every (lambda (p) (typep p '(simple-array double-float (*)))) (asem-tf:model-params (asem-tf:init-model *cfg* 3)))
           "the FP64 reference changed type")))

(deftest fp32-forward-backward-matches-fp64
  (let* ((m64 (asem-tf:init-model *cfg* 3)) (m32 (fp32-copy-of m64)))
    (multiple-value-bind (l64 g64 p64) (step64 m64 *x* *y*)
      (multiple-value-bind (l32 g32 p32) (step32 m32 *x* *y*)
        (check (finite32-p l32) "FP32 loss not finite")
        (check (< (abs (- l32 l64)) (* 1d-5 (abs l64))) (format nil "loss ~a vs ~a" l32 l64))
        (check (< (rel-l2 p32 p64) 1d-5) (format nil "probabilities differ by ~,2e (relative L2)" (rel-l2 p32 p64)))
        (loop for a across g32 for b across g64 for n in (asem-tf:param-names *cfg*)
              do (check (< (rel-l2 a b) 1d-4) (format nil "gradient ~a differs by ~,2e (relative L2)" n (rel-l2 a b))))
        (check (every #'all-finite32-p g32) "NaN/Inf in an FP32 gradient")))))

(defun fd-check32 (cfg x y)
  "Worst relative error between FP32 analytic gradients and central differences (eps 3e-3), sampled over every tensor."
  (let* ((m (asem-tf32:init-model cfg 3)) (g (nth-value 1 (step32 m x y))) (worst 0d0) (eps 3f-3)
         (state (sb-ext:seed-random-state 5)))
    (flet ((loss () (let ((acts (asem-tf32:make-acts cfg (length x)))) (asem-tf32:forward m x acts) (asem-tf32:loss-of-acts acts y))))
      (loop for p across (asem-tf32:model-params m) for gr across g
            do (dotimes (s 5)
                 (let* ((i (random (length p) state)) (old (aref p i)))
                   (setf (aref p i) (+ old eps)) (let ((up (loss)))
                     (setf (aref p i) (- old eps)) (let* ((dn (loss)) (num (/ (- up dn) (* 2 eps))) (ana (aref gr i)))
                       (setf (aref p i) old)
                       (setf worst (max worst (/ (abs (- num ana)) (max 1d-2 (abs num) (abs ana))))))))))
      worst)))

(deftest fp32-gradients-match-finite-differences
  (let ((w (fd-check32 *cfg32* *x* *y*)))
    (check (< w 2d-2) (format nil "worst relative error ~,3e" w))))

(deftest fp32-sgemm-matches-lisp-kernels
  (unless (skip-unless-sgemm)
    (let ((state (sb-ext:seed-random-state 5)) (old (asem-tf32:blas-min-work)))
      (setf (asem-tf32:blas-min-work) 0)
      (unwind-protect
           (dolist (shape '((1 1 1) (3 5 7) (33 17 65) (64 64 64) (200 3 150) (256 256 256) (96 1024 40)))
             (destructuring-bind (m k n) shape
               (dolist (kind '(:nn :nt :tn))
                 (dolist (acc '(nil t))
                   (multiple-value-bind (la lb lc)
                       (ecase kind (:nn (values (* m k) (* k n) (* m n))) (:nt (values (* m k) (* n k) (* m n))) (:tn (values (* m k) (* m n) (* k n))))
                     (let* ((a (rnd32 la state)) (b (rnd32 lb state)) (c0 (rnd32 lc state)) (cl (copy-seq c0)) (cb (copy-seq c0)))
                       (ecase kind
                         (:nn (asem-tf32::mm-nn-lisp a b cl m k n acc) (asem-tf32::mm-nn a b cb m k n acc))
                         (:nt (asem-tf32::mm-nt-lisp a b cl m k n acc) (asem-tf32::mm-nt a b cb m k n acc))
                         (:tn (asem-tf32::mm-tn-lisp a b cl m k n acc) (asem-tf32::mm-tn a b cb m k n acc)))
                       (check (all-finite32-p cb) (format nil "~a ~a: NaN/Inf from sgemm" kind shape))
                       (check (close32-p cb cl) (format nil "~a ~a acc=~a: worst error ~,3e" kind shape acc (worst32 cb cl))))))))))
        (setf (asem-tf32:blas-min-work) old))))

(deftest fp32-attention-blas-matches-lisp
  (unless (skip-unless-sgemm)
    (let ((state (sb-ext:seed-random-state 11)))
      (dolist (shape '((1 1 1) (5 2 3) (17 3 4) (64 4 16) (96 8 8) (33 1 1)))
        (destructuring-bind (tl nh hd) shape
          (let* ((d (* nh hd)) (scale (/ 1f0 (sqrt (float hd 1f0))))
                 (q (rnd32 (* tl d) state)) (k (rnd32 (* tl d) state)) (vv (rnd32 (* tl d) state)) (da (rnd32 (* tl d) state))
                 (p1 (asem-tf32:new-dvec (* nh tl tl))) (p2 (asem-tf32:new-dvec (* nh tl tl)))
                 (a1 (asem-tf32:new-dvec (* tl d))) (a2 (asem-tf32:new-dvec (* tl d))))
            (asem-tf32::attn-fwd-lisp q k vv p1 a1 tl d nh hd scale)
            (asem-tf32::attn-fwd-blas q k vv p2 a2 tl d nh hd scale)
            (check (close32-p p2 p1) (format nil "~a: probabilities differ by ~,3e" shape (worst32 p2 p1)))
            (check (close32-p a2 a1) (format nil "~a: outputs differ by ~,3e" shape (worst32 a2 a1)))
            (dotimes (h nh)
              (dotimes (i tl)
                (let ((row (+ (* h tl tl) (* i tl))) (sum 0d0))
                  (dotimes (j tl)
                    (when (> j i) (check (zerop (aref p2 (+ row j))) (format nil "~a: mass above the diagonal" shape)))
                    (incf sum (aref p2 (+ row j))))
                  (check (< (abs (- sum 1d0)) 1d-5) (format nil "~a: row sums to ~a" shape sum)))))
            (let ((g1 (list (asem-tf32:new-dvec (* tl d)) (asem-tf32:new-dvec (* tl d)) (asem-tf32:new-dvec (* tl d))))
                  (g2 (list (asem-tf32:new-dvec (* tl d)) (asem-tf32:new-dvec (* tl d)) (asem-tf32:new-dvec (* tl d)))))
              (asem-tf32::attn-bwd-lisp p1 q k vv da (first g1) (second g1) (third g1) (asem-tf32:new-dvec tl) tl d nh hd scale)
              (asem-tf32::attn-bwd-blas p1 q k vv da (first g2) (second g2) (third g2) (asem-tf32:new-dvec (* tl tl)) tl d nh hd scale)
              (loop for x in g2 for y in g1 for name in '(dq dk dv)
                    do (check (close32-p x y 1f-3 1f-4) (format nil "~a: ~a differs by ~,3e" shape name (worst32 x y)))))))))))

(deftest fp32-blas-training-step-matches-lisp-and-fp64
  (unless (skip-unless-sgemm)
    (let* ((m64 (asem-tf:init-model *cfg* 3)) (m32 (fp32-copy-of m64)) (old (asem-tf32:blas-min-work)))
      (unwind-protect
           (progn
             (setf (asem-tf32:blas-min-work) 1000000000000)
             (multiple-value-bind (lo go) (step32 m32 *x* *y*)
               (setf (asem-tf32:blas-min-work) 0)
               (multiple-value-bind (lb gb) (step32 m32 *x* *y*)
                 (check (< (abs (- lo lb)) (* 1f-5 (abs lo))) (format nil "loss ~a vs ~a" lo lb))
                 (loop for a across gb for b across go for n in (asem-tf:param-names *cfg*)
                       do (check (< (rel-l2 a b) 1d-4) (format nil "gradient ~a: BLAS vs Lisp ~,2e" n (rel-l2 a b))))
                 (check (every #'all-finite32-p gb) "NaN/Inf in a BLAS gradient"))))
        (setf (asem-tf32:blas-min-work) old)))))

(deftest fp32-nan-and-infinity-are-not-hidden
  (let ((m (asem-tf32:init-model *cfg32* 3)))
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :inexact)
      (setf (aref (svref (asem-tf32:model-params m) 0) 16) (- sb-ext:single-float-positive-infinity sb-ext:single-float-positive-infinity))   ; token 1, dim 0
      (let ((loss (step32 m *x* *y*)))
        (check (not (finite32-p loss)) "a NaN weight must show up in the loss, not vanish")))))

;;; Convergence: both precisions learn the same task to about the same loss.
(defparameter *corpus*
  (with-output-to-string (o)
    (dotimes (i 60)
      (format o "User: What color is the sky?~%Assistant: The sky is blue.~%User: What color is grass?~%Assistant: Grass is green.~%~%"))))

(deftest fp32-and-fp64-converge-to-the-same-loss
  (flet ((run32 () (asem-train32:train *corpus* :steps 200 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 8 :lr 0.01f0
                                      :eval-every 50 :val-examples 20 :verbose nil))
         (run64 () (asem-train:train *corpus* :steps 200 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 8 :lr 0.01d0
                                     :eval-every 50 :val-examples 20 :verbose nil)))
    (let* ((s32 (asem-train32:lm-stats (run32))) (s64 (asem-train:lm-stats (run64)))
           (v32 (getf s32 :val-loss)) (v64 (getf s64 :val-loss)))
      (check (finite32-p v32) "FP32 validation loss not finite")
      (check (< v32 (getf s32 :unigram)) (format nil "FP32 val ~a not below unigram ~a" v32 (getf s32 :unigram)))
      (check (< v32 1.0) (format nil "FP32 val loss ~a" v32))
      (check (< (abs (- v32 v64)) (+ 0.05 (* 0.1 v64))) (format nil "FP32 val ~a vs FP64 val ~a" v32 v64)))))

(deftest fp32-blas-training-converges
  (unless (skip-unless-sgemm)
    (let ((old (asem-tf32:blas-min-work)))
      (setf (asem-tf32:blas-min-work) 0)
      (unwind-protect
           (let* ((s (asem-train32:lm-stats (asem-train32:train *corpus* :steps 200 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 8
                                                                           :lr 0.01f0 :eval-every 50 :val-examples 20 :verbose nil)))
                  (v (getf s :val-loss)))
             (check (and (finite32-p v) (< v 1.0)) (format nil "FP32+BLAS val loss ~a" v)))
        (setf (asem-tf32:blas-min-work) old)))))

;;; Checkpoints: FP32 round-trips bit for bit, and FP64 <-> FP32 files load into either precision.
(deftest fp32-checkpoint-round-trips
  (let* ((lm32 (asem-train32:train *corpus* :steps 10 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 4 :eval-every 10 :val-examples 10 :verbose nil))
         (lm64 (asem-train:train *corpus* :steps 10 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 4 :eval-every 10 :val-examples 10 :verbose nil))
         (p32 (format nil "/tmp/asem-fp32-test-~d.model" (get-universal-time)))
         (p64 (format nil "/tmp/asem-fp64-test-~d.model" (get-universal-time))))
    (asem-train32:save-model lm32 p32) (asem-train:save-model lm64 p64)
    (let ((back (asem-train32:load-model p32)))
      (check (every #'equalp (asem-tf32:model-params (asem-train32:lm-model lm32)) (asem-tf32:model-params (asem-train32:lm-model back)))
             "FP32 parameters differ after reload")
      (check-equal (asem-train32:generate lm32 "User: What color" 20 :seed 3) (asem-train32:generate back "User: What color" 20 :seed 3)))
    (let ((as32 (asem-train32:load-model p64)) (as64 (asem-train:load-model p32)))
      (check (every (lambda (a b) (every (lambda (x y) (= x (float y 1f0))) a b))
                    (asem-tf32:model-params (asem-train32:lm-model as32)) (asem-tf:model-params (asem-train:lm-model lm64)))
             "FP64 file loaded into FP32 does not hold the rounded weights")
      (check (every (lambda (a b) (every (lambda (x y) (<= (abs (- x y)) (* 1d-7 (abs y)))) a b))
                    (asem-tf:model-params (asem-train:lm-model as64)) (asem-tf32:model-params (asem-train32:lm-model lm32)))
             "FP32 file loaded into FP64 differs by more than 1e-7 relative"))
    (delete-file p32) (delete-file p64)))

(run-all)
