;;;; Accelerate/CBLAS (cblas_dgemm) against the pure-Lisp kernels, which stay the reference and the fallback.
;;;; Close, not identical: |blas - lisp| <= atol + rtol * |lisp| with rtol 1e-10, atol 1e-12 for double-float.
;;;; Skipped (and passes) when no CBLAS can be loaded.
(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/tf.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(use-package :asem-tf)

(defparameter *rtol* 1d-10)
(defparameter *atol* 1d-12)

(defun rnd-vec (n state &optional (scale 1d0))
  (let ((v (new-dvec n))) (dotimes (i n v) (setf (aref v i) (* scale (- (random 2d0 state) 1d0))))))

(defun finite-p (x) (and (= x x) (< (abs x) most-positive-double-float)))

(defun all-finite-p (v) (every #'finite-p v))

(defun within-tolerance-p (blas lisp &optional (rtol *rtol*) (atol *atol*))
  (and (= (length blas) (length lisp))
       (every (lambda (x y) (and (finite-p x) (<= (abs (- x y)) (+ atol (* rtol (abs y)))))) blas lisp)))

(defun worst-error (blas lisp)
  (loop for x across blas for y across lisp maximize (abs (- x y))))

(defun skip-unless-blas ()
  (unless (blas-init)
    (format t "  no CBLAS library: skipped~%")
    t))

(defun kernel-sizes (kind m k n)
  (ecase kind
    (:nn (values (* m k) (* k n) (* m n)))
    (:nt (values (* m k) (* n k) (* m n)))
    (:tn (values (* m k) (* m n) (* k n)))))

(defun run-kernels (kind a b c m k n acc)
  "Both paths on copies of C; returns (values lisp-result blas-result)."
  (let ((c-lisp (copy-seq c)) (c-blas (copy-seq c)))
    (ecase kind
      (:nn (asem-tf::mm-nn-lisp a b c-lisp m k n acc) (asem-tf::mm-nn a b c-blas m k n acc))
      (:nt (asem-tf::mm-nt-lisp a b c-lisp m k n acc) (asem-tf::mm-nt a b c-blas m k n acc))
      (:tn (asem-tf::mm-tn-lisp a b c-lisp m k n acc) (asem-tf::mm-tn a b c-blas m k n acc)))
    (values c-lisp c-blas)))

(defun compare-all (shapes state scale)
  (let ((old (blas-min-work)))
    (setf (blas-min-work) 0)          ; force BLAS for every shape
    (unwind-protect
         (dolist (shape shapes)
           (destructuring-bind (m k n) shape
             (dolist (kind '(:nn :nt :tn))
               (dolist (acc '(nil t))
                 (multiple-value-bind (la lb lc) (kernel-sizes kind m k n)
                   (let ((a (rnd-vec la state scale)) (b (rnd-vec lb state scale)) (c0 (rnd-vec lc state scale)))
                     (multiple-value-bind (c-lisp c-blas) (run-kernels kind a b c0 m k n acc)
                       (check (all-finite-p c-blas) (format nil "~a ~a acc=~a: NaN/Inf in BLAS result" kind shape acc))
                       (check (within-tolerance-p c-blas c-lisp)
                              (format nil "~a ~a acc=~a scale=~a: worst error ~,3e" kind shape acc scale (worst-error c-blas c-lisp))))))))))
      (setf (blas-min-work) old))))

(deftest blas-small-and-rectangular-shapes
  (unless (skip-unless-blas)
    (compare-all '((1 1 1) (1 64 1) (3 5 7) (7 1 5) (8 16 16) (33 17 65) (200 3 150) (2 300 2) (64 64 64))
                 (sb-ext:seed-random-state 5) 1d0)))

(deftest blas-large-shapes
  (unless (skip-unless-blas)
    (compare-all '((256 256 256) (512 128 384) (96 1024 40)) (sb-ext:seed-random-state 6) 1d0)))

(deftest blas-values-near-zero
  (unless (skip-unless-blas)
    (compare-all '((8 16 16) (33 17 65)) (sb-ext:seed-random-state 7) 1d-8)
    (compare-all '((8 16 16) (33 17 65)) (sb-ext:seed-random-state 8) 1d-100)))

(deftest blas-nan-and-infinity-are-not-hidden
  (unless (skip-unless-blas)
    (let* ((m 4) (k 4) (n 4) (a (rnd-vec 16 (sb-ext:seed-random-state 9))) (b (rnd-vec 16 (sb-ext:seed-random-state 10)))
           (old (blas-min-work)))
      (setf (blas-min-work) 0)
      (unwind-protect
           (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :inexact)
             (setf (aref a 5) (sb-kernel:make-double-float -524288 0))   ; quiet NaN
             (let ((c (new-dvec 16))) (asem-tf::mm-nn a b c m k n nil)
               (check (not (all-finite-p c)) "a NaN in A must reach C, not be dropped"))
             (setf (aref a 5) sb-ext:double-float-positive-infinity)
             (let ((c (new-dvec 16))) (asem-tf::mm-nn a b c m k n nil)
               (check (not (all-finite-p c)) "an Inf in A must reach C, not be dropped")))
        (setf (blas-min-work) old)))))

;;; Forward/backward equivalence: same model, same batch, BLAS on and off. Loss and every gradient tensor agree.
(defun train-step (cfg x y threshold)
  (setf (blas-min-work) threshold)
  (let* ((m (init-model cfg 3)) (acts (make-acts cfg (length x))) (grads (make-grads cfg)))
    (forward m x acts)
    (let ((loss (backward m x y acts grads)))
      (values loss grads (copy-seq (acts-probs acts))))))

(deftest blas-forward-backward-equivalence
  (unless (skip-unless-blas)
    (let* ((cfg (make-config :vocab 11 :ctx 8 :dim 16 :heads 2 :layers 2 :ff 32))
           (x (make-array 8 :element-type 'fixnum :initial-contents '(1 2 3 4 5 6 7 8)))
           (y (make-array 8 :element-type 'fixnum :initial-contents '(2 3 4 5 6 7 8 9)))
           (old (blas-min-work)))
      (unwind-protect
           (multiple-value-bind (l1 g1 p1) (train-step cfg x y 1000000000000)
             (multiple-value-bind (l2 g2 p2) (train-step cfg x y 0)
               (check (finite-p l2) "BLAS loss is not finite")
               (check (<= (abs (- l1 l2)) (+ *atol* (* *rtol* (abs l1)))) (format nil "loss ~a vs ~a" l1 l2))
               (check (within-tolerance-p p2 p1) "probabilities differ")
               (check (every (lambda (a b) (within-tolerance-p a b 1d-9 1d-11)) g2 g1) "gradients differ")
               (check (every #'all-finite-p g2) "NaN/Inf in a BLAS gradient")))
        (setf (blas-min-work) old)))))

;;; Attention kernels alone, several head counts and lengths (tl not a multiple of anything, hd of 1).
(deftest blas-attention-matches-lisp
  (unless (skip-unless-blas)
    (let ((state (sb-ext:seed-random-state 11)))
      (dolist (shape '((1 1 1) (5 2 3) (17 3 4) (64 4 16) (96 8 8) (33 1 1)))   ; tl, heads, head dim
        (destructuring-bind (tl nh hd) shape
          (let* ((d (* nh hd)) (scale (/ 1d0 (sqrt (float hd 1d0))))
                 (q (rnd-vec (* tl d) state)) (k (rnd-vec (* tl d) state)) (vv (rnd-vec (* tl d) state))
                 (da (rnd-vec (* tl d) state))
                 (p1 (new-dvec (* nh tl tl))) (p2 (new-dvec (* nh tl tl))) (a1 (new-dvec (* tl d))) (a2 (new-dvec (* tl d))))
            (asem-tf::attn-fwd-lisp q k vv p1 a1 tl d nh hd scale)
            (asem-tf::attn-fwd-blas q k vv p2 a2 tl d nh hd scale)
            (check (within-tolerance-p p2 p1) (format nil "~a: probabilities differ by ~,3e" shape (worst-error p2 p1)))
            (check (within-tolerance-p a2 a1) (format nil "~a: outputs differ by ~,3e" shape (worst-error a2 a1)))
            (check (all-finite-p p2) (format nil "~a: NaN/Inf in probabilities" shape))
            (dotimes (h nh)                          ; causal: nothing above the diagonal, every row sums to 1
              (dotimes (i tl)
                (let ((row (+ (* h tl tl) (* i tl))) (sum 0d0))
                  (dotimes (j tl)
                    (when (> j i) (check (zerop (aref p2 (+ row j))) (format nil "~a: mass above the diagonal" shape)))
                    (incf sum (aref p2 (+ row j))))
                  (check (< (abs (- sum 1d0)) 1d-12) (format nil "~a: row sums to ~a" shape sum)))))
            (let ((g1 (list (new-dvec (* tl d)) (new-dvec (* tl d)) (new-dvec (* tl d))))
                  (g2 (list (new-dvec (* tl d)) (new-dvec (* tl d)) (new-dvec (* tl d)))))
              (asem-tf::attn-bwd-lisp p1 q k vv da (first g1) (second g1) (third g1) (new-dvec tl) tl d nh hd scale)
              (asem-tf::attn-bwd-blas p1 q k vv da (first g2) (second g2) (third g2) (new-dvec (* tl tl)) tl d nh hd scale)
              (loop for x in g2 for y in g1 for name in '(dq dk dv)
                    do (check (within-tolerance-p x y 1d-9 1d-11)
                              (format nil "~a: ~a differs by ~,3e" shape name (worst-error x y)))))))))))

(run-all)
