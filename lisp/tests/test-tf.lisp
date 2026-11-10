(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/tf.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(use-package :asem-tf)

(defun ivec (&rest xs) (make-array (length xs) :element-type 'fixnum :initial-contents xs))

(defun loss-at (m x y)
  (let ((acts (make-acts (model-config m) (length x))))
    (forward m x acts)
    (loss-of-acts acts y)))

(defun grad-check (cfg x y &key (samples 6) (eps 1d-6))
  "Worst relative error between analytic and central-difference gradients over sampled entries of every tensor."
  (let* ((m (init-model cfg 3)) (acts (make-acts cfg (length x))) (grads (make-grads cfg)) (worst 0d0)
         (state (sb-ext:seed-random-state 5)))
    (forward m x acts)
    (backward m x y acts grads)
    (loop for p across (model-params m) for g across grads
          do (dotimes (s samples)
               (let* ((i (random (length p) state)) (old (aref p i)))
                 (setf (aref p i) (+ old eps))
                 (let ((up (loss-at m x y)))
                   (setf (aref p i) (- old eps))
                   (let* ((dn (loss-at m x y)) (num (/ (- up dn) (* 2 eps))) (ana (aref g i)))
                     (setf (aref p i) old)
                     (setf worst (max worst (/ (abs (- num ana)) (max 1d-4 (abs num) (abs ana))))))))))
    worst))

(deftest parameter-count
  (let* ((cfg (make-config :vocab 11 :ctx 5 :dim 8 :heads 2 :layers 2 :ff 16)) (m (init-model cfg)))
    ;; tok + pos + layers*(2 gains + 4 attn + 2 mlp) + gf + head
    (check-equal (param-count m) (+ (* 11 8) (* 5 8) (* 2 (+ 8 (* 4 64) 8 (* 8 16) (* 16 8))) 8 (* 8 11)))))

(deftest gradients-match-numerical-differentiation
  (let ((cfg (make-config :vocab 11 :ctx 5 :dim 8 :heads 2 :layers 2 :ff 16)))
    (check (< (grad-check cfg (ivec 1 4 2 9 0) (ivec 4 2 9 0 7)) 1d-5) "2-layer, 2-head gradients")
    (check (< (grad-check (make-config :vocab 7 :ctx 6 :dim 6 :heads 3 :layers 1 :ff 9) (ivec 6 5 4 3 2 1) (ivec 5 4 3 2 1 0)) 1d-5)
           "1-layer, 3-head gradients")))

(deftest initial-loss-near-uniform
  (let* ((cfg (make-config :vocab 50 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32)) (m (init-model cfg 1))
         (l (loss-at m (ivec 1 2 3 4 5 6 7 8) (ivec 2 3 4 5 6 7 8 9))))
    (check (< (abs (- l (log 50d0))) 0.6d0) (format nil "initial loss ~f vs ln 50 = ~f" l (log 50d0)))))

(deftest model-is-causal
  (let* ((cfg (make-config :vocab 20 :ctx 6 :dim 8 :heads 2 :layers 2 :ff 16)) (m (init-model cfg 2))
         (a (make-acts cfg 6)) (b (make-acts cfg 6)))
    (forward m (ivec 1 2 3 4 5 6) a)
    (forward m (ivec 1 2 3 9 8 7) b)           ; same first three tokens
    (check-equal (subseq (acts-probs a) 0 (* 3 20)) (subseq (acts-probs b) 0 (* 3 20)))
    (check (not (equalp (subseq (acts-probs a) (* 3 20)) (subseq (acts-probs b) (* 3 20)))))))

(deftest same-seed-same-model
  (let* ((cfg (make-config :vocab 20 :ctx 6 :dim 8 :heads 2 :layers 1 :ff 16))
         (a (init-model cfg 7)) (b (init-model cfg 7)) (c (init-model cfg 8)))
    (check (every #'equalp (model-params a) (model-params b)))
    (check (notevery #'equalp (model-params a) (model-params c)))))

(deftest nan-weight-shows-up-in-the-loss
  (let* ((cfg (make-config :vocab 11 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32)) (m (init-model cfg 3)))
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :inexact)
      (setf (aref (svref (model-params m) 0) 16) (- sb-ext:double-float-positive-infinity sb-ext:double-float-positive-infinity))
      (let ((l (loss-at m (ivec 1 2 3 4 5 6 7 8) (ivec 2 3 4 5 6 7 8 9))))
        (check (/= l l) "a NaN weight must give a NaN loss, not a clamped finite one")))))

(run-all)
