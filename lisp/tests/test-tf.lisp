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


;;; ---- answer-only (row-selective) head and loss, used by supervised fine-tuning (src/sft.lisp)
(defun masked-loss-full (m x rows targets)
  "Reference: full FORWARD, then the summed cross-entropy at ROWS only."
  (let ((acts (make-acts (model-config m) (length x))) (v (config-vocab (model-config m))) (s 0d0))
    (forward m x acts)
    (loop for r across rows for tg across targets do (decf s (log (aref (acts-probs acts) (+ (* r v) tg)))))
    s))

(deftest forward-rows-matches-forward-at-those-rows
  (let* ((cfg (make-config :vocab 13 :ctx 7 :dim 8 :heads 2 :layers 2 :ff 16)) (m (init-model cfg 4))
         (x (ivec 3 1 4 1 5 9 2)) (rows (ivec 6 2)) (full (make-acts cfg 7)) (part (make-acts cfg 7 2)))
    (forward m x full) (forward-rows m x part rows)
    (check (equalp (subseq (acts-probs part) 0 13) (subseq (acts-probs full) (* 6 13) (* 7 13))) "row 6")
    (check (equalp (subseq (acts-probs part) 13 26) (subseq (acts-probs full) (* 2 13) (* 3 13))) "row 2")))

(deftest padding-after-the-answer-row-changes-nothing
  (let* ((cfg (make-config :vocab 13 :ctx 9 :dim 8 :heads 2 :layers 2 :ff 16)) (m (init-model cfg 4))
         (a (make-acts cfg 9 1)) (b (make-acts cfg 9 1)) (c (make-acts cfg 5 1)) (rows (ivec 4)))
    (forward-rows m (ivec 3 1 4 1 5 0 0 0 0) a rows)
    (forward-rows m (ivec 3 1 4 1 5 12 7 7 2) b rows)
    (forward-rows m (ivec 3 1 4 1 5) c rows)
    (check (equalp (subseq (acts-probs a) 0 13) (subseq (acts-probs b) 0 13)) "different padding tokens")
    (check (equalp (subseq (acts-probs a) 0 13) (subseq (acts-probs c) 0 13)) "padded vs unpadded")))

(deftest masked-gradients-match-numerical-differentiation
  (let* ((cfg (make-config :vocab 11 :ctx 6 :dim 8 :heads 2 :layers 2 :ff 16)) (m (init-model cfg 3))
         (x (ivec 1 4 2 9 0 3)) (rows (ivec 3 5)) (tg (ivec 7 2)) (acts (make-acts cfg 6 2)) (grads (make-grads cfg))
         (state (sb-ext:seed-random-state 5)) (worst 0d0) (eps 1d-6))
    (forward-rows m x acts rows)
    (let ((l (backward-rows m x rows tg acts grads)))
      (check-close l (masked-loss-full m x rows tg) 1d-12))
    (loop for p across (model-params m) for g across grads
          do (dotimes (s 6)
               (let* ((i (random (length p) state)) (old (aref p i)))
                 (setf (aref p i) (+ old eps))
                 (let ((up (masked-loss-full m x rows tg)))
                   (setf (aref p i) (- old eps))
                   (let* ((dn (masked-loss-full m x rows tg)) (num (/ (- up dn) (* 2 eps))) (ana (aref g i)))
                     (setf (aref p i) old)
                     (setf worst (max worst (/ (abs (- num ana)) (max 1d-4 (abs num) (abs ana))))))))))
    (check (< worst 1d-5) (format nil "worst relative gradient error ~e" worst))))

(deftest masked-rows-get-no-gradient-from-other-positions
  ;; the token at the last position is never a prediction source for row 2: its embedding gradient must be zero
  (let* ((cfg (make-config :vocab 11 :ctx 5 :dim 8 :heads 2 :layers 1 :ff 16)) (m (init-model cfg 3))
         (x (ivec 1 2 3 4 10)) (acts (make-acts cfg 5 1)) (grads (make-grads cfg)))
    (forward-rows m x acts (ivec 2))
    (backward-rows m x (ivec 2) (ivec 5) acts grads)
    (check (every #'zerop (subseq (svref grads 0) (* 10 8) (* 11 8))) "token after the row: zero embedding gradient")
    (check (every #'zerop (subseq (svref grads 1) (* 3 8))) "positions after the row: zero position gradient")
    (check (notevery #'zerop (subseq (svref grads 0) (* 3 8) (* 4 8))) "token at the row has a gradient")))


(deftest weighted-masked-gradients-match-numerical-differentiation
  (let* ((cfg (make-config :vocab 11 :ctx 6 :dim 8 :heads 2 :layers 1 :ff 16)) (m (init-model cfg 3))
         (x (ivec 1 4 2 9 0 3)) (rows (ivec 1 3 5)) (tg (ivec 7 2 4)) (w (let ((v (new-dvec 3))) (setf (aref v 0) 0.25d0 (aref v 1) 1d0 (aref v 2) 0.5d0) v))
         (acts (make-acts cfg 6 3)) (grads (make-grads cfg)) (state (sb-ext:seed-random-state 6)) (worst 0d0) (eps 1d-6))
    (flet ((ref () (let ((a (make-acts cfg 6)) (s 0d0)) (forward m x a)
                     (loop for r across rows for t0 across tg for k from 0 do (decf s (* (aref w k) (log (aref (acts-probs a) (+ (* r 11) t0)))))) s)))
      (forward-rows m x acts rows)
      (check-close (backward-rows m x rows tg acts grads w) (ref) 1d-12)
      (loop for p across (model-params m) for g across grads
            do (dotimes (s 5)
                 (let* ((i (random (length p) state)) (old (aref p i)))
                   (setf (aref p i) (+ old eps))
                   (let ((up (ref)))
                     (setf (aref p i) (- old eps))
                     (let* ((dn (ref)) (num (/ (- up dn) (* 2 eps))) (ana (aref g i)))
                       (setf (aref p i) old)
                       (setf worst (max worst (/ (abs (- num ana)) (max 1d-4 (abs num) (abs ana)))))))))))
    (check (< worst 1d-5) (format nil "worst relative error ~e" worst))))

(run-all)
