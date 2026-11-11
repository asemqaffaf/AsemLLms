;;;; Dense retrieval loss: optional sequence weighting must agree with its numerical objective in both precisions.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "tf" "train" "pretrain" "sft" "tf32" "train32" "pretrain32" "sft32"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(defun weighted-example (&optional (normalization :sequence))
  (asem-sft:multi-ex "weighted" '(1 4 2 9 3 6) '(1 3 4) '(2 3 6) :normalization normalization))
(defun weighted-model ()
  (asem-tf:init-model (asem-tf:make-config :vocab 13 :ctx 8 :dim 8 :heads 2 :layers 1 :ff 16 :relpos 2) 17))
(defun multi-loss (m e)
  (asem-sft:run-multi m e (make-hash-table) nil))

(deftest historical-token-mode-remains-the-default
  (let ((a (asem-sft:multi-ex "weighted" '(1 4 2 9 3 6) '(1 3 4) '(2 3 6))) (b (weighted-example :token)))
    (check (equalp a b))
    (multiple-value-bind (loss pred scores mass) (asem-sft:run-multi (weighted-model) a (make-hash-table) nil)
      (declare (ignore loss pred scores))
      (check-equal mass 3))))

(deftest sequence-objective-and-gradient-have-unit-total-weight
  (let* ((m (weighted-model)) (cfg (asem-tf:model-config m)) (gt (asem-tf:make-grads cfg)) (gs (asem-tf:make-grads cfg)))
    (multiple-value-bind (lt pt ignored nt) (asem-sft:run-multi m (weighted-example :token) (make-hash-table) gt)
      (declare (ignore ignored))
      (multiple-value-bind (ls ps ignored ns) (asem-sft:run-multi m (weighted-example) (make-hash-table) gs)
        (declare (ignore ignored))
        (check-close ls (/ lt 3) 1d-12)
        (check-equal pt ps) (check-equal nt 3) (check-equal ns 1)
        (loop for a across gt for b across gs
              do (dotimes (i (length a)) (check-close (aref b i) (/ (aref a i) 3) 1d-12)))))
    (dolist (k '(1 2 3))
      (multiple-value-bind (loss pred scores mass)
          (asem-sft:run-multi m (asem-sft:multi-ex "unequal" '(1 4 2 9 3 6) (subseq '(1 3 4) 0 k) (subseq '(2 3 6) 0 k)
                                                 :normalization :sequence) (make-hash-table) nil)
        (declare (ignore loss pred scores))
        (check-equal mass 1)))))

(deftest sequence-loss-matches-finite-differences-for-every-tensor
  (let* ((m (weighted-model)) (e (weighted-example)) (grads (asem-tf:make-grads (asem-tf:model-config m)))
         (rng (sb-ext:seed-random-state 41)) (eps 1d-6) (worst 0d0))
    (asem-sft:run-multi m e (make-hash-table) grads)
    (loop for p across (asem-tf:model-params m) for g across grads
          do (dotimes (trial 5)
               (let* ((i (random (length p) rng)) (old (aref p i)))
                 (setf (aref p i) (+ old eps))
                 (let ((up (multi-loss m e)))
                   (setf (aref p i) (- old eps))
                   (let* ((down (multi-loss m e)) (numerical (/ (- up down) (* 2 eps))) (analytic (aref g i)))
                     (setf (aref p i) old)
                     (setf worst (max worst (/ (abs (- numerical analytic)) (max 1d-4 (abs numerical) (abs analytic))))))))))
    (check (< worst 2d-5) (format nil "worst relative gradient error ~e" worst))))

(deftest derived-fp32-agrees-with-fp64-for-sequence-weighting
  (let* ((m64 (weighted-model))
         (m32 (asem-tf32:init-model (asem-tf32:make-config :vocab 13 :ctx 8 :dim 8 :heads 2 :layers 1 :ff 16 :relpos 2) 17))
         (g64 (asem-tf:make-grads (asem-tf:model-config m64))) (g32 (asem-tf32:make-grads (asem-tf32:model-config m32)))
         (e32 (asem-sft32:multi-ex "weighted" '(1 4 2 9 3 6) '(1 3 4) '(2 3 6) :normalization :sequence)))
    (loop for a across (asem-tf:model-params m64) for b across (asem-tf32:model-params m32)
          do (dotimes (i (length a)) (setf (aref b i) (float (aref a i) 1f0))))
    (check-close (asem-sft32:run-multi m32 e32 (make-hash-table) g32)
                 (asem-sft:run-multi m64 (weighted-example) (make-hash-table) g64) 1d-5)
    (let ((delta 0d0) (scale 0d0))
      (loop for a across g64 for b across g32
            do (dotimes (i (length a)) (incf delta (expt (- (aref a i) (aref b i)) 2)) (incf scale (expt (aref a i) 2))))
      (check (< (sqrt (/ delta (max scale 1d-30))) 1d-4)))))

(deftest malformed-or-unknown-weighting-is-refused
  (dolist (args '((nil nil :normalization :sequence) ((1) (2 3) :normalization :token) ((1) (2) :normalization :other)))
    (let ((failed nil))
      (handler-case (apply #'asem-sft:multi-ex "bad" '(1 2 3) args) (error () (setf failed t)))
      (check failed))))

(deftest normalized-retrieval-and-copy-replay-resume-exactly-in-fp32
  ;; Unequal answer lengths and copy replay exercise the unit-mass denominator;
  ;; accumulation, worker partitioning and epoch crossings must survive resume.
  (let* ((train (concatenate 'vector
                            (loop for k from 1 to 3
                                  collect (asem-sft32:multi-ex
                                           (format nil "normalized-~d" k) '(1 4 2 9 3 6)
                                           (subseq '(1 3 4) 0 k) (subseq '(2 3 6) 0 k)
                                           :normalization :sequence))
                            (asem-sft32:make-copy-examples 2 31 :lo 1 :hi 13 :min-len 2 :max-len 4)))
         (root (format nil "/tmp/asem-multi-resume-~d-~d/" (sb-posix:getpid) (get-universal-time)))
         (continuous (format nil "~acontinuous" root)) (resumed (format nil "~aresumed" root))
         (new-model (lambda () (asem-tf32:init-model
                               (asem-tf32:make-config :vocab 13 :ctx 16 :dim 8 :heads 2 :layers 1 :ff 16 :relpos 2) 53)))
         (args (list :steps 9 :batch 3 :accum 2 :workers 2 :lr 3f-3 :warmup 2
                     :eval-every 3 :log-every 0 :train-probe 5 :seed 23)))
    (unwind-protect
         (progn
           (apply #'asem-sft32:sft continuous train train '(1 2 3 4) :init-model (funcall new-model) args)
           (apply #'asem-sft32:sft resumed train train '(1 2 3 4) :init-model (funcall new-model) :stop-at 4 args)
           (check-equal (getf (asem-sft32::read-state (format nil "~a/state.sexp" resumed)) :step) 4)
           (apply #'asem-sft32:sft resumed train train '(1 2 3 4) :resume t args)
           (check-equal (getf (asem-sft32::read-state (format nil "~a/state.sexp" resumed)) :step) 9)
           (dolist (file '("last.ckpt" "last-m1.ckpt" "last-m2.ckpt"))
             (check-equal (asem-sha256:sha256-file-hex (format nil "~a/~a" resumed file))
                          (asem-sha256:sha256-file-hex (format nil "~a/~a" continuous file))))
           (let ((trained (asem-pretrain32:load-checkpoint (format nil "~a/last.ckpt" continuous))))
             (check (not (every #'equalp (asem-tf32:model-params trained)
                                (asem-tf32:model-params (funcall new-model))))
                    "the resume comparison must contain actual parameter updates")))
      (when (probe-file root) (sb-ext:delete-directory root :recursive t)))))

(run-all)
