;;;; Lisp kernels vs the C kernels in native/libtlm.so, per product shape. Run: sbcl --script bin/bench-native.lisp
;;;; Use it to pick TLM_MIN_WORK (the smallest m*k*n for which the C call wins on your machine).
(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/tf.lisp" here)))
(in-package :asem-tf)

(unless (native-active-p) (format t "native/build/libtlm.so not found: run `make -C native` first~%") (sb-ext:exit :code 1))
(setf (native-min-work) 0)

(defun fill-random (v state) (dotimes (i (length v) v) (setf (aref v i) (- (random 2d0 state) 1d0))))

(defun now-us ()
  (multiple-value-bind (sec usec) (sb-ext:get-time-of-day) (+ (* sec 1000000) usec)))

(defun time-it (fn reps)
  "Best of 5 runs of REPS calls, in microseconds per call."
  (funcall fn)
  (let ((best most-positive-fixnum))
    (dotimes (r 5)
      (let ((t0 (now-us)))
        (dotimes (i reps) (funcall fn))
        (setf best (min best (- (now-us) t0)))))
    (/ (float best 1d0) reps)))

(let ((state (sb-ext:seed-random-state 1)))
  (format t "~&~8a ~16a ~12a ~12a ~8a~%" "kernel" "m k n" "lisp us" "C us" "C/lisp")
  (dolist (shape '((8 16 16) (32 64 64) (32 64 128) (32 64 1401) (128 128 128) (256 256 256) (512 512 512)))
    (destructuring-bind (m k n) shape
      (dolist (kind '(:nn :nt :tn))
        (multiple-value-bind (la lb lc)
            (ecase kind (:nn (values (* m k) (* k n) (* m n))) (:nt (values (* m k) (* n k) (* m n)))
              (:tn (values (* m k) (* m n) (* k n))))
          (let* ((a (fill-random (new-dvec la) state)) (b (fill-random (new-dvec lb) state)) (c (new-dvec lc))
                 (reps (max 1 (floor 20000000 (* m k n))))
                 (fl (ecase kind (:nn (lambda () (mm-nn-lisp a b c m k n nil))) (:nt (lambda () (mm-nt-lisp a b c m k n nil)))
                       (:tn (lambda () (mm-tn-lisp a b c m k n nil)))))
                 (fc (ecase kind (:nn (lambda () (mm-nn a b c m k n nil))) (:nt (lambda () (mm-nt a b c m k n nil)))
                       (:tn (lambda () (mm-tn a b c m k n nil)))))
                 (tl (time-it fl reps)) (tc (time-it fc reps)))
            (format t "~8a ~16a ~12,2f ~12,2f ~8,2f~%" kind (format nil "~d ~d ~d" m k n) tl tc (/ tc tl))))))))
