;;;; The C kernels and the TLM1 file, against the Lisp code. Skipped (and passes) when native/libtlm.so is not built.
(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/tf.lisp" here))
  (load (merge-pathnames "../src/tlm.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(use-package :asem-tf)

(defun bits-equal (a b)
  "Every element EQL (so -0.0 and 0.0 differ): the results must be identical, not close."
  (and (= (length a) (length b)) (every #'eql a b)))

(defun rnd-vec (n state)
  (let ((v (new-dvec n))) (dotimes (i n v) (setf (aref v i) (- (random 2d0 state) 1d0)))))

(defun skip-unless-native ()
  (unless (native-active-p)
    (format t "  native library not loaded (make -C native): skipped~%")
    t))

(defun kernel-sizes (kind m k n)
  (ecase kind
    (:nn (values (* m k) (* k n) (* m n)))
    (:nt (values (* m k) (* n k) (* m n)))
    (:tn (values (* m k) (* m n) (* k n)))))

(defun compare-kernel (kind m k n acc state)
  (multiple-value-bind (la lb lc) (kernel-sizes kind m k n)
    (let* ((a (rnd-vec la state)) (b (rnd-vec lb state)) (c0 (rnd-vec lc state))
           (c-lisp (copy-seq c0)) (c-native (copy-seq c0)))
      (ecase kind
        (:nn (asem-tf::mm-nn-lisp a b c-lisp m k n acc) (asem-tf::mm-nn a b c-native m k n acc))
        (:nt (asem-tf::mm-nt-lisp a b c-lisp m k n acc) (asem-tf::mm-nt a b c-native m k n acc))
        (:tn (asem-tf::mm-tn-lisp a b c-lisp m k n acc) (asem-tf::mm-tn a b c-native m k n acc)))
      (bits-equal c-lisp c-native))))

(deftest kernels-bit-identical-to-lisp
  (unless (skip-unless-native)
    (let ((state (sb-ext:seed-random-state 11))
          (shapes '((1 1 1) (3 5 7) (32 64 64) (32 64 1401) (7 13 4) (4 1 9) (17 33 5) (2 128 3)))
          (old (native-min-work)))
      (setf (native-min-work) 0)
      (unwind-protect
           (dolist (kind '(:nn :nt :tn))
             (dolist (acc '(nil t))
               (loop for (m k n) in shapes
                     do (check (compare-kernel kind m k n acc state)
                               (format nil "~a ~a ~a: C result differs from Lisp" kind (list m k n) (if acc "acc" "set"))))))
        (setf (native-min-work) old)))))

;;; The dispatch only switches above the threshold; force it to 0 / huge and train one step each way.
(defun one-step (cfg x y threshold)
  (setf (native-min-work) threshold)
  (let* ((m (init-model cfg 3)) (acts (make-acts cfg (length x))) (grads (make-grads cfg)))
    (forward m x acts)
    (let ((loss (backward m x y acts grads)))
      (values loss grads (copy-seq (acts-probs acts))))))

(deftest training-step-identical-with-and-without-native
  (unless (skip-unless-native)
    (let* ((cfg (make-config :vocab 11 :ctx 8 :dim 16 :heads 2 :layers 2 :ff 32))
           (x (make-array 8 :element-type 'fixnum :initial-contents '(1 2 3 4 5 6 7 8)))
           (y (make-array 8 :element-type 'fixnum :initial-contents '(2 3 4 5 6 7 8 9)))
           (old (native-min-work)))
      (unwind-protect
           (multiple-value-bind (l1 g1 p1) (one-step cfg x y 1000000000000)
             (multiple-value-bind (l2 g2 p2) (one-step cfg x y 0)
               (check (eql l1 l2) "loss differs")
               (check (bits-equal p1 p2) "probabilities differ")
               (check (every #'bits-equal g1 g2) "gradients differ")))
        (setf (native-min-work) old)))))

;;; The shared file: Lisp writes it, C mmaps it, and the logits agree with last-logits to the last bit.
(deftest tlm-file-loads-in-c-with-identical-logits
  (unless (skip-unless-native)
    (let* ((cfg (make-config :vocab 13 :ctx 9 :dim 12 :heads 3 :layers 2 :ff 20))
           (m (init-model cfg 4))
           (path (format nil "/tmp/asem-test-~d.tlm" (random 1000000 (make-random-state t))))
           (fn (sb-sys:find-foreign-symbol-address "tlm_logits_file")))
      (check fn "tlm_logits_file not found in the library")
      (when fn
        (save-tlm m path)
        (unwind-protect
             (loop for len from 1 to 9
                   do (let* ((ids (make-array len :element-type 'fixnum
                                                  :initial-contents (loop for i from 1 to len collect (mod (+ 3 (* 7 i)) 13))))
                             (i32 (make-array len :element-type '(signed-byte 32) :initial-contents (coerce ids 'list)))
                             (out (new-dvec 13))
                             (expected (last-logits m ids))
                             (rc (sb-sys:with-pinned-objects (i32 out)
                                   (sb-alien:alien-funcall
                                    (sb-alien:sap-alien (sb-sys:int-sap fn)
                                                        (function sb-alien:int sb-alien:c-string sb-sys:system-area-pointer
                                                                  sb-alien:long sb-sys:system-area-pointer sb-alien:long))
                                    path (sb-sys:vector-sap i32) len (sb-sys:vector-sap out) 13))))
                        (check (= rc 0) (format nil "tlm_logits_file returned ~d" rc))
                        (check (bits-equal expected out) (format nil "logits differ for a prefix of ~d tokens" len))))
          (ignore-errors (delete-file path)))))))

(run-all)
