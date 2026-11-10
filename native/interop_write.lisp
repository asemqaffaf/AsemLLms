;;;; Used by interop_test.sh: write a Lisp model as TLM1 plus the logits Lisp computes for several prefixes.
(load (merge-pathnames "../lisp/src/util.lisp" *load-truename*))
(load (merge-pathnames "../lisp/src/tf.lisp" *load-truename*))
(load (merge-pathnames "../lisp/src/tlm.lisp" *load-truename*))
(in-package :asem-tf)
(defun interop-write (dir)
  (let* ((cfg (make-config :vocab 13 :ctx 9 :dim 12 :heads 3 :layers 2 :ff 20))
         (m (init-model cfg 4)))
    (save-tlm m (format nil "~a/lisp.tlm" dir))
    (with-open-file (out (format nil "~a/logits.bin" dir) :direction :output :if-exists :supersede
                                                          :element-type '(unsigned-byte 64))
      (loop for len from 1 to 9
            do (let ((ids (make-array len :element-type 'fixnum
                                          :initial-contents (loop for i from 1 to len collect (mod (+ 3 (* 7 i)) 13)))))
                 (loop for x across (last-logits m ids)
                       do (write-byte (logior (ash (ldb (byte 32 0) (sb-kernel:double-float-high-bits x)) 32)
                                              (sb-kernel:double-float-low-bits x))
                                      out)))))))
