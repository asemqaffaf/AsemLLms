;;;; Single-float (FP32) transformer: tf.lisp, derived. See derive.lisp. Package ASEM-TF32 has the same API as ASEM-TF.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :asem-derive)
    (load (merge-pathnames "derive.lisp" (or *load-truename* *compile-file-truename*)))))
(asem-derive:load-derived (merge-pathnames "tf.lisp" (or *load-truename* *compile-file-truename*)))
