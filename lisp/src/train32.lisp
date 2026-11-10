;;;; Single-float (FP32) trainer: train.lisp, derived on top of tf32.lisp. Package ASEM-TRAIN32.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :asem-derive)
    (load (merge-pathnames "derive.lisp" (or *load-truename* *compile-file-truename*)))))
(asem-derive:load-derived (merge-pathnames "train.lisp" (or *load-truename* *compile-file-truename*)))
