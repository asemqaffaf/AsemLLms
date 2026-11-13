;;;; Single-float (FP32) pretraining: pretrain.lisp, derived on top of tf32.lisp / train32.lisp. Package ASEM-PRETRAIN32.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :asem-derive)
    (load (merge-pathnames "derive.lisp" (or *load-truename* *compile-file-truename*)))))
(asem-derive:load-derived (merge-pathnames "pretrain.lisp" (or *load-truename* *compile-file-truename*)))
