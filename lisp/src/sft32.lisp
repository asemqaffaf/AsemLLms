;;;; Single-float (FP32) supervised fine-tuning: sft.lisp, derived on top of tf32 / train32 / pretrain32. Package ASEM-SFT32.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :asem-derive)
    (load (merge-pathnames "derive.lisp" (or *load-truename* *compile-file-truename*)))))
(asem-derive:load-derived (merge-pathnames "sft.lisp" (or *load-truename* *compile-file-truename*)))
