;;; FP32 version of bench.lisp (src/tf32.lisp, src/train32.lisp). sbcl --script lisp/bin/bench32.lisp [dim layers heads ctx batch steps] [workers]
;;; One JSON line, same fields as tinyllm/scripts/bench.escript.
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "tf32" "train32")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*root* (merge-pathnames "../../" here)))
(in-package :asem-train32)
(let* ((a (mapcar #'parse-integer (cdr sb-ext:*posix-argv*)))
       (dim (or (nth 0 a) 64)) (layers (or (nth 1 a) 2)) (heads (or (nth 2 a) 4)) (ctx (or (nth 3 a) 32))
       (batch (or (nth 4 a) 8)) (steps (or (nth 5 a) 30)) (workers (or (nth 6 a) 1))
       (lm (train (asem-util:read-file (merge-pathnames "dataset/chat_clean.txt" cl-user::*root*))
                  :steps steps :workers workers :ctx ctx :dim dim :heads heads :layers layers :ff (* 2 dim)
                  :batch batch :eval-every steps :val-examples 8 :verbose nil))
       (cfg (asem-tf32:model-params (lm-model lm)))
       (c (asem-tf32:model-config (lm-model lm)))
       (tps (getf (lm-stats lm) :tokens-per-sec)))
  (declare (ignore cfg))
  (format t "{\"impl\":\"lisp-fp32\",\"params\":~d,\"vocab\":~d,\"dim\":~d,\"layers\":~d,\"ctx\":~d,\"batch\":~d,\"workers\":~d,\"tokens_per_sec\":~,1f,\"flops_per_sec\":~,3e}~%"
          (asem-tf32:param-count (lm-model lm)) (asem-tf32:config-vocab c) dim layers ctx batch workers tps
          (* tps (asem-tf32:flops-per-token c))))
