;;; Is a learning rate stable? Reads the "train" events of a short probe run (pretrain.lisp ... log-every=10) and prints a JSON verdict.
;;;   sbcl --script lisp/bin/check-stability.lisp OUT/log.jsonl      exit code 0 = stable, 1 = not stable
;;; Rules (see STABILITY-VERDICT): finite values, loss fell by >= 1.0, never > 0.5 above its running minimum, gradient norm after step 100 within 10x of its median.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "tf32" "train32" "pretrain32")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(let* ((path (or (second sb-ext:*posix-argv*) (progn (format *error-output* "usage: check-stability.lisp LOG.jsonl~%") (sb-ext:exit :code 2))))
       (events (with-open-file (in path :external-format :utf-8)
                 (loop for line = (read-line in nil) while line
                       when (search "\"event\":\"train\"" line)
                         collect (let ((o (asem-json:parse-json line)))
                                   (list (asem-json:json-get o "step") (float (asem-json:json-get o "train_loss") 1d0) (float (asem-json:json-get o "grad_norm") 1d0))))))
       (v (asem-pretrain32::stability-verdict events)))
  (asem-json:write-json v *standard-output*) (terpri)
  (sb-ext:exit :code (if (eq (asem-json:json-get v "stable") :true) 0 1)))
