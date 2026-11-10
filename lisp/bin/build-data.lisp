;;; sbcl --script lisp/bin/build-data.lisp build IN.jsonl OUT-DIR [vocab-size] [max-docs] [min-edu-score]
;;; sbcl --script lisp/bin/build-data.lisp verify OUT-DIR
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "bpe" "data")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(in-package :asem-data)
(let ((args (cdr sb-ext:*posix-argv*)))
  (cond ((and (equal (first args) "build") (>= (length args) 3))
         (let* ((opts (append (when (fourth args) (list :vocab-size (parse-integer (fourth args))))
                              (when (fifth args) (list :max-docs (parse-integer (fifth args))))
                              (when (sixth args) (list :min-edu-score (float (read-from-string (sixth args)) 1d0)))))
                (m (apply #'build (lambda (cb) (read-jsonl (second args) cb)) (third args) opts)))
           (verify (third args))
           (format t "version ~a~%stats ~s~%" (json-get m "version") (cdr (json-get m "stats")))
           (format t "train ~d docs / ~d tokens, val ~d docs / ~d tokens~%"
                   (json-get (json-get (json-get m "splits") "train") "documents") (json-get (json-get (json-get m "splits") "train") "tokens")
                   (json-get (json-get (json-get m "splits") "val") "documents") (json-get (json-get (json-get m "splits") "val") "tokens"))))
        ((and (equal (first args) "verify") (second args))
         (format t "ok, version ~a~%" (verify (second args))))
        (t (format *error-output* "usage: build IN.jsonl OUT [vocab] [max-docs] [min-edu] | verify OUT~%") (sb-ext:exit :code 2))))
