;;; sbcl --script lisp/bin/generate.lisp MODEL "prompt" [tokens] [temperature]
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "tf" "train")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(in-package :asem-train)
(let* ((args (cdr sb-ext:*posix-argv*)) (lm (load-model (first args))))
  (format t "~a~%" (generate lm (or (second args) "User: Hello!
Assistant:") (if (third args) (parse-integer (third args)) 60)
                             :temperature (if (fourth args) (float (read-from-string (fourth args)) 1d0) 0.7d0))))
