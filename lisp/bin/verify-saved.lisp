;;;; Independent verifier. Run from repository root; refuses to overwrite outputs.
;;;; sbcl --script lisp/bin/verify-saved.lisp mode=phase2 out=NEW.json
;;;; sbcl --script lisp/bin/verify-saved.lisp mode=binding bundle=DIR out=NEW.json
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "verify-saved"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(defun opt (name default)
  (let ((arg (find-if (lambda (x) (and (> (length x) (length name)) (string= (concatenate 'string name "=") x :end2 (1+ (length name))))) (cdr sb-ext:*posix-argv*))))
    (if arg (subseq arg (1+ (length name))) default)))
(let* ((mode (opt "mode" "phase2")) (out (opt "out" nil)))
  (when (and out (probe-file out)) (error "Refusing to overwrite ~a" out))
  (asem-verify-saved:run-self-checks)
  (let ((report (cond ((equal mode "phase2") (asem-verify-saved:verify-phase2))
                      ((equal mode "fixtures") (asem-verify-saved:verify-fixtures (or (opt "fixtures" nil) (error "fixtures=JSON required"))))
                      ((equal mode "binding") (asem-verify-saved:verify-binding (or (opt "bundle" nil) (error "bundle=DIR required"))))
                      (t (error "mode must be phase2 or binding")))))
    (if out (asem-verify-saved:write-new-json out report) (progn (asem-json:write-json report *standard-output*) (terpri)))))
