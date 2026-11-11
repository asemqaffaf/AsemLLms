;;;; One-time recovery of a diagnostic written with CL F exponents. No inference.
;;;; Preserve raw bytes; only replace numeric exponent markers outside JSON strings.
;;;; sbcl --script lisp/bin/normalize-fp32-audit.lisp raw=FILE out=NEW.json record=NEW-LINEAGE.json
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "verify-saved")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(defun opt (key)
  (let ((p (concatenate 'string key "=")))
    (let ((s (find-if (lambda (s) (and (>= (length s) (length p)) (string= p s :end2 (length p)))) (cdr sb-ext:*posix-argv*)))) (and s (subseq s (length p))))))
(let* ((raw (or (opt "raw") (error "raw= required"))) (out (or (opt "out") (error "out= required")))
       (record (or (opt "record") (error "record= required"))) (text nil) (count 0) (quoted nil) (escaped nil))
  (when (or (probe-file out) (probe-file record)) (error "Refusing to overwrite recovery outputs"))
  (with-open-file (s raw :external-format :utf-8) (let ((b (make-string (file-length s)))) (setf text (subseq b 0 (read-sequence b s)))))
  (let ((normalized (with-output-to-string (s)
                      (loop for c across text for i from 0 do
                        (cond (quoted (write-char c s)
                                      (cond (escaped (setf escaped nil)) ((char= c #\\) (setf escaped t)) ((char= c #\") (setf quoted nil))))
                              ((char= c #\") (setf quoted t) (write-char c s))
                              ((and (find c "fF") (> i 0) (digit-char-p (char text (1- i))) (< (1+ i) (length text))
                                    (find (char text (1+ i)) "+-0123456789")) (incf count) (write-char #\e s))
                              (t (write-char c s)))))))
    (let ((value (asem-json:parse-json normalized)))
      (asem-verify-saved:write-new-json out value)
      (asem-verify-saved:write-new-json record
        (asem-verify-saved:obj "raw_file" raw "raw_sha256" (asem-sha256:sha256-file-hex raw)
                              "normalized_file" out "normalized_sha256" (asem-sha256:sha256-file-hex out)
                              "numeric_exponent_markers_replaced" count "model_inference_repeated" :false
                              "normalizer_source_sha256" (asem-sha256:sha256-file-hex "lisp/bin/normalize-fp32-audit.lisp")
                              "fixed_writer_sha256" (asem-sha256:sha256-file-hex "lisp/src/json.lisp")
                              "note" "Raw output unchanged. Only unquoted numeric F exponent markers were changed to E before parsing and standard JSON reserialization."))))
  (format t "Normalized ~d numeric exponent markers; raw result preserved.~%" count))
