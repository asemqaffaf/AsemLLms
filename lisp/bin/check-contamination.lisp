;;; Does any evaluation question (or reading passage) appear verbatim in the training text?
;;;   sbcl --dynamic-space-size 4GB --script lisp/bin/check-contamination.lisp EVAL.jsonl CORPUS.jsonl
;;; Text is lowercased and whitespace-collapsed first. A hit means the item can be answered by recall of its own wording,
;;; so it should be dropped or rewritten before its score is believed. Exit code 1 if any item is contaminated.
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "json")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(in-package :asem-json)

(defun squash (s)
  (let ((out (make-string-output-stream)) (space nil))
    (loop for c across (string-downcase s)
          do (if (member c '(#\Space #\Newline #\Tab #\Return)) (setf space t)
                 (progn (when space (write-char #\Space out) (setf space nil)) (write-char c out))))
    (get-output-stream-string out)))

(let* ((args (cdr sb-ext:*posix-argv*)) (eval-path (first args)) (corpus-path (second args)))
  (unless (and eval-path corpus-path) (format *error-output* "usage: check-contamination.lisp EVAL.jsonl CORPUS.jsonl~%") (sb-ext:exit :code 2))
  (let ((needles nil) (hits nil) (docs 0))
    (with-open-file (in eval-path :external-format :utf-8)
      (loop for line = (read-line in nil) while line
            when (plusp (length (string-trim " " line)))
              do (let ((o (parse-json line)))
                   (push (list (json-get o "id") (squash (json-get o "q"))) needles)
                   (when (json-get o "context") (push (list (json-get o "id") (squash (json-get o "context"))) needles)))))
    (with-open-file (in corpus-path :external-format :utf-8)
      (loop for line = (read-line in nil) while line
            do (incf docs)
               (let ((text (squash (json-get (parse-json line) "text"))))
                 (dolist (n needles)
                   (when (and (not (assoc (first n) hits :test #'equal)) (search (second n) text))
                     (push (cons (first n) (second n)) hits))))))
    (format t "checked ~d evaluation strings against ~d documents: ~d contaminated~%" (length needles) docs (length hits))
    (dolist (h (reverse hits)) (format t "  ~a: ~s~%" (car h) (cdr h)))
    (sb-ext:exit :code (if hits 1 0))))
