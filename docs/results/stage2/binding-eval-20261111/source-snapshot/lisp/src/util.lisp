;;;; Small shared helpers: text files, whitespace, splitting, JSON writing.
(defpackage :asem-util
  (:use :cl)
  (:export #:read-file #:write-file #:split-lines #:py-strip #:py-whitespace-p #:count-words
           #:split-on #:string-prefix-p #:join #:utf8-length #:json-string #:script-dir))
(in-package :asem-util)

(defun script-dir ()
  (directory-namestring (or *load-truename* *default-pathname-defaults*)))

(defun read-file (path)
  "Whole file as a string (UTF-8; invalid bytes become ?)."
  (with-open-file (s path :external-format '(:utf-8 :replacement #\?))
    (let* ((buf (make-string (file-length s)))
           (n (read-sequence buf s)))
      (subseq buf 0 n))))

(defun write-file (path string)
  (with-open-file (s path :direction :output :if-exists :supersede :external-format :utf-8)
    (write-string string s))
  path)

(defun py-whitespace-p (c)
  "Characters Python's str.strip()/str.split() treat as whitespace."
  (let ((code (char-code c)))
    (or (<= 9 code 13) (<= #x1c code #x20) (= code #x85) (= code #xa0) (= code #x1680)
        (<= #x2000 code #x200a) (= code #x2028) (= code #x2029) (= code #x202f)
        (= code #x205f) (= code #x3000))))

(defun py-strip (s)
  (let ((start (position-if-not #'py-whitespace-p s))
        (end (position-if-not #'py-whitespace-p s :from-end t)))
    (if start (subseq s start (1+ end)) "")))

(defun count-words (s)
  "Number of whitespace-separated words, like len(s.split())."
  (let ((n 0) (in nil))
    (loop for c across s
          do (if (py-whitespace-p c)
                 (setf in nil)
                 (unless in (setf in t) (incf n))))
    n))

(defun split-on (s ch)
  "Split on every occurrence of CH, keeping empty fields (like Python str.split(ch))."
  (loop with start = 0
        for pos = (position ch s :start start)
        collect (subseq s start pos)
        while pos
        do (setf start (1+ pos))))

(defun split-lines (s)
  (split-on s #\Newline))

(defun string-prefix-p (prefix s)
  (and (>= (length s) (length prefix)) (string= prefix s :end2 (length prefix))))

(defun join (sep strings)
  (with-output-to-string (o)
    (loop for (s . rest) on strings
          do (write-string s o)
             (when rest (write-string sep o)))))

(defun utf8-length (s)
  (loop for c across s
        sum (let ((code (char-code c)))
              (cond ((< code #x80) 1) ((< code #x800) 2) ((< code #x10000) 3) (t 4)))))

(defun json-string (s)
  (with-output-to-string (o)
    (write-char #\" o)
    (loop for c across s
          do (case c
               (#\" (write-string "\\\"" o))
               (#\\ (write-string "\\\\" o))
               (#\Newline (write-string "\\n" o))
               (t (if (< (char-code c) 32)
                      (format o "\\u~4,'0x" (char-code c))
                      (write-char c o)))))
    (write-char #\" o)))
