;;;; Chat-data cleaner: drops malformed, duplicate and context-free dialogues; deterministic output.
;;;; See that file for what each rule is for.
(defpackage :asem-clean
  (:use :cl :asem-util)
  (:export #:clean #:reject-reason #:parse #:line-text #:norm #:*max-lines* #:*max-exchange-repeats* #:main))
(in-package :asem-clean)

(defparameter *max-lines* 12)
(defparameter *max-exchange-repeats* 2)

(defparameter *stopwords*
  (let ((h (make-hash-table :test 'equal)))
    (dolist (w (split-on "a an the is are was were be do does did i you he she it we they me my your this that these those to of in on for and or but not no yes can could should would will it's i'm what why how when where who which with at by from as if so there their them its have has had than then too also very just about up out more most some any" #\Space))
      (setf (gethash w h) t))
    h))

(defparameter *pronouns* '("it" "they" "them" "that" "this" "those" "these" "its" "their"))

(defun word-char-p (c) (or (char<= #\a c #\z) (char<= #\A c #\Z) (char<= #\0 c #\9) (char= c #\_)))

(defun runs (s pred)
  "Maximal runs of characters satisfying PRED."
  (loop with i = 0 with n = (length s) with out = nil
        while (< i n)
        do (if (funcall pred (char s i))
               (let ((j (or (position-if-not pred s :start i) n)))
                 (push (subseq s i j) out)
                 (setf i j))
               (incf i))
        finally (return (nreverse out))))

(defun lower-letter-p (c) (char<= #\a c #\z))

(defun content-words (text)
  (remove-if (lambda (w) (or (gethash w *stopwords*) (<= (length w) 2)))
             (runs (string-downcase text) #'lower-letter-p)))

(defun has-pronoun-p (text)
  (some (lambda (r) (member (string-downcase r) *pronouns* :test #'string=))
        (runs text #'word-char-p)))

(defun norm (text)
  "Lower-case, keep only a-z, 0-9 and space, strip."
  (py-strip (remove-if-not (lambda (c) (or (char<= #\a c #\z) (char<= #\0 c #\9) (char= c #\Space)))
                           (string-downcase text))))

(defun parse (text)
  "Dialogues: blocks of lines separated by whitespace-only lines."
  (let ((trimmed (string-trim '(#\Newline) text)) (blocks nil) (cur nil))
    (dolist (line (split-lines trimmed))
      (if (string= (py-strip line) "")
          (when cur (push (nreverse cur) blocks) (setf cur nil))
          (push line cur)))
    (when cur (push (nreverse cur) blocks))
    (nreverse blocks)))

(defun line-text (line)
  (let ((p (search ": " line)))
    (if p (py-strip (subseq line (+ p 2))) "")))

(defun non-ascii-p (lines)
  (some (lambda (line) (some (lambda (c) (not (<= #x20 (char-code c) #x7e))) line)) lines))

(defun alternate (list) "Elements at even then odd positions."
  (loop for x in list for i from 0 if (evenp i) collect x into a else collect x into b
        finally (return (values a b))))

(defun reject-reason (lines)
  "First rule the dialogue breaks (a keyword) or NIL. No cross-dialogue state."
  (let ((n (length lines)))
    (when (or (zerop n) (oddp n)) (return-from reject-reason :malformed))
    (loop for line in lines for i from 0
          for prefix = (if (evenp i) "User: " "Assistant: ")
          do (when (or (not (string-prefix-p prefix line)) (string= (line-text line) ""))
               (return-from reject-reason :malformed)))
    (when (> n *max-lines*) (return-from reject-reason :too-long))
    (when (non-ascii-p lines) (return-from reject-reason :non-ascii))
    (multiple-value-bind (ul bl) (alternate lines)
      (let ((users (mapcar #'line-text ul)) (bots (mapcar #'line-text bl)))
        (when (< (length (remove-duplicates (mapcar #'norm bots) :test #'string=)) (length bots))
          (return-from reject-reason :degenerate))
        (when (or (some (lambda (u b) (string= (norm u) (norm b))) users bots)
                  (some (lambda (b) (< (count-words b) 2)) bots))
          (return-from reject-reason :degenerate))
        (when (and (has-pronoun-p (first users)) (< (length (content-words (first users))) 2))
          (return-from reject-reason :no-context))))
    nil))

(defun reason-name (k) (string-downcase (substitute #\_ #\- (symbol-name k))))

(defun clean (text)
  "-> (values cleaned-text report-alist)"
  (let* ((dialogues (parse text))
         (removed (make-hash-table))
         (kept nil)
         (seen (make-hash-table :test 'equal))
         (exchange-count (make-hash-table :test 'equal)))
    (dolist (lines dialogues)
      (let ((reason (reject-reason lines)))
        (unless reason
          (let* ((key (mapcar (lambda (l) (norm (line-text l))) lines))
                 (exchanges (loop for (u b) on key by #'cddr collect (cons u b))))
            (cond ((gethash key seen) (setf reason :duplicate))
                  ((some (lambda (e) (>= (gethash e exchange-count 0) *max-exchange-repeats*)) exchanges)
                   (setf reason :boilerplate))
                  (t (setf (gethash key seen) t)
                     (dolist (e exchanges) (incf (gethash e exchange-count 0)))
                     (push lines kept)))))
        (when reason (incf (gethash reason removed 0)))))
    (setf kept (nreverse kept))
    (let* ((out (concatenate 'string (join (format nil "~%~%") (mapcar (lambda (d) (join (string #\Newline) d)) kept))
                             (string #\Newline)))
           (users (loop for d in kept append (mapcar #'line-text (alternate d))))
           (uniq (make-hash-table :test 'equal))
           (rm (sort (loop for k being the hash-keys of removed using (hash-value v)
                           collect (cons (reason-name k) v))
                     #'string< :key #'car)))
      (dolist (u users) (setf (gethash u uniq) t))
      (values out
              `(("bytes_in" . ,(utf8-length text)) ("bytes_out" . ,(utf8-length out))
                ("dialogues_in" . ,(length dialogues)) ("dialogues_out" . ,(length kept))
                ("max_exchange_repeats" . ,(loop for v being the hash-values of exchange-count maximize v into m
                                                 finally (return (or m 0))))
                ("removed" . ,rm)
                ("unique_user_turns_out" . ,(hash-table-count uniq))
                ("user_turns_out" . ,(length users)))))))

(defun report-json (report)
  "Same layout as Python's json.dumps(report, indent=2, sort_keys=True)."
  (with-output-to-string (o)
    (format o "{~%")
    (loop for ((k . v) . rest) on report
          do (format o "  ~a: " (json-string k))
             (if (listp v)
                 (if (null v)
                     (write-string "{}" o)
                     (progn (format o "{~%")
                            (loop for ((rk . rv) . rrest) on v
                                  do (format o "    ~a: ~d~a~%" (json-string rk) rv (if rrest "," "")))
                            (write-string "  }" o)))
                 (format o "~d" v))
             (format o "~a~%" (if rest "," "")))
    (format o "}")))

(defun main (args)
  (unless (= (length args) 2)
    (format *error-output* "usage: sbcl --script lisp/bin/clean-chat.lisp IN OUT~%")
    (sb-ext:exit :code 2))
  (multiple-value-bind (out report) (clean (read-file (first args)))
    (write-file (second args) out)
    (write-file (concatenate 'string (second args) ".report.json")
                (concatenate 'string (report-json report) (string #\Newline)))
    (format t "~a~%" (report-json report))))
