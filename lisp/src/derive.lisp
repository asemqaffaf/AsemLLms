;;;; Derives the single-float (FP32) version of a double-float source file at load time, so FP32 can never drift from
;;;; the FP64 reference it is derived from. Text rules, applied in this order to the file's text:
;;;;   package names   asem-tf -> asem-tf32, asem-train -> asem-train32 (so the FP32 code lives in its own packages)
;;;;   types           every "double-float" -> "single-float" (also in most-positive-double-float and friends)
;;;;   literals        1.5d0 -> 1.5f0 (a double below 1e-37 becomes 1f-37); 1l0 / long-float are host doubles and stay
;;;;   foreign calls   cblas_dgemm -> cblas_sgemm, sb-alien:double -> sb-alien:float
;;;;   exact forms     (native-init) is dropped (libtlm.so is double only); HOST-FLOAT coerces to single-float;
;;;;                   ELT-BYTES is 4; PI is the single-float constant
;;;; Then the forms are read (with single-float as the default float format) and evaluated one by one.
(defpackage :asem-derive (:use :cl) (:export #:load-derived))
(in-package :asem-derive)

(defun constituent-p (c) (or (alphanumericp c) (find c "-_*+/<>=!?%$&:.")))

(defun replace-all (text old new)
  (with-output-to-string (out)
    (let ((i 0) (n (length old)))
      (loop (let ((j (search old text :start2 i)))
              (cond (j (write-string text out :start i :end j) (write-string new out) (setf i (+ j n)))
                    (t (write-string text out :start i) (return))))))))

(defun replace-word (text old new)
  "Replace OLD where it is a whole token (not part of a longer symbol)."
  (with-output-to-string (out)
    (let ((i 0) (n (length old)))
      (loop (let ((j (search old text :start2 i)))
              (cond ((null j) (write-string text out :start i) (return))
                    ((and (or (= j 0) (not (constituent-p (char text (1- j)))))
                          (or (>= (+ j n) (length text)) (not (constituent-p (char text (+ j n))))))
                     (write-string text out :start i :end j) (write-string new out) (setf i (+ j n)))
                    (t (write-string text out :start i :end (+ j 1)) (setf i (+ j 1)))))))))

(defun rename-package-words (text old new)
  "asem-tf -> asem-tf32 wherever it is a symbol prefix or package name, never inside asem-tf-something."
  (with-output-to-string (out)
    (let ((i 0) (n (length old)))
      (loop (let ((j (search old text :start2 i)))
              (cond ((null j) (write-string text out :start i) (return))
                    ((and (or (= j 0) (not (constituent-p (char text (1- j)))) (char= (char text (1- j)) #\:))
                          (or (>= (+ j n) (length text)) (not (find (char text (+ j n)) "-_abcdefghijklmnopqrstuvwxyz0123456789"))))
                     (write-string text out :start i :end j) (write-string new out) (setf i (+ j n)))
                    (t (write-string text out :start i :end (+ j 1)) (setf i (+ j 1)))))))))

(defun double-literal-p (tok)
  "digits[.digits]d[+-]digits, e.g. 1d0 0.5d0 1d-8 .5d0"
  (let ((d (position #\d tok)))
    (and d (> d 0) (< (1+ d) (length tok))
         (let ((mant (subseq tok 0 d)) (ex (subseq tok (1+ d))))
           (and (every (lambda (c) (or (digit-char-p c) (char= c #\.))) mant) (some #'digit-char-p mant)
                (let ((e (if (find (char ex 0) "+-") (subseq ex 1) ex)))
                  (and (plusp (length e)) (every #'digit-char-p e))))))))

(defun convert-token (tok out)
  "Write TOK to OUT, turning a double literal (optionally signed) into a single one."
  (let* ((sign (and (plusp (length tok)) (find (char tok 0) "+-")))
         (body (if sign (subseq tok 1) tok)))
    (cond ((double-literal-p body)
           (let* ((d (position #\d body)) (ex (parse-integer body :start (1+ d))))
             (when sign (write-char (char tok 0) out))
             (if (< ex -37)
                 (write-string "1f-37" out)
                 (progn (write-string body out :end d) (write-char #\f out) (write-string body out :start (1+ d))))))
          (t (write-string tok out)))))

(defun convert-literals (text)
  (with-output-to-string (out)
    (let ((i 0) (n (length text)))
      (loop while (< i n)
            do (let ((c (char text i)))
                 (cond ((and (constituent-p c) (or (= i 0) (not (constituent-p (char text (1- i))))))
                        (let ((j (or (position-if-not #'constituent-p text :start i) n)))
                          (convert-token (subseq text i j) out) (setf i j)))
                       (t (write-char c out) (incf i))))))))

(defun derive-text (text)
  (let* ((s (rename-package-words text "asem-tf" "asem-tf32"))
         (s (rename-package-words s "asem-train" "asem-train32"))
         (s (replace-all s "double-float" "single-float"))
         (s (replace-all s "cblas_dgemm" "cblas_sgemm"))
         (s (replace-all s "sb-alien:double" "sb-alien:float"))
         (s (replace-all s "(native-init)" "nil"))
         (s (replace-all s "(defmacro host-float (x) x)" "(defmacro host-float (x) `(coerce ,x 'single-float))"))
         (s (replace-all s "(defmacro elt-bytes () 8)" "(defmacro elt-bytes () 4)"))
         (s (replace-word s "pi" "3.1415927f0")))
    (convert-literals s)))

(defun load-derived (path)
  "Evaluate the FP32 derivation of the Lisp file at PATH."
  (let* ((text (with-open-file (in path :external-format :utf-8)
                 (let ((s (make-string (file-length in)))) (subseq s 0 (read-sequence s in)))))
         (*package* *package*) (*read-default-float-format* 'single-float))
    (with-input-from-string (in (derive-text text))
      (loop for form = (read in nil in) until (eq form in) do (eval form)))
    t))
