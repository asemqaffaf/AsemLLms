;;;; Byte-level BPE tokenizer. Ids: 0 = <|endoftext|>, 1..256 = raw bytes, 257.. = merges.
;;;; Any UTF-8 text round-trips exactly; there is no <unk>.
(defpackage :asem-bpe
  (:use :cl)
  (:export #:pretokenize #:train-bpe #:bpe-encode #:bpe-decode #:save-bpe #:load-bpe #:bpe-vocab-size
           #:bpe-eos #:make-bpe-codec #:bpe))
(in-package :asem-bpe)

(defstruct (bpe (:constructor %make-bpe)) merges ranks cache bytes)   ; merges: vector of (a . b)

(defun bpe-vocab-size (b) (+ 257 (length (bpe-merges b))))
(defun bpe-eos (b) (declare (ignore b)) 0)

(defun word-class (c)
  (cond ((alpha-char-p c) :letter) ((digit-char-p c) :digit)
        ((member c '(#\Space #\Tab #\Newline #\Return #\Page #\Vt)) :space) (t :other)))

(defun pretokenize (text)
  "GPT-2-style pieces: an optional leading space plus a run of letters / digits / other symbols;
whitespace runs (a trailing space is left to start the next word). Concatenating them gives TEXT."
  (let ((n (length text)) (i 0) (out nil))
    (loop while (< i n)
          do (let* ((c (char text i)) (cls (word-class c)))
               (flet ((run (from cl) (or (position-if-not (lambda (x) (eq (word-class x) cl)) text :start from) n)))
                 (cond ((and (char= c #\Space) (< (1+ i) n) (not (eq (word-class (char text (1+ i))) :space)))
                        (let ((e (run (1+ i) (word-class (char text (1+ i)))))) (push (subseq text i e) out) (setf i e)))
                       ((eq cls :space)
                        (let ((e (run i :space)))
                          (when (and (< e n) (> (- e i) 1) (char= (char text (1- e)) #\Space)) (decf e))
                          (push (subseq text i e) out) (setf i e)))
                       (t (let ((e (run i cls))) (push (subseq text i e) out) (setf i e)))))))
    (nreverse out)))

(defun piece-symbols (piece)
  "Initial symbol ids of a piece: its UTF-8 bytes + 1."
  (map 'simple-vector #'1+ (sb-ext:string-to-octets piece :external-format :utf-8)))

(defun pair-key (a b) (+ (* a 65536) b))

(defun merge-symbols (syms a b new)
  (let ((out (make-array (length syms) :fill-pointer 0)) (i 0) (n (length syms)))
    (loop while (< i n)
          do (if (and (< (1+ i) n) (= (svref syms i) a) (= (svref syms (1+ i)) b))
                 (progn (vector-push new out) (incf i 2))
                 (progn (vector-push (svref syms i) out) (incf i))))
    (coerce out 'simple-vector)))

(defun finish (merges)
  (let ((ranks (make-hash-table)) (bytes (make-array (+ 257 (length merges)))))
    (loop for (a . b) across merges for r from 0
          do (setf (gethash (pair-key a b) ranks) r))
    (setf (svref bytes 0) (make-array 0 :element-type '(unsigned-byte 8)))
    (loop for id from 1 to 256 do (setf (svref bytes id) (make-array 1 :element-type '(unsigned-byte 8) :initial-element (1- id))))
    (loop for (a . b) across merges for r from 0
          do (setf (svref bytes (+ 257 r)) (concatenate '(vector (unsigned-byte 8)) (svref bytes a) (svref bytes b))))
    (%make-bpe :merges merges :ranks ranks :cache (make-hash-table :test 'equal) :bytes bytes)))

(defun train-bpe (texts vocab-size)
  "Learn VOCAB-SIZE - 257 merges from an iterable (list) of strings. Deterministic: ties go to the smaller pair."
  (let ((freq (make-hash-table :test 'equal)) (words nil))
    (dolist (text texts) (dolist (p (pretokenize text)) (incf (gethash p freq 0))))
    (maphash (lambda (p c) (push (cons (piece-symbols p) c) words)) freq)
    (setf words (sort words (lambda (x y) (or (> (cdr x) (cdr y)) (and (= (cdr x) (cdr y)) (lexless (car x) (car y)))))))
    (let ((merges (make-array 0 :adjustable t :fill-pointer 0)) (words (coerce words 'simple-vector)))
      (loop while (< (+ 257 (length merges)) vocab-size)
            do (let ((counts (make-hash-table)) (best nil) (best-n 1))
                 (loop for (syms . c) across words
                       do (loop for i from 0 below (1- (length syms))
                                do (incf (gethash (pair-key (svref syms i) (svref syms (1+ i))) counts 0) c)))
                 (maphash (lambda (k v) (when (or (> v best-n) (and (= v best-n) best (< k best))) (setf best k best-n v))) counts)
                 (unless best (return))
                 (let ((a (floor best 65536)) (b (mod best 65536)) (new (+ 257 (length merges))))
                   (vector-push-extend (cons a b) merges)
                   (loop for i from 0 below (length words)
                         do (let ((w (svref words i)))
                              (when (> (length (car w)) 1)
                                (setf (svref words i) (cons (merge-symbols (car w) a b new) (cdr w)))))))))
      (finish (coerce merges 'simple-vector)))))

(defun lexless (a b)
  (loop for i from 0 below (min (length a) (length b))
        do (cond ((< (svref a i) (svref b i)) (return-from lexless t)) ((> (svref a i) (svref b i)) (return-from lexless nil))))
  (< (length a) (length b)))

(defun encode-piece (bpe piece)
  (or (gethash piece (bpe-cache bpe))
      (setf (gethash piece (bpe-cache bpe))
            (let ((syms (piece-symbols piece)) (ranks (bpe-ranks bpe)))
              (loop
                (let ((best nil) (best-rank nil))
                  (loop for i from 0 below (1- (length syms))
                        do (let ((r (gethash (pair-key (svref syms i) (svref syms (1+ i))) ranks)))
                             (when (and r (or (null best-rank) (< r best-rank))) (setf best-rank r best i))))
                  (unless best (return (coerce syms 'list)))
                  (setf syms (merge-symbols syms (svref syms best) (svref syms (1+ best)) (+ 257 best-rank))))))))) 

(defun bpe-encode (bpe text)
  (loop for p in (pretokenize text) append (encode-piece bpe p)))

(defun bpe-decode (bpe ids)
  (let ((octets (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
    (dolist (id ids)
      (loop for b across (svref (bpe-bytes bpe) id) do (vector-push-extend b octets)))
    (sb-ext:octets-to-string (coerce octets '(simple-array (unsigned-byte 8) (*))) :external-format '(:utf-8 :replacement #\?))))

(defun save-bpe (bpe path)
  (with-open-file (s path :direction :output :if-exists :supersede :external-format :utf-8)
    (format s "asem-bpe 1~%~d~%" (bpe-vocab-size bpe))
    (loop for (a . b) across (bpe-merges bpe) do (format s "~d ~d~%" a b)))
  path)

(defun load-bpe (path)
  (with-open-file (s path :external-format :utf-8)
    (unless (string= (read-line s) "asem-bpe 1") (error "not an asem-bpe file: ~a" path))
    (let* ((vocab (parse-integer (read-line s))) (merges (loop for line = (read-line s nil) while line
                                                              collect (let ((sp (position #\Space line)))
                                                                        (cons (parse-integer line :end sp)
                                                                              (parse-integer line :start (1+ sp)))))))
      (unless (= vocab (+ 257 (length merges))) (error "corrupt tokenizer file (~d merges for vocab ~d)" (length merges) vocab))
      (finish (coerce merges 'simple-vector)))))
