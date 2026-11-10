;;;; The Parquet reader's building blocks against hand-made encodings (no network). The real-file check, a FineWeb-Edu row
;;;; compared with what the dataset server returns for the same id, is documented in README.md.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "parquet")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(defun oct (&rest bytes) (make-array (length bytes) :element-type '(unsigned-byte 8) :initial-contents bytes))
(defun str-oct (s) (map '(simple-array (unsigned-byte 8) (*)) #'char-code s))
(defun cat (&rest vs) (apply #'concatenate '(simple-array (unsigned-byte 8) (*)) vs))
(defun octs->string (v) (map 'string #'code-char v))

(deftest snappy-literals-and-copies
  ;; "abc" literal, then a 2-byte-offset copy of length 9 at offset 3 (overlapping): "abcabcabcabc"
  (check-equal (octs->string (asem-parquet:snappy-decompress (cat (oct 12) (oct (ash 2 2)) (str-oct "abc") (oct (logior (ash 8 2) 2) 3 0))))
               "abcabcabcabc")
  ;; 1-byte-offset copy (tag kind 1): length 4 + ((tag>>2)&7), offset (tag>>5)<<8 | next byte
  (check-equal (octs->string (asem-parquet:snappy-decompress (cat (oct 10) (oct (ash 1 2)) (str-oct "xy") (oct (logior 1 (ash 4 2) (ash 0 5)) 2))))
               "xyxyxyxyxy")
  ;; a literal longer than 60 bytes uses the extra-length bytes
  (let* ((text (make-string 100 :initial-element #\q)))
    (check-equal (octs->string (asem-parquet:snappy-decompress (cat (oct 100) (oct (ash 60 2) 99) (str-oct text)))) text))
  (check (handler-case (progn (asem-parquet:snappy-decompress (cat (oct 5) (oct (ash 0 2)) (str-oct "a"))) nil) (error () t))
         "a block that ends short of its declared length is an error")
  (check (handler-case (progn (asem-parquet:snappy-decompress (cat (oct 4) (oct (logior (ash 3 2) 2) 9 0))) nil) (error () t))
         "a copy before the start of the output is an error"))

(defun pack-bits (values bit-width)
  "LSB-first bit packing, as Parquet's bit-packed runs."
  (let* ((n (length values)) (bytes (make-array (ceiling (* n bit-width) 8) :element-type '(unsigned-byte 8) :initial-element 0)))
    (loop for v in values for i from 0
          do (dotimes (b bit-width)
               (when (logbitp b v) (let ((bp (+ (* i bit-width) b))) (setf (aref bytes (ash bp -3)) (logior (aref bytes (ash bp -3)) (ash 1 (logand bp 7))))))))
    bytes))

(deftest rle-hybrid-runs-and-bit-packing
  (check-equal (coerce (asem-parquet::rle-decode (oct 10 3) 0 2 3 5) 'list) '(3 3 3 3 3))                 ; RLE run: 5 x 3
  (let ((packed (pack-bits '(0 1 2 3 4 5 6 7) 3)))                                                          ; one bit-packed group of 8
    (check-equal (coerce (asem-parquet::rle-decode (cat (oct 3) packed) 0 (1+ (length packed)) 3 8) 'list) '(0 1 2 3 4 5 6 7)))
  ;; a run followed by a packed group, and a request for fewer values than the group holds
  (let ((packed (pack-bits '(1 0 1 1 0 0 1 0) 1)))
    (check-equal (coerce (asem-parquet::rle-decode (cat (oct 6 1) (oct 3) packed) 0 (+ 3 (length packed)) 1 6) 'list) '(1 1 1 1 0 1)) ; 3 x 1 then 1 0 1 ...
    (check-equal (coerce (asem-parquet::rle-decode (cat (oct 3) packed) 0 (1+ (length packed)) 1 5) 'list) '(1 0 1 1 0)))
  (check (handler-case (progn (asem-parquet::rle-decode (oct 10 3) 0 2 3 9) nil) (error () t)) "asking for more values than encoded is an error"))

(deftest thrift-compact-structs
  ;; field 1 i32 = 5, field 2 binary "hi", field 4 list<i32> [1, -1], field 6 nested struct {1: i64 = -2}, stop
  (let* ((buf (cat (oct #x15 10) (oct #x18 2) (str-oct "hi") (oct #x29 #x25 2 1) (oct #x2c #x16 3 0) (oct 0)))
         (s (asem-parquet::tr-struct (asem-parquet::make-tr buf))))
    (check-equal (asem-parquet:tget s 1) 5)
    (check-equal (octs->string (asem-parquet:tget s 2)) "hi")
    (check-equal (coerce (asem-parquet:tget s 4) 'list) '(1 -1))
    (check-equal (asem-parquet:tget (asem-parquet:tget s 6) 1) -2)
    (check-equal (asem-parquet:tget s 9 :missing) :missing)))

(deftest plain-values-decode
  (multiple-value-bind (v end) (asem-parquet::plain-values (cat (oct 2 0 0 0) (str-oct "ab") (oct 0 0 0 0)) 0 2 6)
    (check-equal (list (octs->string (aref v 0)) (length (aref v 1)) end) '("ab" 0 10)))
  (multiple-value-bind (v) (asem-parquet::plain-values (oct 0 0 0 0 0 0 #xf0 #x3f  0 0 0 0 0 0 0 #xc0) 0 2 5)    ; 1.0d0 and -2.0d0
    (check-equal (coerce v 'list) '(1d0 -2d0)))
  (multiple-value-bind (v) (asem-parquet::plain-values (oct #xfe #xff #xff #xff #xff #xff #xff #xff) 0 1 2)       ; int64 -2
    (check-equal (aref v 0) -2)))

(run-all)
