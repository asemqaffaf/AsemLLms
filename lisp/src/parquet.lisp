;;;; A small Parquet reader in plain Common Lisp (no libraries): Thrift compact footer, Snappy pages, PLAIN and dictionary
;;;; encodings, the RLE/bit-packed hybrid for levels and dictionary indices. It covers what the FineWeb-Edu shards use
;;;; (flat schema, Snappy, data pages v1/v2, BYTE_ARRAY / DOUBLE / INT32 / INT64 columns). Anything else signals an error.
;;;; Remote files are read with HTTP range requests through curl, so a 2 GB shard is never downloaded whole.
(defpackage :asem-parquet
  (:use :cl)
  (:export #:snappy-decompress #:open-source #:source-size #:read-range #:read-footer #:footer-columns #:footer-row-groups
           #:column-index #:chunk-info #:read-column #:read-row-group #:tget))
(in-package :asem-parquet)

(deftype octets () '(simple-array (unsigned-byte 8) (*)))

;;; ---------------------------------------------------------------- Snappy (raw block format)
(defun snappy-decompress (src &optional (start 0) (end (length src)))
  (declare (type octets src) (type fixnum start end) (optimize (speed 3) (safety 1)))
  (let ((pos start) (ulen 0))
    (declare (type fixnum pos ulen))
    (let ((shift 0))
      (declare (type fixnum shift))
      (loop (let ((b (aref src pos)))
              (incf pos) (incf ulen (ash (logand b 127) shift)) (incf shift 7)
              (when (< b 128) (return)))))
    (let ((out (make-array ulen :element-type '(unsigned-byte 8))) (op 0))
      (declare (type fixnum op))
      (flet ((copy (off len)
               (declare (type fixnum off len))
               (when (or (zerop off) (> off op)) (error "snappy: bad copy offset ~d at ~d" off op))
               (dotimes (i len) (setf (aref out (+ op i)) (aref out (+ op i (- off)))))
               (incf op len)))
        (loop while (< pos end)
              do (let* ((tag (aref src pos)) (kind (logand tag 3)))
                   (declare (type fixnum tag kind))
                   (incf pos)
                   (case kind
                     (0 (let ((len (ash tag -2)))
                          (declare (type fixnum len))
                          (if (< len 60)
                              (incf len)
                              (let ((nb (- len 59)) (v 0))
                                (declare (type fixnum nb v))
                                (dotimes (i nb) (incf v (ash (aref src (+ pos i)) (* 8 i))))
                                (incf pos nb) (setf len (1+ v))))
                          (replace out src :start1 op :start2 pos :end2 (+ pos len))
                          (incf op len) (incf pos len)))
                     (1 (let ((len (+ 4 (logand (ash tag -2) 7))) (off (+ (ash (ash tag -5) 8) (aref src pos))))
                          (incf pos) (copy off len)))
                     (2 (let ((len (1+ (ash tag -2))) (off (+ (aref src pos) (ash (aref src (1+ pos)) 8))))
                          (incf pos 2) (copy off len)))
                     (t (let ((len (1+ (ash tag -2)))
                              (off (+ (aref src pos) (ash (aref src (+ pos 1)) 8) (ash (aref src (+ pos 2)) 16) (ash (aref src (+ pos 3)) 24))))
                          (incf pos 4) (copy off len)))))))
      (unless (= op ulen) (error "snappy: produced ~d bytes, header said ~d" op ulen))
      out)))

;;; ---------------------------------------------------------------- Thrift compact protocol (generic reader)
;;; A struct becomes an alist (field-id . value); lists become vectors; binary becomes an octet vector.
(defstruct (tr (:constructor make-tr (buf &optional (pos 0)))) (buf nil :type octets) (pos 0 :type fixnum))

(defun tr-byte (r) (prog1 (aref (tr-buf r) (tr-pos r)) (incf (tr-pos r))))
(defun tr-varint (r)
  (let ((n 0) (shift 0))
    (loop (let ((b (tr-byte r))) (incf n (ash (logand b 127) shift)) (incf shift 7) (when (< b 128) (return n))))))
(defun zigzag (n) (logxor (ash n -1) (- (logand n 1))))

(defun tr-value (r type)
  (case type
    (1 t) (2 nil)
    (3 (let ((b (tr-byte r))) (if (>= b 128) (- b 256) b)))
    ((4 5 6) (zigzag (tr-varint r)))
    (7 (let ((lo 0)) (dotimes (i 8) (incf lo (ash (tr-byte r) (* 8 i)))) (sb-kernel:make-double-float (let ((hi (ldb (byte 32 32) lo))) (if (>= hi #x80000000) (- hi #x100000000) hi)) (ldb (byte 32 0) lo))))
    (8 (let* ((n (tr-varint r)) (p (tr-pos r)) (v (subseq (tr-buf r) p (+ p n)))) (setf (tr-pos r) (+ p n)) v))
    ((9 10) (let* ((h (tr-byte r)) (et (logand h 15)) (n (ash h -4)))
              (when (= n 15) (setf n (tr-varint r)))
              (let ((v (make-array n)))
                (dotimes (i n v)
                  (setf (aref v i) (if (member et '(1 2)) (= (tr-byte r) 1) (tr-value r et)))))))
    (12 (tr-struct r))
    (t (error "thrift: unsupported type ~d" type))))

(defun tr-struct (r)
  (let ((last 0) (out nil))
    (loop (let ((b (tr-byte r)))
            (when (zerop b) (return (nreverse out)))
            (let* ((delta (ash b -4)) (type (logand b 15))
                   (id (if (zerop delta) (zigzag (tr-varint r)) (+ last delta))))
              (setf last id)
              (push (cons id (tr-value r type)) out))))))

(defun tget (struct id &optional default) (let ((c (assoc id struct))) (if c (cdr c) default)))

;;; ---------------------------------------------------------------- sources: local file or URL (HTTP range via curl)
(defstruct source url path (size 0) (tmp nil))

(defun run-curl (args)
  (let ((p (sb-ext:run-program "/usr/bin/curl" args :output nil :error nil :wait t)))
    (zerop (sb-ext:process-exit-code p))))

(defun read-file-octets (path &optional (start 0) end)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let* ((e (or end (file-length in))) (v (make-array (- e start) :element-type '(unsigned-byte 8))))
      (file-position in start) (read-sequence v in) v)))

(defun open-source (location)
  "LOCATION is a local path or an http(s) URL."
  (if (and (> (length location) 4) (string= "http" location :end2 4))
      (let* ((tmp (format nil "/tmp/asem-parquet-~d-~d.part" (sb-posix:getpid) (random 1000000 (make-random-state t))))
             (hdr (with-output-to-string (o)
                    (sb-ext:run-program "/usr/bin/curl" (list "-sIL" "-m" "60" location) :output o :error nil :wait t)))
             (size (let ((best 0))   ; the last content-length / x-linked-size in the redirect chain
                     (dolist (line (asem-util:split-lines hdr) best)
                       (let ((l (string-downcase (string-right-trim '(#\Return) line))))
                         (dolist (key '("content-length: " "x-linked-size: "))
                           (when (and (> (length l) (length key)) (string= key l :end2 (length key)))
                             (let ((n (parse-integer l :start (length key) :junk-allowed t))) (when (and n (> n 100000)) (setf best n))))))))))
        (when (zerop size) (error "cannot determine the size of ~a" location))
        (make-source :url location :size size :tmp tmp))
      (make-source :path location :size (with-open-file (in location :element-type '(unsigned-byte 8)) (file-length in)))))

(defun source-size* (s) (source-size s))

(defun read-range (s start end)
  "Octets [START, END) of the source."
  (when (> end (source-size s)) (setf end (source-size s)))
  (if (source-path s)
      (read-file-octets (source-path s) start end)
      (dotimes (attempt 14 (error "range ~d-~d of ~a failed after 14 attempts" start end (source-url s)))
        (when (run-curl (list "-sL" "-m" "300" "-r" (format nil "~d-~d" start (1- end)) "-o" (source-tmp s) (source-url s)))
          (let ((v (ignore-errors (read-file-octets (source-tmp s)))))
            (when (and v (= (length v) (- end start))) (return v))))
        (sleep (min 30 (* 2 (1+ attempt)))))))

;;; ---------------------------------------------------------------- footer
(defun read-footer (s)
  "Parsed FileMetaData (an alist, see the Parquet spec): 2 schema, 3 num_rows, 4 row_groups."
  (let* ((size (source-size s)) (tail (read-range s (- size 8) size)))
    (unless (equalp (subseq tail 4) #(80 65 82 49)) (error "not a Parquet file"))
    (let* ((len (+ (aref tail 0) (ash (aref tail 1) 8) (ash (aref tail 2) 16) (ash (aref tail 3) 24)))
           (meta (read-range s (- size 8 len) (- size 8))))
      (tr-struct (make-tr meta)))))

(defun footer-columns (footer)
  "List of (name physical-type repetition) for the leaf columns of a flat schema."
  (let ((schema (tget footer 2)))
    (loop for i from 1 below (length schema)
          collect (let ((e (aref schema i)))
                    (list (map 'string #'code-char (tget e 4)) (tget e 1) (tget e 3 0))))))

(defun footer-row-groups (footer) (tget footer 4))

(defun column-index (footer name)
  (or (position name (footer-columns footer) :key #'first :test #'string=) (error "no column ~a" name)))

(defun chunk-info (footer rg col)
  "Plist for column COL of row group RG: :start :size :codec :num-values :type :nullable."
  (let* ((group (aref (footer-row-groups footer) rg)) (chunk (aref (tget group 1) col)) (md (tget chunk 3))
         (dict (tget md 11)) (data (tget md 9)) (col-def (nth col (footer-columns footer))))
    (list :start (if (and dict (plusp dict)) (min dict data) data) :size (tget md 7) :codec (tget md 4)
          :num-values (tget md 5) :type (tget md 1) :nullable (= 1 (third col-def)) :encodings (tget md 2)
          :rows (tget group 3))))

;;; ---------------------------------------------------------------- RLE / bit-packed hybrid
(defun rle-decode (buf start end bit-width count)
  "COUNT integers from the hybrid encoding in BUF[START,END)."
  (declare (type octets buf) (type fixnum start end bit-width count))
  (let ((out (make-array count :initial-element 0)) (pos start) (n 0))
    (declare (type fixnum pos n))
    (flet ((varint ()
             (let ((v 0) (shift 0))
               (loop (let ((b (aref buf pos))) (incf pos) (incf v (ash (logand b 127) shift)) (incf shift 7) (when (< b 128) (return v)))))))
      (loop while (and (< n count) (< pos end))
            do (let ((h (varint)))
                 (if (logbitp 0 h)
                     (let* ((groups (ash h -1)) (vals (* groups 8)) (bitpos (* pos 8)))
                       (dotimes (i vals)
                         (when (< n count)
                           (let ((v 0))
                             (dotimes (b bit-width)
                               (let ((bp (+ bitpos (* i bit-width) b)))
                                 (when (logbitp (logand bp 7) (aref buf (ash bp -3))) (setf v (logior v (ash 1 b))))))
                             (setf (aref out n) v) (incf n))))
                       (incf pos (* groups bit-width)))
                     (let* ((run (ash h -1)) (nb (ceiling bit-width 8)) (v 0))
                       (dotimes (i nb) (incf v (ash (aref buf (+ pos i)) (* 8 i))))
                       (incf pos nb)
                       (dotimes (i run) (when (< n count) (setf (aref out n) v) (incf n))))))))
    (unless (= n count) (error "rle: decoded ~d of ~d values" n count))
    out))

;;; ---------------------------------------------------------------- PLAIN values
(defun u32 (buf p) (+ (aref buf p) (ash (aref buf (+ p 1)) 8) (ash (aref buf (+ p 2)) 16) (ash (aref buf (+ p 3)) 24)))
(defun u64 (buf p) (+ (u32 buf p) (ash (u32 buf (+ p 4)) 32)))

(defun plain-values (buf pos count type)
  "COUNT PLAIN values of physical TYPE starting at POS: (values vector end-pos)."
  (let ((out (make-array count)))
    (ecase type
      (6 (dotimes (i count) (let ((len (u32 buf pos))) (setf (aref out i) (subseq buf (+ pos 4) (+ pos 4 len))) (incf pos (+ 4 len)))))
      (5 (dotimes (i count) (let ((lo (u32 buf pos)) (hi (u32 buf (+ pos 4))))
                              (setf (aref out i) (sb-kernel:make-double-float (if (>= hi #x80000000) (- hi #x100000000) hi) lo)) (incf pos 8))))
      (2 (dotimes (i count) (let ((v (u64 buf pos))) (setf (aref out i) (if (logbitp 63 v) (- v (ash 1 64)) v)) (incf pos 8))))
      (1 (dotimes (i count) (let ((v (u32 buf pos))) (setf (aref out i) (if (logbitp 31 v) (- v (ash 1 32)) v)) (incf pos 4)))))
    (values out pos)))

;;; ---------------------------------------------------------------- column chunks
(defun page-bytes (buf pos size codec)
  (ecase codec
    (0 (subseq buf pos (+ pos size)))
    (1 (snappy-decompress buf pos (+ pos size)))))

(defun read-column (buf base info)
  "All values of one column chunk as a simple-vector (NIL for nulls). BUF holds the file bytes from offset BASE on."
  (let* ((type (getf info :type)) (codec (getf info :codec)) (nullable (getf info :nullable)) (total (getf info :num-values))
         (pos (- (getf info :start) base)) (dict nil) (out (make-array total)) (n 0))
    (loop while (< n total)
          do (let* ((r (make-tr buf pos)) (h (tr-struct r)) (hend (tr-pos r)) (ptype (tget h 1))
                    (usize (tget h 2)) (csize (tget h 3)))
               (declare (ignorable usize))
               (setf pos (+ hend csize))
               (case ptype
                 (2 (setf dict (plain-values (page-bytes buf hend csize codec) 0 (tget (tget h 7) 1) type)))
                 ((0 3)
                  (let* ((v2 (= ptype 3)) (dh (tget h (if v2 8 5))) (nv (tget dh 1)) (enc (tget dh (if v2 4 2)))
                         (page nil) (p 0) (defs nil) (nonnull nv))
                    (if v2
                        (let* ((dl (tget dh 5)) (rl (tget dh 6)) (compressed (tget dh 7 t)) (lv (+ dl rl)))
                          (when (plusp rl) (error "parquet: repeated columns are not supported"))
                          (setf page (concatenate 'octets (subseq buf hend (+ hend lv))
                                                  (if compressed (page-bytes buf (+ hend lv) (- csize lv) codec) (subseq buf (+ hend lv) (+ hend csize)))))
                          (when (and nullable (plusp dl)) (setf defs (rle-decode page 0 dl 1 nv)))
                          (setf p lv))
                        (progn (setf page (page-bytes buf hend csize codec))
                               (when nullable
                                 (let ((dl (u32 page 0))) (setf defs (rle-decode page 4 (+ 4 dl) 1 nv)) (setf p (+ 4 dl))))))
                    (when defs (setf nonnull (count 1 defs)))
                    (let ((vals (case enc
                                  (0 (plain-values page p nonnull type))
                                  ((2 8) (unless dict (error "parquet: dictionary page missing"))
                                   (let* ((bw (aref page p)) (idx (rle-decode page (1+ p) (length page) bw nonnull)) (v (make-array nonnull)))
                                     (dotimes (i nonnull) (setf (aref v i) (aref dict (aref idx i))))
                                     v))
                                  (t (error "parquet: unsupported encoding ~d" enc)))))
                      (if defs
                          (let ((k 0)) (dotimes (i nv) (setf (aref out (+ n i)) (when (= 1 (aref defs i)) (prog1 (aref vals k) (incf k))))))
                          (replace out vals :start1 n))
                      (incf n nv))))
                 (t (error "parquet: unexpected page type ~d" ptype)))))
    out))

(defun read-row-group (s footer rg names)
  "Hash table name -> value vector for the named columns of row group RG, with one range request."
  (let* ((cols (mapcar (lambda (nm) (column-index footer nm)) names))
         (infos (mapcar (lambda (c) (chunk-info footer rg c)) cols))
         (lo (reduce #'min (mapcar (lambda (i) (getf i :start)) infos)))
         (hi (reduce #'max (mapcar (lambda (i) (+ (getf i :start) (getf i :size))) infos)))
         (buf (read-range s lo hi)) (h (make-hash-table :test 'equal)))
    (loop for nm in names for info in infos do (setf (gethash nm h) (read-column buf lo info)))
    h))
