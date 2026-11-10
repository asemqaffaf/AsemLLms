;;;; Corpus preparation in Common Lisp: JSONL documents -> filter -> deduplicate -> hash split ->
;;;; byte-level BPE -> uint16 shards + manifest.json with SHA-256 and a dataset version.
;;;; Input is JSONL with "text", "score" and "language_score"; there is no downloader (no Parquet/HTTP).
;;;; Splits and the dataset version come from SHA-256, so they are reproducible and verifiable.
(defpackage :asem-data
  (:use :cl :asem-util :asem-sha256 :asem-json :asem-bpe)
  (:export #:default-config #:check-doc #:normalize #:make-deduper #:duplicate-p #:split-of #:build #:verify
           #:read-jsonl #:load-manifest))
(in-package :asem-data)

(defun default-config ()
  (list :min-edu-score 3d0 :min-language-score 0.9d0 :min-chars 500 :max-chars 200000 :min-alpha-ratio 0.6d0
        :min-mean-word-len 3d0 :max-mean-word-len 10d0 :max-dup-line-ratio 0.3d0 :max-symbol-ratio 0.1d0
        :val-permille 5 :vocab-size 8192 :tokenizer-docs 5000 :shard-tokens 5000000 :near-dedup t :max-docs 100000))

(defun cfg (c k) (getf c k))

;;; ------------------------------------------------------------------ filters
(defparameter *symbols* "#{}<>|\\^~`")

(defun mean-word-length (text)
  "Mean length of [A-Za-z']+ runs, and how many there are."
  (let ((total 0) (count 0) (run 0))
    (loop for c across text
          do (if (or (char<= #\a c #\z) (char<= #\A c #\Z) (char= c #\'))
                 (incf run)
                 (when (plusp run) (incf total run) (incf count) (setf run 0))))
    (when (plusp run) (incf total run) (incf count))
    (values (if (plusp count) (/ total count 1d0) 0d0) count)))

(defun check-doc (doc c)
  "Reason keyword the document is rejected for, or NIL. DOC is (:text s :score x :language-score y)."
  (let* ((text (getf doc :text)) (n (length text)))
    (when (< n (cfg c :min-chars)) (return-from check-doc :too-short))
    (when (> n (cfg c :max-chars)) (return-from check-doc :too-long))
    (let ((s (getf doc :score)) (l (getf doc :language-score)))
      (when (and s (< s (cfg c :min-edu-score))) (return-from check-doc :low-edu-score))
      (when (and l (< l (cfg c :min-language-score))) (return-from check-doc :low-language-score)))
    (let ((nonspace (count-if-not #'py-whitespace-p text)))
      (when (zerop nonspace) (return-from check-doc :empty))
      (when (< (/ (count-if #'alpha-char-p text) nonspace 1d0) (cfg c :min-alpha-ratio)) (return-from check-doc :low-alpha-ratio))
      (when (> (/ (count-if (lambda (ch) (find ch *symbols*)) text) nonspace 1d0) (cfg c :max-symbol-ratio))
        (return-from check-doc :symbol-heavy)))
    (multiple-value-bind (mean count) (mean-word-length text)
      (when (zerop count) (return-from check-doc :no-words))
      (unless (<= (cfg c :min-mean-word-len) mean (cfg c :max-mean-word-len)) (return-from check-doc :odd-word-length)))
    (let* ((lines (remove "" (mapcar #'py-strip (split-lines text)) :test #'string=)))
      (when (>= (length lines) 5)
        (let ((uniq (make-hash-table :test 'equal)))
          (dolist (l lines) (setf (gethash l uniq) t))
          (when (> (- 1d0 (/ (hash-table-count uniq) (length lines) 1d0)) (cfg c :max-dup-line-ratio))
            (return-from check-doc :repeated-lines)))))
    nil))

;;; ------------------------------------------------------------------ dedup
(defun normalize (text)
  "Lower-case, every run of non-word characters becomes one space, stripped."
  (let ((out (make-string-output-stream)) (pending nil) (any nil))
    (loop for c across (string-downcase text)
          do (if (or (alphanumericp c) (char= c #\_))
                 (progn (when (and pending any) (write-char #\Space out)) (setf pending nil any t) (write-char c out))
                 (setf pending t)))
    (get-output-stream-string out)))

(defun hash-key (text) (sha256-hex text))

(defparameter *num-perm* 128)
(defparameter *bands* 16)
(defparameter *rows* 8)

(defun fnv32 (string)
  (let ((h 2166136261))
    (loop for c across string do (setf h (ldb (byte 32 0) (* (logxor h (char-code c)) 16777619))))
    h))

(defun perm-params (seed)
  (let ((state (sb-ext:seed-random-state seed)))
    (values (coerce (loop repeat *num-perm* collect (1+ (random (ash 1 31) state))) '(simple-array (unsigned-byte 64) (*)))
            (coerce (loop repeat *num-perm* collect (random (ash 1 31) state)) '(simple-array (unsigned-byte 64) (*))))))

(defun minhash-signature (norm-text a b)
  (declare (type (simple-array (unsigned-byte 64) (*)) a b))
  (let* ((words (coerce (split-on norm-text #\Space) 'simple-vector)) (nw (length words))
         (wh (map 'simple-vector #'fnv32 words)) (k 5)
         (ngrams (max 1 (- nw k -1)))
         (sig (make-array *num-perm* :element-type '(unsigned-byte 64) :initial-element (1- (ash 1 32)))))
    (dotimes (g ngrams)
      (let ((x 2166136261))
        (loop for j from g below (min nw (+ g k))
              do (setf x (ldb (byte 32 0) (* (logxor x (svref wh j)) 16777619))))
        (dotimes (p *num-perm*)
          (let ((v (ldb (byte 32 0) (+ (* (aref a p) x) (aref b p)))))
            (when (< v (aref sig p)) (setf (aref sig p) v))))))
    sig))

(defstruct (deduper (:constructor %make-deduper)) near exact bands a b)

(defun make-deduper (&key (near t) (seed 1))
  (multiple-value-bind (a b) (perm-params seed)
    (%make-deduper :near near :exact (make-hash-table :test 'equal) :a a :b b
                   :bands (coerce (loop repeat *bands* collect (make-hash-table)) 'simple-vector))))

(defun duplicate-p (d text)
  "-> :exact, :near or NIL; remembers TEXT when it is new."
  (let* ((norm (normalize text)) (h (hash-key norm)))
    (when (gethash h (deduper-exact d)) (return-from duplicate-p :exact))
    (let ((keys nil))
      (when (deduper-near d)
        (let ((sig (minhash-signature norm (deduper-a d) (deduper-b d))))
          (setf keys (loop for band below *bands*
                           collect (let ((x 2166136261))
                                     (loop for r below *rows* do (setf x (ldb (byte 62 0) (* (logxor x (aref sig (+ (* band *rows*) r))) 1099511628211))))
                                     x)))
          (loop for k in keys for band from 0
                do (when (gethash k (svref (deduper-bands d) band)) (return-from duplicate-p :near)))))
      (setf (gethash h (deduper-exact d)) t)
      (loop for k in keys for band from 0 do (setf (gethash k (svref (deduper-bands d) band)) t))
      nil)))

(defun split-of (text val-permille &optional (test-permille 0))
  "Document-level split by content hash: buckets [0, val) are :val, [val, val+test) are :test, the rest :train."
  (let* ((hex (hash-key (normalize text))) (bucket (mod (parse-integer hex :end 8 :radix 16) 1000)))
    (cond ((< bucket val-permille) :val)
          ((< bucket (+ val-permille test-permille)) :test)
          (t :train))))

;;; ------------------------------------------------------------------ IO
(defun read-jsonl (path fn)
  "Call FN with (:text :score :language-score) for every line."
  (with-open-file (in path :external-format '(:utf-8 :replacement #\?))
    (loop for line = (read-line in nil) while line
          do (unless (string= (py-strip line) "")
               (let* ((o (parse-json line)) (s (json-get o "score")) (l (json-get o "language_score")))
                 (funcall fn (list :text (json-get o "text" "")
                                   :score (and (realp s) (float s 1d0)) :language-score (and (realp l) (float l 1d0)))))))))

(defstruct (shard-writer (:constructor make-shard-writer (dir split limit)))
  dir split limit (buf (make-array 0 :element-type '(unsigned-byte 32) :adjustable t :fill-pointer 0)) (shards nil) (total 0))

(defun write-shard (w n)
  (let* ((name (format nil "~(~a~)_~5,'0d.bin" (shard-writer-split w) (length (shard-writer-shards w))))
         (path (concatenate 'string (shard-writer-dir w) "/" name))
         (bytes (make-array (* 2 n) :element-type '(unsigned-byte 8))))
    (dotimes (i n)
      (let ((v (aref (shard-writer-buf w) i)))
        (setf (aref bytes (* 2 i)) (ldb (byte 8 0) v) (aref bytes (1+ (* 2 i))) (ldb (byte 8 8) v))))   ; little endian
    (with-open-file (out path :direction :output :if-exists :supersede :element-type '(unsigned-byte 8))
      (write-sequence bytes out))
    (push (list :file name :tokens n :sha256 (sha256-octets-hex bytes)) (shard-writer-shards w))
    (incf (shard-writer-total w) n)
    (let ((rest (subseq (shard-writer-buf w) n)))
      (setf (fill-pointer (shard-writer-buf w)) 0)
      (loop for x across rest do (vector-push-extend x (shard-writer-buf w))))))

(defun shard-add (w ids)
  (dolist (id ids) (vector-push-extend id (shard-writer-buf w)))
  (loop while (>= (length (shard-writer-buf w)) (shard-writer-limit w)) do (write-shard w (shard-writer-limit w))))

(defun shard-close (w)
  (when (plusp (length (shard-writer-buf w))) (write-shard w (length (shard-writer-buf w))))
  (setf (shard-writer-shards w) (nreverse (shard-writer-shards w))))

;;; ------------------------------------------------------------------ build
(defun config-json (c)
  (cons :obj (loop for (k v) on c by #'cddr
                   collect (cons (substitute #\_ #\- (string-downcase (symbol-name k)))
                                 (cond ((eq v t) :true) ((null v) :false) (t v))))))

(defun utf8-bytes (s) (length (sb-ext:string-to-octets s :external-format :utf-8)))

(defun build (read-docs out-dir &rest overrides)
  "READ-DOCS is a function taking a callback; it calls the callback once per document.
Returns the manifest (an :obj)."
  (let* ((c (append overrides (default-config))) (t0 (get-internal-real-time))
         (dedup (make-deduper :near (cfg c :near-dedup))) (stats (make-hash-table))
         (pending-train nil) (pending-val nil) (pending-test nil) (bpe nil) (writers nil)
         (test-pm (or (cfg c :test-permille) 0))          ; absent from the config unless asked for: old manifests are unchanged
         (split-names (if (plusp test-pm) '(:train :val :test) '(:train :val)))
         (bytes (list :train 0 :val 0 :test 0)) (docs (list :train 0 :val 0 :test 0)) (seen 0))
    (ensure-directories-exist (concatenate 'string out-dir "/"))
    (labels ((count! (k) (incf (gethash k stats 0)))
             (start! ()
               (unless bpe (setf bpe (train-bpe (reverse pending-train) (cfg c :vocab-size))))
               (setf writers (loop for sp in split-names append (list sp (make-shard-writer out-dir sp (cfg c :shard-tokens))))))
             (flush! ()
               (start!) (emit :train (reverse pending-train)) (emit :val (reverse pending-val)) (emit :test (reverse pending-test))
               (setf pending-train nil pending-val nil pending-test nil))
             (emit (split texts)
               (dolist (tx texts)
                 (shard-add (getf writers split) (append (bpe-encode bpe tx) (list (bpe-eos bpe))))
                 (incf (getf bytes split) (utf8-bytes tx)) (incf (getf docs split))))
             (on-doc (doc)
               (when (>= seen (cfg c :max-docs)) (return-from on-doc))
               (incf seen) (count! :read)
               (let ((reason (check-doc doc c)))
                 (when reason (count! (intern (format nil "DROPPED-~a" reason) :keyword)) (return-from on-doc)))
               (let ((dup (duplicate-p dedup (getf doc :text))))
                 (when dup (count! (intern (format nil "DROPPED-~a-DUPLICATE" dup) :keyword)) (return-from on-doc)))
               (count! :kept)
               (let ((split (split-of (getf doc :text) (cfg c :val-permille) test-pm)) (text (getf doc :text)))
                 (cond ((null writers)
                        (ecase split (:train (push text pending-train)) (:val (push text pending-val)) (:test (push text pending-test)))
                        (when (>= (length pending-train) (cfg c :tokenizer-docs))
                          (flush!)))
                       (t (emit split (list text)))))))
      (funcall read-docs #'on-doc)
      (unless writers
        (unless pending-train (error "no training documents survived filtering; relax the filters or read more documents"))
        (flush!))
      (dolist (sp split-names) (shard-close (getf writers sp)))
      (when (zerop (shard-writer-total (getf writers :val))) (error "validation split is empty; read more documents or raise val-permille"))
      (when (and (plusp test-pm) (zerop (shard-writer-total (getf writers :test))))
        (error "test split is empty; read more documents or raise test-permille")))
    (let* ((tok-path (concatenate 'string out-dir "/tokenizer.bpe"))
           (tok-sha (progn (save-bpe bpe tok-path) (sha256-file-hex tok-path)))
           (splits (loop for s in split-names
                         collect (let ((w (getf writers s)))
                                   (cons (string-downcase (symbol-name s))
                                         (list :obj (cons "documents" (getf docs s)) (cons "tokens" (shard-writer-total w))
                                               (cons "bytes" (getf bytes s))
                                               (cons "bytes_per_token" (/ (getf bytes s) (max 1 (shard-writer-total w)) 1d0))
                                               (cons "shards" (coerce (mapcar (lambda (sh) (list :obj (cons "file" (getf sh :file)) (cons "tokens" (getf sh :tokens))
                                                                                                 (cons "sha256" (getf sh :sha256))))
                                                                              (shard-writer-shards w)) 'simple-vector)))))))
           (m (list :obj (cons "format" 1) (cons "config" (config-json c)) (cons "dtype" "uint16")
                    (cons "tokenizer" (list :obj (cons "file" "tokenizer.bpe") (cons "sha256" tok-sha)
                                            (cons "vocab_size" (bpe-vocab-size bpe)) (cons "eos_id" 0)))
                    (cons "stats" (cons :obj (loop for k being the hash-keys of stats using (hash-value v)
                                                   collect (cons (substitute #\_ #\- (string-downcase (symbol-name k))) v))))
                    (cons "splits" (cons :obj splits))
                    (cons "seconds" (/ (round (* 10 (/ (- (get-internal-real-time) t0) internal-time-units-per-second))) 10d0)))))
      (let ((m (append m (list (cons "version" (version-of m))))))
        (with-open-file (o (concatenate 'string out-dir "/manifest.json") :direction :output :if-exists :supersede :external-format :utf-8)
          (write-json m o) (terpri o))
        m))))

(defun version-of (m)
  "Hash of everything that determines the data bytes (not timings)."
  (flet ((shas (split) (map 'list (lambda (s) (json-get s "sha256"))
                            (json-get (json-get (json-get m "splits") split) "shards"))))
    (subseq (sha256-hex (with-output-to-string (o)
                          (write-json (list* :obj (cons "config" (json-get m "config"))
                                             (cons "tokenizer" (json-get (json-get m "tokenizer") "sha256"))
                                             (cons "train" (coerce (shas "train") 'simple-vector))
                                             (cons "val" (coerce (shas "val") 'simple-vector))
                                             (cons "format" (json-get m "format"))
                                             (when (json-get (json-get m "splits") "test")      ; only when there is a test split
                                               (list (cons "test" (coerce (shas "test") 'simple-vector)))))
                                      o)))
            0 16)))

(defun load-manifest (dir)
  (parse-json (read-file (concatenate 'string dir "/manifest.json"))))

(defun verify (dir)
  "Re-hash the tokenizer and every shard; signal an error on any mismatch. Returns the version."
  (let ((m (load-manifest dir)))
    (unless (string= (sha256-file-hex (concatenate 'string dir "/" (json-get (json-get m "tokenizer") "file")))
                     (json-get (json-get m "tokenizer") "sha256"))
      (error "tokenizer does not match manifest"))
    (dolist (split (remove-if-not (lambda (sp) (json-get (json-get m "splits") sp)) '("train" "val" "test")))
      (loop for sh across (json-get (json-get (json-get m "splits") split) "shards")
            do (unless (string= (sha256-file-hex (concatenate 'string dir "/" (json-get sh "file"))) (json-get sh "sha256"))
                 (error "~a does not match manifest" (json-get sh "file")))))
    (unless (string= (version-of m) (json-get m "version")) (error "manifest version hash is inconsistent"))
    (json-get m "version")))
