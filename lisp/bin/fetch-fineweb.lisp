;;; Python-free FineWeb-Edu import straight from the official Parquet shards (HTTP range requests, one row group
;;; of ~3 MB at a time), filtered and written as the JSONL that bin/build-data.lisp reads.
;;;   sbcl --script lisp/bin/fetch-fineweb.lisp OUT.jsonl [max-tokens=10000000] [min-score=4.0] [min-lang=0.95]
;;;        [shards=sample/10BT/000_00000.parquet[,sample/10BT/000_00001.parquet,...]] [start=0] [resume=1]
;;; OUT.jsonl.progress records the next row group after each one; resume=1 appends to OUT.jsonl from there, so a network failure
;;; (the program then stops cleanly, keeping what it wrote) never costs the data already downloaded.
;;; Reading stops once max-tokens (FineWeb's own token_count, a Llama-style tokenizer) have been written. START is the
;;; first row group to read, to resume or to take a different slice. Shards are read from the start in order, so the
;;; same arguments always give the same file.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "json" "parquet")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(in-package :asem-parquet)

(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))

(defun split-commas (s)
  (let ((out nil) (start 0))
    (loop for i = (position #\, s :start start)
          do (push (subseq s start i) out) (if i (setf start (1+ i)) (return)))
    (nreverse out)))

(defun write-jsonl-line (o text score lang id url dump)
  "One compact JSON object on one line (ASEM-JSON:WRITE-JSON pretty-prints over many lines, which is not JSONL)."
  (let ((*read-default-float-format* 'double-float))
    (format o "{\"text\":~a,\"score\":~a,\"language_score\":~a,\"id\":~a,\"url\":~a,\"dump\":~a}~%"
            (asem-json::json-string text) score lang (asem-json::json-string id) (asem-json::json-string url) (asem-json::json-string dump))))

(defun octets-string (v) (and v (sb-ext:octets-to-string v :external-format '(:utf-8 :replacement #\?))))

;; The output stream is closed normally on every exit (never WITH-OPEN-FILE's abort, which would delete a new file on an error).
(defun read-progress (path)
  (when (probe-file path)
    (with-open-file (in path) (let ((*read-eval* nil)) (read in)))))

(defun write-progress (path shard-index rg docs tokens rows)
  (with-open-file (o path :direction :output :if-exists :supersede)
    (format o "(~d ~d ~d ~d ~d)~%" shard-index rg docs tokens rows)))

(let* ((args (remove-if (lambda (a) (find #\= a)) (cdr sb-ext:*posix-argv*)))
       (out-path (or (first args) (progn (format *error-output* "usage: fetch-fineweb.lisp OUT.jsonl [key=value ...]~%") (sb-ext:exit :code 2))))
       (progress-path (concatenate 'string out-path ".progress"))
       (max-tokens (parse-integer (opt-arg "max-tokens" "10000000")))
       (min-score (float (read-from-string (opt-arg "min-score" "4.0")) 1d0)) (min-lang (float (read-from-string (opt-arg "min-lang" "0.95")) 1d0))
       (shards (split-commas (opt-arg "shards" "sample/10BT/000_00000.parquet")))
       (start (parse-integer (opt-arg "start" "0")))
       (resume (equal (opt-arg "resume" "0") "1"))
       (prog (and resume (read-progress progress-path)))
       (shard-start (if prog (first prog) 0)) (rg-start (if prog (second prog) start))
       (docs (if prog (third prog) 0)) (tokens (if prog (fourth prog) 0)) (rows-read (if prog (fifth prog) 0))
       (t0 (get-universal-time)) (failed nil)
       (o (open out-path :direction :output :if-exists (if prog :append :supersede) :if-does-not-exist :create :external-format :utf-8)))
  (when prog (format t "resuming at shard ~d, row group ~d with ~d docs / ~,2f M tokens~%" shard-start rg-start docs (/ tokens 1d6)))
  (unwind-protect
       (block fetching
         (loop for shard in shards for si from 0
               do (when (>= si shard-start)
                    (let* ((url (format nil "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/~a" shard))
                           (src (open-source url)) (footer (read-footer src)) (ngroups (length (footer-row-groups footer))))
                      (format t "~a: ~d row groups, ~d rows~%" shard ngroups (tget footer 3))
                      (loop for rg from (if (= si shard-start) rg-start 0) below ngroups
                            do (let ((h (handler-case (read-row-group src footer rg '("text" "id" "url" "dump" "score" "language_score" "token_count"))
                                          (error (e) (format t "stopping at shard ~d row group ~d: ~a~%" si rg e) (setf failed t) (return-from fetching)))))
                                 (let ((texts (gethash "text" h)) (scores (gethash "score" h)) (langs (gethash "language_score" h)) (counts (gethash "token_count" h)))
                                   (dotimes (i (length texts))
                                     (incf rows-read)
                                     (let ((s (aref scores i)) (l (aref langs i)) (tx (aref texts i)))
                                       (when (and s l tx (>= s min-score) (>= l min-lang))
                                         (write-jsonl-line o (octets-string tx) s l (or (octets-string (aref (gethash "id" h) i)) "")
                                                           (or (octets-string (aref (gethash "url" h) i)) "") (or (octets-string (aref (gethash "dump" h) i)) ""))
                                         (incf docs) (incf tokens (or (aref counts i) 0))))))
                                 (finish-output o)
                                 (write-progress progress-path si (1+ rg) docs tokens rows-read)
                                 (when (>= tokens max-tokens) (return-from fetching))
                                 (when (zerop (mod (1+ rg) 10))
                                   (format t "  ~a row group ~d: ~d of ~d rows kept, ~,2f M tokens, ~d s~%" shard (1+ rg) docs rows-read (/ tokens 1d6) (- (get-universal-time) t0))
                                   (finish-output *standard-output*))))))))
    (close o))
  (format t "~a ~d documents, ~,2f M tokens (FineWeb count) from ~d rows to ~a in ~d s~%" (if failed "STOPPED EARLY:" "wrote") docs (/ tokens 1d6) rows-read out-path (- (get-universal-time) t0))
  (when failed (format t "rerun the same command with resume=1 to continue~%") (sb-ext:exit :code 3)))
