(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "sha256" "json" "bpe" "data")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)
(use-package :asem-data) (use-package :asem-sha256) (use-package :asem-json) (use-package :asem-bpe) (use-package :asem-util)

(defparameter *words* (coerce (asem-util:split-on "the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no she other its may these two like him into time look write go see number way could people my than first water call find long down day did get come made part school science history river garden planet energy language music mountain library teacher student market weather bridge forest ocean" #\Space) 'vector))

(defun make-doc-text (state &optional (n-words 160))
  (with-output-to-string (o)
    (dotimes (s (floor n-words 10))
      (dotimes (w (+ 6 (random 9 state)))
        (let ((word (aref *words* (random (length *words*) state))))
          (write-string (if (zerop w) (string-capitalize word) word) o)
          (unless (= w 0) nil)
          (write-char #\Space o)))
      (write-string ". " o))))

(defun make-docs (n &key (seed 0) (words 160))
  (let ((state (sb-ext:seed-random-state seed)))
    (loop repeat n collect (list :text (make-doc-text state words) :score 3.5d0 :language-score 0.99d0))))

(defun feed (docs) (lambda (cb) (dolist (d docs) (funcall cb d))))
(defparameter *cfg* (list :min-chars 200 :vocab-size 300 :tokenizer-docs 100 :shard-tokens 20000 :val-permille 100))
(defun tmpdir (name) (format nil "/tmp/asem-lisp-data-~a-~d" name (get-universal-time)))
(defun build-in (dir docs &rest cfg) (apply #'build (feed docs) dir (append cfg *cfg*)))
(defun stat (m key) (or (json-get (json-get m "stats") key) 0))
(defun rm-rf (dir) (sb-ext:run-program "/bin/rm" (list "-rf" dir)))

(deftest sha256-standard-vectors
  (check-equal (sha256-hex "abc") "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  (check-equal (sha256-hex "") "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
  (check-equal (sha256-hex "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
  (check-equal (sha256-octets-hex (make-array 1000000 :element-type '(unsigned-byte 8) :initial-element 97))
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
  ;; incremental == one shot, across block boundaries
  (let* ((s (make-sha256)) (data (string-octets (make-string 200 :initial-element #\x))))
    (sha256-update s data 0 63) (sha256-update s data 63 64) (sha256-update s data 64 200)
    (check-equal (sha256-final-hex s) (sha256-octets-hex data))))

(deftest json-parse-and-write
  (let ((v (parse-json "{\"a\": [1, 2.5, -3e2, true, null], \"s\": \"x\\n\\u00e9\\ud83d\\ude00\\\\\", \"o\": {}}")))
    (check-equal (json-get v "a") (vector 1 2.5d0 -300d0 :true :null))
    (check-equal (json-get v "s") (format nil "x~%é~a\\" (code-char #x1f600)))
    (check-equal (json-get v "o") '(:obj)))
  (let ((text (with-output-to-string (o) (write-json (list :obj (cons "b" 1) (cons "a" (vector "x" (list :obj (cons "k" :false))))) o))))
    (check-equal (parse-json text) (list :obj (cons "a" (vector "x" (list :obj (cons "k" :false)))) (cons "b" 1)))   ; keys come back sorted
    (check (< (search "\"a\"" text) (search "\"b\"" text)) "keys are written sorted"))
  (check (handler-case (progn (parse-json "{\"a\": }") nil) (error () t)) "malformed JSON is an error")
  (check (handler-case (progn (parse-json "[1] x") nil) (error () t)) "trailing data is an error"))

(deftest filters
  (let ((c (append (list :min-chars 200) (default-config))) (good (make-doc-text (sb-ext:seed-random-state 1))))
    (flet ((reason (&rest over) (check-doc (append over (list :text good :score 3.5d0 :language-score 0.99d0)) c)))
      (check-equal (reason) nil)
      (check-equal (reason :text "short text.") :too-short)
      (check-equal (reason :score 1.2d0) :low-edu-score)
      (check-equal (reason :language-score 0.5d0) :low-language-score)
      (check-equal (reason :text (apply #'concatenate 'string (loop repeat 100 collect "12345 67890 "))) :low-alpha-ratio)
      (check-equal (reason :text (concatenate 'string (make-string 400 :initial-element #\{) (apply #'concatenate 'string (loop repeat 200 collect " word ")))) :symbol-heavy)
      (check-equal (reason :text (apply #'concatenate 'string (loop repeat 100 collect "a b c d e f g "))) :odd-word-length)
      (check-equal (reason :text (apply #'concatenate 'string
                                        (loop repeat 5 collect (format nil "~{~a~%~}" (append (loop repeat 10 collect "same line here") (list "another distinct line of words" "and one more here"))))))
                   :repeated-lines)
      (check-equal (check-doc (list :text good) c) nil))))     ; missing scores skip those rules

(deftest dedup
  (let* ((d (make-deduper)) (state (sb-ext:seed-random-state 5)) (a (make-doc-text state 400)))
    (check-equal (duplicate-p d a) nil)
    (check-equal (duplicate-p d a) :exact)
    (check-equal (duplicate-p d (string-upcase (substitute #\! #\. a))) :exact)
    (let ((near (concatenate 'string (subseq a 0 (- (length a) 40)) " completely different ending words appended here now")))
      (check-equal (duplicate-p d near) :near))
    (check-equal (duplicate-p d (make-doc-text state 400)) nil))
  (let ((d (make-deduper :near nil)) (a (make-doc-text (sb-ext:seed-random-state 9) 400)))
    (duplicate-p d a)
    (check-equal (duplicate-p d (concatenate 'string a " one more sentence.")) nil)))

(deftest normalize-text
  (check-equal (normalize "Hello,  WORLD!! it's") "hello world it s")
  (check-equal (normalize "  ...  ") ""))

(deftest bpe-roundtrip-and-determinism
  (let* ((texts (loop for d in (make-docs 30 :seed 2) collect (getf d :text)))
         (b (train-bpe texts 320)) (b2 (train-bpe texts 320)))
    (check-equal (bpe-vocab-size b) 320)
    (dolist (s (list "Hello, world!" (format nil "line one~%line two~%") "  leading and trailing  " "café 日本語 emoji \U0001F600 end"
                     "" "the water cycle" (car texts)))
      (check-equal (bpe-decode b (bpe-encode b s)) s))
    (check (every (lambda (id) (< id 320)) (bpe-encode b (car texts))))
    (check (< (length (bpe-encode b (car texts))) (* 0.7 (length (car texts)))) "merges compress the text it was trained on")
    (check-equal (bpe-encode b (car texts)) (bpe-encode b2 (car texts)))        ; deterministic training
    (let ((path (format nil "/tmp/asem-bpe-~d.bpe" (get-universal-time))))
      (save-bpe b path)
      (check-equal (bpe-encode (load-bpe path) (car texts)) (bpe-encode b (car texts)))
      (delete-file path))))

(deftest pretokenize-concatenates-back
  (dolist (s (list "Hello, world!" (format nil "a  b~%~%c") " x" "x " "123 abc!?" "\t tabs \t"))
    (check-equal (apply #'concatenate 'string (pretokenize s)) s)))

(deftest build-outputs-and-verify
  (let* ((dir (tmpdir "a")) (m (build-in dir (make-docs 400))))
    (check (stringp (verify dir)))
    (dolist (s '("train" "val"))
      (let ((sp (json-get (json-get m "splits") s)))
        (check (plusp (json-get sp "tokens")))
        (check-equal (reduce #'+ (map 'list (lambda (x) (json-get x "tokens")) (json-get sp "shards"))) (json-get sp "tokens"))
        (check (> (json-get sp "bytes_per_token") 1d0))))
    (let ((shards (json-get (json-get (json-get m "splits") "train") "shards")))
      (check (> (length shards) 1) "shard-tokens 20000 forces several shards")
      (check (every (lambda (s) (= 20000 (json-get s "tokens"))) (subseq shards 0 (1- (length shards)))))
      (with-open-file (in (format nil "~a/~a" dir (json-get (aref shards 0) "file")) :element-type '(unsigned-byte 8))
        (check-equal (file-length in) 40000)                          ; uint16 little endian
        (let ((buf (make-array 40000 :element-type '(unsigned-byte 8))))
          (read-sequence buf in)
          (check (loop for i below 20000 always (< (+ (aref buf (* 2 i)) (* 256 (aref buf (1+ (* 2 i))))) 300))))))
    (rm-rf dir)))

(deftest build-stats-account-for-every-document
  (let* ((dir (tmpdir "b")) (docs (make-docs 60 :seed 3))
         (docs (append docs (list (copy-list (first docs)) (list :text "tiny" :score 4d0)
                                  (list :text (getf (first (make-docs 1 :seed 99)) :text) :score 0.5d0))))
         (m (build-in dir docs)))
    (check-equal (stat m "read") (length docs))
    (check-equal (stat m "dropped_exact_duplicate") 1)
    (check-equal (stat m "dropped_too_short") 1)
    (check-equal (stat m "dropped_low_edu_score") 1)
    (let ((dropped (loop for (k . v) in (cdr (json-get m "stats")) when (search "dropped_" k) sum v)))
      (check-equal (+ (stat m "kept") dropped) (stat m "read")))
    (check-equal (+ (json-get (json-get (json-get m "splits") "train") "documents") (json-get (json-get (json-get m "splits") "val") "documents"))
                 (stat m "kept"))
    (rm-rf dir)))

(deftest build-is-reproducible-and-version-tracks-the-recipe
  (let* ((docs (make-docs 300 :seed 4)) (a (tmpdir "r1")) (b (tmpdir "r2")) (c (tmpdir "r3"))
         (ma (build-in a docs)) (mb (build-in b docs)) (mc (build-in c docs :min-edu-score 3.2d0)))
    (check-equal (json-get ma "version") (json-get mb "version"))
    (let ((f (json-get (aref (json-get (json-get (json-get ma "splits") "train") "shards") 0) "file")))
      (check-equal (read-file (format nil "~a/~a" a f)) (read-file (format nil "~a/~a" b f))))
    (check (string/= (json-get ma "version") (json-get mc "version")))
    (mapc #'rm-rf (list a b c))))

(deftest verify-detects-corruption
  (let* ((dir (tmpdir "v")) (m (build-in dir (make-docs 300 :seed 6)))
         (f (format nil "~a/~a" dir (json-get (aref (json-get (json-get (json-get m "splits") "val") "shards") 0) "file"))))
    (with-open-file (o f :direction :output :if-exists :overwrite :element-type '(unsigned-byte 8))
      (write-byte 255 o) (write-byte 255 o))
    (check (handler-case (progn (verify dir) nil) (error () t)) "a changed shard must fail verification")
    (rm-rf dir)))

(deftest split-depends-on-content-only
  (let* ((docs (loop for d in (make-docs 200 :seed 11) collect (getf d :text)))
         (a (mapcar (lambda (x) (split-of x 100)) docs)) (b (reverse (mapcar (lambda (x) (split-of x 100)) (reverse docs)))))
    (check-equal a b)
    (check (< 0 (count :val a) 100))))

(deftest empty-input-is-an-error
  (let ((dir (tmpdir "e")))
    (check (handler-case (progn (build-in dir (list (list :text "tiny"))) nil) (error () t)))
    (rm-rf dir)))

(run-all)
