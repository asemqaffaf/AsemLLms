;;; Paired R8 pilot: identical initialization, examples, data order and optimizer; only dense-retrieval normalization changes.
;;; TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/retrieval-weighting-pilot.lisp
;;;   data=data/fineweb/stage1 init=CKPT out=NEW-DIR [steps=300] [workers=4]
;;; Selection uses fresh :new entities / :unseen development wording, seed 18001; never reads the locked test.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(defun save-data (path value)
  (with-open-file (out path :direction :output :if-exists :error :if-does-not-exist :create)
    (with-standard-io-syntax (let ((*print-readably* t)) (write value :stream out)))))
(let* ((data (or (opt-arg "data" nil) (error "data=DIR required")))
       (init (or (opt-arg "init" nil) (error "init=CKPT required")))
       (out (or (opt-arg "out" nil) (error "out=NEW-DIR required")))
       (steps (parse-integer (opt-arg "steps" "300")))
       (workers (parse-integer (opt-arg "workers" "4")))
       (train-n (parse-integer (opt-arg "train-n" "15000")))
       (copy-n (parse-integer (opt-arg "copy-n" "3000")))
       (probe-n (parse-integer (opt-arg "probe-n" "300")))
       (seed (parse-integer (opt-arg "seed" "801")))
       (probe-seed (parse-integer (opt-arg "probe-seed" "18001")))
       (manifest (asem-data:load-manifest data))
       (bpe (asem-bpe:load-bpe (format nil "~a/~a" data (asem-json:json-get (asem-json:json-get manifest "tokenizer") "file"))))
       (model (asem-pretrain32:load-checkpoint init))
       (ctx (asem-tf32:config-ctx (asem-tf32:model-config model)))
       (asem-retrieval:*broad-train* t) (asem-retrieval:*augment* t) (asem-retrieval:*reverse* t) (asem-retrieval:*reasoning* t)
       (asem-retrieval:*vocab-words* (asem-retrieval:vocab-words-of bpe)))
  (when (probe-file out) (error "Refusing to overwrite pilot directory ~a" out))
  (unless (and (plusp steps) (plusp workers) (plusp train-n) (not (minusp copy-n)) (plusp probe-n)) (error "Invalid pilot size"))
  (ensure-directories-exist (format nil "~a/" out))
  (let* ((seqs (asem-retrieval:train-sequences train-n '(1 2 3) seed bpe ctx :options t))
         (copies (asem-sft32:make-copy-examples copy-n 77 :lo 1 :hi 8192 :min-len 10 :max-len 40))
         (probe (asem-retrieval:probe-items 3 :new :unseen probe-n probe-seed :options t
                  :fits (lambda (prompt answer)
                          (<= (+ (length (asem-bpe:bpe-encode bpe prompt)) (length (asem-bpe:bpe-encode bpe (format nil " ~a" answer)))) ctx))))
         (dev (map 'vector (lambda (it)
                            (asem-sft32:generation-ex (getf it :id) (getf it :kind) "selection"
                              (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :question) :options (getf it :options))
                              (getf it :answer) bpe ctx)) probe))
         (train-file (format nil "~a/retrieval-data.sexp" out)) (probe-file (format nil "~a/selection-data.sexp" out)))
    (unless (= (length seqs) train-n) (error "Could not generate the requested training count"))
    (save-data train-file seqs) (save-data probe-file probe)
    (save-data (format nil "~a/recipe.sexp" out)
               (list :init init :init-sha256 (asem-sha256:sha256-file-hex init) :steps steps :workers workers
                     :seed seed :probe-seed probe-seed :train-n train-n :copy-n copy-n :copy-seed 77 :probe-n probe-n
                     :selection-entities :new :selection-wording :unseen :lr 1f-4 :warmup 30 :batch 32
                     :train-sha256 (asem-sha256:sha256-file-hex train-file) :selection-sha256 (asem-sha256:sha256-file-hex probe-file)
                     :primary-endpoint :fixed-final :note "Development pilot; previously inspected template family; no locked test. Normalization also changes relative copy replay weight."))
    (dolist (normalization '(:token :sequence))
      (let* ((train (concatenate 'vector
                     (loop for (ids rows targets) in seqs for i from 0
                           collect (asem-sft32:multi-ex (format nil "r8-~d" i) ids rows targets :normalization normalization)) copies))
             (arm-out (format nil "~a/~(~a~)" out normalization)))
        (asem-sft32:sft arm-out train dev (asem-sft32:letter-token-ids bpe)
                       :init-model (asem-pretrain32:load-checkpoint init) :steps steps :batch 32 :workers workers
                       :lr 1f-4 :warmup 30 :eval-every 100 :log-every 20 :train-probe 100 :seed seed
                       :header (list :experiment "R8 normalization pilot" :normalization (string-downcase (symbol-name normalization))
                                     :init init :init-sha256 (asem-sha256:sha256-file-hex init)
                                     :retrieval-data-sha256 (asem-sha256:sha256-file-hex train-file)
                                     :selection-data-sha256 (asem-sha256:sha256-file-hex probe-file)
                                     :selection-seed probe-seed :selection-wording "unseen" :primary-endpoint "fixed final")))
      (sb-ext:gc :full t))))
