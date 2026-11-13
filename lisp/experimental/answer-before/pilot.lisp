;;;; Pilot 2 (answer-before-entity binding). Isolated experimental driver; the stable generator and trainer are not modified.
;;;; Protocol: docs/results/stage2/mission-20261111/pilot2/protocol.md (frozen before this file was first run).
;;;;   sbcl --dynamic-space-size 6GB --script lisp/experimental/answer-before/pilot.lisp mode=prepare
;;;;   TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 6GB --script lisp/experimental/answer-before/pilot.lisp mode=train arm=control|treatment
(let ((src (merge-pathnames "../../src/" (directory-namestring *load-truename*))))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis"))
    (load (merge-pathnames (format nil "~a.lisp" f) src))))
(defpackage :answer-before (:use :cl))
(in-package :answer-before)

(defparameter +seed+ 111302)
(defparameter +dense+ 15000)
(defparameter +copy+ 3000)
(defparameter +steps+ 500)
(defparameter +bundle+ "docs/results/stage2/mission-20261111/pilot2")
(defparameter +run+ "data/runs/retrieval/p2-answer-before")
(defparameter +init+ "data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt")
(defparameter +reference+ "33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1")

(defun opt (key)
  (let ((p (concatenate 'string key "=")))
    (let ((a (find-if (lambda (a) (and (>= (length a) (length p)) (string= p a :end2 (length p)))) (cdr sb-ext:*posix-argv*)))) (and a (subseq a (length p))))))
(defun obj (&rest x) (apply #'asem-qa-analysis::obj x))
(defun qsym (name) (find-symbol name :asem-qagen))
(defun qf (name &rest args) (apply (symbol-function (find-symbol name :asem-qagen)) args))
(defun rf (name &rest args) (apply (symbol-function (find-symbol name :asem-retrieval)) args))
(defun rng-for (i attempt stream) (sb-ext:seed-random-state (+ (* +seed+ 1000003) (* i 7919) (* attempt 104729) stream)))

(defun person-first-p (frame)
  (< (search "{P}" frame) (or (search "{V}" frame) (search "{aV}" frame))))

(defun answer-before-fact-set (rng)
  "Four people, one attribute type. Every question's answer precedes the queried entity in its sentence:
reverse (value -> person) on a person-first frame, or extraction (person -> value) on a value-first frame when the type has one."
  (let* ((ty (qf "PICK" (symbol-value (qsym "*ATTR-TYPES*")) rng)) (spec (cdr ty))
         (names (rf "POOL" :train '*names*)) (pool (rf "POOL" :train (getf spec :pool)))
         (people (qf "SAMPLE-N" names 4 rng)) (vals (qf "SAMPLE-N" pool 4 rng))
         (frames (rf "FRAMES-OF" spec :frames :seen))
         (pf (remove-if-not #'person-first-p frames)) (vf (remove-if #'person-first-p frames))
         (qext (rf "FRAMES-OF" spec :q-extract :seen)) (qrev (rf "FRAMES-OF" spec :q-reverse :seen))
         (sents nil) (qa nil) (nrev 0) (next 0))
    (loop for p in people for v in vals do
      (if (and vf (zerop (random 2 rng)))
          (progn (incf next) (push (qf "FILL-FRAME" (qf "PICK" vf rng) :p p :v v) sents) (push (cons (qf "FILL-FRAME" (qf "PICK" qext rng) :p p) v) qa))
          (progn (incf nrev) (push (qf "FILL-FRAME" (qf "PICK" pf rng) :p p :v v) sents) (push (cons (qf "FILL-FRAME" (qf "PICK" qrev rng) :v v) p) qa))))
    (setf sents (nreverse sents) qa (nreverse qa))
    (multiple-value-bind (s qq) (rf "AUGMENTED" sents qa people (car ty) (append people vals) rng people ty)
      (values (format nil "~{~a~^ ~}" (qf "SHUFFLE" s rng)) qq
              (lambda (ans) (if (member ans people :test #'string=) people vals))
              (lambda (ans) (if (member ans people :test #'string=) (lambda () (qf "PICK" names rng)) (lambda () (qf "PICK" pool rng))))
              people))))

(defun build (passage qa cands padder rng bpe ctx)
  "Same packing as stable TRAIN-SEQUENCES with option lines. Returns (ids rows targets texts n-questions)."
  (let ((ids (cons 0 (asem-bpe:bpe-encode bpe (format nil "Text: ~a~%" passage)))) (rows nil) (targets nil) (texts (list passage)) (nq 0))
    (dolist (x (qf "SHUFFLE" qa rng))
      (let* ((opts (rf "OPTIONS-FOR" (cdr x) (funcall cands (cdr x)) (funcall padder (cdr x)) rng))
             (qtoks (asem-bpe:bpe-encode bpe (format nil "Q: ~a~%~@[~a~]The answer is" (car x) (and opts (rf "OPTIONS-BLOCK" opts)))))
             (atoks (asem-bpe:bpe-encode bpe (format nil " ~a" (cdr x))))
             (tail (asem-bpe:bpe-encode bpe (format nil ".~%"))))
        (when (> (+ (length ids) (length qtoks) (length atoks) (length tail)) ctx) (return))
        (setf ids (append ids qtoks))
        (loop for tk in atoks for k from 0 do (push (+ (length ids) k -1) rows) (push tk targets))
        (setf ids (append ids atoks tail))
        (incf nq) (push (car x) texts) (push (cdr x) texts) (dolist (o opts) (push o texts))))
    (list ids (nreverse rows) (nreverse targets) texts nq)))

(defun words-of (text)
  (let ((out nil) (start nil))
    (loop for i from 0 to (length text)
          do (if (and (< i (length text)) (alpha-char-p (char text i)))
                 (unless start (setf start i))
                 (when start (push (subseq text start i) out) (setf start nil))))
    out))

(defun clean-p (seq excluded)
  (notany (lambda (text) (some (lambda (w) (gethash w excluded)) (words-of text))) (fourth seq)))

(defun set-recipe (bpe)
  (setf asem-retrieval:*augment* t asem-retrieval:*reverse* t asem-retrieval:*reasoning* t
        asem-retrieval:*broad-train* t asem-retrieval:*vocab-words* (asem-retrieval:vocab-words-of bpe)))

(defun save-sexp (path x)
  (with-open-file (s path :direction :output :if-exists :error :if-does-not-exist :create)
    (with-standard-io-syntax (write x :stream s :readably t))))
(defun read-sexp (path) (with-open-file (s path) (let ((*read-eval* nil)) (read s))))
(defun write-json (path x)
  (with-open-file (s path :direction :output :if-exists :error :if-does-not-exist :create :external-format :utf-8)
    (asem-json:write-json x s) (terpri s)))

(defun prepare (bpe ctx)
  (let ((excluded (make-hash-table :test 'equalp)) (control nil) (treatment nil) (redraws 0) (replaced 0)
        (kinds (make-hash-table)) (qc 0) (qt 0) (tc 0) (tt 0) (qrep 0) (trep 0) (examples nil))
    (with-open-file (in (format nil "~a/d1-entities.txt" +bundle+))
      (loop for line = (read-line in nil) while line do (unless (zerop (length line)) (setf (gethash line excluded) t))))
    (dotimes (i +dense+)
      (loop for attempt from 0 do
        (let* ((rng (rng-for i attempt 0)) (level (qf "PICK" '(1 2 3) rng)))
          (multiple-value-bind (passage qa kind cands padder) (asem-retrieval:make-fact-set level :train :seen rng)
            (let* ((swap (and (= level 1) (eq kind :attr)))
                   (c (build passage qa cands padder rng bpe ctx))
                   (tr (if swap
                           (let ((rng2 (rng-for i attempt 1)))
                             (multiple-value-bind (p2 qa2 c2 pad2) (answer-before-fact-set rng2) (build p2 qa2 c2 pad2 rng2 bpe ctx)))
                           c)))
              (if (and (second c) (second tr) (clean-p c excluded) (clean-p tr excluded))
                  (progn (push (subseq c 0 3) control) (push (subseq tr 0 3) treatment)
                         (incf (gethash kind kinds 0)) (incf qc (fifth c)) (incf qt (fifth tr)) (incf tc (length (third c))) (incf tt (length (third tr)))
                         (when swap (incf replaced) (incf qrep (fifth tr)) (incf trep (length (third tr)))
                               (when (< (length examples) 6) (push (asem-bpe:bpe-decode bpe (first tr)) examples)))
                         (return))
                  (incf redraws)))))))
    (ensure-directories-exist (format nil "~a/" +run+))
    (save-sexp (format nil "~a/data-control.sexp" +run+) (nreverse control))
    (save-sexp (format nil "~a/data-treatment.sexp" +run+) (nreverse treatment))
    (write-json (format nil "~a/data-audit.json" +bundle+)
                (obj "schema" "asem-answer-before-data-1" "seed" +seed+ "dense_sequences" +dense+ "copy_sequences" +copy+ "copy_seed" 77
                     "replaced_level1_attribute_sequences" replaced "redraws_for_excluded_entities_or_empty" redraws
                     "excluded_entity_count" (hash-table-count excluded)
                     "control_questions" qc "treatment_questions" qt "control_answer_tokens" tc "treatment_answer_tokens" tt
                     "treatment_questions_in_replaced_sequences" qrep "treatment_answer_tokens_in_replaced_sequences" trep
                     "sequences_by_drawn_kind" (cons :obj (loop for k being the hash-keys of kinds using (hash-value v) collect (cons (string-downcase (symbol-name k)) v)))
                     "replaced_examples" (coerce (nreverse examples) 'vector)
                     "sha256" (obj "data-control.sexp" (asem-sha256:sha256-file-hex (format nil "~a/data-control.sexp" +run+))
                                   "data-treatment.sexp" (asem-sha256:sha256-file-hex (format nil "~a/data-treatment.sexp" +run+))
                                   "protocol.md" (asem-sha256:sha256-file-hex (format nil "~a/protocol.md" +bundle+))
                                   "pilot.lisp" (asem-sha256:sha256-file-hex "lisp/experimental/answer-before/pilot.lisp")
                                   "retrieval.lisp" (asem-sha256:sha256-file-hex "lisp/src/retrieval.lisp")
                                   "sft.lisp" (asem-sha256:sha256-file-hex "lisp/src/sft.lisp"))))
    (format t "Prepared ~d sequences per arm; ~d replaced; ~d redraws.~%" +dense+ replaced redraws)))

(defun train (arm bpe ctx)
  (let* ((out (format nil "~a/~a" +run+ arm)) (audit (asem-json:parse-json (with-open-file (s (format nil "~a/data-audit.json" +bundle+)) (let ((text (make-string (file-length s)))) (subseq text 0 (read-sequence text s))))))
         (data (format nil "~a/data-~a.sexp" +run+ arm)))
    (when (probe-file (format nil "~a/" out)) (error "Refusing to overwrite training output ~a" out))
    (unless (equal (asem-json:json-get (asem-json:json-get audit "sha256") (format nil "data-~a.sexp" arm)) (asem-sha256:sha256-file-hex data)) (error "Data changed"))
    (unless (equal (asem-json:json-get (asem-json:json-get audit "sha256") "protocol.md") (asem-sha256:sha256-file-hex (format nil "~a/protocol.md" +bundle+))) (error "Protocol changed"))
    (let* ((model (asem-pretrain32:load-checkpoint +init+))
           (train (concatenate 'vector
                    (loop for (ids rows targets) in (read-sexp data) for i from 0 collect (asem-sft32:multi-ex (format nil "r~d" i) ids rows targets))
                    (asem-sft32:make-copy-examples +copy+ 77 :lo 1 :hi 8192 :min-len 10 :max-len 40)))
           (fits (lambda (prompt answer) (<= (+ (length (asem-bpe:bpe-encode bpe prompt)) (length (asem-bpe:bpe-encode bpe (format nil " ~a" answer)))) ctx)))
           (probe (let ((asem-retrieval:*augment* nil)) (asem-retrieval:probe-items 3 :new :seen 300 7001 :fits fits :options t)))
           (dev (coerce (loop for it in probe
                              collect (asem-sft32:generation-ex (getf it :id) "L3" (getf it :kind)
                                                                (asem-retrieval:retrieval-prompt (getf it :passage) (getf it :question) :options (getf it :options))
                                                                (getf it :answer) bpe ctx))
                        'vector)))
      (asem-sft32:sft out train dev '(316 367 338 443) :init-model model
        :steps +steps+ :batch 32 :workers 8 :lr 5f-4 :warmup 50 :eval-every 250 :log-every 20 :train-probe 300 :seed +seed+
        :header (list :mission "answer-before-binding" :arm arm :init +init+ :init-sha256 +reference+
                      :protocol-sha256 (asem-sha256:sha256-file-hex (format nil "~a/protocol.md" +bundle+))
                      :data-sha256 (asem-sha256:sha256-file-hex data)
                      :driver-sha256 (asem-sha256:sha256-file-hex "lisp/experimental/answer-before/pilot.lisp"))))))

(let* ((mode (or (opt "mode") (error "mode=prepare|train required")))
       (bpe (asem-bpe:load-bpe "data/fineweb/stage1/tokenizer.bpe")) (ctx 128))
  (unless (equal +reference+ (asem-sha256:sha256-file-hex +init+)) (error "Reference checkpoint changed"))
  (set-recipe bpe)
  (cond ((equal mode "prepare") (prepare bpe ctx))
        ((equal mode "train") (let ((arm (opt "arm"))) (unless (member arm '("control" "treatment") :test #'equal) (error "arm=control|treatment")) (train arm bpe ctx)))
        (t (error "Unknown mode")))
  (unless (equal +reference+ (asem-sha256:sha256-file-hex +init+)) (error "Reference checkpoint changed during run")))
