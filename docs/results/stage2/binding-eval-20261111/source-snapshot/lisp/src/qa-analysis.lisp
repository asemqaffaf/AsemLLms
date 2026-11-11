;;;; Additive QA development diagnostics. No locked-test evaluation entry point.
;;;; Familiar questions change ONLY question wording; passage, options and keys stay fixed.
(defpackage :asem-qa-analysis
  (:use :cl)
  (:export #:read-development #:familiar-items #:evaluate #:condition-item #:score-predictions))
(in-package :asem-qa-analysis)

(defun obj (&rest kvs)
  (cons :obj (loop for (k v) on kvs by #'cddr collect (cons k v))))
(defun arr (xs) (coerce xs 'vector))
(defun jb (b) (if b :true :false))

(defun data-form (s)
  "Read structured generator metadata as data, disabling reader evaluation."
  (with-standard-io-syntax
    (let ((*read-eval* nil) (*package* (find-package :keyword)))
      (multiple-value-bind (value end) (read-from-string s)
        (unless (every (lambda (c) (find c '(#\Space #\Tab #\Newline #\Return))) (subseq s end))
          (error "Trailing data in generator metadata"))
        (labels ((valid (x) (or (null x) (stringp x) (numberp x) (keywordp x)
                               (and (consp x) (valid (car x)) (valid (cdr x))))))
          (unless (valid value) (error "Unexpected value in generator metadata")))
        value))))

(defun read-development (path lock-file)
  "Read DEV only, retaining structured query/facts. Refuse test names/hash before parsing items."
  (when (search "test" (string-downcase (file-namestring path)))
    (error "Development diagnostics refuse test-named files: ~a" path))
  (let ((locked (with-open-file (in lock-file) (string-trim '(#\Space #\Return #\Newline) (read-line in)))))
    (when (string-equal locked (asem-sha256:sha256-file-hex path))
      (error "Development diagnostics refuse locked-test content")))
  (with-open-file (in path :external-format :utf-8)
    (loop for line = (read-line in nil) while line
          unless (zerop (length (string-trim '(#\Space #\Tab) line)))
            collect (let ((o (asem-json:parse-json line)))
                      (flet ((g (k) (asem-json:json-get o k)))
                        (unless (equal (g "split") "dev") (error "Development diagnostics require split=dev"))
                        (let ((queries (data-form (g "query"))))
                          (list :id (g "id") :split "dev" :category (g "category") :subtype (g "subtype")
                                :passage (g "passage") :question (g "question") :options (coerce (g "options") 'list)
                                :answer (g "answer") :cf-question (g "cf_question") :cf-answer (g "cf_answer")
                                :qframe (g "qframe") :query (first queries) :cf-query (second queries)
                                :facts (data-form (g "facts")))))))))

(defun familiar-question (it query)
  "First TRAIN frame for the exact structured query; no key-dependent template choice."
  (let* ((cat (getf it :category)) (sub (getf it :subtype))
         (type (intern (string-upcase (first (asem-util:split-on sub #\-))) :keyword))
         (spec (cdr (assoc type (append asem-qagen::*attr-types* asem-qagen::*num-types* asem-qagen::*relations*))))
         (qtype (case (first query)
                  (:extract :q-extract) (:reverse :q-reverse) (:rel-subj :q-subj) (:rel-obj :q-obj)
                  (:max :q-max) (:min :q-min) (:first :q-first) (:last :q-last) (:after :q-after) (:before :q-before)
                  (otherwise (error "Unsupported structured query ~s" query))))
         (frame (first (getf (getf (if (equal cat "temporal") asem-qagen::*temporal* spec) qtype) :train))))
    (unless frame (error "No training frame for ~s ~s" cat query))
    (values (apply #'asem-qagen::fill-frame frame
                   (case (first query)
                     (:extract (list :p (second query))) (:reverse (list :v (second query)))
                     (:rel-obj (list :a (second query))) (:rel-subj (list :b (second query)))
                     ((:first :last) (list :p (getf (getf it :facts) :person)))
                     ((:after :before) (list :p (getf (getf it :facts) :person) :x (second query)))))
            frame)))

(defun familiar-items (items)
  (mapcar (lambda (it)
            (let ((copy (copy-list it)))
              (multiple-value-bind (q frame) (familiar-question it (getf it :query))
                (setf (getf copy :question) q (getf copy :qframe) frame))
              (setf (getf copy :cf-question) (familiar-question it (getf it :cf-query)))
              copy))
          items))

(defun condition-item (items it condition)
  "Historical conditions, including the original contaminated mismatched-question selection."
  (let ((copy (copy-list it)))
    (ecase condition
      (:real nil)
      (:counterfactual (setf (getf copy :question) (getf it :cf-question) (getf copy :answer) (getf it :cf-answer)))
      (:no-question (setf (getf copy :omit-question) t))
      (:no-passage (setf (getf copy :omit-passage) t))
      (:mismatched (setf (getf copy :question) (asem-sft32::mismatched-question items it 0)
                         (getf copy :mismatch-literally-unchanged) (equal (getf copy :question) (getf it :question))))
      (:reversed (setf copy (asem-sft32::reverse-options it))))
    copy))

(defun score-predictions (scores lengths)
  "Return summed-score and mean-token-score argmax, with deterministic first-index ties."
  (let ((means (mapcar (lambda (s n) (/ s n)) scores lengths)))
    (values (position (reduce #'max scores) scores) (position (reduce #'max means) means) means)))

(defun answer-boundary (decoded)
  "First answer-ending period/newline, independent of the gold text."
  (position-if (lambda (c) (member c '(#\. #\Newline #\Return))) decoded))

(defun greedy-answer (model bpe e cache &key (max-tokens 8))
  "Bounded autoregressive argmax, independent of gold length: EOT, period/newline, eight tokens or context."
  (let* ((cfg (asem-tf32:model-config model)) (v (asem-tf32:config-vocab cfg))
         (ids (coerce (asem-sft32:ex-ids e) 'list)) (generated nil) (reason "token_limit") (text ""))
    (dotimes (k max-tokens)
      (declare (ignorable k))
      (when (> (length ids) (asem-tf32:config-ctx cfg)) (setf reason "context_limit") (return))
      (let* ((acts (asem-sft32::acts-for cache cfg (length ids)))
             (x (asem-sft32::padded-x (asem-sft32::ivec-of ids) (asem-tf32::acts-tlen acts))))
        (asem-tf32:forward-rows model x acts (asem-sft32::ivec-of (list (1- (length ids)))))
        (let* ((logits (asem-tf32::acts-logits acts)) (best 0))
          (dotimes (j v) (when (> (aref logits j) (aref logits best)) (setf best j)))
          (push best generated) (setf ids (append ids (list best)))
          (when (= best 0) (setf reason "eot") (return))
          (setf text (asem-bpe:bpe-decode bpe (reverse generated)))
          (let ((boundary (answer-boundary text)))
            (when boundary (setf text (subseq text 0 boundary) reason "period_or_newline") (return))))))
    (values (nreverse generated) (string-trim '(#\Space #\Tab #\Return #\Newline) text) reason)))

(defun predict-item (model bpe it cache)
  (let* ((ctx (asem-tf32:config-ctx (asem-tf32:model-config model))) (letters (asem-sft32:letter-token-ids bpe))
         (prompt (asem-qagen:item-prompt it :passage (not (getf it :omit-passage)) :question (not (getf it :omit-question))))
         ;; ENCODE-ITEMS validates prompt/answer token boundaries and every option's context length.
         (e (aref (asem-sft32:encode-items (list it) bpe ctx :mode :text :prompt-fn (lambda (x) (declare (ignore x)) prompt)) 0))
         (lengths (mapcar #'length (asem-sft32::ex-opts e)))
         (scores (mapcar (lambda (tail) (asem-sft32::score-continuation model e tail cache)) (asem-sft32::ex-opts e)))
         (letter-ex (asem-sft32::copy-ex e)))
    (setf (asem-sft32::ex-mode letter-ex) :letter
          (asem-sft32::ex-tgt letter-ex) (asem-sft32::ivec-of (list (nth (getf it :answer) letters))))
    (multiple-value-bind (loss letter-pred) (asem-sft32::run-example model letter-ex cache letters)
      (declare (ignore loss))
      (multiple-value-bind (sum-pred mean-pred means) (score-predictions scores lengths)
        (multiple-value-bind (all-generated decoded reason) (greedy-answer model bpe e cache)
          (let* ((generated (subseq all-generated 0 (min (length all-generated) (length (asem-sft32::ex-tgt e)))))
                 (gold-text (nth (getf it :answer) (getf it :options))) (exact (string= decoded gold-text)))
            (list :id (getf it :id) :category (getf it :category) :subtype (getf it :subtype) :key (getf it :answer)
              :question (getf it :question) :sum-pred sum-pred :mean-pred mean-pred :letter-pred letter-pred
              :prefix-exact (equal generated (coerce (asem-sft32::ex-tgt e) 'list))
              :generated generated :generated-text (asem-bpe:bpe-decode bpe generated)
              :gold-text gold-text :bounded-generated all-generated :bounded-text decoded :termination-reason reason
              :bounded-exact exact :terminated-exact (and exact (member reason '("eot" "period_or_newline") :test #'equal))
              :prompt-tokens (1- (length (asem-sft32:ex-ids e))) :option-token-lengths lengths
              :scores scores :means means :mismatch-literally-unchanged (getf it :mismatch-literally-unchanged))))))))

(defun row-json (r)
  (obj "id" (getf r :id) "category" (getf r :category) "subtype" (getf r :subtype) "key" (getf r :key)
       "question" (getf r :question) "summed_option_prediction" (getf r :sum-pred)
       "mean_token_option_prediction" (getf r :mean-pred) "letter_prediction" (getf r :letter-pred)
       "gold_length_greedy_prefix_exact" (jb (getf r :prefix-exact))
       "generated_token_ids" (arr (getf r :generated)) "generated_text" (getf r :generated-text) "gold_text" (getf r :gold-text)
       "bounded_free_generation_exact" (jb (getf r :bounded-exact))
       "bounded_free_generation_terminated_exact" (jb (getf r :terminated-exact))
       "bounded_generated_text" (getf r :bounded-text) "bounded_generated_token_ids" (arr (getf r :bounded-generated))
       "generation_termination_reason" (getf r :termination-reason)
       "prompt_tokens_without_eot" (getf r :prompt-tokens) "option_token_lengths" (arr (getf r :option-token-lengths))
       "summed_option_logprob" (arr (getf r :scores)) "mean_token_option_logprob" (arr (getf r :means))
       "mismatched_question_literally_unchanged" (jb (getf r :mismatch-literally-unchanged))))

(defun accuracy-obj (rows metric)
  (let* ((n (length rows))
         (k (count-if (lambda (r) (if (member metric '(:prefix-exact :bounded-exact :terminated-exact)) (getf r metric) (= (getf r :key) (getf r metric)))) rows))
         (ci (asem-sft32:wilson k n)))
    (obj "n" n "correct" k "accuracy" (/ k (max 1 n) 1d0) "ci95_low" (first ci) "ci95_high" (second ci))))

(defun summary (rows)
  (cons :obj
        (loop for (name metric) in '(("summed_option_likelihood" :sum-pred) ("mean_token_option_likelihood" :mean-pred)
                                    ("letter_A_to_D" :letter-pred) ("gold_length_greedy_prefix_exact" :prefix-exact)
                                    ("bounded_free_generation_exact" :bounded-exact) ("bounded_free_generation_terminated_exact" :terminated-exact))
              collect
              (let* ((overall (accuracy-obj rows metric))
                     (categories (cons :obj (loop for cat in asem-qagen:+categories+
                                                 collect (cons cat (accuracy-obj (remove-if-not (lambda (r) (equal (getf r :category) cat)) rows) metric))))))
                (cons name (append overall (list (cons "by_category" categories))))))))

(defun evaluate (model bpe items &key (workers 4) (controls t) (paired t))
  "Per-item and aggregate diagnostics. All rows remain development evidence."
  (unless (and (plusp workers) items) (error "Need positive workers and nonempty development items"))
  (unless (every (lambda (it) (equal (getf it :split) "dev")) items) (error "Only development items may be evaluated"))
  (let ((conditions (if controls '(:real :counterfactual :no-question :no-passage :mismatched :reversed) '(:real))) (results nil))
    (labels ((run (source condition label)
               (let* ((variants (mapcar (lambda (it) (condition-item source it condition)) source))
                      (parts (asem-train32::parallel-map
                              (lambda (part)
                                ;; BPE-ENCODE memoizes. Workers share only immutable ranks/bytes/merges,
                                ;; while each owns its encoding cache and model activation buffers.
                                (let ((cache (make-hash-table)) (worker-bpe (asem-bpe::copy-bpe bpe)))
                                  (setf (asem-bpe::bpe-cache worker-bpe) (make-hash-table :test 'equal))
                                  (mapcar (lambda (it) (predict-item model worker-bpe it cache)) part)))
                              (asem-train32::shards variants workers)))
                      (rows (apply #'append parts)))
                 ;; Shards are interleaved; restore input order for stable paired comparisons.
                 (setf rows (mapcar (lambda (it) (find (getf it :id) rows :key (lambda (r) (getf r :id)) :test #'equal)) variants))
                 (push (cons label (obj "summary" (summary rows) "items" (arr (mapcar #'row-json rows))
                                        "mismatch_literally_unchanged_n" (count-if (lambda (r) (getf r :mismatch-literally-unchanged)) rows))) results)
                 (format *error-output* "QA analysis ~a: ~d items complete~%" label (length rows)))))
      (dolist (c conditions) (run items c (string-downcase (symbol-name c))))
      (when paired
        (let ((familiar (familiar-items items)))
          (run familiar :real "familiar-question-real")
          (when controls (run familiar :counterfactual "familiar-question-counterfactual"))))
      (cons :obj (nreverse results)))))
