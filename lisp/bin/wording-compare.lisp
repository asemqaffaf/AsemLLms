;;;; Paired cross-checkpoint comparison on saved wording-eval JSON diagnostics (Control vs Treatment or Init vs Endpoint).
;;;;   sbcl --script lisp/bin/wording-compare.lisp a=CTRL_2X2.json b=TRT_2X2.json out=COMPARE.json [policy=seeded]
;;;; Reuses ASEM-WORDING::SUMMARIZE-CELL-ROWS and ASEM-WORDING::SUMMARIZE-PAIR so Wilson CIs, paired CIs, and
;;;; exact two-sided McNemar p-values use the exact same verified statistical routines.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval" "tf32" "train32" "pretrain32" "sft32" "qa-analysis" "wording"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))

(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name))
                                     (string= name a :end2 (length name))
                                     (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))

(defun g (obj key) (asem-json:json-get obj key))

(defun json-row->plist (o)
  (list :id (g o "id") :category (g o "category") :subtype (g o "subtype") :key (g o "key")
        :gold-text (g o "gold_text")
        :sum-pred (g o "summed_option_prediction")
        :mean-pred (g o "mean_token_option_prediction")
        :letter-pred (g o "letter_prediction")
        :prefix-exact (eq (g o "gold_length_greedy_prefix_exact") :true)
        :bounded-exact (eq (g o "bounded_free_generation_exact") :true)
        :terminated-exact (eq (g o "bounded_free_generation_terminated_exact") :true)))

(defun cell-rows (doc pol cell-key &optional cf)
  (let* ((pobj (g (g doc "policies") pol))
         (cobj (and pobj (g (g pobj (if cf "counterfactual_cells" "cells")) cell-key))))
    (when cobj (map 'list #'json-row->plist (g cobj "items")))))

(defun read-json-file (path)
  (with-open-file (in path :external-format :utf-8)
    (let* ((buf (make-string (file-length in)))
           (n (read-sequence buf in)))
      (asem-json:parse-json (subseq buf 0 n)))))

(let* ((a-path (or (opt-arg "a" nil) (error "a=FIRST.json required")))
       (b-path (or (opt-arg "b" nil) (error "b=SECOND.json required")))
       (out (or (opt-arg "out" nil) (error "out=COMPARE.json required")))
       (pol (opt-arg "policy" "seeded"))
       (a-doc (read-json-file a-path))
       (b-doc (read-json-file b-path)))
  (when (probe-file out) (error "Refusing to overwrite existing output ~a" out))
  (unless (and (equal (g a-doc "dataset_kind") (g b-doc "dataset_kind"))
               (equal (g a-doc "item_count") (g b-doc "item_count")))
    (error "Incompatible wording-eval files (~a/~a vs ~a/~a)"
           (g a-doc "dataset_kind") (g a-doc "item_count")
           (g b-doc "dataset_kind") (g b-doc "item_count")))
  (let* ((cell-names '("p-unf/q-unf" "p-unf/q-fam" "p-fam/q-unf" "p-fam/q-fam"))
         (cells (cons :obj
                      (loop for c in cell-names
                            for ra = (cell-rows a-doc pol c nil)
                            for rb = (cell-rows b-doc pol c nil)
                            when (and ra rb)
                              collect (cons c (asem-qa-analysis::obj
                                               "first" (asem-wording::summarize-cell-rows ra)
                                               "second" (asem-wording::summarize-cell-rows rb)
                                               "paired_second_minus_first" (asem-wording::summarize-pair ra rb))))))
         (cf-cells (cons :obj
                         (loop for c in cell-names
                               for ra = (cell-rows a-doc pol c t)
                               for rb = (cell-rows b-doc pol c t)
                               when (and ra rb)
                                 collect (cons c (asem-qa-analysis::obj
                                                  "paired_second_minus_first" (asem-wording::summarize-pair ra rb))))))
         (report (asem-qa-analysis::obj
                  "schema" "asem-wording-pair-compare-1"
                  "policy" pol
                  "dataset_kind" (g a-doc "dataset_kind")
                  "items" (g a-doc "items")
                  "item_count" (g a-doc "item_count")
                  "first_file" a-path "first_checkpoint" (g a-doc "checkpoint") "first_sha256" (g a-doc "checkpoint_sha256")
                  "second_file" b-path "second_checkpoint" (g b-doc "checkpoint") "second_sha256" (g b-doc "checkpoint_sha256")
                  "cells" cells
                  "counterfactual_cells" cf-cells)))
    (ensure-directories-exist out)
    (with-open-file (stream out :direction :output :if-exists :error :if-does-not-exist :create :external-format :utf-8)
      (asem-json:write-json report stream) (terpri stream))
    (format *error-output* "Wrote paired comparison to ~a~%" out)))
