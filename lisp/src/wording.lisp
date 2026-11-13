;;;; Paired 2x2 wording diagnostic: passage wording (familiar / unfamiliar) x question wording (familiar / unfamiliar).
;;;;
;;;; Every development passage sentence is parsed back to its frame and slot bindings (person, value, number, agent, patient,
;;;; place) using the qagen frame inventory, then re-rendered with TRAINING frames. Sentence order, entities, numbers, options,
;;;; option order, answer key and counterfactual key are untouched; only frames change. Questions are rendered from the
;;;; item's structured query with training question frames. "Unfamiliar" cells use the original development strings verbatim,
;;;; so the unfamiliar/unfamiliar cell is exactly the original item.
;;;;
;;;; Frame choice never depends on the answer: :seeded draws each frame from a random state seeded by (item id, slot), and
;;;; :first takes the first training frame (the historical familiar-question policy). Intros carry no facts; non-empty
;;;; development intros map to the closest-meaning training intro (*INTRO-MAP*).
;;;;
;;;; Development only. Nothing here reads or names the locked test file.
(defpackage :asem-wording
  (:use :cl)
  (:export #:+cells+ #:frame-entries #:parse-sentence #:parse-passage #:parse-question #:render-passage #:render-question
           #:cell-item #:cell-items #:probe-to-qa-item #:records-semantics #:derived-facts #:item-type #:fnv32 #:check-cell-item
           #:paired-diff #:factorial-effects #:evaluate-2x2))
(in-package :asem-wording)

(defparameter +cells+ '((:u :u) (:u :f) (:f :u) (:f :f))
  "(passage-wording question-wording): :u = unfamiliar (original development wording), :f = familiar (training frames).")

(defun qg (sym) (symbol-value (find-symbol (string sym) :asem-qagen)))
(defun ret-new-pool (kind)
  (let ((pkg (find-package :asem-retrieval)))
    (and pkg (let ((fn (find-symbol "NEW-POOL" pkg))) (and fn (fboundp fn) (funcall fn kind))))))

;;; ------------------------------------------------------------------ frame inventory
;;; Entry: (:kind K :type T :role R :split S :index I :frame STRING). K: :attr :num :rel :tmp :intro.
;;; Fact roles: :frames (attr/num/rel), :first :middle :last (tmp), :people :relation :temporal (intro).
;;; Question roles: :q-extract :q-reverse :q-max :q-min :q-subj :q-obj :q-first :q-last :q-after :q-before.
(defvar *entries* nil)

(defun frame-entries ()
  (or *entries*
      (setf *entries*
            (let ((out nil))
              (flet ((add (kind type role by)
                       (dolist (split '(:train :dev :test))
                         (loop for f in (getf by split) for i from 0
                               unless (string= f "")
                                 do (push (list :kind kind :type type :role role :split split :index i :frame f) out)))))
                (dolist (ty (qg '*attr-types*)) (dolist (r '(:frames :q-extract :q-reverse)) (add :attr (car ty) r (getf (cdr ty) r))))
                (dolist (ty (qg '*num-types*)) (dolist (r '(:frames :q-extract :q-max :q-min)) (add :num (car ty) r (getf (cdr ty) r))))
                (dolist (ty (qg '*relations*)) (dolist (r '(:frames :q-subj :q-obj)) (add :rel (car ty) r (getf (cdr ty) r))))
                (loop for (r by) on (qg '*temporal*) by #'cddr do (add :tmp :temporal r by))
                (loop for (r by) on (qg '*intros*) by #'cddr do (add :intro r r by)))
              (nreverse out)))))

(defparameter +fact-roles+ '(:frames :first :middle :last :people :relation :temporal))
(defun fact-entry-p (e) (member (getf e :role) +fact-roles+))

;;; ------------------------------------------------------------------ frame matching
(defun frame-tokens (frame)
  "FRAME as a list of literal strings and slot keywords (:p :v :av :n :a :b :x)."
  (let ((out nil) (i 0) (n (length frame)))
    (loop while (< i n)
          do (let ((open (position #\{ frame :start i)))
               (cond ((null open) (push (subseq frame i) out) (setf i n))
                     (t (when (> open i) (push (subseq frame i open) out))
                        (let ((close (position #\} frame :start open)))
                          (push (intern (string-upcase (subseq frame (1+ open) close)) :keyword) out)
                          (setf i (1+ close)))))))
    (nreverse out)))

(defun slot-pool (entry slot)
  (ecase slot
    ((:p :a :b) (append (qg '*names*) (ret-new-pool :names)))
    (:x (qg '*places*))
    ((:v :av) (let ((spec (cdr (assoc (getf entry :type) (qg '*attr-types*)))))
                (unless spec (error "slot ~s outside an attribute frame: ~s" slot (getf entry :frame)))
                (append (qg (getf spec :pool)) (when (eq (getf entry :type) :town) (ret-new-pool :towns)))))
    (:n nil)))

(defun text-at-p (s pos text)
  "TEXT occurs in S at POS; at position 0 the first character is compared case-insensitively (fill-frame upcases it)."
  (let ((end (+ pos (length text))))
    (and (<= end (length s)) (plusp (length text))
         (if (zerop pos)
             (and (char-equal (char s 0) (char text 0)) (string= s text :start1 1 :end1 end :start2 1))
             (string= s text :start1 pos :end1 end)))))

(defun match-frame (entry sentence)
  "All binding plists (slot -> canonical value) under which ENTRY's frame renders exactly SENTENCE."
  (let ((results nil))
    (labels ((bind (bindings slot value)
               (let* ((key (if (eq slot :av) :v slot)) (old (getf bindings key :unbound)))
                 (cond ((eq old :unbound) (list* key value bindings))
                       ((equal old value) bindings)
                       (t :conflict))))
             (walk (tokens pos bindings)
               (cond ((null tokens) (when (= pos (length sentence)) (push bindings results)))
                     ((stringp (first tokens))
                      (when (text-at-p sentence pos (first tokens)) (walk (rest tokens) (+ pos (length (first tokens))) bindings)))
                     ((eq (first tokens) :n)
                      (let ((end (or (position-if-not #'digit-char-p sentence :start pos) (length sentence))))
                        (when (> end pos)
                          (let ((b (bind bindings :n (parse-integer sentence :start pos :end end))))
                            (unless (eq b :conflict) (walk (rest tokens) end b))))))
                     (t (dolist (c (slot-pool entry (first tokens)))
                          (let ((text (if (eq (first tokens) :av) (format nil "~a ~a" (asem-qagen::article c) c) c)))
                            (when (text-at-p sentence pos text)
                              (let ((b (bind bindings (first tokens) c)))
                                (unless (eq b :conflict) (walk (rest tokens) (+ pos (length text)) b))))))))))
      (let ((tokens (frame-tokens (getf entry :frame))))
        ;; cheap prefilter: every literal piece must occur somewhere (case-insensitively for a sentence-initial literal)
        (when (every (lambda (tk) (or (keywordp tk) (search tk sentence :test #'char-equal))) tokens)
          (walk tokens 0 nil))))
    results))

(defun canonical-bindings (bindings)
  "Sorted (slot . value) alist; independent of binding order."
  (sort (loop for (k v) on bindings by #'cddr collect (cons k v)) #'string< :key (lambda (x) (symbol-name (car x)))))

(defun semantics (entry bindings)
  "Wording-free meaning of a matched sentence: kind, type, role and canonical bindings (frame text and split excluded)."
  (list (getf entry :kind) (getf entry :type) (getf entry :role) (canonical-bindings bindings)))

(defun parse-sentence (sentence splits &key (roles :facts))
  "Unique (values entry bindings) whose frame (split in SPLITS) renders SENTENCE; ROLES :facts or :questions.
Errors when no frame matches or when matches disagree in meaning."
  (let ((hits nil))
    (dolist (e (frame-entries))
      (when (and (member (getf e :split) splits) (if (eq roles :facts) (fact-entry-p e) (not (fact-entry-p e))))
        (dolist (b (match-frame e sentence)) (push (cons e b) hits))))
    (unless hits (error "No ~(~a~) frame of ~s renders ~s" roles splits sentence))
    (let ((meanings (remove-duplicates (mapcar (lambda (h) (semantics (car h) (cdr h))) hits) :test #'equal)))
      (when (rest meanings) (error "Ambiguous sentence ~s: ~s" sentence meanings)))
    (values (car (first hits)) (cdr (first hits)))))

(defun parse-passage (passage splits)
  "List of (entry . bindings), one per sentence, in passage order."
  (mapcar (lambda (s) (multiple-value-bind (e b) (parse-sentence s splits) (cons e b))) (asem-qagen::sentences-of passage)))

(defun records-semantics (records) (mapcar (lambda (r) (semantics (car r) (cdr r))) records))

(defun parse-question (question splits)
  (multiple-value-bind (e b) (parse-sentence question splits :roles :questions) (cons e b)))

;;; ------------------------------------------------------------------ frame choice (never depends on the answer)
(defun fnv32 (string)
  "32-bit FNV-1a over the UTF-8 octets of STRING (portable, deterministic)."
  (let ((h 2166136261))
    (loop for octet across (sb-ext:string-to-octets string :external-format :utf-8)
          do (setf h (logand #xffffffff (* (logxor h octet) 16777619))))
    h))

(defun choose (candidates policy id slot)
  (ecase policy
    (:first (first candidates))
    (:seeded (nth (random (length candidates) (sb-ext:seed-random-state (fnv32 (format nil "~a|~a" id slot)))) candidates))))

(defun entries-for (kind type role split)
  (sort (remove-if-not (lambda (e) (and (eq (getf e :kind) kind) (eq (getf e :type) type) (eq (getf e :role) role) (eq (getf e :split) split)))
                       (frame-entries))
        #'< :key (lambda (e) (getf e :index))))

(defparameter *intro-map*
  '(("A record of four members follows." . "These facts are about four people.")
    ("Events from the workshop are listed." . "This is what happened at the club.")
    ("{P} ran errands on Monday." . "{P} had a busy day."))
  "Development intro -> closest-meaning training intro. Intros state no fact any question depends on.")

(defun fill-bindings (frame bindings)
  (apply #'asem-qagen::fill-frame frame
         (loop for (k v) on bindings by #'cddr append (if (eq k :n) (list :n v :v v) (list k v)))))

(defun render-record (record split policy id slot)
  "Re-render a parsed sentence with a SPLIT frame of the same kind, type and role, keeping every binding."
  (destructuring-bind (entry . bindings) record
    (let ((frame (if (eq (getf entry :kind) :intro)
                     (if (eq split (getf entry :split)) (getf entry :frame)
                         (or (cdr (assoc (getf entry :frame) *intro-map* :test #'string=))
                             (error "No intro mapping for ~s" (getf entry :frame))))
                     (getf (choose (or (entries-for (getf entry :kind) (getf entry :type) (getf entry :role) split)
                                       (error "No ~a frame for ~s" split (semantics entry bindings)))
                                   policy id slot)
                           :frame))))
      (fill-bindings frame bindings))))

(defun render-passage (passage id &key (from :dev) (to :train) (policy :seeded))
  (let ((records (parse-passage passage (list from))))
    (flet ((render-with (pol suffix)
             (asem-qagen::join-sentences
              (loop for record in records for i from 0
                    collect (render-record record to pol id (format nil "passage-~d~a" i suffix))))))
      (if (eq policy :first)
          (render-with :first "")
          (let ((limit (- (length passage) (if (> (length passage) 220) 10 0))))
            (or (loop for attempt from 0 below 16
                      for cand = (render-with :seeded (if (zerop attempt) "" (format nil "-~d" attempt)))
                      when (<= (length cand) limit) return cand)
                (render-with :first "")))))))

;;; ------------------------------------------------------------------ questions from the structured query
(defun item-type (it)
  "Fact type of an item (:town ... :height, :teach ..., :temporal)."
  (if (equal (getf it :category) "temporal") :temporal
      (intern (string-upcase (first (asem-util:split-on (getf it :subtype) #\-))) :keyword)))

(defun query-role (query)
  (ecase (first query)
    (:extract :q-extract) (:reverse :q-reverse) (:rel-subj :q-subj) (:rel-obj :q-obj) (:max :q-max) (:min :q-min)
    (:first :q-first) (:last :q-last) (:after :q-after) (:before :q-before)))

(defun query-bindings (it query)
  (ecase (first query)
    (:extract (list :p (second query))) (:reverse (list :v (second query)))
    (:rel-subj (list :b (second query))) (:rel-obj (list :a (second query)))
    ((:max :min) nil)
    ((:first :last) (list :p (getf (getf it :facts) :person)))
    ((:after :before) (list :p (getf (getf it :facts) :person) :x (second query)))))

(defun question-kind (it)
  (let ((ty (item-type it)))
    (cond ((eq ty :temporal) :tmp)
          ((assoc ty (qg '*attr-types*)) :attr)
          ((assoc ty (qg '*num-types*)) :num)
          ((assoc ty (qg '*relations*)) :rel)
          (t (error "Unknown item type ~s" ty)))))

(defun render-question (it query &key (split :train) (policy :seeded))
  "Question text for QUERY with a SPLIT frame. The frame is seeded by (id, role), so a real and counterfactual question of
the same role share their frame, as in the development generator."
  (let* ((role (query-role query))
         (entry (choose (or (entries-for (question-kind it) (item-type it) role split) (error "No ~a ~a frame" split role))
                        policy (getf it :id) (format nil "question-~(~a~)" role))))
    (values (fill-bindings (getf entry :frame) (query-bindings it query)) (getf entry :frame))))

(defun question-query (record)
  "Structured query recovered from a parsed question."
  (destructuring-bind (entry . b) record
    (ecase (getf entry :role)
      (:q-extract (list :extract (getf b :p))) (:q-reverse (list :reverse (getf b :v)))
      (:q-subj (list :rel-subj (getf b :b))) (:q-obj (list :rel-obj (getf b :a)))
      (:q-max (list :max)) (:q-min (list :min)) (:q-first (list :first)) (:q-last (list :last))
      (:q-after (list :after (getf b :x))) (:q-before (list :before (getf b :x))))))

;;; ------------------------------------------------------------------ cells
(defun cell-name (cell) (format nil "~(~a~a~)" (if (eq (first cell) :f) "P-fam" "P-unf") (if (eq (second cell) :f) "/Q-fam" "/Q-unf")))

(defun cell-item (it cell &key (policy :seeded))
  "Copy of development item IT in CELL = (passage-wording question-wording). Options, order, keys, facts and queries unchanged."
  (unless (equal (getf it :split) "dev") (error "Wording cells are built from development items only"))
  (let ((copy (copy-list it)))
    (when (eq (first cell) :f)
      (setf (getf copy :passage) (render-passage (getf it :passage) (getf it :id) :policy policy)))
    (when (eq (second cell) :f)
      (multiple-value-bind (q frame) (render-question it (getf it :query) :policy policy)
        (setf (getf copy :question) q (getf copy :qframe) frame))
      (setf (getf copy :cf-question) (render-question it (getf it :cf-query) :policy policy)))
    (setf (getf copy :cell) (cell-name cell) (getf copy :policy) policy)
    copy))

(defun cell-items (items &key (policy :seeded))
  "Alist cell-name -> list of cell items (same order and ids as ITEMS)."
  (mapcar (lambda (cell) (cons (cell-name cell) (mapcar (lambda (it) (cell-item it cell :policy policy)) items))) +cells+))

(defun derived-facts (it records)
  "Facts recomputed from the parsed passage text alone (independent of the stored :facts)."
  (let* ((cat (getf it :category)) (ty (item-type it))
         (facts (remove-if (lambda (r) (eq (getf (car r) :kind) :intro)) records))
         (typed (remove-if-not (lambda (r) (eq (getf (car r) :type) ty)) facts)))
    (cond ((member cat '("extract" "reverse") :test #'string=)
           (list :people (mapcar (lambda (r) (getf (cdr r) :p)) typed)
                 :values (mapcar (lambda (r) (princ-to-string (or (getf (cdr r) :v) (getf (cdr r) :n)))) typed)))
          ((string= cat "relation") (list :pairs (mapcar (lambda (r) (list (getf (cdr r) :a) (getf (cdr r) :b))) typed)))
          ((string= cat "compare") (list :people (mapcar (lambda (r) (getf (cdr r) :p)) typed) :numbers (mapcar (lambda (r) (getf (cdr r) :n)) typed)))
          ((string= cat "temporal")
           (let ((people (remove-duplicates (mapcar (lambda (r) (getf (cdr r) :p)) facts) :test #'string=)))
             (unless (= 1 (length people)) (error "Temporal passage names ~d people" (length people)))
             (unless (and (eq (getf (car (first facts)) :role) :first) (eq (getf (car (car (last facts))) :role) :last)
                          (every (lambda (r) (eq (getf (car r) :role) :middle)) (butlast (rest facts))))
               (error "Temporal sentence roles out of narrative order"))
             (list :person (first people) :order (mapcar (lambda (r) (getf (cdr r) :x)) facts))))
          (t (error "Unknown category ~a" cat)))))

(defun probe-to-qa-item (p)
  "Convert an :unseen retrieval probe item (from ASEM-RETRIEVAL:PROBE-ITEMS) into a full QA development item with parsed
structured queries, facts and option indices, so all four 2x2 wording cells share the exact same underlying item."
  (let* ((qrec (parse-question (getf p :question) '(:dev)))
         (cfrec (parse-question (getf p :cf-question) '(:dev)))
         (q (question-query qrec)) (cfq (question-query cfrec))
         (kind (getf p :kind)) (qtype (getf (car qrec) :type))
         (cat (cond ((equal kind "cmp") "compare")
                    ((equal kind "tmp") "temporal")
                    ((equal kind "rel") "relation")
                    ((eq (first q) :reverse) "reverse")
                    (t "extract")))
         (sub (cond ((equal cat "compare") (format nil "~(~a-~a~)" qtype (if (eq (first q) :max) "max" "min")))
                    ((equal cat "temporal") (string-downcase (symbol-name (first q))))
                    ((equal cat "relation") (format nil "~(~a~)-chain-~a" qtype (if (eq (first q) :rel-subj) "subj" "obj")))
                    (t (string-downcase (symbol-name qtype)))))
         (precs (parse-passage (getf p :passage) '(:dev)))
         (ans (position (getf p :answer) (getf p :options) :test #'string=))
         (cf-ans (position (getf p :cf-answer) (getf p :options) :test #'string=))
         (it (list :id (getf p :id) :split "dev" :probe-p t :kind kind :category cat :subtype sub
                   :passage (getf p :passage) :question (getf p :question) :options (getf p :options)
                   :answer ans :cf-question (getf p :cf-question) :cf-answer cf-ans
                   :qframe (getf (car qrec) :frame) :query q :cf-query cfq)))
    (unless (and ans cf-ans (/= ans cf-ans)) (error "Invalid option indices for probe ~s" p))
    (setf (getf it :facts) (derived-facts it precs))
    it))

(defun check-cell-item (original variant)
  "List of problems proving VARIANT differs from ORIGINAL only in wording; NIL when sound."
  (let ((problems nil) (cell (getf variant :cell)))
    (flet ((bad (fmt &rest args) (push (apply #'format nil fmt args) problems)))
      (dolist (k '(:id :split :category :subtype :options :answer :cf-answer :query :cf-query :facts))
        (unless (equal (getf original k) (getf variant k)) (bad "~a changed ~s" cell k)))
      (let* ((pfam (search "P-fam" cell :test #'char-equal)) (qfam (search "Q-fam" cell :test #'char-equal))
             (orig-rec (parse-passage (getf original :passage) '(:dev)))
             (var-rec (handler-case (parse-passage (getf variant :passage) (if pfam '(:train) '(:dev)))
                        (error (e) (bad "~a passage is not ~a wording: ~a" cell (if pfam "training" "development") e) nil))))
        ;; same facts in the same order: identical wording-free semantics (kind, type, role, every binding; intros by kind)
        (when var-rec
          (unless (equal (records-semantics orig-rec) (records-semantics var-rec)) (bad "~a passage meaning changed" cell)))
        ;; wording actually changed (or not) as the cell claims: every sentence belongs to exactly one split
        (dolist (s (asem-qagen::sentences-of (getf variant :passage)))
          (let ((dev (ignore-errors (parse-sentence s '(:dev)))) (train (ignore-errors (parse-sentence s '(:train)))))
            (unless (if pfam (and train (not dev)) (and dev (not train))) (bad "~a sentence ~s has the wrong wording split" cell s))))
        (when (and pfam (equal (getf variant :passage) (getf original :passage))) (bad "~a familiar passage is unchanged" cell))
        ;; answer recomputed from the rendered text alone
        (let ((facts (and var-rec (handler-case (derived-facts variant var-rec) (error (e) (bad "~a ~a" cell e) nil)))))
          (when facts
            (unless (equal (asem-qagen::solve (getf variant :category) facts (getf variant :query)) (nth (getf variant :answer) (getf variant :options)))
              (bad "~a solver on rendered text disagrees with the key" cell))
            (unless (equal (asem-qagen::solve (getf variant :category) facts (getf variant :cf-query)) (nth (getf variant :cf-answer) (getf variant :options)))
              (bad "~a solver on rendered text disagrees with the counterfactual key" cell))))
        ;; questions: same structured query, wording from the claimed split
        (loop for (qk queryk) in '((:question :query) (:cf-question :cf-query))
              do (let* ((text (getf variant qk)) (split (if qfam :train :dev))
                        (rec (ignore-errors (parse-question text (list split))))
                        (other (ignore-errors (parse-question text (list (if qfam :dev :train))))))
                   (cond ((null rec) (bad "~a ~(~a~) ~s is not ~(~a~) wording" cell qk text split))
                         (t (unless (equal (question-query rec) (getf variant queryk)) (bad "~a ~(~a~) asks ~s, not ~s" cell qk (question-query rec) (getf variant queryk)))
                            (unless (eq (getf (car rec) :type) (item-type variant)) (bad "~a ~(~a~) has the wrong fact type" cell qk))
                            (when (and (eq (getf (car rec) :kind) :tmp) (not (equal (getf (cdr rec) :p) (getf (getf variant :facts) :person))))
                              (bad "~a temporal question names another person" cell))))
                   (when other (bad "~a ~(~a~) ~s also parses as the other split" cell qk text))))
        (when (and qfam (equal (getf variant :question) (getf original :question))) (bad "~a familiar question is unchanged" cell))
        (unless pfam (unless (equal (getf variant :passage) (getf original :passage)) (bad "~a unfamiliar passage is not verbatim" cell)))
        (unless qfam (unless (and (equal (getf variant :question) (getf original :question)) (equal (getf variant :cf-question) (getf original :cf-question)))
                       (bad "~a unfamiliar question is not verbatim" cell)))
        (dolist (o (getf variant :options))
          (unless (= (asem-qagen::word-boundary-count o (getf original :passage))
                     (asem-qagen::word-boundary-count o (getf variant :passage)))
            (bad "~a option ~s occurrence count changed" cell o)))
        (unless (getf variant :probe-p)
          (dolist (p (asem-qagen:validate-item variant)) (bad "~a validate-item: ~a" cell p)))))
    (nreverse problems)))

;;; ------------------------------------------------------------------ paired 2x2 statistics & evaluation
(defun obj (&rest kvs) (apply #'asem-qa-analysis::obj kvs))
(defun arr (xs) (coerce xs 'vector))

(defparameter +metrics+
  '(("gold_length_greedy_prefix_exact" :prefix-exact)
    ("summed_option_likelihood" :sum-pred)
    ("mean_token_option_likelihood" :mean-pred)
    ("letter_A_to_D" :letter-pred)
    ("bounded_free_generation_exact" :bounded-exact)
    ("bounded_free_generation_terminated_exact" :terminated-exact)))

(defun row-hit-p (r metric)
  (if (member metric '(:prefix-exact :bounded-exact :terminated-exact))
      (and (getf r metric) t)
      (= (getf r :key) (getf r metric))))

(defun binom (n k)
  (if (or (< k 0) (> k n)) 0
      (let ((k (min k (- n k))) (res 1))
        (dotimes (i k res) (setf res (/ (* res (- n i)) (1+ i)))))))

(defun mcnemar-p (gained lost)
  (let ((d (+ gained lost)) (m (min gained lost)))
    (if (zerop d) 1d0
        (float (min 1 (/ (* 2 (loop for k from 0 to m sum (binom d k))) (ash 1 d))) 1d0))))

(defun contrast-ci (diffs &optional (z 1.96d0) (clamp t))
  "Mean, standard error and 95% confidence interval for a paired per-item contrast vector DIFFS."
  (let ((n (length diffs)))
    (if (zerop n)
        (values 0d0 0d0 0d0 0d0)
        (let* ((mean (/ (reduce #'+ diffs) n 1d0))
               (ss (loop for d in diffs sum (let (( dev (- (float d 1d0) mean))) (* dev dev))))
               (se (/ (sqrt ss) n))
               (lo (- mean (* z se)))
               (hi (+ mean (* z se))))
          (values mean se (if clamp (max -1d0 lo) lo) (if clamp (min 1d0 hi) hi))))))

(defun check-paired-rows (rows-a rows-b)
  (unless (= (length rows-a) (length rows-b)) (error "Paired row lists have unequal lengths"))
  (loop for a in rows-a for b in rows-b
        do (unless (and (equal (getf a :id) (getf b :id))
                        (equal (getf a :category) (getf b :category))
                        (eql (getf a :key) (getf b :key))
                        (equal (getf a :gold-text) (getf b :gold-text)))
             (error "Paired row mismatch between ~s and ~s" (getf a :id) (getf b :id)))))

(defun paired-diff (rows-a rows-b metric)
  "Paired comparison (ROWS-B minus ROWS-A) on METRIC: counts, difference, 95% CI and exact two-sided McNemar p."
  (check-paired-rows rows-a rows-b)
  (let ((n (length rows-a)) (k1 0) (k2 0) (both 0) (neither 0) (gained 0) (lost 0) (diffs nil))
    (loop for a in rows-a for b in rows-b
          for ha = (row-hit-p a metric) for hb = (row-hit-p b metric)
          do (when ha (incf k1)) (when hb (incf k2))
             (cond ((and ha hb) (incf both) (push 0d0 diffs))
                   ((and (not ha) (not hb)) (incf neither) (push 0d0 diffs))
                   ((and (not ha) hb) (incf gained) (push 1d0 diffs))
                   (t (incf lost) (push -1d0 diffs))))
    (multiple-value-bind (mean se lo hi) (contrast-ci (nreverse diffs))
      (obj "n" n "first_correct" k1 "second_correct" k2
           "first_accuracy" (/ k1 (max 1 n) 1d0) "second_accuracy" (/ k2 (max 1 n) 1d0)
           "both_correct" both "both_wrong" neither "gained" gained "lost" lost
           "discordant" (+ gained lost) "diff_count" (- gained lost)
           "diff" mean "se" se "ci95_low" lo "ci95_high" hi
           "mcnemar_exact_two_sided_p" (mcnemar-p gained lost)))))

(defun factorial-effects (uu uf fu ff metric)
  "2x2 paired main effects and interaction on METRIC across (P-unf/Q-unf, P-unf/Q-fam, P-fam/Q-unf, P-fam/Q-fam)."
  (check-paired-rows uu uf) (check-paired-rows uu fu) (check-paired-rows uu ff)
  (let ((p-main nil) (q-main nil) (inter nil))
    (loop for r-uu in uu for r-uf in uf for r-fu in fu for r-ff in ff
          for y-uu = (if (row-hit-p r-uu metric) 1d0 0d0)
          for y-uf = (if (row-hit-p r-uf metric) 1d0 0d0)
          for y-fu = (if (row-hit-p r-fu metric) 1d0 0d0)
          for y-ff = (if (row-hit-p r-ff metric) 1d0 0d0)
          do (push (* 0.5d0 (+ (- y-ff y-uf) (- y-fu y-uu))) p-main)
             (push (* 0.5d0 (+ (- y-ff y-fu) (- y-uf y-uu))) q-main)
             (push (- (- y-ff y-fu) (- y-uf y-uu)) inter))
    (flet ((pack (xs clamp)
             (multiple-value-bind (m se lo hi) (contrast-ci (nreverse xs) 1.96d0 clamp)
               (obj "n" (length uu) "effect" m "se" se "ci95_low" lo "ci95_high" hi))))
      (obj "passage_main_effect" (pack p-main t)
           "question_main_effect" (pack q-main t)
           "passage_x_question_interaction" (pack inter nil)))))

(defun numeric-subtype-p (sub)
  (member (first (asem-util:split-on sub #\-)) '("age" "score" "height") :test #'string-equal))

(defun group-filter (group)
  "Predicate selecting rows of semantic question GROUP."
  (cond ((string= group "comparison") (lambda (r) (equal (getf r :category) "compare")))
        ((string= group "temporal") (lambda (r) (equal (getf r :category) "temporal")))
        ((string= group "relationship") (lambda (r) (equal (getf r :category) "relation")))
        ((string= group "attribute") (lambda (r) (member (getf r :category) '("extract" "reverse") :test #'equal)))
        ((string= group "attribute_categorical")
         (lambda (r) (and (member (getf r :category) '("extract" "reverse") :test #'equal)
                          (not (numeric-subtype-p (getf r :subtype))))))
        ((string= group "attribute_numeric")
         (lambda (r) (and (equal (getf r :category) "extract") (numeric-subtype-p (getf r :subtype)))))
        ((string= group "attribute_forward") (lambda (r) (equal (getf r :category) "extract")))
        ((string= group "attribute_reverse") (lambda (r) (equal (getf r :category) "reverse")))
        (t (error "Unknown question group ~a" group))))

(defparameter +question-groups+
  '("comparison" "temporal" "attribute" "relationship"
    "attribute_categorical" "attribute_numeric" "attribute_forward" "attribute_reverse"))

(defun summarize-cell-rows (rows)
  (cons :obj
        (loop for (name metric) in +metrics+
              collect
              (let ((overall (asem-qa-analysis::accuracy-obj rows metric))
                    (groups (cons :obj (loop for g in +question-groups+
                                             collect (cons g (asem-qa-analysis::accuracy-obj (remove-if-not (group-filter g) rows) metric)))))
                    (cats (cons :obj (loop for c in asem-qagen:+categories+
                                           collect (cons c (asem-qa-analysis::accuracy-obj (remove-if-not (lambda (r) (equal (getf r :category) c)) rows) metric))))))
                (cons name (append overall (list (cons "by_question_group" groups) (cons "by_category" cats))))))))

(defun summarize-pair (rows-a rows-b)
  (cons :obj
        (loop for (name metric) in +metrics+
              collect
              (let ((overall (paired-diff rows-a rows-b metric))
                    (groups (cons :obj (loop for g in +question-groups+
                                             for fn = (group-filter g)
                                             collect (cons g (paired-diff (remove-if-not fn rows-a) (remove-if-not fn rows-b) metric)))))
                    (cats (cons :obj (loop for c in asem-qagen:+categories+
                                           for fn = (lambda (r) (equal (getf r :category) c))
                                           collect (cons c (paired-diff (remove-if-not fn rows-a) (remove-if-not fn rows-b) metric))))))
                (cons name (append overall (list (cons "by_question_group" groups) (cons "by_category" cats))))))))

(defun summarize-factorial (uu uf fu ff)
  (cons :obj
        (loop for (name metric) in +metrics+
              collect
              (let ((overall (factorial-effects uu uf fu ff metric))
                    (groups (cons :obj (loop for g in +question-groups+
                                             for fn = (group-filter g)
                                             collect (cons g (factorial-effects (remove-if-not fn uu) (remove-if-not fn uf)
                                                                                (remove-if-not fn fu) (remove-if-not fn ff) metric)))))
                    (cats (cons :obj (loop for c in asem-qagen:+categories+
                                           for fn = (lambda (r) (equal (getf r :category) c))
                                           collect (cons c (factorial-effects (remove-if-not fn uu) (remove-if-not fn uf)
                                                                              (remove-if-not fn fu) (remove-if-not fn ff) metric))))))
                (cons name (append overall (list (cons "by_question_group" groups) (cons "by_category" cats))))))))

(defun run-cell-predictions (model bpe variants workers)
  (let* ((parts (asem-train32::parallel-map
                 (lambda (part)
                   (let ((cache (make-hash-table)) (worker-bpe (asem-bpe::copy-bpe bpe)))
                     (setf (asem-bpe::bpe-cache worker-bpe) (make-hash-table :test 'equal))
                     (mapcar (lambda (it) (asem-qa-analysis::predict-item model worker-bpe it cache)) part)))
                 (asem-train32::shards variants workers)))
         (rows (apply #'append parts)))
    (mapcar (lambda (it) (find (getf it :id) rows :key (lambda (r) (getf r :id)) :test #'equal)) variants)))

(defun evaluate-2x2 (model bpe items &key (workers 4) (policy :seeded) (counterfactual t))
  "Run the paired 2x2 passage x question wording diagnostic on development ITEMS."
  (unless (and (plusp workers) items) (error "Need positive workers and nonempty development items"))
  (unless (every (lambda (it) (equal (getf it :split) "dev")) items) (error "Only development items may be evaluated"))
  (let ((cells (cell-items items :policy policy)) (rows-by-cell nil) (cf-by-cell nil))
    ;; Verify every cell item preserves meaning and answers before running the model.
    (dolist (cell cells)
      (loop for orig in items for var in (cdr cell)
            for problems = (check-cell-item orig var)
            when problems do (error "Semantic check failed for ~a (~a): ~{~a~^; ~}" (getf orig :id) (car cell) problems)))
    (dolist (cell cells)
      (let ((rows (run-cell-predictions model bpe (cdr cell) workers)))
        (push (cons (car cell) rows) rows-by-cell)
        (format *error-output* "2x2 ~a (~a): ~d items complete~%" (car cell) policy (length rows)))
      (when counterfactual
        (let* ((cf-items (mapcar (lambda (it) (asem-qa-analysis:condition-item (cdr cell) it :counterfactual)) (cdr cell)))
               (cf-rows (run-cell-predictions model bpe cf-items workers)))
          (push (cons (car cell) cf-rows) cf-by-cell)
          (format *error-output* "2x2 ~a counterfactual (~a): ~d items complete~%" (car cell) policy (length cf-rows)))))
    (setf rows-by-cell (nreverse rows-by-cell) cf-by-cell (nreverse cf-by-cell))
    (flet ((cell-rows (name &optional cf) (cdr (assoc name (if cf cf-by-cell rows-by-cell) :test #'string=))))
      (let ((uu (cell-rows "p-unf/q-unf")) (uf (cell-rows "p-unf/q-fam"))
            (fu (cell-rows "p-fam/q-unf")) (ff (cell-rows "p-fam/q-fam")))
        (obj "policy" (string-downcase (symbol-name policy))
             "cells" (cons :obj (loop for (name . rows) in rows-by-cell
                                      collect (cons name (obj "summary" (summarize-cell-rows rows)
                                                              "items" (arr (mapcar #'asem-qa-analysis::row-json rows))))))
             "paired_differences"
             (obj "p_unf_q_unf_to_p_fam_q_fam" (summarize-pair uu ff)
                  "p_unf_q_unf_to_p_fam_q_unf" (summarize-pair uu fu)
                  "p_unf_q_unf_to_p_unf_q_fam" (summarize-pair uu uf)
                  "p_unf_q_fam_to_p_fam_q_fam" (summarize-pair uf ff)
                  "p_fam_q_unf_to_p_fam_q_fam" (summarize-pair fu ff)
                  "p_unf_q_fam_to_p_fam_q_unf" (summarize-pair uf fu))
             "factorial_effects" (summarize-factorial uu uf fu ff)
             "counterfactual_cells"
             (if counterfactual
                 (cons :obj (loop for (name . rows) in cf-by-cell
                                  collect (cons name (obj "summary" (summarize-cell-rows rows)
                                                          "items" (arr (mapcar #'asem-qa-analysis::row-json rows))))))
                 :null))))))
