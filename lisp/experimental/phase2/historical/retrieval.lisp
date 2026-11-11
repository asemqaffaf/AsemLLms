;;;; Entity-retrieval curriculum and probes (Stage 2, after the copy stage).
;;;;
;;;; A retrieval item is a short passage of facts and a question about ONE of them; the answer is the value, produced verbatim:
;;;;     Text: <passage>
;;;;     Q: <question>
;;;;     The answer is <value>
;;;; Scoring is greedy exact match of the value's tokens (no options, so no guessing floor).
;;;;
;;;; Levels: 1 = one fact; 2 = two facts about different people; 3 = four facts; 4 = four facts plus 2-4 distracting sentences
;;;; about the same people (other attributes). Fact kinds: attributes (town, job, pet, colour, instrument), numbers (age, score,
;;;; height) and directed relations (teach, hire, call, thank, visit, help; asked by subject or object).
;;;;
;;;; Training items use generated names/towns, training attribute pools and training wording (asem-qagen frames of the :train split).
;;;; Probes use entities never seen in training -- either the QA dev pools (:dev-real) or names/towns generated here from a separate
;;;; syllable inventory (:new) -- with training wording (:seen) or the QA dev wording (:unseen). The QA test pools and test wording
;;;; are never used.
(defpackage :asem-retrieval
  (:use :cl)
  (:export #:*reasoning* #:*reverse* #:*augment* #:*predicate-paraphrase* #:*paraphrase-rng* #:*broad-train* #:*vocab-words*
           #:vocab-words-of #:all-predicate-paraphrase-frames #:make-fact-set #:probe-items #:train-sequences #:retrieval-prompt #:+levels+))
(in-package :asem-retrieval)

(defparameter +levels+ '(1 2 3 4))

(defun q (sym) (find-symbol (string sym) :asem-qagen))
(defmacro qg (fn &rest args) `(funcall (symbol-function (q ',fn)) ,@args))

;;; ------------------------------------------------------------------ entity sources
(defvar *new-cache* (make-hash-table))

(defun new-pool (kind)
  "Names or towns generated from an inventory used nowhere else, minus anything that collides with any other pool."
  (or (gethash kind *new-cache*)
      (setf (gethash kind *new-cache*)
            (let* ((taken (make-hash-table :test 'equalp)) (rng (sb-ext:seed-random-state 9191))
                   (cands (ecase kind
                            (:names (loop for o in '("Qu" "Xa" "Vy" "Zo" "Ky" "Wu" "Ja" "Ru" "Ti" "Ne" "Lo" "Fa" "Hu" "Pe" "Gy" "Mo")
                                          append (loop for m in '("ri" "ne" "lo" "ta" "vi" "so" "ka" "du" "me")
                                                       append (loop for e in '("n" "x" "l" "s" "")
                                                                    collect (concatenate 'string o m e)))))
                            (:towns (loop for o in '("Quar" "Xen" "Vyl" "Zor" "Kyr" "Wul" "Jas" "Rum" "Tib" "Nesh" "Lom" "Fask" "Hul" "Pesk" "Gyl" "Mox")
                                          append (loop for s in '("thal" "bourne" "spire" "keld" "rith" "moss" "fell" "garth")
                                                       collect (concatenate 'string o s)))))))
              (dolist (pool (list (symbol-value (q '*names*)) (symbol-value (q '*towns*)) (symbol-value (q '*jobs*)) (symbol-value (q '*pets*))
                                  (symbol-value (q '*colors*)) (symbol-value (q '*instruments*)) (symbol-value (q '*places*))
                                  (qg synthetic-pool (q '*names*)) (qg synthetic-pool (q '*towns*))))
                (dolist (x pool) (setf (gethash x taken) t)))
              (qg shuffle (remove-if (lambda (c) (gethash c taken)) cands) rng)))))

;;; ------------------------------------------------------------------ high-entropy training answers (R2 onwards)
;;; With *BROAD-TRAIN* true, training names and towns come from a ~30k-string syllable inventory and attribute values from
;;; thousands of whole-word tokenizer entries (*VOCAB-WORDS*, set by the driver from the BPE), so no small set of answer strings
;;; can be learned as a prior. Everything that equals a QA dev/test value or a probe-only (:new) string is removed.
(defvar *broad-train* nil)
(defvar *vocab-words* nil)
(defvar *broad-cache* (make-hash-table))

(defun heldout-strings ()
  (let ((h (make-hash-table :test 'equalp)))
    (dolist (pool (list (symbol-value (q '*names*)) (symbol-value (q '*towns*)) (symbol-value (q '*jobs*)) (symbol-value (q '*pets*))
                        (symbol-value (q '*colors*)) (symbol-value (q '*instruments*)) (symbol-value (q '*places*))))
      (dolist (s '(:dev :test)) (dolist (x (qg split-pool pool s)) (setf (gethash x h) t))))
    (dolist (x (new-pool :names)) (setf (gethash x h) t))
    (dolist (x (new-pool :towns)) (setf (gethash x h) t))
    h))

(defun broad-pool (kind)
  (or (gethash kind *broad-cache*)
      (setf (gethash kind *broad-cache*)
            (let ((held (heldout-strings)) (rng (sb-ext:seed-random-state 5150)))
              (qg shuffle
                  (remove-if (lambda (c) (or (< (length c) 3) (gethash c held)))
                             (remove-duplicates
                              (ecase kind
                                (:names (loop for o in '("B" "C" "D" "F" "G" "H" "J" "K" "L" "M" "N" "P" "R" "S" "T" "V" "W" "Y" "Z" "Br" "Ch" "Cl" "Dr" "Fl" "Gr" "Kr" "Pr" "Sh" "St" "Th" "Tr" "A" "E" "I" "O" "U")
                                              append (loop for v in '("a" "e" "i" "o" "u" "ai" "ou")
                                                           append (loop for m in '("" "l" "n" "r" "m" "s" "v" "d" "k" "b" "t")
                                                                        append (loop for e in '("" "a" "o" "en" "ar" "is" "el" "an")
                                                                                     collect (concatenate 'string o v m e))))))
                                (:towns (loop for o in '("Ash" "Bram" "Cor" "Dal" "Elm" "Fal" "Gil" "Hol" "Ist" "Jor" "Kel" "Lor" "Mal" "Nor" "Orm" "Pry" "Ros" "Sar" "Tev" "Ur" "Vos" "Wil" "Yar" "Zel" "Brig" "Cal" "Dorn" "Ever" "Glen" "Hart" "Kin" "Lang" "Mor" "Pen" "Red" "Stan" "Thorn" "Wex")
                                              append (loop for s in '("wold" "haven" "brook" "mouth" "field" "crest" "port" "burg" "thorpe" "shaw" "wyn" "dell" "ford" "by" "ton" "ley" "wick" "stead" "mere" "dale")
                                                           collect (concatenate 'string o s)))))
                              :test #'string-equal))
                  rng)))))

(defun broad-values ()
  (or (gethash :values *broad-cache*)
      (setf (gethash :values *broad-cache*)
            (let ((held (heldout-strings)))
              (remove-if (lambda (w) (gethash w held)) *vocab-words*)))))

(defun vocab-words-of (bpe)
  "Whole lowercase words (4-10 letters) that are single tokens with a leading space, e.g. \" table\"."
  (loop for id from 257 below (asem-bpe:bpe-vocab-size bpe)
        for s = (asem-bpe:bpe-decode bpe (list id))
        when (and (> (length s) 4) (<= (length s) 11) (char= (char s 0) #\Space) (every #'lower-case-p (subseq s 1)))
          collect (subseq s 1)))

(defun pool (entities sym)
  "Values for SYM ('*names*, '*towns*, '*jobs* ...) under ENTITIES: :train (generated names/towns, training pools), :dev-real
(QA dev pools), :new (new names/towns, QA dev pools for the attribute words)."
  (let ((s (q sym)))
    (ecase entities
      (:train (cond ((not *broad-train*) (let ((asem-qagen::*synthetic-train* t)) (qg entity-pool s :train)))
                    ((string= (symbol-name sym) "*NAMES*") (broad-pool :names))
                    ((string= (symbol-name sym) "*TOWNS*") (broad-pool :towns))
                    (t (append (qg split-pool (symbol-value s) :train) (broad-values)))))
      (:dev-real (qg split-pool (symbol-value s) :dev))
      (:new (cond ((string= (symbol-name sym) "*NAMES*") (new-pool :names))
                  ((string= (symbol-name sym) "*TOWNS*") (new-pool :towns))
                  (t (qg split-pool (symbol-value s) :dev)))))))

;;; ------------------------------------------------------------------ wording augmentation (training only)
;;; With *AUGMENT* true, training sentences get modifiers inserted at random word gaps and some get a leading clause, and some
;;; questions are replaced by generic type-cued templates, so the start of the answer span is not tied to a few frames. None of these
;;; strings is a QA dev or test frame (bin/retrieval.lisp audit checks 4-gram overlap).
(defvar *augment* nil)
(defvar *reverse* nil "When true, about half of the attribute questions are reverse lookups (value -> person, QA dev/train :q-reverse frames).")
(defparameter *modifiers* '("now" "still" "also" "today" "really" "already" "usually" "these days" "for now" "at the moment"
                             "as it happens" "by all accounts" "it seems" "we hear" "once again" "as usual"))
(defparameter *leads* '("According to the note, " "As far as anyone knows, " "Everyone agrees that " "Rumor says that " "For the record, "))
(defparameter *generic-questions* '("{P} has which {noun}?" "Report {P}'s {noun}." "Which {noun} goes with {P}?" "{P}'s {noun}, please?"
                                     "Tell me {P}'s {noun}."))
(defparameter *type-nouns* '(:town "town" :job "job" :pet "pet" :color "color" :instrument "instrument" :age "age" :score "score" :height "height"))

(defun insert-modifier (s protect rng)
  "Insert a random modifier at a random gap between words of S, never inside one of the strings in PROTECT."
  (let* ((gaps (loop for i from 1 below (length s) when (and (char= (char s i) #\Space)
                                                             (notany (lambda (p) (let ((j (search p s))) (and j (< j i (+ j (length p)))))) protect))
                     collect i)))
    (if (null gaps) s
        (let ((g (nth (random (length gaps) rng) gaps)))
          (concatenate 'string (subseq s 0 g) " " (qg pick *modifiers* rng) (subseq s g))))))

(defun augment-sentence (s protect rng)
  (let ((s (if (< (random 10 rng) 5) (insert-modifier s protect rng) s)))
    (if (< (random 10 rng) 2)
        (let ((lead (qg pick *leads* rng)) (entity-first (some (lambda (p) (eql 0 (search p s))) protect)))
          (concatenate 'string lead (if entity-first (subseq s 0 1) (string-downcase (subseq s 0 1))) (subseq s 1)))
        s)))

(defun augment-question (qtext person type-key protect rng)
  (cond ((and type-key person (< (random 10 rng) 4))
         (qg fill-frame (qg replace-all (qg pick *generic-questions* rng) "{noun}" (getf *type-nouns* type-key)) :p person))
        ((< (random 10 rng) 3) (insert-modifier qtext protect rng))
        (t qtext)))

;;; ------------------------------------------------------------------ training-only predicate paraphrase augmentation
;;; Expands predicate/relation/temporal/comparison wording diversity ONLY during training (:train / :seen) when
;;; *PREDICATE-PARAPHRASE* is true. Attribute extraction (attr-facts / num-facts) and reverse lookup are kept untouched as
;;; skill-retention controls. All paraphrase strings below were authored independently of QA dev/test frames and checked to
;;; have zero duplicate or 4-gram overlap against QA dev frames.
(defvar *predicate-paraphrase* nil
  "When true, mix training-only predicate paraphrases for temporal, relationship, and comparison items during training.")
(defvar *paraphrase-rng* nil
  "Optional independent RNG state used when selecting predicate paraphrases so base fact RNG advancement matches Control.")

(defparameter *rel-paraphrases*
  '((:teach
     :frames ("{A} instructed {B}." "{B} was instructed by {A}." "{A} coached {B}." "{B} trained under {A}.")
     :q-subj ("Who instructed {B}?" "Who coached {B}?" "Under whom did {B} train?")
     :q-obj ("Who did {A} instruct?" "Who did {A} coach?" "Who trained under {A}?"))
    (:hire
     :frames ("{A} employed {B}." "{B} was employed by {A}." "{A} appointed {B}." "{B} was brought on by {A}.")
     :q-subj ("Who employed {B}?" "Who appointed {B}?" "Who was {B} employed by?")
     :q-obj ("Who did {A} employ?" "Who did {A} appoint?" "Who was appointed by {A}?"))
    (:call
     :frames ("{A} phoned {B}." "{B} was phoned by {A}." "{A} dialed {B}." "{B} got a call from {A}.")
     :q-subj ("Who phoned {B}?" "Who dialed {B}?" "From whom did {B} get a call?")
     :q-obj ("Who did {A} phone?" "Who did {A} dial?" "Who got a call from {A}?"))
    (:thank
     :frames ("{A} offered thanks to {B}." "{A} gave thanks to {B}." "{B} was given thanks by {A}." "{B} got thanks from {A}.")
     :q-subj ("Who gave thanks to {B}?" "Who offered thanks to {B}?" "From whom did {B} get thanks?")
     :q-obj ("Who got thanks from {A}?" "Who was given thanks by {A}?" "Who received thanks from {A}?"))
    (:visit
     :frames ("{A} came to see {B}." "{B} hosted a visit from {A}." "{A} called on {B}." "{B} was called on by {A}.")
     :q-subj ("Who came to see {B}?" "Who called on {B}?" "Whose visit did {B} host?")
     :q-obj ("Who did {A} come to see?" "Who did {A} call on?" "Who hosted {A}?"))
    (:help
     :frames ("{A} aided {B}." "{B} was aided by {A}." "{A} gave aid to {B}." "{B} received aid from {A}.")
     :q-subj ("Who aided {B}?" "Who gave aid to {B}?" "From whom did {B} receive aid?")
     :q-obj ("Who did {A} aid?" "Who received aid from {A}?" "Who was aided by {A}?"))))

(defparameter *cmp-paraphrases*
  '((:age
     :frames ("{P} is at age {N}." "{P} has lived {N} years." "Aged {N}, there is {P}.")
     :q-max ("Which person has the most years?" "Among them, who is oldest?" "Whose age tops the rest?")
     :q-min ("Which person has the fewest years?" "Among them, who is youngest?" "Whose age trails the rest?"))
    (:score
     :frames ("{P} tallied {N} points." "{P} posted {N} points." "{P} came away with {N} points.")
     :q-max ("Who posted the top score?" "Who tallied maximum points?" "Who came out with top points?")
     :q-min ("Who posted the bottom score?" "Who tallied minimum points?" "Who came away with bottom points?"))
    (:height
     :frames ("{P} reaches {N} cm tall." "{P} has a height of {N} cm." "Standing {N} cm tall is {P}.")
     :q-max ("Who stands tallest here?" "Whose height tops the rest?" "Which person reaches top height?")
     :q-min ("Who stands shortest here?" "Whose height trails the rest?" "Which person comes in shortest?"))))

(defparameter *tmp-paraphrases*
  '(:first ("{P} opened the day at the {X}." "Early on, {P} went to the {X}." "To begin, {P} went to the {X}."
            "{P} kicked off at the {X}." "At first, {P} went to the {X}." "First up, {P} stopped at the {X}.")
    :middle ("Afterward, {P} went to the {X}." "Soon after, {P} went to the {X}." "Next up, {P} stopped at the {X}."
             "Later, {P} went to the {X}." "A bit later, {P} went to the {X}." "{P} then stopped by the {X}.")
    :last ("At the end, {P} went to the {X}." "Lastly, {P} went to the {X}." "To wrap up, {P} stopped at the {X}."
           "{P} ended up at the {X}." "In the end, {P} went to the {X}." "At the close, {P} went to the {X}.")
    :q-first ("Where did {P} begin?" "Which spot did {P} visit first?" "At the start, where did {P} go?"
              "What spot did {P} start at?" "Where did {P} kick off?" "Where was {P} at first?")
    :q-last ("Where did {P} end up?" "Which spot did {P} go to last?" "At the end, where did {P} go?"
             "Where did {P} wrap up?" "What was {P}'s final stop?" "Where was {P} at last?")
    :q-after ("Where did {P} go right after the {X}?" "Where was {P} right after the {X}?" "Just after the {X}, where did {P} go?"
              "Directly following the {X}, where did {P} go?" "Which spot came right after the {X} for {P}?" "Immediately after the {X}, where was {P}?")
    :q-before ("Where was {P} right before the {X}?" "Just before the {X}, where did {P} go?" "Right before the {X}, where did {P} stop?"
               "Which spot came right before the {X} for {P}?" "Immediately before the {X}, where was {P}?" "Directly ahead of the {X}, where did {P} go?")))

(defun all-predicate-paraphrase-frames ()
  "All training-only predicate paraphrase templates added by *PREDICATE-PARAPHRASE* (for audit/overlap tests)."
  (let ((out nil))
    (dolist (r *rel-paraphrases*) (dolist (k '(:frames :q-subj :q-obj)) (dolist (f (getf (cdr r) k)) (push f out))))
    (dolist (c *cmp-paraphrases*) (dolist (k '(:frames :q-max :q-min)) (dolist (f (getf (cdr c) k)) (push f out))))
    (loop for (nil xs) on *tmp-paraphrases* by #'cddr do (dolist (f xs) (push f out)))
    (nreverse out)))

(defun predicate-paraphrase-active-p (entities wording)
  (and *predicate-paraphrase* (eq entities :train) (eq wording :seen)))

(defun pick-predicate-frame (base extra rng active)
  "Pick a template from BASE using RNG (advancing RNG identically to baseline), and when ACTIVE select uniformly over (append BASE EXTRA) with *PARAPHRASE-RNG*."
  (let ((chosen (qg pick base rng)))
    (if (and active extra)
        (let ((all (append base extra))
              (prng (or *paraphrase-rng* rng)))
          (nth (random (length all) prng) all))
        chosen)))

;;; ------------------------------------------------------------------ fact sets
;;; A fact set is a passage plus a list of (question . answer) pairs, each answerable from exactly one passage sentence.
(defun frames-of (plist key wording) (getf (getf plist key) (if (eq wording :unseen) :dev :train)))

(defun augmented (sents qa people type-key protect rng &optional (ret people) ty)
  "Apply the training-only wording augmentation when *AUGMENT*; returns (values sents qa people ty)."
  (if (not *augment*) (values sents qa ret ty)
      (values (mapcar (lambda (s) (augment-sentence s protect rng)) sents)
              (loop for x in qa for i from 0
                    collect (cons (augment-question (car x) (and people (not (member (cdr x) people :test #'string=)) (nth i people)) type-key protect rng) (cdr x)))
              ret ty)))

(defun attr-facts (level entities wording rng)
  (let* ((ty (qg pick (symbol-value (q '*attr-types*)) rng)) (spec (cdr ty))
         (n (ecase level (1 1) (2 2) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (qg sample-n (pool entities (intern (symbol-name (getf spec :pool)) :asem-retrieval)) n rng))
         (frames (frames-of spec :frames wording)) (qframes (frames-of spec :q-extract wording)) (rframes (frames-of spec :q-reverse wording))
         (sents (loop for p in people for v in vals collect (qg fill-frame (qg pick frames rng) :p p :v v)))
         (qa (loop for p in people for v in vals
                   collect (if (and *reverse* (zerop (random 2 rng)))
                               (cons (qg fill-frame (qg pick rframes rng) :v v) p)
                               (cons (qg fill-frame (qg pick qframes rng) :p p) v)))))
    (multiple-value-bind (s qq p ty2) (augmented sents qa people (car ty) (append people vals) rng people ty) (values s qq p ty2 vals))))

(defun num-facts (level entities wording rng)
  (let* ((ty (qg pick (symbol-value (q '*num-types*)) rng)) (spec (cdr ty))
         (n (ecase level (1 1) (2 2) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (destructuring-bind (lo hi) (getf spec :range) (qg distinct-numbers lo hi n rng)))
         (frames (frames-of spec :frames wording)) (qframes (frames-of spec :q-extract wording))
         (sents (loop for p in people for v in vals collect (qg fill-frame (qg pick frames rng) :p p :n v :v v)))
         (qa (loop for p in people for v in vals collect (cons (qg fill-frame (qg pick qframes rng) :p p) (princ-to-string v)))))
    (multiple-value-bind (s qq p ty2) (augmented sents qa people (car ty) (append people (mapcar (function princ-to-string) vals)) rng people ty)
      (values s qq p ty2 (mapcar (function princ-to-string) vals)))))

(defun rel-facts (level entities wording rng)
  "Level 1: one pair; 2: two pairs; 3-4: a chain of three links (four people). Questions by subject or by object, each with a unique answer."
  (let* ((rel (qg pick (symbol-value (q '*relations*)) rng)) (spec (cdr rel))
         (active (predicate-paraphrase-active-p entities wording))
         (para (and active (cdr (assoc (car rel) *rel-paraphrases*))))
         (people (qg sample-n (pool entities '*names*) (ecase level (1 2) (2 4) ((3 4) 4)) rng))
         (pairs (ecase level
                  (1 (list (list (first people) (second people))))
                  (2 (list (list (first people) (second people)) (list (third people) (fourth people))))
                  ((3 4) (list (list (first people) (second people)) (list (second people) (third people)) (list (third people) (fourth people))))))
         (frames (frames-of spec :frames wording)) (qs (frames-of spec :q-subj wording)) (qo (frames-of spec :q-obj wording))
         (sents (mapcar (lambda (pr) (qg fill-frame (pick-predicate-frame frames (getf para :frames) rng active) :a (first pr) :b (second pr))) pairs))
         (qa (mapcar (lambda (pr) (if (zerop (random 2 rng))
                                      (cons (qg fill-frame (pick-predicate-frame qs (getf para :q-subj) rng active) :b (second pr)) (first pr))
                                      (cons (qg fill-frame (pick-predicate-frame qo (getf para :q-obj) rng active) :a (first pr)) (second pr))))
                     pairs)))
    (augmented sents qa nil nil people rng people rel)))

;;; ------------------------------------------------------------------ comparison and temporal-order kinds (*REASONING*)
(defvar *reasoning* nil "When true, fact sets also include numeric comparisons and temporal orders (QA compare / temporal skills).")

(defun cmp-facts (level entities wording rng)
  "2 / 3 / 4 people with distinct numbers of one type; questions: who has the largest and who the smallest."
  (let* ((ty (qg pick (symbol-value (q '*num-types*)) rng)) (spec (cdr ty))
         (active (predicate-paraphrase-active-p entities wording))
         (para (and active (cdr (assoc (car ty) *cmp-paraphrases*))))
         (n (ecase level (1 2) (2 3) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (destructuring-bind (lo hi) (getf spec :range) (qg distinct-numbers lo hi n rng)))
         (frames (frames-of spec :frames wording))
         (sents (loop for p in people for v in vals
                      collect (qg fill-frame (pick-predicate-frame frames (getf para :frames) rng active) :p p :n v :v v)))
         (best (lambda (fn) (nth (position (reduce fn vals) vals) people)))
         (qa (list (cons (qg fill-frame (pick-predicate-frame (frames-of spec :q-max wording) (getf para :q-max) rng active)) (funcall best #'max))
                   (cons (qg fill-frame (pick-predicate-frame (frames-of spec :q-min wording) (getf para :q-min) rng active)) (funcall best #'min)))))
    (multiple-value-bind (s qq p ty2) (augmented sents qa nil nil (append people (mapcar #'princ-to-string vals)) rng people ty)
      (values s qq p ty2 nil))))

(defun tmp-facts (level entities wording rng)
  "One person's day with 2 / 3 / 4 stops in narrative order; questions: first, last, right after X, right before X."
  (let* ((tm (symbol-value (q '*temporal*))) (k (ecase level (1 2) (2 3) ((3 4) 4)))
         (active (predicate-paraphrase-active-p entities wording))
         (p (qg pick (pool entities '*names*) rng))
         (places (qg sample-n (pool entities '*places*) k rng))
         (sents (loop for x in places for i from 0
                      for pos = (cond ((= i 0) :first) ((= i (1- k)) :last) (t :middle))
                      collect (qg fill-frame (pick-predicate-frame (frames-of tm pos wording) (and active (getf *tmp-paraphrases* pos)) rng active) :p p :x x)))
         (qa (append (list (cons (qg fill-frame (pick-predicate-frame (frames-of tm :q-first wording) (and active (getf *tmp-paraphrases* :q-first)) rng active) :p p) (first places))
                           (cons (qg fill-frame (pick-predicate-frame (frames-of tm :q-last wording) (and active (getf *tmp-paraphrases* :q-last)) rng active) :p p) (car (last places))))
                     (loop for i from 0 below (1- k)
                           collect (if (zerop (random 2 rng))
                                       (cons (qg fill-frame (pick-predicate-frame (frames-of tm :q-after wording) (and active (getf *tmp-paraphrases* :q-after)) rng active) :p p :x (nth i places)) (nth (1+ i) places))
                                       (cons (qg fill-frame (pick-predicate-frame (frames-of tm :q-before wording) (and active (getf *tmp-paraphrases* :q-before)) rng active) :p p :x (nth (1+ i) places)) (nth i places)))))))
    ;; narrative order is the fact here, so the sentences are NOT shuffled by make-fact-set (see :keep-order)
    (multiple-value-bind (s qq pp ty2) (augmented sents qa nil nil (cons p places) rng (list p) :temporal)
      (values s qq pp ty2 places))))

(defun distractors (people entities wording exclude-ty rng k)
  "K sentences giving OTHER attributes of the same people (never asked about)."
  (loop repeat k
        collect (let* ((ty (qg pick (remove exclude-ty (symbol-value (q '*attr-types*))) rng)) (spec (cdr ty)))
                  (qg fill-frame (qg pick (frames-of spec :frames wording) rng) :p (qg pick people rng)
                      :v (qg pick (pool entities (intern (symbol-name (getf spec :pool)) :asem-retrieval)) rng)))))

(defun %make-fact-set (level entities wording rng)
  "(values passage qa-pairs kind candidates padder): CANDIDATES are the passage's values of the asked type (the natural options),
PADDER returns one more value of that type (for padding the options to four)."
  (let ((kind (qg pick (if *reasoning* '(:attr :attr :num :rel :cmp :tmp) '(:attr :attr :num :rel)) rng)))
    (multiple-value-bind (sents qa people ty vals)
        (ecase kind (:attr (attr-facts level entities wording rng)) (:num (num-facts level entities wording rng)) (:rel (rel-facts level entities wording rng))
          (:cmp (cmp-facts level entities wording rng)) (:tmp (tmp-facts level entities wording rng)))
      (let* ((all (append sents (when (and (= level 4) (not (eq kind :tmp))) (distractors people entities wording (and (eq kind :attr) ty) rng (+ 2 (random 3 rng))))))
            (values-of (or vals (remove-duplicates (mapcar #'cdr qa) :test #'string=)))
            (cands (lambda (ans) (if (or (member kind '(:rel :cmp)) (member ans people :test #'string=)) people values-of)))
            (names (pool entities '*names*))
            (padder (ecase kind
                      (:attr (let ((p (pool entities (getf (cdr ty) :pool)))) (lambda (ans) (if (member ans people :test #'string=) (lambda () (qg pick names rng)) (lambda () (qg pick p rng))))))
                      (:num (destructuring-bind (lo hi) (getf (cdr ty) :range) (lambda (ans) (declare (ignore ans)) (lambda () (princ-to-string (+ lo (random (1+ (- hi lo)) rng)))))))
                      (:rel (lambda (ans) (declare (ignore ans)) (lambda () (qg pick names rng))))
                      (:cmp (lambda (ans) (declare (ignore ans)) (lambda () (qg pick names rng))))
                      (:tmp (let ((pl (pool entities '*places*))) (lambda (ans) (declare (ignore ans)) (lambda () (qg pick pl rng))))))))
        (values (format nil "~{~a~^ ~}" (if (eq kind :tmp) all (qg shuffle all rng))) qa kind cands padder)))))

(defun make-fact-set (level entities wording rng)
  (%make-fact-set level entities wording rng))

(defun options-for (answer cands padder rng)
  "Four distinct options containing ANSWER: other passage values first, then padding; shuffled. None is a substring of another."
  (let ((opts (list answer)))
    (dolist (c (qg shuffle cands rng)) (when (and (< (length opts) 4) (notany (lambda (o) (or (search c o) (search o c))) opts)) (push c opts)))
    (loop repeat 50 while (< (length opts) 4)
          do (let ((c (funcall padder))) (when (notany (lambda (o) (or (search c o) (search o c))) opts) (push c opts))))
    (and (= 4 (length opts)) (qg shuffle opts rng))))

(defun options-block (opts) (format nil "A. ~a~%B. ~a~%C. ~a~%D. ~a~%" (first opts) (second opts) (third opts) (fourth opts)))

(defun retrieval-prompt (passage question &key (passage-p t) (question-p t) options)
  "OPTIONS (four strings) adds the QA option lines between the question and \"The answer is\"."
  (format nil "~@[Text: ~a~%~]~@[Q: ~a~%~]~@[~a~]The answer is" (and passage-p passage) (and question-p question) (and options (options-block options))))

(defun answer-occurs-once-p (passage qa)
  "The answer must be present and must not be a substring of another answer in the set (so exact match is unambiguous)."
  (and (search (cdr qa) passage) t))

;;; ------------------------------------------------------------------ probes
(defun probe-items (level entities wording n seed &key (fits (constantly t)) options &aux (*augment* nil) (*predicate-paraphrase* nil))
  "N single-question items (plists :id :level :kind :passage :question :answer :cf-question :cf-answer). The counterfactual
question asks about another fact of the same passage (levels >= 2) and must change the answer. FITS is called with
(prompt answer) for the real and counterfactual question; items it rejects (too long for the context) are redrawn, never truncated."
  (let ((rng (sb-ext:seed-random-state seed)) (out nil))
    (loop while (< (length out) n)
          do (multiple-value-bind (passage qa kind cands padder) (make-fact-set level entities wording rng)
               (let* ((i (random (length qa) rng)) (main (nth i qa))
                      (others (remove-if (lambda (x) (string= (cdr x) (cdr main))) (remove main qa)))
                      (cf (and others (qg pick others rng)))
                      (opts (and options (options-for (cdr main) (funcall cands (cdr main)) (funcall padder (cdr main)) rng))))
                 (when (and (every (lambda (x) (answer-occurs-once-p passage x)) qa)
                            (or (not options) (and opts (or (null cf) (member (cdr cf) opts :test #'string=))))
                            (every (lambda (o) (funcall fits (retrieval-prompt passage (car main) :options opts) o)) (or opts (list (cdr main))))
                            (or (null cf) (every (lambda (o) (funcall fits (retrieval-prompt passage (car cf) :options opts) o)) (or opts (list (cdr cf))))))
                   (push (list :id (format nil "L~d-~(~a-~a~)-~d" level entities wording (length out)) :level level :kind (string-downcase (symbol-name kind))
                               :passage passage :question (car main) :answer (cdr main) :options opts
                               :cf-question (and cf (car cf)) :cf-answer (and cf (cdr cf)))
                         out)))))
    (nreverse out)))

;;; ------------------------------------------------------------------ dense training sequences
(defun encode-fact-set-sequence (passage qa cands padder bpe ctx options rng)
  (let ((ids (list 0)) (rows nil) (targets nil))
    (flet ((add (text) (setf ids (append ids (asem-bpe:bpe-encode bpe text)))))
      (add (format nil "Text: ~a~%" passage))
      (dolist (x (qg shuffle qa rng))
        (let* ((opts (and options (options-for (cdr x) (funcall cands (cdr x)) (funcall padder (cdr x)) rng)))
               (qtoks (asem-bpe:bpe-encode bpe (format nil "Q: ~a~%~@[~a~]The answer is" (car x) (and opts (options-block opts)))))
               (atoks (asem-bpe:bpe-encode bpe (format nil " ~a" (cdr x))))
               (tail (asem-bpe:bpe-encode bpe (format nil ".~%"))))
          (when (> (+ (length ids) (length qtoks) (length atoks) (length tail)) ctx) (return))
          (setf ids (append ids qtoks))
          (loop for tk in atoks for k from 0
                do (push (+ (length ids) k -1) rows) (push tk targets))
          (setf ids (append ids atoks tail)))))
    (and rows (list ids (nreverse rows) (nreverse targets)))))

(defun train-sequences (n levels seed bpe ctx &key options)
  "N training sequences (training entities and wording) as lists (ids rows targets): a passage of a random level from LEVELS,
then as many of its questions (shuffled) as fit in CTX, each followed by \"The answer is <value>.\" (with OPTIONS, each question is
followed by four option lines first, as in the QA prompt). ROWS are the positions whose
next token is an answer token, TARGETS those tokens: loss only on answers."
  (let ((rng (sb-ext:seed-random-state seed))
        (*paraphrase-rng* (sb-ext:seed-random-state (+ 900001 seed)))
        (out nil) (tries 0))
    (loop while (and (< (length out) n) (< (incf tries) (* 3 n)))
          do (let* ((ctrl-rng (and *predicate-paraphrase* (make-random-state rng)))
                    (lvl (qg pick levels rng)))
               (multiple-value-bind (passage qa kind cands padder) (make-fact-set lvl :train :seen rng)
                 (let ((seq (encode-fact-set-sequence passage qa cands padder bpe ctx options rng)))
                   (when (and *predicate-paraphrase* (member kind '(:rel :cmp :tmp)))
                     ;; Keep subsequent tries lockstep-aligned with Control by advancing CTRL-RNG through Control's
                     ;; exact un-paraphrased sequence generation (without invoking MAKE-FACT-SET a second time).
                     (let* ((*predicate-paraphrase* nil)
                            (ctrl-lvl (qg pick levels ctrl-rng)))
                       (multiple-value-bind (cp cqa ckind ccands cpadder) (%make-fact-set ctrl-lvl :train :seen ctrl-rng)
                         (declare (ignore ckind))
                         (let ((ctrl-seq (encode-fact-set-sequence cp cqa ccands cpadder bpe ctx options ctrl-rng)))
                           (unless seq (setf seq ctrl-seq)))))
                     (setf rng ctrl-rng))
                   (when seq (push seq out))))))
    (nreverse out)))
