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
  (:export #:*augment* #:*broad-train* #:*vocab-words* #:vocab-words-of #:make-fact-set #:probe-items #:train-sequences #:retrieval-prompt #:+levels+))
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

;;; ------------------------------------------------------------------ fact sets
;;; A fact set is a passage plus a list of (question . answer) pairs, each answerable from exactly one passage sentence.
(defun frames-of (plist key wording) (getf (getf plist key) (if (eq wording :unseen) :dev :train)))

(defun augmented (sents qa people type-key protect rng &optional (ret people) ty)
  "Apply the training-only wording augmentation when *AUGMENT*; returns (values sents qa people ty)."
  (if (not *augment*) (values sents qa ret ty)
      (values (mapcar (lambda (s) (augment-sentence s protect rng)) sents)
              (loop for x in qa for i from 0
                    collect (cons (augment-question (car x) (and people (nth i people)) type-key protect rng) (cdr x)))
              ret ty)))

(defun attr-facts (level entities wording rng)
  (let* ((ty (qg pick (symbol-value (q '*attr-types*)) rng)) (spec (cdr ty))
         (n (ecase level (1 1) (2 2) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (qg sample-n (pool entities (intern (symbol-name (getf spec :pool)) :asem-retrieval)) n rng))
         (frames (frames-of spec :frames wording)) (qframes (frames-of spec :q-extract wording))
         (sents (loop for p in people for v in vals collect (qg fill-frame (qg pick frames rng) :p p :v v)))
         (qa (loop for p in people for v in vals collect (cons (qg fill-frame (qg pick qframes rng) :p p) v))))
    (augmented sents qa people (car ty) (append people vals) rng people ty)))

(defun num-facts (level entities wording rng)
  (let* ((ty (qg pick (symbol-value (q '*num-types*)) rng)) (spec (cdr ty))
         (n (ecase level (1 1) (2 2) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (destructuring-bind (lo hi) (getf spec :range) (qg distinct-numbers lo hi n rng)))
         (frames (frames-of spec :frames wording)) (qframes (frames-of spec :q-extract wording))
         (sents (loop for p in people for v in vals collect (qg fill-frame (qg pick frames rng) :p p :n v :v v)))
         (qa (loop for p in people for v in vals collect (cons (qg fill-frame (qg pick qframes rng) :p p) (princ-to-string v)))))
    (augmented sents qa people (car ty) (append people (mapcar (function princ-to-string) vals)) rng people ty)))

(defun rel-facts (level entities wording rng)
  "Level 1: one pair; 2: two pairs; 3-4: a chain of three links (four people). Questions by subject or by object, each with a unique answer."
  (let* ((rel (qg pick (symbol-value (q '*relations*)) rng)) (spec (cdr rel))
         (people (qg sample-n (pool entities '*names*) (ecase level (1 2) (2 4) ((3 4) 4)) rng))
         (pairs (ecase level
                  (1 (list (list (first people) (second people))))
                  (2 (list (list (first people) (second people)) (list (third people) (fourth people))))
                  ((3 4) (list (list (first people) (second people)) (list (second people) (third people)) (list (third people) (fourth people))))))
         (frames (frames-of spec :frames wording)) (qs (frames-of spec :q-subj wording)) (qo (frames-of spec :q-obj wording))
         (sents (mapcar (lambda (pr) (qg fill-frame (qg pick frames rng) :a (first pr) :b (second pr))) pairs))
         (qa (mapcar (lambda (pr) (if (zerop (random 2 rng))
                                      (cons (qg fill-frame (qg pick qs rng) :b (second pr)) (first pr))
                                      (cons (qg fill-frame (qg pick qo rng) :a (first pr)) (second pr))))
                     pairs)))
    (augmented sents qa nil nil people rng people rel)))

(defun distractors (people entities wording exclude-ty rng k)
  "K sentences giving OTHER attributes of the same people (never asked about)."
  (loop repeat k
        collect (let* ((ty (qg pick (remove exclude-ty (symbol-value (q '*attr-types*))) rng)) (spec (cdr ty)))
                  (qg fill-frame (qg pick (frames-of spec :frames wording) rng) :p (qg pick people rng)
                      :v (qg pick (pool entities (intern (symbol-name (getf spec :pool)) :asem-retrieval)) rng)))))

(defun make-fact-set (level entities wording rng)
  "(values passage qa-pairs kind) for one passage of LEVEL."
  (let ((kind (qg pick '(:attr :attr :num :rel) rng)))
    (multiple-value-bind (sents qa people ty)
        (ecase kind (:attr (attr-facts level entities wording rng)) (:num (num-facts level entities wording rng)) (:rel (rel-facts level entities wording rng)))
      (let ((all (append sents (when (= level 4) (distractors people entities wording (and (eq kind :attr) ty) rng (+ 2 (random 3 rng)))))))
        (values (format nil "~{~a~^ ~}" (qg shuffle all rng)) qa kind)))))

(defun retrieval-prompt (passage question &key (passage-p t) (question-p t))
  (format nil "~@[Text: ~a~%~]~@[Q: ~a~%~]The answer is" (and passage-p passage) (and question-p question)))

(defun answer-occurs-once-p (passage qa)
  "The answer must be present and must not be a substring of another answer in the set (so exact match is unambiguous)."
  (and (search (cdr qa) passage) t))

;;; ------------------------------------------------------------------ probes
(defun probe-items (level entities wording n seed &key (fits (constantly t)))
  "N single-question items (plists :id :level :kind :passage :question :answer :cf-question :cf-answer). The counterfactual
question asks about another fact of the same passage (levels >= 2) and must change the answer. FITS is called with
(prompt answer) for the real and counterfactual question; items it rejects (too long for the context) are redrawn, never truncated."
  (let ((rng (sb-ext:seed-random-state seed)) (out nil))
    (loop while (< (length out) n)
          do (multiple-value-bind (passage qa kind) (make-fact-set level entities wording rng)
               (let* ((i (random (length qa) rng)) (main (nth i qa))
                      (others (remove-if (lambda (x) (string= (cdr x) (cdr main))) (remove main qa)))
                      (cf (and others (qg pick others rng))))
                 (when (and (every (lambda (x) (answer-occurs-once-p passage x)) qa)
                            (funcall fits (retrieval-prompt passage (car main)) (cdr main))
                            (or (null cf) (funcall fits (retrieval-prompt passage (car cf)) (cdr cf))))
                   (push (list :id (format nil "L~d-~(~a-~a~)-~d" level entities wording (length out)) :level level :kind (string-downcase (symbol-name kind))
                               :passage passage :question (car main) :answer (cdr main)
                               :cf-question (and cf (car cf)) :cf-answer (and cf (cdr cf)))
                         out)))))
    (nreverse out)))

;;; ------------------------------------------------------------------ dense training sequences
(defun train-sequences (n levels seed bpe ctx)
  "N training sequences (training entities and wording) as lists (ids rows targets): a passage of a random level from LEVELS,
then as many of its questions (shuffled) as fit in CTX, each followed by \"The answer is <value>.\". ROWS are the positions whose
next token is an answer token, TARGETS those tokens: loss only on answers."
  (let ((rng (sb-ext:seed-random-state seed)) (out nil) (tries 0))
    (loop while (and (< (length out) n) (< (incf tries) (* 3 n)))
          do (multiple-value-bind (passage qa) (make-fact-set (qg pick levels rng) :train :seen rng)
               (let* ((ids (list 0)) (rows nil) (targets nil))
                 (flet ((add (text) (setf ids (append ids (asem-bpe:bpe-encode bpe text)))))
                   (add (format nil "Text: ~a~%" passage))
                   (dolist (x (qg shuffle qa rng))
                     (let* ((qtoks (asem-bpe:bpe-encode bpe (format nil "Q: ~a~%The answer is" (car x))))
                            (atoks (asem-bpe:bpe-encode bpe (format nil " ~a" (cdr x))))
                            (tail (asem-bpe:bpe-encode bpe (format nil ".~%"))))
                       (when (> (+ (length ids) (length qtoks) (length atoks) (length tail)) ctx) (return))
                       (setf ids (append ids qtoks))
                       (loop for tk in atoks for k from 0
                             do (push (+ (length ids) k -1) rows) (push tk targets))
                       (setf ids (append ids atoks tail)))))
                 (when rows (push (list ids (nreverse rows) (nreverse targets)) out)))))
    (nreverse out)))
