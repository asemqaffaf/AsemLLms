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
  (:export #:make-fact-set #:probe-items #:train-sequences #:retrieval-prompt #:+levels+))
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

(defun pool (entities sym)
  "Values for SYM ('*names*, '*towns*, '*jobs* ...) under ENTITIES: :train (generated names/towns, training pools), :dev-real
(QA dev pools), :new (new names/towns, QA dev pools for the attribute words)."
  (let ((s (q sym)))
    (ecase entities
      (:train (let ((asem-qagen::*synthetic-train* t)) (qg entity-pool s :train)))
      (:dev-real (qg split-pool (symbol-value s) :dev))
      (:new (cond ((string= (symbol-name sym) "*NAMES*") (new-pool :names))
                  ((string= (symbol-name sym) "*TOWNS*") (new-pool :towns))
                  (t (qg split-pool (symbol-value s) :dev)))))))

;;; ------------------------------------------------------------------ fact sets
;;; A fact set is a passage plus a list of (question . answer) pairs, each answerable from exactly one passage sentence.
(defun frames-of (plist key wording) (getf (getf plist key) (if (eq wording :unseen) :dev :train)))

(defun attr-facts (level entities wording rng)
  (let* ((ty (qg pick (symbol-value (q '*attr-types*)) rng)) (spec (cdr ty))
         (n (ecase level (1 1) (2 2) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (qg sample-n (pool entities (intern (symbol-name (getf spec :pool)) :asem-retrieval)) n rng))
         (frames (frames-of spec :frames wording)) (qframes (frames-of spec :q-extract wording))
         (sents (loop for p in people for v in vals collect (qg fill-frame (qg pick frames rng) :p p :v v)))
         (qa (loop for p in people for v in vals collect (cons (qg fill-frame (qg pick qframes rng) :p p) v))))
    (values sents qa people ty)))

(defun num-facts (level entities wording rng)
  (let* ((ty (qg pick (symbol-value (q '*num-types*)) rng)) (spec (cdr ty))
         (n (ecase level (1 1) (2 2) ((3 4) 4)))
         (people (qg sample-n (pool entities '*names*) n rng))
         (vals (destructuring-bind (lo hi) (getf spec :range) (qg distinct-numbers lo hi n rng)))
         (frames (frames-of spec :frames wording)) (qframes (frames-of spec :q-extract wording))
         (sents (loop for p in people for v in vals collect (qg fill-frame (qg pick frames rng) :p p :n v :v v)))
         (qa (loop for p in people for v in vals collect (cons (qg fill-frame (qg pick qframes rng) :p p) (princ-to-string v)))))
    (values sents qa people ty)))

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
    (values sents qa people rel)))

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
(defun probe-items (level entities wording n seed)
  "N single-question items (plists :id :level :kind :passage :question :answer :cf-question :cf-answer). The counterfactual
question asks about another fact of the same passage (levels >= 2) and must change the answer."
  (let ((rng (sb-ext:seed-random-state seed)) (out nil))
    (loop while (< (length out) n)
          do (multiple-value-bind (passage qa kind) (make-fact-set level entities wording rng)
               (let* ((i (random (length qa) rng)) (main (nth i qa))
                      (others (remove-if (lambda (x) (string= (cdr x) (cdr main))) (remove main qa)))
                      (cf (and others (qg pick others rng))))
                 (when (every (lambda (x) (answer-occurs-once-p passage x)) qa)
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
