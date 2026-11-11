;;;; Read-only Phase 2 inventory for overlap audits; never installs training overrides.
(defpackage :asem-phase2-frames (:use :cl) (:export #:all-predicate-paraphrase-frames))
(in-package :asem-phase2-frames)
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

