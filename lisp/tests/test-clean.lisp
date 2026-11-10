(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/clean.lisp" here))
  (load (merge-pathnames "harness.lisp" here))
  (defparameter cl-user::*root* (merge-pathnames "../../" here)))
(in-package :harness)
(use-package :asem-clean)
(use-package :asem-util)

(defun d (&rest lines) (join (string #\Newline) lines))
(defun lines (&rest ls) ls)
(defparameter *good* (lines "User: Why is the sky blue?" "Assistant: Sunlight scatters off the atmosphere."
                            "User: Is it cloudy today?" "Assistant: Check a weather app for updates."))

(deftest good-dialogue-passes (check (null (reject-reason *good*))))

(deftest malformed
  (check-equal (reject-reason (lines "User: Hi there friend?")) :malformed)
  (check-equal (reject-reason (lines "User: Q one?" "Assistant: Answer one here." "Assistant: And more words.")) :malformed)
  (check-equal (reject-reason (lines "Assistant: Hello there friend." "User: Hi?")) :malformed)
  (check-equal (reject-reason (lines "User: " "Assistant: Some words here.")) :malformed))

(deftest too-long-non-ascii-degenerate
  (check-equal (reject-reason (loop for i below 14 collect (format nil "~a: line number ~d about topic ~d" (if (evenp i) "User" "Assistant") i i))) :too-long)
  (check-equal (reject-reason (lines "User: What is café culture?" "Assistant: It is the coffee shop tradition.")) :non-ascii)
  (check-equal (reject-reason (lines "User: Tell me something." "Assistant: Tell me something.")) :degenerate)
  (check-equal (reject-reason (lines "User: Hello again?" "Assistant: Ok.")) :degenerate)
  (check-equal (reject-reason (lines "User: Q one?" "Assistant: Same reply text." "User: Q two?" "Assistant: Same reply text.")) :degenerate))

(deftest pronoun-opening
  (check-equal (reject-reason (lines "User: Is it healthy?" "Assistant: Yes, they contain vitamins.")) :no-context)
  (check-equal (reject-reason (lines "User: Do you recommend it?" "Assistant: Yes, a must-watch.")) :no-context)
  (check (null (reject-reason (lines "User: Is broccoli healthy for dogs?" "Assistant: Yes, in small amounts.")))))

(deftest duplicates-and-boilerplate
  (let ((shouted (d "User: WHY IS THE SKY BLUE?" "Assistant: Sunlight scatters off the atmosphere!"
                    "User: Is it cloudy today" "Assistant: Check a weather app for updates")))
    (multiple-value-bind (out rep) (clean (d (join (string #\Newline) *good*) "" shouted))
      (declare (ignore out))
      (check-equal (cdr (assoc "dialogues_out" rep :test #'string=)) 1)
      (check-equal (cdr (assoc "removed" rep :test #'string=)) '(("duplicate" . 1)))))
  (flet ((filler (i) (d (format nil "User: Topic number ~d question about stuff?" i)
                        (format nil "Assistant: Answer ~d with details." i)
                        "User: Please help me." "Assistant: I can help you.")))
    (multiple-value-bind (out rep) (clean (join (format nil "~%~%") (loop for i below 5 collect (filler i))))
      (declare (ignore out))
      (check-equal (cdr (assoc "dialogues_out" rep :test #'string=)) *max-exchange-repeats*)
      (check-equal (cdr (assoc "removed" rep :test #'string=)) '(("boilerplate" . 3))))))

(deftest real-dataset-matches-committed-output
  (let* ((text (read-file (merge-pathnames "dataset/chat.txt" cl-user::*root*)))
         (expected (read-file (merge-pathnames "dataset/chat_clean.txt" cl-user::*root*))))
    (multiple-value-bind (out rep) (clean text)
      (check (string= out expected) "cleaned text differs from dataset/chat_clean.txt (the committed cleaned data)")
      (check-equal (cdr (assoc "dialogues_out" rep :test #'string=)) 543)
      (multiple-value-bind (again rep2) (clean out)
        (check (string= again out) "cleaning clean data must change nothing")
        (check-equal (cdr (assoc "removed" rep2 :test #'string=)) nil)))))

(run-all)
