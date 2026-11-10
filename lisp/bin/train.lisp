;;; sbcl --script lisp/bin/train.lisp CORPUS [steps] [model-out] [workers]
;;; Trains on a text file (word tokenizer, ctx 32, dim 64, 2 layers) and prints samples.
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "tf" "train")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(in-package :asem-train)
(let* ((args (cdr sb-ext:*posix-argv*))
       (corpus (or (first args) "dataset/chat_clean.txt"))
       (steps (if (second args) (parse-integer (second args)) 1500))
       (out (or (third args) "model.lisp-model"))
       (workers (if (fourth args) (parse-integer (fourth args)) 4))
       (lm (train (asem-util:read-file corpus) :steps steps :workers workers :ctx 32 :dim 64 :heads 4 :layers 2
                  :ff 128 :batch 8 :lr 0.003d0 :eval-every 100 :val-examples 100)))
  (save-model lm out)
  (format t "~%saved ~a~%stats: ~s~%" out (lm-stats lm))
  (dolist (p '("User: How do I cook pasta?
Assistant:" "User: Hello!
Assistant:"))
    (format t "~%--- prompt ~s ---~%~a~%" p (generate lm p 60 :temperature 0.6d0 :seed 42))))
