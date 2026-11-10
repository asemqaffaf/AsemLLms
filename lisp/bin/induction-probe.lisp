;;; Does a checkpoint copy from its context? Mean next-token loss on the first and on the second copy of repeated random
;;; token sequences (induction heads make the second copy nearly free). Forward passes only, deterministic.
;;;   sbcl --dynamic-space-size 4GB --script lisp/bin/induction-probe.lisp ckpt=PATH [len=40] [n=50] [precision=32|64]
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*here* here))
(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(let* ((p32 (not (equal (opt-arg "precision" "32") "64"))) (here cl-user::*here*))
  (dolist (f (if p32 '("tf32" "train32" "pretrain32") '("tf" "train" "pretrain"))) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (let* ((tf (if p32 "ASEM-TF32" "ASEM-TF")) (f (lambda (s) (symbol-function (find-symbol s tf))))
         (model (funcall (symbol-function (find-symbol "LOAD-CHECKPOINT" (if p32 "ASEM-PRETRAIN32" "ASEM-PRETRAIN"))) (opt-arg "ckpt" nil)))
         (cfg (funcall (funcall f "MODEL-CONFIG") model)) (v (funcall (funcall f "CONFIG-VOCAB") cfg))
         (len (parse-integer (opt-arg "len" "40"))) (n (parse-integer (opt-arg "n" "50"))) (rng (sb-ext:seed-random-state 17))
         (first-sum 0d0) (second-sum 0d0))
    (dotimes (k n)
      (let* ((seq (loop repeat len collect (+ 300 (random 7000 rng)))) (ids (append (list 0) seq seq))
             (x (coerce (butlast ids) '(simple-array fixnum (*)))) (y (rest ids))
             (acts (funcall (funcall f "MAKE-ACTS") cfg (length x))))
        (funcall (funcall f "FORWARD") model x acts)
        (loop for tg in y for i from 0
              do (let ((lp (- (log (max 1d-30 (float (aref (funcall (funcall f "ACTS-PROBS") acts) (+ (* i v) tg)) 1d0))))))
                   (cond ((< i len) (incf first-sum lp)) ((> i len) (incf second-sum lp)))))))
    (format t "{\"checkpoint\":~s,\"sequences\":~d,\"length\":~d,\"loss_first_copy\":~,4f,\"loss_second_copy\":~,4f,\"uniform\":~,4f}~%"
            (opt-arg "ckpt" nil) n len (/ first-sum (* n len)) (/ second-sum (* n (1- len))) (log (float v 1d0)))))
