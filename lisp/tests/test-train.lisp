(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "tf" "train")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here))
  (defparameter cl-user::*root* (merge-pathnames "../../" here)))
(in-package :harness)
(use-package :asem-train)
(use-package :asem-tf)

(defun nl (s) (substitute #\Newline #\| s))

(deftest tokenizer-splits-like-the-erlang-regex
  (check-equal (tokenize "Hello, world!") '("Hello" "," " world" "!"))
  (check-equal (tokenize (nl "User: hi|Assistant: it's ok")) (list "User" ":" " hi" (string #\Newline) "Assistant" ":" " it's" " ok"))
  (check-equal (tokenize "a  b") '("a" " " " b"))               ; two spaces: the first is a token of its own
  (check-equal (tokenize " ") '(" "))
  (check-equal (tokenize "") nil))

(deftest tokenizer-roundtrips
  (dolist (s (list "Hello, world!" (nl "User: How do I cook pasta?|Assistant: Boil water.|") "  odd   spacing\t(tabs)  ")) 
    (check-equal (apply #'concatenate 'string (tokenize s)) s)))

(deftest vocab-order-and-limits
  (let ((v (build-vocab (tokenize "b a b c b a d") 3 1)))
    (check-equal v '("<unk>" " a" " b")))                         ; counts: " a" 2, " b" 2 (tie -> alphabetical), rest 1
  (check-equal (first (build-vocab (tokenize "x y z") 10 1)) "<unk>")
  (check-equal (length (build-vocab (tokenize "a a b b c c") 10 3)) 1))   ; min-count removes everything

(defparameter *corpus*
  (with-output-to-string (o)
    (dotimes (i 60)
      (format o "User: What color is the sky?~%Assistant: The sky is blue.~%User: What color is grass?~%Assistant: Grass is green.~%~%"))))

(deftest training-beats-the-unigram-baseline
  (let* ((lm (train *corpus* :steps 120 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 8 :lr 0.01d0
                            :eval-every 40 :val-examples 20 :verbose nil))
         (st (lm-stats lm)))
    (check (< (getf st :val-loss) (getf st :unigram)) (format nil "val ~f vs unigram ~f" (getf st :val-loss) (getf st :unigram)))
    (check (< (getf st :val-loss) 1.0d0) (format nil "val loss ~f" (getf st :val-loss)))))

(deftest training-is-deterministic-and-thread-count-independent
  (flet ((run (w) (lm-stats (train *corpus* :steps 8 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 6 :eval-every 8
                                             :val-examples 10 :workers w :verbose nil))))
    (let ((a (run 1)) (b (run 1)) (c (run 3)))
      (check-equal (getf a :val-loss) (getf b :val-loss))              ; same seed -> identical
      (check-close (getf a :val-loss) (getf c :val-loss) 1d-9))))      ; threads only reorder float sums

(deftest save-load-roundtrip
  (let* ((lm (train *corpus* :steps 10 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 4 :eval-every 10
                             :val-examples 10 :verbose nil))
         (path (format nil "/tmp/asem-lisp-test-~d.model" (get-universal-time))))
    (save-model lm path)
    (let ((lm2 (load-model path)))
      (check-equal (lm-vocab lm) (lm-vocab lm2))
      (check (every #'equalp (model-params (lm-model lm)) (model-params (lm-model lm2))) "parameters differ after reload")
      (check-equal (generate lm "User: What color" 20 :seed 3) (generate lm2 "User: What color" 20 :seed 3)))
    (delete-file path)))

(deftest generation-properties
  (let ((lm (train *corpus* :steps 150 :ctx 8 :dim 16 :heads 2 :layers 1 :ff 32 :batch 8 :lr 0.01d0 :eval-every 150
                            :val-examples 10 :verbose nil)))
    (let ((out (generate lm "User: What color is the sky?" 12 :temperature 0.3d0 :seed 1)))
      (check (string= "User: What color is the sky?" (subseq out 0 28)) "output starts with the prompt")
      (check (not (search "<unk>" out)) "never emits <unk>")
      (check (search "Assistant" out) "a trained model continues the pattern"))
    (check-equal (generate lm "User:" 15 :seed 9) (generate lm "User:" 15 :seed 9))))

(run-all)
