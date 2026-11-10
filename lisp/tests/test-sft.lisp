;;;; Supervised fine-tuning (src/sft.lisp), in double precision on tiny models: label alignment, refusal to truncate,
;;;; the shuffled-label control, that a small model can fit a handful of examples, exact resume, and checkpoint selection on dev.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "tf" "train" "pretrain" "sft" "tf32" "train32" "pretrain32" "sft32"))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(defvar *tmp* (format nil "/tmp/asem-test-sft-~d/" (sb-posix:getpid)))
(defvar *items* (asem-qagen:generate-split :train 60 11))
(defvar *bpe* (asem-bpe:train-bpe (mapcar #'asem-qagen:item-prompt *items*) 1200))
(defvar *letters* (asem-sft:letter-token-ids *bpe*))
(defun last-eval-loss (dir)
  (with-open-file (in (format nil "~a/log.jsonl" dir))
    (let ((v nil)) (loop for line = (read-line in nil) while line
                         for o = (asem-json:parse-json line)
                         when (equal (asem-json:json-get o "event") "eval") do (setf v (asem-json:json-get o "dev_loss")))
      v)))
(defun tiny-model (seed)
  (asem-tf:init-model (asem-tf:make-config :vocab (asem-bpe:bpe-vocab-size *bpe*) :ctx 256 :dim 32 :heads 2 :layers 2 :ff 64) seed))

(deftest letters-are-single-tokens-and-targets-align
  (let ((exs (asem-sft:encode-items *items* *bpe* 256)))
    (check-equal (length (remove-duplicates *letters*)) 4)
    (check (every (lambda (e it) (and (= (asem-sft:ex-target e) (nth (getf it :answer) *letters*))
                                      (= (asem-sft:ex-answer e) (getf it :answer))
                                      (= 0 (aref (asem-sft:ex-ids e) 0))
                                      (equal (coerce (subseq (asem-sft:ex-ids e) 1) 'list) (asem-bpe:bpe-encode *bpe* (asem-qagen:item-prompt it)))))
                  exs *items*)
           "target = token of the key's letter; ids = end-of-text + prompt")))

(deftest refuses-to-truncate
  (let ((err nil))
    (handler-case (asem-sft:encode-items *items* *bpe* 20) (error () (setf err t)))
    (check err "a prompt longer than the context must be an error, not truncated")))

(deftest shuffled-labels-are-seeded-and-content-free
  (let* ((exs (asem-sft:encode-items (asem-qagen:generate-split :train 400 5) *bpe* 256))
         (a (asem-sft:shuffle-label-targets exs *letters* 7)) (b (asem-sft:shuffle-label-targets exs *letters* 7))
         (agree (count t (map 'list (lambda (x y) (= (asem-sft:ex-answer x) (asem-sft:ex-answer y))) exs a))))
    (check (every (lambda (x y) (= (asem-sft:ex-target x) (asem-sft:ex-target y))) a b) "same seed, same labels")
    (check (every (lambda (x y) (equalp (asem-sft:ex-ids x) (asem-sft:ex-ids y))) exs a) "prompts unchanged")
    (check (every (lambda (x) (= (asem-sft:ex-target x) (nth (asem-sft:ex-answer x) *letters*))) a) "target matches the new letter")
    (check (< 60 agree 140) (format nil "agreement with the real key ~d/400 should be near 100 (chance)" agree))))

(deftest a-tiny-model-fits-a-few-examples
  ;; memorising 8 examples is only possible if loss, label and gradient line up
  (let* ((exs (subseq (asem-sft:encode-items *items* *bpe* 256) 0 8)) (dir (format nil "~afit" *tmp*))
         (st (asem-sft:sft dir exs exs *letters* :init-model (tiny-model 3) :steps 150 :batch 8 :workers 1 :lr 3d-3 :warmup 10
                                                 :eval-every 50 :log-every 0 :train-probe 8)))
    (check (= (getf st :best-acc) 1l0) (format nil "training accuracy on 8 examples: ~a" (getf st :best-acc)))))

(deftest resume-is-exact-in-fp32
  ;; checkpoints and Adam moments are stored as float32, so an FP32 run resumes bit for bit (one worker)
  (let* ((exs (asem-sft32:encode-items *items* *bpe* 256)) (dev (subseq exs 0 10))
         (a (format nil "~aone" *tmp*)) (b (format nil "~atwo" *tmp*))
         (model (lambda () (asem-tf32:init-model (asem-tf32:make-config :vocab (asem-bpe:bpe-vocab-size *bpe*) :ctx 256 :dim 32 :heads 2 :layers 2 :ff 64) 5)))
         (args (list :steps 8 :batch 4 :workers 1 :lr 1f-3 :warmup 2 :eval-every 4 :log-every 0 :train-probe 4)))
    (apply #'asem-sft32:sft a exs dev *letters* :init-model (funcall model) args)
    (apply #'asem-sft32:sft b exs dev *letters* :init-model (funcall model) :stop-at 3 args)
    (check-equal (getf (asem-sft::read-state (format nil "~a/state.sexp" b)) :step) 3)
    (apply #'asem-sft32:sft b exs dev *letters* :resume t args)
    (check-equal (asem-sha256:sha256-file-hex (format nil "~a/last.ckpt" b)) (asem-sha256:sha256-file-hex (format nil "~a/last.ckpt" a)))
    (check-equal (asem-sha256:sha256-file-hex (format nil "~a/last-m1.ckpt" b)) (asem-sha256:sha256-file-hex (format nil "~a/last-m1.ckpt" a)))
    (check-equal (asem-sha256:sha256-file-hex (format nil "~a/last-m2.ckpt" b)) (asem-sha256:sha256-file-hex (format nil "~a/last-m2.ckpt" a)))))

(deftest fp32-and-fp64-sft-agree
  ;; the derived FP32 trainer follows the FP64 one: same data order, losses within single-precision tolerance after 6 steps
  (let* ((e64 (asem-sft:encode-items *items* *bpe* 256)) (e32 (asem-sft32:encode-items *items* *bpe* 256))
         (cfg64 (asem-tf:make-config :vocab (asem-bpe:bpe-vocab-size *bpe*) :ctx 256 :dim 32 :heads 2 :layers 2 :ff 64))
         (cfg32 (asem-tf32:make-config :vocab (asem-bpe:bpe-vocab-size *bpe*) :ctx 256 :dim 32 :heads 2 :layers 2 :ff 64))
         (s64 (asem-sft:sft (format nil "~ap64" *tmp*) e64 (subseq e64 0 12) *letters* :init-model (asem-tf:init-model cfg64 9)
                            :steps 6 :batch 6 :workers 1 :lr 1d-3 :warmup 2 :eval-every 6 :log-every 0 :train-probe 4))
         (s32 (asem-sft32:sft (format nil "~ap32" *tmp*) e32 (subseq e32 0 12) *letters* :init-model (let ((m (asem-tf32:init-model cfg32 1)) (src (asem-tf:init-model cfg64 9)))
                                                                                 (loop for p across (asem-tf32:model-params m) for q across (asem-tf:model-params src)
                                                                                       do (dotimes (i (length p)) (setf (aref p i) (coerce (aref q i) 'single-float))))
                                                                                 m)
                              :steps 6 :batch 6 :workers 1 :lr 1f-3 :warmup 2 :eval-every 6 :log-every 0 :train-probe 4)))
    (declare (ignorable s32 s64))
    (check-close (last-eval-loss (format nil "~ap32" *tmp*)) (last-eval-loss (format nil "~ap64" *tmp*)) 1d-3)))

(deftest best-checkpoint-is-chosen-on-dev-accuracy
  (let* ((exs (asem-sft:encode-items *items* *bpe* 256)) (dir (format nil "~asel" *tmp*))
         (st (asem-sft:sft dir (subseq exs 0 40) (subseq exs 40) *letters* :init-model (tiny-model 2) :steps 30 :batch 8 :workers 2
                           :lr 2d-3 :warmup 5 :eval-every 5 :log-every 0 :train-probe 8))
         (evals (with-open-file (in (format nil "~a/log.jsonl" dir))
                  (loop for line = (read-line in nil) while line
                        for o = (asem-json:parse-json line)
                        when (equal (asem-json:json-get o "event") "eval")
                          collect (cons (asem-json:json-get o "step") (asem-json:json-get o "dev_acc")))))
         (best (reduce #'max (mapcar #'cdr evals))))
    (check (< (abs (- best (getf st :best-acc))) 1d-6) "best accuracy recorded = maximum logged dev accuracy")
    (check (< (abs (- (cdr (assoc (getf st :best-step) evals)) best)) 1d-6) "best step has that accuracy")
    (multiple-value-bind (m h) (asem-pretrain:load-checkpoint (format nil "~a/best.ckpt" dir))
      (declare (ignore m))
      (check-equal (asem-json:json-get h "step") (getf st :best-step)))))

(deftest text-mode-targets-are-the-answer-tokens-and-fit
  (let* ((exs (asem-sft:encode-items *items* *bpe* 256 :mode :text)))
    (check (every (lambda (e it) (equal (coerce (asem-sft::ex-tgt e) 'list) (asem-bpe:bpe-encode *bpe* (format nil " ~a" (nth (getf it :answer) (getf it :options))))))
                  exs *items*) "target = tokens of the key's text")
    (let* ((few (subseq exs 0 8))
           (st (asem-sft:sft (format nil "~atext" *tmp*) few few *letters* :init-model (tiny-model 3) :steps 120 :batch 8 :workers 1 :lr 3d-3
                             :warmup 10 :eval-every 40 :log-every 0 :train-probe 8)))
      (check (= (getf st :best-acc) 1l0) (format nil "option-scoring accuracy on 8 memorised examples: ~a" (getf st :best-acc))))))

(deftest auxiliary-prompt-loss-trains-and-keeps-answer-metrics
  (let* ((exs (subseq (asem-sft:encode-items *items* *bpe* 256) 0 8))
         (st (asem-sft:sft (format nil "~aaux" *tmp*) exs exs *letters* :init-model (tiny-model 3) :steps 150 :batch 8 :workers 2 :lr 3d-3
                           :warmup 10 :eval-every 50 :log-every 0 :train-probe 8 :aux-lm 0.5d0))
         (cache (make-hash-table)) (e (aref exs 0)) (m (asem-pretrain:load-checkpoint (format nil "~aaux/last.ckpt" *tmp*))))
    (check (= (getf st :best-acc) 1l0) (format nil "letter accuracy with the auxiliary loss: ~a" (getf st :best-acc)))
    ;; the answer loss reported during training equals the evaluation loss on the same weights
    (multiple-value-bind (l1) (asem-sft::run-example m e cache *letters*)
      (multiple-value-bind (l2) (asem-sft::run-example m e cache *letters* :grads (asem-tf:make-grads (asem-tf:model-config m)) :aux 0.5d0)
        (check-close l2 l1 1d-9)))))

(deftest wilson-interval
  (destructuring-bind (lo hi) (asem-sft:wilson 900 1000)
    (check-close lo 0.8798d0 2d-4) (check-close hi 0.9171d0 2d-4))
  (destructuring-bind (lo hi) (asem-sft:wilson 0 10) (check (< (abs lo) 1d-12)) (check (< 0.27 hi 0.28))))

(unwind-protect (run-all) (sb-ext:delete-directory *tmp* :recursive t))
