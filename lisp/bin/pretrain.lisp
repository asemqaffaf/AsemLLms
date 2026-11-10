;;; Pretrain an AsemLLM from scratch on BPE shards built by bin/build-data.lisp.
;;;   sbcl --dynamic-space-size 12GB --script lisp/bin/pretrain.lisp data=DIR out=DIR [size=1m|3m|10m|25m] [steps=1000]
;;;        [batch=16] [ctx=128] [workers=4] [lr=0.003] [warmup=100] [clip=1.0] [seed=1] [eval-every=250] [val-windows=64]
;;;        [eval=lisp/eval/asem_eval.jsonl] [precision=32|64]
;;; With TLM_BLAS=1 the matrix products and attention use Accelerate. Writes OUT/log.jsonl, OUT/best.ckpt, OUT/last.ckpt.
(let ((here (directory-namestring *load-truename*)))
  (require :sb-posix)
  (dolist (f '("util" "sha256" "json" "bpe" "data")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (defparameter cl-user::*here* here))

(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))

(let* ((p32 (not (equal (opt-arg "precision" "32") "64"))) (here cl-user::*here*))
  (dolist (f (if p32 '("tf32" "train32" "pretrain32") '("tf" "train" "pretrain")))
    (load (merge-pathnames (format nil "../src/~a.lisp" f) here)))
  (let* ((pkg (if p32 "ASEM-PRETRAIN32" "ASEM-PRETRAIN"))
         (fn (symbol-function (find-symbol "PRETRAIN" pkg)))
         (data (or (opt-arg "data" nil) (error "data=DIR is required"))) (out (or (opt-arg "out" nil) (error "out=DIR is required")))
         (num (lambda (name default) (parse-integer (opt-arg name default))))
         (flt (lambda (name default) (float (read-from-string (opt-arg name default)) (if p32 1f0 1d0))))
         (eval-file (opt-arg "eval" (namestring (merge-pathnames "../eval/asem_eval.jsonl" here)))))
    (funcall fn data out :size (intern (string-upcase (opt-arg "size" "1m")) :keyword) :ctx (funcall num "ctx" "128")
             :steps (funcall num "steps" "1000") :batch (funcall num "batch" "16") :workers (funcall num "workers" "4")
             :lr (funcall flt "lr" "0.003") :warmup (funcall num "warmup" "100") :clip (funcall flt "clip" "1.0")
             :seed (funcall num "seed" "1") :eval-every (funcall num "eval-every" "250") :val-windows (funcall num "val-windows" "64")
             :eval-file (and (probe-file eval-file) eval-file))))
