;;;; A small GPT-style transformer with hand-written backprop, in Common Lisp (SBCL).
;;;; Pre-RMSNorm, multi-head causal attention, ReLU MLP, residuals, learned positions.
;;;; All tensors are flat (simple-array double-float (*)), row-major. Mirrors tinyllm_tf.erl.
(defpackage :asem-tf
  (:use :cl)
  (:export #:make-config #:config-vocab #:config-ctx #:config-dim #:config-heads #:config-layers #:config-ff
           #:model #:model-config #:model-params #:init-model #:param-count #:make-acts #:forward #:backward
           #:make-grads #:zero-grads #:copy-model-params #:last-logits #:softmax-row #:dvec #:new-dvec
           #:param-names #:flops-per-token #:acts-probs #:gen-kernel #:tuned-config))
(in-package :asem-tf)

(deftype dvec () '(simple-array double-float (*)))
(deftype ivec () '(simple-array fixnum (*)))
(deftype idx () '(integer 0 #.(ash 1 40)))

(declaim (inline new-dvec))
(defun new-dvec (n) (make-array n :element-type 'double-float :initial-element 0d0))

(defstruct config vocab ctx dim heads layers ff)

(defstruct model config params)       ; params: simple-vector of dvec, see PARAM-NAMES

;;; ---------------------------------------------------------------- parameter layout
;;; 0 tok  1 pos  then per layer l: g1 wq wk wv wo g2 w1 w2   then gf, head
(defun layer-base (l) (+ 2 (* 8 l)))
(defun gf-index (cfg) (+ 2 (* 8 (config-layers cfg))))
(defun head-index (cfg) (+ 3 (* 8 (config-layers cfg))))

(defun param-shapes (cfg)
  (let ((v (config-vocab cfg)) (tt (config-ctx cfg)) (d (config-dim cfg)) (f (config-ff cfg)))
    (append (list (list "tok" (* v d)) (list "pos" (* tt d)))
            (loop for l below (config-layers cfg)
                  append (list (list (format nil "l~d.g1" l) d) (list (format nil "l~d.wq" l) (* d d))
                               (list (format nil "l~d.wk" l) (* d d)) (list (format nil "l~d.wv" l) (* d d))
                               (list (format nil "l~d.wo" l) (* d d)) (list (format nil "l~d.g2" l) d)
                               (list (format nil "l~d.w1" l) (* d f)) (list (format nil "l~d.w2" l) (* f d))))
            (list (list "gf" d) (list "head" (* d v))))))

(defun param-names (cfg) (mapcar #'first (param-shapes cfg)))

(defun param-count (m)
  (reduce #'+ (map 'list #'length (model-params m))))

(defun make-grads (cfg)
  (coerce (loop for (nil n) in (param-shapes cfg) collect (new-dvec n)) 'simple-vector))

(defun zero-grads (g) (loop for v across g do (fill (the dvec v) 0d0)))

(defun copy-model-params (m)
  (map 'simple-vector #'copy-seq (model-params m)))

(defun normal (state)
  (let ((u1 (- 1d0 (random 1d0 state))) (u2 (random 1d0 state)))
    (* (sqrt (* -2d0 (log u1))) (cos (* 2d0 pi u2)))))

(defun init-model (cfg &optional (seed 1))
  (let* ((state (sb-ext:seed-random-state seed))
         (d (config-dim cfg)) (f (config-ff cfg))
         (params
           (coerce
            (loop for (name n) in (param-shapes cfg)
                  collect (let ((v (new-dvec n))
                                (scale (cond ((member name '("tok" "pos") :test #'string=) 0.1d0)
                                             ((string= name "head") (/ 0.5d0 (sqrt d)))
                                             ((search ".w2" name) (/ 1d0 (sqrt f)))
                                             (t (/ 1d0 (sqrt d))))))
                            (if (or (search ".g" name) (string= name "gf"))
                                (fill v 1d0)
                                (dotimes (i n) (setf (aref v i) (* scale (normal state)))))
                            v))
            'simple-vector)))
    (make-model :config cfg :params params)))

(defun flops-per-token (cfg)
  "Training FLOPs per token: 6 per matmul parameter + attention (12 * layers * ctx * dim)."
  (let ((d (config-dim cfg)) (f (config-ff cfg)) (l (config-layers cfg)))
    (+ (* 6 (+ (* l (+ (* 4 d d) (* 2 d f))) (* d (config-vocab cfg))))
       (* 12 l (config-ctx cfg) d))))

;;; ---------------------------------------------------------------- kernels
(defmacro with-fast (&body body)
  `(locally (declare (optimize (speed 3) (safety 0) (debug 0))) ,@body))

;;; The three matmul kernels are GENERATED: the functions below build the source of each kernel as a list
;;; (code is data) for a given unroll factor, and DEF-KERNELS expands into the winners recorded in
;;; src/tuned.sexp by `bin/tune.lisp` (which compiles every variant at runtime, checks it against a plain
;;; reference, times it and keeps the fastest). Every variant adds in the same order as the plain loops, so
;;; results are bit-identical whatever the factor.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *kernel-defaults* '(:mm-nn 4 :mm-nt 4 :mm-tn 4))

  (defun tuned-config ()
    (let ((path (merge-pathnames "tuned.sexp" (or *load-truename* *compile-file-truename*))))
      (append (when (probe-file path)
                (with-open-file (in path) (let ((*read-eval* nil) (*package* (find-package :asem-tf))) (read in))))
              *kernel-defaults*)))

  (defun %syms (prefix n) (loop for q below n collect (intern (format nil "~a~d" prefix q) :asem-tf)))

  (defun gen-mm-nn (name u)
    "C(m x n) = [C +] A(m x k) B(k x n); U rows of B are folded into each pass over a row of C."
    (let ((as (%syms "A" u)) (bs (%syms "B" u)))
      `(defun ,name (a b c m k n acc)
         (declare (type dvec a b c) (type idx m k n))
         (with-fast
           (unless acc (fill c 0d0))
           (dotimes (i m)
             (let ((co (* i n)) (ao (* i k)) (p 0))
               (declare (type idx co ao p))
               (loop while (<= (+ p ,u) k)
                     do (let* (,@(loop for s in as for q from 0 collect `(,s (aref a (+ ao p ,q))))
                               (,(first bs) (* p n))
                               ,@(loop for (prev cur) on bs while cur collect `(,cur (+ ,prev n))))
                          (declare (type double-float ,@as) (type idx ,@bs))
                          (dotimes (j n)
                            (setf (aref c (+ co j))
                                  ,(let ((e '(aref c (+ co j))))
                                     (loop for s in as for bq in bs do (setf e `(+ ,e (* ,s (aref b (+ ,bq j))))))
                                     e))))
                        (incf p ,u))
               (loop while (< p k)
                     do (let ((aip (aref a (+ ao p))) (bo (* p n)))
                          (declare (type double-float aip) (type idx bo))
                          (dotimes (j n) (incf (aref c (+ co j)) (* aip (aref b (+ bo j))))))
                        (incf p))))))))

  (defun gen-mm-nt (name u)
    "C(m x n) = [C +] A B^T, B is (n x k); U output columns share each load of A."
    (let ((ss (%syms "S" u)) (bs (%syms "B" u)))
      `(defun ,name (a b c m k n acc)
         (declare (type dvec a b c) (type idx m k n))
         (with-fast
           (unless acc (fill c 0d0))
           (dotimes (i m)
             (let ((ao (* i k)) (co (* i n)) (j 0))
               (declare (type idx ao co j))
               (loop while (<= (+ j ,u) n)
                     do (let* (,@(loop for s in ss collect `(,s 0d0))
                               (,(first bs) (* j k))
                               ,@(loop for (prev cur) on bs while cur collect `(,cur (+ ,prev k))))
                          (declare (type double-float ,@ss) (type idx ,@bs))
                          (dotimes (p k)
                            (let ((ap (aref a (+ ao p))))
                              (declare (type double-float ap))
                              ,@(loop for s in ss for bq in bs collect `(incf ,s (* ap (aref b (+ ,bq p)))))))
                          ,@(loop for s in ss for q from 0 collect `(incf (aref c (+ co j ,q)) ,s)))
                        (incf j ,u))
               (loop while (< j n)
                     do (let ((s 0d0) (bo (* j k)))
                          (declare (type double-float s) (type idx bo))
                          (dotimes (p k) (incf s (* (aref a (+ ao p)) (aref b (+ bo p)))))
                          (incf (aref c (+ co j)) s))
                        (incf j))))))))

  (defun gen-mm-tn (name u)
    "C(k x n) = [C +] A^T B, A is (m x k), B is (m x n); U rows of A/B are consumed per pass."
    (let ((as (%syms "A" u)) (aos (%syms "AO" u)) (bos (%syms "BO" u)))
      `(defun ,name (a b c m k n acc)
         (declare (type dvec a b c) (type idx m k n))
         (with-fast
           (unless acc (fill c 0d0))
           (let ((i 0))
             (declare (type idx i))
             (loop while (<= (+ i ,u) m)
                   do (let* ((,(first aos) (* i k)) ,@(loop for (prev cur) on aos while cur collect `(,cur (+ ,prev k)))
                             (,(first bos) (* i n)) ,@(loop for (prev cur) on bos while cur collect `(,cur (+ ,prev n))))
                        (declare (type idx ,@aos ,@bos))
                        (dotimes (p k)
                          (let (,@(loop for s in as for ao in aos collect `(,s (aref a (+ ,ao p))))
                                (co (* p n)))
                            (declare (type double-float ,@as) (type idx co))
                            (dotimes (j n)
                              (setf (aref c (+ co j))
                                    ,(let ((e '(aref c (+ co j))))
                                       (loop for s in as for bo in bos do (setf e `(+ ,e (* ,s (aref b (+ ,bo j))))))
                                       e))))))
                      (incf i ,u))
             (loop while (< i m)
                   do (dotimes (p k)
                        (let ((aip (aref a (+ (* i k) p))) (co (* p n)) (bo (* i n)))
                          (declare (type double-float aip) (type idx co bo))
                          (dotimes (j n) (incf (aref c (+ co j)) (* aip (aref b (+ bo j)))))))
                      (incf i)))))))

  (defun gen-kernel (kind name u)
    (ecase kind (:mm-nn (gen-mm-nn name u)) (:mm-nt (gen-mm-nt name u)) (:mm-tn (gen-mm-tn name u)))))

(defmacro def-kernels ()
  (let ((cfg (tuned-config)))
    `(progn ,(gen-kernel :mm-nn 'mm-nn (getf cfg :mm-nn))
            ,(gen-kernel :mm-nt 'mm-nt (getf cfg :mm-nt))
            ,(gen-kernel :mm-tn 'mm-tn (getf cfg :mm-tn)))))

(def-kernels)

(defun rmsnorm-fwd (h g out r rows d)
  (declare (type dvec h g out r) (type fixnum rows d))
  (with-fast
    (dotimes (i rows)
      (let ((s 0d0) (o (* i d)))
        (declare (type double-float s) (type fixnum o))
        (dotimes (j d) (incf s (* (aref h (+ o j)) (aref h (+ o j)))))
        (let ((ri (/ 1d0 (sqrt (+ (/ s d) 1d-5)))))
          (declare (type double-float ri))
          (setf (aref r i) ri)
          (dotimes (j d) (setf (aref out (+ o j)) (* ri (aref g j) (aref h (+ o j))))))))))

(defun rmsnorm-bwd (h g r dn dh dg rows d acc)
  "dh = (or += when ACC) gradient wrt the norm input; dg += gradient wrt gain."
  (declare (type dvec h g r dn dh dg) (type fixnum rows d))
  (with-fast
    (dotimes (i rows)
      (let ((s 0d0) (o (* i d)) (ri (aref r i)))
        (declare (type double-float s ri) (type fixnum o))
        (dotimes (j d) (incf s (* (aref dn (+ o j)) (aref g j) (aref h (+ o j)))))
        (let ((c (/ (* ri ri ri s) d)))
          (declare (type double-float c))
          (dotimes (j d)
            (let ((v (- (* ri (aref g j) (aref dn (+ o j))) (* (aref h (+ o j)) c))))
              (if acc (incf (aref dh (+ o j)) v) (setf (aref dh (+ o j)) v)))
            (incf (aref dg j) (* (aref dn (+ o j)) (aref h (+ o j)) ri))))))))

;;; ---------------------------------------------------------------- activations
(defstruct (acts (:constructor %make-acts))
  tlen hs hmid r1 n1 q k v att a r2 n2 rr tmp rf nf logits probs dlog dh dhmid dn dq dk dv da dnorm drr)

(defun make-acts (cfg tlen)
  (let* ((d (config-dim cfg)) (f (config-ff cfg)) (l (config-layers cfg)) (h (config-heads cfg))
         (v (config-vocab cfg)))
    (flet ((td () (new-dvec (* tlen d))) (per-layer (fn) (coerce (loop repeat l collect (funcall fn)) 'simple-vector)))
      (%make-acts :tlen tlen
                 :hs (coerce (loop repeat (1+ l) collect (td)) 'simple-vector)
                 :hmid (per-layer #'td) :r1 (per-layer (lambda () (new-dvec tlen))) :n1 (per-layer #'td)
                 :q (per-layer #'td) :k (per-layer #'td) :v (per-layer #'td)
                 :att (per-layer (lambda () (new-dvec (* h tlen tlen)))) :a (per-layer #'td)
                 :r2 (per-layer (lambda () (new-dvec tlen))) :n2 (per-layer #'td)
                 :rr (per-layer (lambda () (new-dvec (* tlen f))))
                 :tmp (td) :rf (new-dvec tlen) :nf (td) :logits (new-dvec (* tlen v)) :probs (new-dvec (* tlen v))
                 :dlog (new-dvec (* tlen v)) :dh (td) :dhmid (td) :dn (td) :dq (td) :dk (td) :dv (td) :da (td)
                 :dnorm (td) :drr (new-dvec (* tlen f))))))

(defun softmax-row (src so dst do n temp)
  "dst[do..do+n) = softmax(src[so..so+n) / temp)"
  (declare (type dvec src dst) (type fixnum so do n) (type double-float temp))
  (with-fast
    (let ((mx most-negative-double-float) (s 0d0))
      (declare (type double-float mx s))
      (dotimes (j n) (setf mx (max mx (aref src (+ so j)))))
      (dotimes (j n)
        (let ((e (exp (/ (- (aref src (+ so j)) mx) temp))))
          (setf (aref dst (+ do j)) e) (incf s e)))
      (dotimes (j n) (setf (aref dst (+ do j)) (/ (aref dst (+ do j)) s))))))

;;; ---------------------------------------------------------------- forward
(defun forward (m x acts)
  "X: ivec of token ids (length = tlen of ACTS). Fills ACTS, returns nothing; call LOSS-OF-ACTS."
  (declare (type ivec x))
  (let* ((cfg (model-config m)) (ps (model-params m))
         (d (config-dim cfg)) (f (config-ff cfg)) (nh (config-heads cfg)) (v (config-vocab cfg))
         (hd (floor d nh)) (tl (acts-tlen acts)) (scale (/ 1d0 (sqrt (float hd 1d0))))
         (tok (svref ps 0)) (pos (svref ps 1)) (h0 (svref (acts-hs acts) 0)))
    (declare (type fixnum d f nh v hd tl) (type double-float scale) (type dvec tok pos h0))
    (dotimes (i tl)
      (dotimes (j d)
        (setf (aref h0 (+ (* i d) j)) (+ (aref tok (+ (* (aref x i) d) j)) (aref pos (+ (* i d) j))))))
    (dotimes (l (config-layers cfg))
      (let* ((b (layer-base l)) (hin (svref (acts-hs acts) l)) (hout (svref (acts-hs acts) (1+ l)))
             (g1 (svref ps b)) (wq (svref ps (+ b 1))) (wk (svref ps (+ b 2))) (wv (svref ps (+ b 3)))
             (wo (svref ps (+ b 4))) (g2 (svref ps (+ b 5))) (w1 (svref ps (+ b 6))) (w2 (svref ps (+ b 7)))
             (n1 (svref (acts-n1 acts) l)) (q (svref (acts-q acts) l)) (k (svref (acts-k acts) l))
             (vv (svref (acts-v acts) l)) (p (svref (acts-att acts) l)) (a (svref (acts-a acts) l))
             (hmid (svref (acts-hmid acts) l)) (n2 (svref (acts-n2 acts) l)) (rr (svref (acts-rr acts) l))
             (tmp (acts-tmp acts)))
        (declare (type dvec hin hout g1 wq wk wv wo g2 w1 w2 n1 q k vv p a hmid n2 rr tmp) (type fixnum b))
        (rmsnorm-fwd hin g1 n1 (svref (acts-r1 acts) l) tl d)
        (mm-nn n1 wq q tl d d nil) (mm-nn n1 wk k tl d d nil) (mm-nn n1 wv vv tl d d nil)
        (with-fast
          (fill a 0d0)
          (dotimes (h nh)
            (dotimes (i tl)
              (let ((po (* (+ (* h tl) i) tl)) (mx most-negative-double-float) (s 0d0))
                (declare (type fixnum po) (type double-float mx s))
                (dotimes (j (1+ i))
                  (let ((sc 0d0))
                    (declare (type double-float sc))
                    (dotimes (e hd) (incf sc (* (aref q (+ (* i d) (* h hd) e)) (aref k (+ (* j d) (* h hd) e)))))
                    (setf sc (* sc scale))
                    (setf (aref p (+ po j)) sc)
                    (setf mx (max mx sc))))
                (dotimes (j (1+ i))
                  (let ((ex (exp (- (aref p (+ po j)) mx)))) (setf (aref p (+ po j)) ex) (incf s ex)))
                (dotimes (j (1+ i))
                  (let ((pj (/ (aref p (+ po j)) s)))
                    (declare (type double-float pj))
                    (setf (aref p (+ po j)) pj)
                    (dotimes (e hd)
                      (incf (aref a (+ (* i d) (* h hd) e)) (* pj (aref vv (+ (* j d) (* h hd) e)))))))))))
        (mm-nn a wo tmp tl d d nil)
        (with-fast (dotimes (i (* tl d)) (setf (aref hmid i) (+ (aref hin i) (aref tmp i)))))
        (rmsnorm-fwd hmid g2 n2 (svref (acts-r2 acts) l) tl d)
        (mm-nn n2 w1 rr tl d f nil)
        (with-fast (dotimes (i (* tl f)) (when (< (aref rr i) 0d0) (setf (aref rr i) 0d0))))
        (mm-nn rr w2 tmp tl f d nil)
        (with-fast (dotimes (i (* tl d)) (setf (aref hout i) (+ (aref hmid i) (aref tmp i)))))))
    (let ((hl (svref (acts-hs acts) (config-layers cfg))))
      (rmsnorm-fwd hl (svref ps (gf-index cfg)) (acts-nf acts) (acts-rf acts) tl d)
      (mm-nn (acts-nf acts) (svref ps (head-index cfg)) (acts-logits acts) tl d v nil)
      (dotimes (i tl) (softmax-row (acts-logits acts) (* i v) (acts-probs acts) (* i v) v 1d0)))
    nil))

(defun loss-of-acts (acts y)
  (declare (type ivec y))
  (let ((v (floor (length (the dvec (acts-probs acts))) (acts-tlen acts))) (s 0d0))
    (dotimes (i (acts-tlen acts))
      (decf s (log (max 1d-300 (aref (the dvec (acts-probs acts)) (+ (* i v) (aref y i)))))))
    (/ s (acts-tlen acts))))

(export 'loss-of-acts)

(defun last-logits (m x)
  "Logits for the position after the sequence X (any length up to ctx)."
  (let* ((cfg (model-config m)) (acts (make-acts cfg (length x))) (v (config-vocab cfg)))
    (forward m x acts)
    (let ((out (new-dvec v)))
      (replace out (acts-logits acts) :start2 (* (1- (length x)) v))
      out)))

;;; ---------------------------------------------------------------- backward
(defun backward (m x y acts grads)
  "Accumulate d(mean loss)/d(params) into GRADS. Call FORWARD first. Returns the loss."
  (declare (type ivec x y))
  (let* ((cfg (model-config m)) (ps (model-params m)) (nl (config-layers cfg))
         (d (config-dim cfg)) (f (config-ff cfg)) (nh (config-heads cfg)) (v (config-vocab cfg))
         (hd (floor d nh)) (tl (acts-tlen acts)) (scale (/ 1d0 (sqrt (float hd 1d0))))
         (loss (loss-of-acts acts y))
         (dlog (acts-dlog acts)) (dh (acts-dh acts)) (dhmid (acts-dhmid acts)) (dn (acts-dn acts))
         (dq (acts-dq acts)) (dk (acts-dk acts)) (dv (acts-dv acts)) (da (acts-da acts)) (drr (acts-drr acts))
         (dsc (acts-dnorm acts)))
    (declare (type fixnum d f nh v hd tl nl) (type double-float scale)
             (type dvec dlog dh dhmid dn dq dk dv da drr dsc))
    ;; softmax cross-entropy
    (replace dlog (the dvec (acts-probs acts)))
    (dotimes (i tl) (decf (aref dlog (+ (* i v) (aref y i))) 1d0))
    (let ((inv (/ 1d0 tl))) (dotimes (i (* tl v)) (setf (aref dlog i) (* (aref dlog i) inv))))
    (mm-tn (acts-nf acts) dlog (svref grads (head-index cfg)) tl d v t)
    (mm-nt dlog (svref ps (head-index cfg)) dn tl v d nil)
    (rmsnorm-bwd (svref (acts-hs acts) nl) (svref ps (gf-index cfg)) (acts-rf acts) dn dh
                 (svref grads (gf-index cfg)) tl d nil)
    (loop for l from (1- nl) downto 0
          do (let* ((b (layer-base l)) (hin (svref (acts-hs acts) l))
                    (g1 (svref ps b)) (wq (svref ps (+ b 1))) (wk (svref ps (+ b 2))) (wv (svref ps (+ b 3)))
                    (wo (svref ps (+ b 4))) (g2 (svref ps (+ b 5))) (w1 (svref ps (+ b 6))) (w2 (svref ps (+ b 7)))
                    (n1 (svref (acts-n1 acts) l)) (q (svref (acts-q acts) l)) (k (svref (acts-k acts) l))
                    (vv (svref (acts-v acts) l)) (p (svref (acts-att acts) l)) (a (svref (acts-a acts) l))
                    (hmid (svref (acts-hmid acts) l)) (n2 (svref (acts-n2 acts) l)) (rr (svref (acts-rr acts) l)))
               (declare (type dvec hin g1 wq wk wv wo g2 w1 w2 n1 q k vv p a hmid n2 rr) (type fixnum b))
               ;; MLP: hout = hmid + relu(n2 W1) W2
               (replace dhmid dh)
               (mm-tn rr dh (svref grads (+ b 7)) tl f d t)
               (mm-nt dh w2 drr tl d f nil)
               (with-fast (dotimes (i (* tl f)) (unless (> (aref rr i) 0d0) (setf (aref drr i) 0d0))))
               (mm-tn n2 drr (svref grads (+ b 6)) tl d f t)
               (mm-nt drr w1 dn tl f d nil)
               (rmsnorm-bwd hmid g2 (svref (acts-r2 acts) l) dn dhmid (svref grads (+ b 5)) tl d t)
               ;; attention: hmid = hin + a Wo
               (mm-tn a dhmid (svref grads (+ b 4)) tl d d t)
               (mm-nt dhmid wo da tl d d nil)
               (fill dq 0d0) (fill dk 0d0) (fill dv 0d0)
               (with-fast
                 (dotimes (h nh)
                   (dotimes (i tl)
                     (let ((po (* (+ (* h tl) i) tl)) (sum 0d0))
                       (declare (type fixnum po) (type double-float sum))
                       ;; dp_j = da_i . v_j ; dv_j += p_ij da_i ; sum = sum_j p_ij dp_j
                       (dotimes (j (1+ i))
                         (let ((dp 0d0) (pij (aref p (+ po j))))
                           (declare (type double-float dp pij))
                           (dotimes (e hd)
                             (incf dp (* (aref da (+ (* i d) (* h hd) e)) (aref vv (+ (* j d) (* h hd) e))))
                             (incf (aref dv (+ (* j d) (* h hd) e)) (* pij (aref da (+ (* i d) (* h hd) e)))))
                           (incf sum (* pij dp))
                           (setf (aref dsc j) dp)))
                       (dotimes (j (1+ i))
                         (let ((ds (* scale (aref p (+ po j)) (- (aref dsc j) sum))))
                           (declare (type double-float ds))
                           (dotimes (e hd)
                             (incf (aref dq (+ (* i d) (* h hd) e)) (* ds (aref k (+ (* j d) (* h hd) e))))
                             (incf (aref dk (+ (* j d) (* h hd) e)) (* ds (aref q (+ (* i d) (* h hd) e))))))))))
               (mm-tn n1 dq (svref grads (+ b 1)) tl d d t)
               (mm-tn n1 dk (svref grads (+ b 2)) tl d d t)
               (mm-tn n1 dv (svref grads (+ b 3)) tl d d t)
               (mm-nt dq wq dn tl d d nil)
               (mm-nt dk wk dn tl d d t)
               (mm-nt dv wv dn tl d d t)
               ;; dh for the layer below = dhmid (residual) + path through norm1
               (rmsnorm-bwd hin g1 (svref (acts-r1 acts) l) dn dh (svref grads b) tl d nil)
               (with-fast (dotimes (i (* tl d)) (incf (aref dh i) (aref dhmid i)))))))
    ;; embeddings
    (let ((gt (svref grads 0)) (gp (svref grads 1)))
      (declare (type dvec gt gp))
      (dotimes (i tl)
        (dotimes (j d)
          (incf (aref gt (+ (* (aref x i) d) j)) (aref dh (+ (* i d) j)))
          (incf (aref gp (+ (* i d) j)) (aref dh (+ (* i d) j))))))
    loss))
