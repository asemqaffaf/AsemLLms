;;;; A small GPT-style transformer with hand-written backprop, in Common Lisp (SBCL).
;;;; Pre-RMSNorm, multi-head causal attention, ReLU MLP, residuals, learned positions.
;;;; All tensors are flat (simple-array double-float (*)), row-major. Mirrors tinyllm_tf.erl.
(defpackage :asem-tf
  (:use :cl)
  (:export #:make-config #:config-vocab #:config-ctx #:config-dim #:config-heads #:config-layers #:config-ff
           #:model #:model-config #:model-params #:init-model #:param-count #:make-acts #:forward #:backward
           #:make-grads #:zero-grads #:copy-model-params #:last-logits #:softmax-row #:dvec #:new-dvec
           #:param-names #:flops-per-token #:acts-probs #:gen-kernel #:tuned-config
           #:native-init #:native-active-p #:native-min-work
           #:blas-init #:blas-active-p #:blas-min-work #:host-float #:elt-bytes))
(in-package :asem-tf)

(deftype dvec () '(simple-array double-float (*)))
(deftype ivec () '(simple-array fixnum (*)))
(deftype idx () '(integer 0 #.(ash 1 40)))

;;; Host-side scalars (accumulators, optimizer bias terms) are written with LONG-FLOAT / 0l0 literals, which SBCL
;;; treats as double-float and src/derive.lisp leaves alone when it derives the single-float version of this file.
;;; HOST-FLOAT brings such a value back to the element type; ELT-BYTES is the element size for pointer arithmetic.
(defmacro host-float (x) x)
(defmacro elt-bytes () 8)

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
    `(progn ,(gen-kernel :mm-nn 'mm-nn-lisp (getf cfg :mm-nn))
            ,(gen-kernel :mm-nt 'mm-nt-lisp (getf cfg :mm-nt))
            ,(gen-kernel :mm-tn 'mm-tn-lisp (getf cfg :mm-tn)))))

(def-kernels)

;;; ---------------------------------------------------------------- optional C kernels (sb-alien)
;;; native/libtlm.so (built with `make -C native`) holds the same three products in C. When it can be loaded,
;;; MM-NN / MM-NT / MM-TN call it for products with m*k*n >= *native-min-work*, and run the Lisp kernels above
;;; otherwise (and whenever the library is missing). No Quicklisp, no CFFI: sb-alien is part of SBCL.
;;; The C code adds in the same order as the Lisp kernels, so the results are bit-identical; the tests check
;;; that. (If libtlm was built with BLAS, big products are faster but not bit-identical.)
;;; The vectors are pinned only for the duration of a call, so the GC can still move them in between.
(sb-ext:defglobal **native-nn** 0)
(sb-ext:defglobal **native-nt** 0)
(sb-ext:defglobal **native-tn** 0)
(sb-ext:defglobal **native-min-work** 4096)
(declaim (type (unsigned-byte 62) **native-nn** **native-nt** **native-tn**)
         (type (unsigned-byte 40) **native-min-work**))

(defun native-min-work () **native-min-work**)
(defun (setf native-min-work) (n) (setf **native-min-work** n))
(defun native-active-p () (/= **native-nn** 0))

(defun native-library-path ()
  (or (sb-ext:posix-getenv "TLM_LIB")
      (namestring (merge-pathnames "../../native/build/libtlm.so"
                                   (or *load-truename* *compile-file-truename* *default-pathname-defaults*)))))

(defun native-init (&optional path)
  "Load libtlm and switch the matrix kernels over to it for big products. Returns T on success, NIL (and keeps
using the Lisp kernels) if the library cannot be loaded."
  (handler-case
      (progn
        (sb-alien:load-shared-object (or path (native-library-path)) :dont-save t)
        (let ((nn (sb-sys:find-foreign-symbol-address "tlm_mm_nn"))
              (nt (sb-sys:find-foreign-symbol-address "tlm_mm_nt"))
              (tn (sb-sys:find-foreign-symbol-address "tlm_mm_tn")))
          (when (and nn nt tn)
            (let ((w (sb-ext:posix-getenv "TLM_MIN_WORK")))
              (when w (setf **native-min-work** (or (parse-integer w :junk-allowed t) **native-min-work**))))
            (setf **native-nn** nn **native-nt** nt **native-tn** tn)
            t)))
    (error () nil)))

(defmacro %native-mm (addr a b c m k n acc)
  `(sb-sys:with-pinned-objects (,a ,b ,c)
     (sb-alien:alien-funcall
      (sb-alien:sap-alien (sb-sys:int-sap ,addr)
                          (function sb-alien:void sb-sys:system-area-pointer sb-sys:system-area-pointer
                                    sb-sys:system-area-pointer sb-alien:long sb-alien:long sb-alien:long sb-alien:int))
      (sb-sys:vector-sap ,a) (sb-sys:vector-sap ,b) (sb-sys:vector-sap ,c) ,m ,k ,n (if ,acc 1 0))))

;;; ---------------------------------------------------------------- optional BLAS (Apple Accelerate)
;;; BLAS-INIT loads a CBLAS library (Accelerate on macOS, or the file named by TLM_BLAS_LIB) and sends products with
;;; m*k*n >= *blas-min-work* to cblas_dgemm. It takes precedence over the C kernels above. BLAS adds in a different
;;; order, so the results are close (about 1e-12 relative), not bit-identical; it is therefore off unless asked for:
;;; call (blas-init) or start with TLM_BLAS=1. Same style as the C kernels: sb-alien only, vectors pinned per call.
(sb-ext:defglobal **blas-gemm** 0)
(sb-ext:defglobal **blas-min-work** 16384)
(declaim (type (unsigned-byte 62) **blas-gemm**) (type (unsigned-byte 40) **blas-min-work**))

(defun blas-min-work () **blas-min-work**)
(defun (setf blas-min-work) (n) (setf **blas-min-work** n))
(defun blas-active-p () (/= **blas-gemm** 0))

(defun blas-init (&optional path)
  "Switch big products to cblas_dgemm. Returns T on success, NIL (and keeps the other kernels) if no CBLAS can be loaded."
  (handler-case
      (progn
        (sb-alien:load-shared-object
         (or path (sb-ext:posix-getenv "TLM_BLAS_LIB") "/System/Library/Frameworks/Accelerate.framework/Accelerate")
         :dont-save t)
        (let ((f (sb-sys:find-foreign-symbol-address "cblas_dgemm"))
              (w (sb-ext:posix-getenv "TLM_BLAS_MIN_WORK")))
          (when f
            (when w (setf **blas-min-work** (or (parse-integer w :junk-allowed t) **blas-min-work**)))
            (setf **blas-gemm** f)
            t)))
    (error () nil)))

(defmacro %blas-mm (kind a b c m k n acc)
  "Row-major C (+)= op(A) op(B). CBLAS order 101 = row major, 111 = no transpose, 112 = transpose."
  (multiple-value-bind (ta tb bm bn bk lda ldb)
      (ecase kind                                    ; A is m x k throughout; B is k x n (nn), n x k (nt), m x n (tn)
        (:nn (values 111 111 m n k k n))
        (:nt (values 111 112 m n k k k))
        (:tn (values 112 111 k n m k n)))
    `(sb-sys:with-pinned-objects (,a ,b ,c)
       (sb-alien:alien-funcall
        (sb-alien:sap-alien (sb-sys:int-sap **blas-gemm**)
                            (function sb-alien:void sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:int
                                      sb-alien:double sb-sys:system-area-pointer sb-alien:int sb-sys:system-area-pointer sb-alien:int
                                      sb-alien:double sb-sys:system-area-pointer sb-alien:int))
        101 ,ta ,tb ,bm ,bn ,bk 1d0 (sb-sys:vector-sap ,a) ,lda (sb-sys:vector-sap ,b) ,ldb
        (if ,acc 1d0 0d0) (sb-sys:vector-sap ,c) ,n))))

(defmacro def-mm-dispatch (name kind lisp-name global)
  `(progn
     (declaim (inline ,name))
     (defun ,name (a b c m k n acc)
       (declare (type dvec a b c) (type idx m k n))
       (cond ((and (/= **blas-gemm** 0) (>= (* m k n) **blas-min-work**))
              (%blas-mm ,kind a b c m k n acc))
             ((and (/= ,global 0) (>= (* m k n) **native-min-work**))
              (%native-mm ,global a b c m k n acc))
             (t (,lisp-name a b c m k n acc))))))

(def-mm-dispatch mm-nn :nn mm-nn-lisp **native-nn**)
(def-mm-dispatch mm-nt :nt mm-nt-lisp **native-nt**)
(def-mm-dispatch mm-tn :tn mm-tn-lisp **native-tn**)

(native-init)
(when (equal (sb-ext:posix-getenv "TLM_BLAS") "1") (blas-init))

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
  tlen hs hmid r1 n1 q k v att a r2 n2 rr tmp rf nf logits probs dlog dh dhmid dn dq dk dv da dnorm drr dp)

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
                 :dnorm (td) :drr (new-dvec (* tlen f)) :dp (new-dvec (* tlen tlen))))))

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

;;; ---------------------------------------------------------------- attention kernels
;;; Per head, Q_h, K_h, V_h are tl x hd blocks of the tl x d matrices (row stride d, column offset h*hd) and the
;;; probabilities P_h are a tl x tl block of P (row stride tl). The Lisp versions are the reference and the fallback.
;;; The BLAS versions compute the same numbers with strided cblas_dgemm calls and no transposed copies:
;;;   S = scale Q K^T, masked softmax over j <= i, A = P V        dV = P^T dA, dP = dA V^T,
;;;   dS = scale P (dP - sum_j P dP), dQ = dS K, dK = dS^T Q
;;; They compute whole tl x tl products and write zeros above the diagonal, so the causal mask stays exact.
(defun blas-gemm-strided (kind m n k alpha a aoff lda b boff ldb beta c coff ldc)
  "C[coff..] = alpha op(A) op(B) + beta C, row-major, explicit leading dimensions and element offsets.
KIND :nn, :nt (B transposed) or :tn (A transposed). C is M x N, the inner dimension is K."
  (declare (type dvec a b c) (type fixnum m n k aoff lda boff ldb coff ldc) (type double-float alpha beta))
  (let ((ta (if (eq kind :tn) 112 111)) (tb (if (eq kind :nt) 112 111)))
    (sb-sys:with-pinned-objects (a b c)
      (sb-alien:alien-funcall
       (sb-alien:sap-alien (sb-sys:int-sap **blas-gemm**)
                           (function sb-alien:void sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:int
                                     sb-alien:double sb-sys:system-area-pointer sb-alien:int sb-sys:system-area-pointer sb-alien:int
                                     sb-alien:double sb-sys:system-area-pointer sb-alien:int))
       101 ta tb m n k alpha (sb-sys:sap+ (sb-sys:vector-sap a) (* (elt-bytes) aoff)) lda
       (sb-sys:sap+ (sb-sys:vector-sap b) (* (elt-bytes) boff)) ldb beta (sb-sys:sap+ (sb-sys:vector-sap c) (* (elt-bytes) coff)) ldc))))

(declaim (inline attn-blas-p))
(defun attn-blas-p (tl hd)
  (declare (type fixnum tl hd))
  (and (/= **blas-gemm** 0) (>= (* tl tl hd) **blas-min-work**)))

(defun attn-fwd-lisp (q k vv p a tl d nh hd scale)
  (declare (type dvec q k vv p a) (type fixnum tl d nh hd) (type double-float scale))
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
                      (incf (aref a (+ (* i d) (* h hd) e)) (* pj (aref vv (+ (* j d) (* h hd) e))))))))))))

(defun attn-fwd-blas (q k vv p a tl d nh hd scale)
  (declare (type dvec q k vv p a) (type fixnum tl d nh hd) (type double-float scale))
  (dotimes (h nh)
    (let ((ho (* h hd)) (po (* h tl tl)))
      (declare (type fixnum ho po))
      (blas-gemm-strided :nt tl tl hd scale q ho d k ho d 0d0 p po tl)
      (with-fast
        (dotimes (i tl)
          (let ((row (+ po (* i tl))) (mx most-negative-double-float) (s 0d0))
            (declare (type fixnum row) (type double-float mx s))
            (dotimes (j (1+ i)) (setf mx (max mx (aref p (+ row j)))))
            (dotimes (j (1+ i))
              (let ((ex (exp (- (aref p (+ row j)) mx)))) (setf (aref p (+ row j)) ex) (incf s ex)))
            (dotimes (j (1+ i)) (setf (aref p (+ row j)) (/ (aref p (+ row j)) s)))
            (loop for j of-type fixnum from (1+ i) below tl do (setf (aref p (+ row j)) 0d0)))))
      (blas-gemm-strided :nn tl hd tl 1d0 p po tl vv ho d 0d0 a ho d))))

(defun attn-bwd-lisp (p q k vv da dq dk dv dsc tl d nh hd scale)
  (declare (type dvec p q k vv da dq dk dv dsc) (type fixnum tl d nh hd) (type double-float scale))
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
                             (incf (aref dk (+ (* j d) (* h hd) e)) (* ds (aref q (+ (* i d) (* h hd) e))))))))))))

(defun attn-bwd-blas (p q k vv da dq dk dv dp tl d nh hd scale)
  "DP is a tl x tl scratch matrix (dP, then dS in place)."
  (declare (type dvec p q k vv da dq dk dv dp) (type fixnum tl d nh hd) (type double-float scale))
  (dotimes (h nh)
    (let ((ho (* h hd)) (po (* h tl tl)))
      (declare (type fixnum ho po))
      (blas-gemm-strided :tn tl hd tl 1d0 p po tl da ho d 0d0 dv ho d)
      (blas-gemm-strided :nt tl tl hd 1d0 da ho d vv ho d 0d0 dp 0 tl)
      (with-fast
        (dotimes (i tl)
          (let ((row (* i tl)) (prow (+ po (* i tl))) (sum 0d0))
            (declare (type fixnum row prow) (type double-float sum))
            (dotimes (j (1+ i)) (incf sum (* (aref p (+ prow j)) (aref dp (+ row j)))))
            (dotimes (j (1+ i))
              (setf (aref dp (+ row j)) (* scale (aref p (+ prow j)) (- (aref dp (+ row j)) sum))))
            (loop for j of-type fixnum from (1+ i) below tl do (setf (aref dp (+ row j)) 0d0)))))
      (blas-gemm-strided :nn tl hd tl 1d0 dp 0 tl k ho d 0d0 dq ho d)
      (blas-gemm-strided :tn tl hd tl 1d0 dp 0 tl q ho d 0d0 dk ho d))))

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
        (if (attn-blas-p tl hd)
            (attn-fwd-blas q k vv p a tl d nh hd scale)
            (attn-fwd-lisp q k vv p a tl d nh hd scale))
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
      (let ((pr (aref (the dvec (acts-probs acts)) (+ (* i v) (aref y i)))))
        (decf s (log (if (< pr 1d-300) 1d-300 pr)))))   ; clamp tiny values, but let a NaN through (MAX would hide it)
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
               (if (attn-blas-p tl hd)
                   (attn-bwd-blas p q k vv da dq dk dv (acts-dp acts) tl d nh hd scale)
                   (attn-bwd-lisp p q k vv da dq dk dv dsc tl d nh hd scale))
               (mm-tn n1 dq (svref grads (+ b 1)) tl d d t)
               (mm-tn n1 dk (svref grads (+ b 2)) tl d d t)
               (mm-tn n1 dv (svref grads (+ b 3)) tl d d t)
               (mm-nt dq wq dn tl d d nil)
               (mm-nt dk wk dn tl d d t)
               (mm-nt dv wv dn tl d d t)
               ;; dh for the layer below = dhmid (residual) + path through norm1
               (rmsnorm-bwd hin g1 (svref (acts-r1 acts) l) dn dh (svref grads b) tl d nil)
               (with-fast (dotimes (i (* tl d)) (incf (aref dh i) (aref dhmid i))))))
    ;; embeddings
    (let ((gt (svref grads 0)) (gp (svref grads 1)))
      (declare (type dvec gt gp))
      (dotimes (i tl)
        (dotimes (j d)
          (incf (aref gt (+ (* (aref x i) d) j)) (aref dh (+ (* i d) j)))
          (incf (aref gp (+ (* i d) j)) (aref dh (+ (* i d) j))))))
    loss))
