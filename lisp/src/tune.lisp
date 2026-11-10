;;;; Autotuner: code that writes and measures code. Uses the generators in tf.lisp to build the source of
;;;; many kernel variants, COMPILEs them at runtime, rejects any that does not match a plain reference
;;;; bit for bit, times the rest on the shapes the trainer really uses, and records the fastest in
;;;; src/tuned.sexp (which tf.lisp expands into its kernels on the next load).
;;;; The same idea drives the Erlang side: it generates the source of tinyllm's dot/3 and axpy/3 with
;;;; different unroll factors, has `escript` compile, verify and time each, writes the winner to
;;;; tinyllm_kernels.hrl and keeps it only if `rebar3 eunit` still passes.
(defpackage :asem-tune
  (:use :cl :asem-tf)
  (:export #:tune-lisp #:tune-erlang #:*factors* #:reference-mm #:compile-variant #:matches-reference-p
           #:erlang-source #:write-tuned))
(in-package :asem-tune)

(defparameter *factors* '(1 2 3 4 6 8 12 16))
(defparameter *tie* 1.03d0 "A larger factor must beat the smaller one by this ratio to win; avoids noise flips.")

;;; ------------------------------------------------------------ plain reference implementations
(defun reference-mm (kind a b c m k n acc)
  "Obviously-correct loops with the accumulation order the fast kernels promise."
  (declare (type dvec a b c))
  (unless acc (fill c 0d0))
  (ecase kind
    (:mm-nn (dotimes (i m) (dotimes (p k) (dotimes (j n) (incf (aref c (+ (* i n) j)) (* (aref a (+ (* i k) p)) (aref b (+ (* p n) j))))))))
    (:mm-nt (dotimes (i m) (dotimes (j n) (let ((s 0d0)) (dotimes (p k) (incf s (* (aref a (+ (* i k) p)) (aref b (+ (* j k) p)))))
                                            (incf (aref c (+ (* i n) j)) s)))))
    (:mm-tn (dotimes (i m) (dotimes (p k) (dotimes (j n) (incf (aref c (+ (* p n) j)) (* (aref a (+ (* i k) p)) (aref b (+ (* i n) j))))))))))

(defun sizes (kind m k n)
  "Element counts of A, B, C for KIND with the dims as passed to the kernel."
  (ecase kind
    (:mm-nn (list (* m k) (* k n) (* m n)))
    (:mm-nt (list (* m k) (* n k) (* m n)))
    (:mm-tn (list (* m k) (* m n) (* k n)))))

(defun random-dvec (n state) (let ((v (new-dvec n))) (dotimes (i n v) (setf (aref v i) (- (random 2d0 state) 1d0)))))

(defun compile-variant (kind u)
  "Generate the source for KIND with unroll U, compile it now, return the function."
  (let ((form (gen-kernel kind (intern (format nil "~a-U~d" kind u) :asem-tune) u)))
    (compile nil `(lambda ,(third form) ,@(cdddr form)))))

(defun matches-reference-p (kind fn)
  "Bit-identical to the reference on shapes that exercise every remainder path, with and without accumulation."
  (let ((state (sb-ext:seed-random-state 7)))
    (loop for (m k n) in '((1 1 1) (5 7 9) (3 13 2) (9 5 11) (8 8 8) (2 17 5) (13 3 7) (16 9 33))
          always (loop for acc in '(nil t)
                       always (destructuring-bind (la lb lc) (sizes kind m k n)
                                (let* ((a (random-dvec la state)) (b (random-dvec lb state))
                                       (c1 (random-dvec lc state)) (c2 (copy-seq c1)))
                                  (funcall fn a b c1 m k n acc)
                                  (reference-mm kind a b c2 m k n acc)
                                  (equalp c1 c2)))))))

;;; ------------------------------------------------------------ timing
(defun seconds () (/ (get-internal-real-time) (float internal-time-units-per-second 1d0)))

(defun time-call (fn kind m k n)
  "Best-of-5 seconds per call."
  (let* ((state (sb-ext:seed-random-state 3))
         (sz (sizes kind m k n))
         (a (random-dvec (first sz) state)) (b (random-dvec (second sz) state)) (c (new-dvec (third sz)))
         (reps (max 5 (ceiling 3d7 (* 2d0 m k n)))))
    (dotimes (i 3) (funcall fn a b c m k n nil))
    (loop repeat 5
          minimize (let ((t0 (seconds)))
                     (dotimes (i reps) (funcall fn a b c m k n nil))
                     (/ (- (seconds) t0) reps)))))

(defun training-shapes (kind)
  "(m k n) of the calls the trainer makes (batch window 32, dim 64, ff 128, vocab ~1400)."
  (ecase kind
    (:mm-nn '((32 64 1401) (32 64 64) (32 64 128) (32 128 64)))
    (:mm-nt '((32 1401 64) (32 64 64) (32 128 64) (32 64 128)))
    (:mm-tn '((32 64 1401) (32 64 64) (32 64 128) (32 128 64)))))

(defun pick (results)
  "RESULTS: ((u . seconds) ...) ascending u. The smallest u whose time is within *TIE* of the best."
  (let ((best (reduce #'min results :key #'cdr)))
    (car (find-if (lambda (r) (<= (cdr r) (* best *tie*))) results))))

(defun tune-kind (kind)
  (let ((results nil))
    (dolist (u *factors*)
      (let ((fn (compile-variant kind u)))
        (if (matches-reference-p kind fn)
            (push (cons u (loop for (m k n) in (training-shapes kind) sum (time-call fn kind m k n))) results)
            (format t "  ~a unroll ~d REJECTED: result differs from the reference~%" kind u))))
    (setf results (nreverse results))
    (dolist (r results) (format t "  ~a unroll ~d: ~,3f ms~%" kind (car r) (* 1000 (cdr r))))
    (values (pick results) results)))

(defun write-tuned (plist path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (let ((*package* (find-package :asem-tf)) (*print-pretty* nil))
      (format out ";; written by lisp/bin/tune.lisp - do not edit by hand; machine-specific, re-run to refresh~%")
      (prin1 plist out) (terpri out))))

(defun tune-lisp (path)
  (format t "Lisp kernels (SBCL ~a)~%" (lisp-implementation-version))
  (let ((cfg nil) (old (tuned-config)))
    (dolist (kind '(:mm-nn :mm-nt :mm-tn))
      (let ((u (tune-kind kind)))
        (format t "=> ~a: unroll ~d (was ~d)~%" kind u (getf old kind))
        (setf cfg (append cfg (list kind u)))))
    (write-tuned (append cfg (list :host (machine-type) :sbcl (lisp-implementation-version))) path)
    cfg))

;;; ------------------------------------------------------------ Erlang side
(defun erlang-source (u-dot &optional (u-axpy u-dot))
  "Erlang source of dot/3 unrolled U-DOT times and axpy/3 unrolled U-AXPY times (additions keep the plain
left-to-right order, so results are bit-identical)."
  (flet ((vars (p u) (loop for i from 1 to u collect (format nil "~a~d" p i))))
    (with-output-to-string (s)
      (when (> u-dot 1)
        (format s "dot([~{~a~^,~}|As], [~{~a~^,~}|Bs], Acc) ->~%    dot(As, Bs, Acc~{ + ~a~});~%"
                (vars "A" u-dot) (vars "B" u-dot)
                (loop for a in (vars "A" u-dot) for b in (vars "B" u-dot) collect (format nil "~a * ~a" a b))))
      (format s "dot([A|As], [B|Bs], Acc) -> dot(As, Bs, Acc + A * B);~%dot([], [], Acc) -> Acc.~%~%")
      (when (> u-axpy 1)
        (format s "axpy([~{~a~^,~}|Ys], A, [~{~a~^,~}|Xs]) ->~%    [~{~a~^, ~} | axpy(Ys, A, Xs)];~%"
                (vars "Y" u-axpy) (vars "X" u-axpy)
                (loop for y in (vars "Y" u-axpy) for x in (vars "X" u-axpy) collect (format nil "~a + A * ~a" y x))))
      (format s "axpy([Y|Ys], A, [X|Xs]) -> [Y + A * X | axpy(Ys, A, Xs)];~%axpy([], _, []) -> [].~%"))))

(defun run (program args &key directory)
  (let* ((out (make-string-output-stream))
         (p (sb-ext:run-program program args :search t :output out :error out :directory directory)))
    (values (sb-ext:process-exit-code p) (get-output-stream-string out))))

(defun slurp (path) (with-open-file (in path :if-does-not-exist nil)
                      (when in (let ((s (make-string (file-length in)))) (subseq s 0 (read-sequence s in))))))
(defun spit (path text) (with-open-file (out path :direction :output :if-exists :supersede) (write-string text out)))

(defun tune-erlang (tinyllm-dir tune-dir)
  "Pick the unroll factor of dot/3 and of axpy/3 separately. Returns (u-dot u-axpy), or NIL if the
tests rejected the change."
  (let* ((hrl (merge-pathnames "apps/tinyllm/src/tinyllm_kernels.hrl" tinyllm-dir))
         (bench (merge-pathnames "kernel_bench.escript" tune-dir))
         (dots nil) (axpys nil))
    (dolist (u '(1 2 4 8 16))   ; capped at 16: a 32-wide unroll crashes the BEAM JIT
      (let ((cand (format nil "/tmp/asem_cand_~d.erl" u)))
        (spit cand (format nil "-module(asem_cand_~d).~%-export([dot/3, axpy/3]).~%~a" u (erlang-source u)))
        (multiple-value-bind (code text) (run "escript" (list (namestring bench) cand))
          (let ((fields (and (zerop code) (read-from-string (format nil "(~a)" text)))))
            (cond ((and fields (eq (first fields) 'ok))
                   (format t "  erlang unroll ~2d: dot ~,1f us, axpy ~,1f us (per vector pass)~%" u (second fields) (third fields))
                   (push (cons u (second fields)) dots) (push (cons u (third fields)) axpys))
                  (t (format t "  erlang unroll ~d REJECTED: ~a~%" u (string-trim '(#\Newline) text))))))))
    (let* ((ud (pick (nreverse dots))) (ua (pick (nreverse axpys))) (old (slurp hrl))
           (new (format nil "%% GENERATED by lisp/bin/tune.lisp (dot unroll ~d, axpy unroll ~d) - do not edit; re-run the tuner instead.~%~a"
                        ud ua (erlang-source ud ua))))
      (format t "=> erlang: dot unroll ~d, axpy unroll ~d~%" ud ua)
      (cond ((equal old new) (format t "   already in place~%") (list ud ua))
            (t (spit hrl new)
               (format t "   running rebar3 eunit with the new kernels (about 30 s)...~%")
               (multiple-value-bind (code text) (run "rebar3" '("eunit") :directory (namestring tinyllm-dir))
                 (cond ((zerop code) (format t "   tests pass: kept~%") (list ud ua))
                       (t (when old (spit hrl old))
                          (format t "   TESTS FAILED, restored the previous kernels~%~a~%" (subseq text (max 0 (- (length text) 800))))
                          nil))))))))
