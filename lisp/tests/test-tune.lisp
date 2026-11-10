(let ((here (directory-namestring *load-truename*)))
  (load (merge-pathnames "../src/util.lisp" here))
  (load (merge-pathnames "../src/tf.lisp" here))
  (load (merge-pathnames "../src/tune.lisp" here))
  (load (merge-pathnames "harness.lisp" here)))
(in-package :harness)

(deftest every-generated-kernel-matches-the-reference-bit-for-bit
  (dolist (kind '(:mm-nn :mm-nt :mm-tn))
    (dolist (u asem-tune:*factors*)
      (check (asem-tune:matches-reference-p kind (asem-tune:compile-variant kind u))
             (format nil "~a unroll ~d differs from the plain loops" kind u)))))

(deftest the-checker-rejects-a-wrong-kernel
  (check (not (asem-tune:matches-reference-p :mm-nn (lambda (a b c m k n acc) (declare (ignore a b m k n acc)) (fill c 1d0))))
         "a kernel that ignores its inputs must not pass"))

(deftest generators-return-code-as-data
  (let ((form (asem-tf:gen-kernel :mm-nn 'foo 4)))
    (check (eq (first form) 'defun) "a defun form")
    (check (eq (second form) 'foo) "named as asked")
    (check (> (length (prin1-to-string (asem-tf:gen-kernel :mm-nn 'foo 8)))
              (length (prin1-to-string form))) "a bigger factor generates more code")))

(deftest erlang-source-is-unrolled-and-ordered
  (let ((s (asem-tune:erlang-source 1 4)))
    (check (search "dot([A|As], [B|Bs], Acc) -> dot(As, Bs, Acc + A * B);" s) "plain dot clause")
    (check (not (search "A2" s)) "dot is not unrolled when its factor is 1")
    (check (search "Y1 + A * X1, Y2 + A * X2, Y3 + A * X3, Y4 + A * X4 | axpy(Ys, A, Xs)" s) "axpy unrolled 4 times, in order"))
  (check (search "Acc + A1 * B1 + A2 * B2" (asem-tune:erlang-source 2)) "additions stay left to right"))

(deftest tuned-config-is-valid
  (let ((cfg (asem-tf:tuned-config)))
    (dolist (k '(:mm-nn :mm-nt :mm-tn))
      (check (member (getf cfg k) asem-tune:*factors*) (format nil "~a has a known factor" k)))))

(run-all)
