;;;; Writes an asem-tf model as a TLM1 file (see native/FORMAT.md), the flat page-aligned format that
;;;; native/libtlm.so (and the Erlang NIF) mmap directly. Load after tf.lisp.
;;;;
;;;; tf.lisp stores matrices (in x out) row-major, TLM1 stores them (out x in) row-major, so every weight matrix
;;;; is transposed on the way out. The model is pre-RMSNorm without biases, which the file records
;;;; (norm = 1, no bias flag); the Erlang model is pre-LayerNorm with biases (norm = 0): the two architectures
;;;; are different models, the format carries both.
(in-package :asem-tf)

(unless (member :little-endian *features*)
  (error "TLM1 files are little endian; this Lisp is not"))

(defconstant +tlm-page+ 4096)
(defconstant +tlm-align+ 256)

(defun %align (n a) (* a (ceiling n a)))

(defun %transpose-in-out (w rows cols)
  "W is (rows x cols) row-major; returns its transpose (cols x rows)."
  (declare (type dvec w) (type fixnum rows cols))
  (let ((out (new-dvec (* rows cols))))
    (dotimes (r rows) (dotimes (c cols) (setf (aref out (+ (* c rows) r)) (aref w (+ (* r cols) c)))))
    out))

(defun tlm-tensors (m)
  "List of (name rows cols dvec) in file order, canonical (out x in) layout."
  (let* ((cfg (model-config m)) (ps (model-params m))
         (v (config-vocab cfg)) (tt (config-ctx cfg)) (d (config-dim cfg)) (f (config-ff cfg)))
    (append
     (list (list "tok" v d (svref ps 0)) (list "pos" tt d (svref ps 1))
           (list "lnf.g" 1 d (svref ps (gf-index cfg)))
           (list "out.w" v d (%transpose-in-out (svref ps (head-index cfg)) d v)))
     (loop for l below (config-layers cfg)
           for b = (layer-base l)
           append (flet ((nm (s) (format nil "l~d.~a" l s)))
                    (list (list (nm "ln1.g") 1 d (svref ps b))
                          (list (nm "wq") d d (%transpose-in-out (svref ps (+ b 1)) d d))
                          (list (nm "wk") d d (%transpose-in-out (svref ps (+ b 2)) d d))
                          (list (nm "wv") d d (%transpose-in-out (svref ps (+ b 3)) d d))
                          (list (nm "wo") d d (%transpose-in-out (svref ps (+ b 4)) d d))
                          (list (nm "ln2.g") 1 d (svref ps (+ b 5)))
                          (list (nm "w1") f d (%transpose-in-out (svref ps (+ b 6)) d f))
                          (list (nm "w2") d f (%transpose-in-out (svref ps (+ b 7)) f d))))))))

(defun save-tlm (m path)
  "Write model M to PATH as TLM1 (f64, RMSNorm, no biases). Written to PATH.tmp and renamed, so a process that
has the old file mmap'd is not disturbed."
  (let* ((cfg (model-config m)) (tensors (tlm-tensors m)) (n (length tensors))
         (data-off (%align (+ 128 (* 40 n)) +tlm-page+))
         (offsets (let ((cur data-off))
                    (loop for (nil r c nil) in tensors
                          collect (let ((off (%align cur +tlm-align+))) (setf cur (+ off (* 8 r c))) off))))
         (end (+ (car (last offsets)) (* 8 (second (car (last tensors))) (third (car (last tensors))))))
         (fsize (%align end +tlm-page+))
         (buf (make-array fsize :element-type '(unsigned-byte 8) :initial-element 0))
         (tmp (concatenate 'string (namestring path) ".tmp")))
    (sb-sys:with-pinned-objects (buf)
      (let ((sap (sb-sys:vector-sap buf)))
        (loop for ch across "TLM1" for i from 0 do (setf (sb-sys:sap-ref-8 sap i) (char-code ch)))
        (loop for val in (list 1 1 1 0                     ; version, dtype f64, norm rms, flags (no biases)
                               (config-vocab cfg) (config-ctx cfg) (config-dim cfg) (config-heads cfg)
                               (config-layers cfg) (config-ff cfg) n)
              for i from 4 by 4 do (setf (sb-sys:sap-ref-32 sap i) val))
        (setf (sb-sys:sap-ref-64 sap 48) 128 (sb-sys:sap-ref-64 sap 56) data-off (sb-sys:sap-ref-64 sap 64) fsize)
        (loop for (name r c vec) in tensors for off in offsets for i from 0
              do (let ((e (+ 128 (* 40 i))))
                   (assert (< (length name) 24))
                   (loop for ch across name for j from 0 do (setf (sb-sys:sap-ref-8 sap (+ e j)) (char-code ch)))
                   (setf (sb-sys:sap-ref-32 sap (+ e 24)) r (sb-sys:sap-ref-32 sap (+ e 28)) c
                         (sb-sys:sap-ref-64 sap (+ e 32)) off)
                   (assert (= (length vec) (* r c)))
                   (let ((vec vec)) (declare (type dvec vec))
                     (dotimes (k (* r c)) (setf (sb-sys:sap-ref-double sap (+ off (* 8 k))) (aref vec k))))))))
    (with-open-file (out tmp :direction :output :if-exists :supersede :element-type '(unsigned-byte 8))
      (write-sequence buf out))
    (rename-file tmp path)
    path))

(export 'save-tlm)
(export 'tlm-tensors)
