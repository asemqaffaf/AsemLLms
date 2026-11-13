;;;; Experimental, training-process-only targeted contrastive objective.
;;;; Load normally after SFT for FP64 tests, or via ASEM-DERIVE for FP32 training.
;;;; No model architecture, checkpoint, sampler, or inference decoder changes.
(in-package :asem-sft)
(defvar *inverse-original-run-multi* (symbol-function 'run-multi))
(defun inverse-pair-loss (model prompt gold wrong cache grads coefficient)
  "coefficient * L * softplus(log P(wrong)-log P(gold)); equal token lengths.
Gradient is +c*CE(gold) - c*CE(wrong), accumulated through separate activation buffers."
  (unless (and gold (= (length gold) (length wrong)) (not (equal gold wrong))) (error "Invalid equal-length contrastive pair"))
  (let* ((cfg (model-config model)) (v (config-vocab cfg)) (n (length prompt)) (len (bucket-len (+ n (length gold) -1) (config-ctx cfg)))
         (a (or (gethash (+ 100000 len) cache) (setf (gethash (+ 100000 len) cache) (make-acts cfg len +max-target+))))
         (b (or (gethash (+ 200000 len) cache) (setf (gethash (+ 200000 len) cache) (make-acts cfg len +max-target+))))
         (rows (ivec-of (loop for k below (length gold) collect (+ n k -1))))
         (gg (ivec-of gold)) (ww (ivec-of wrong))
         (xg (padded-x (ivec-of prompt) len gg)) (xw (padded-x (ivec-of prompt) len ww)))
    (forward-rows model xg a rows) (forward-rows model xw b rows)
    (let* ((delta (- (rows-logprob b ww v) (rows-logprob a gg v)))
           (softplus (+ (max 0l0 delta) (log (+ 1l0 (exp (- (abs delta)))))))
           (sigmoid (if (>= delta 0) (/ 1l0 (+ 1l0 (exp (- delta)))) (let ((z (exp delta))) (/ z (+ 1l0 z)))))
           (scale (* coefficient (length gold)))
           (weight (host-float (* scale sigmoid))))
      (when grads
        (let ((positive (new-dvec (length gold))) (negative (new-dvec (length gold))))
          (fill positive weight) (fill negative (- weight))
          (backward-rows model xg rows gg a grads positive)
          (backward-rows model xw rows ww b grads negative)))
      (* scale softplus))))
(defun install-inverse-objective (coefficient annotations)
  "Lexically capture the coefficient/table so worker threads share the intended immutable configuration."
  (unless (>= coefficient 0) (error "Nonnegative objective coefficient required"))
  (let ((original *inverse-original-run-multi*))
    (setf (symbol-function 'run-multi)
          (if (zerop coefficient) original
              (lambda (model e cache grads)
                (multiple-value-bind (loss pred scores mass) (funcall original model e cache grads)
                  (dolist (pair (gethash (ex-id e) annotations))
                    (incf loss (inverse-pair-loss model (getf pair :prompt) (getf pair :gold) (getf pair :wrong) cache grads coefficient)))
                  (values loss pred scores mass)))))))
