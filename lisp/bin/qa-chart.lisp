;;; SVG line chart of one metric from several SFT logs (log.jsonl "eval" or "train" events). No dependencies.
;;;   sbcl --script lisp/bin/qa-chart.lisp out=FILE.svg metric=dev_acc [event=eval] [title=...] [ymin=0] [ymax=1] [chance=0.25]
;;;        runs=LABEL=PATH/log.jsonl,LABEL=PATH/log.jsonl,...  [x=step|epoch]
(let ((here (directory-namestring *load-truename*)))
  (dolist (f '("util" "json")) (load (merge-pathnames (format nil "../src/~a.lisp" f) here))))
(defpackage :qa-chart (:use :cl))
(in-package :qa-chart)

(defun opt-arg (name default)
  (let ((p (find-if (lambda (a) (and (> (length a) (length name)) (string= name a :end2 (length name)) (char= (char a (length name)) #\=)))
                    (cdr sb-ext:*posix-argv*))))
    (if p (subseq p (1+ (length name))) default)))
(defun split (s ch) (let ((out nil) (start 0)) (loop for i = (position ch s :start start) do (push (subseq s start i) out) (if i (setf start (1+ i)) (return))) (nreverse out)))
(defun esc (s) (with-output-to-string (o) (loop for c across s do (case c (#\< (write-string "&lt;" o)) (#\> (write-string "&gt;" o)) (#\& (write-string "&amp;" o)) (t (write-char c o))))))

(defparameter *colors* '("#2a6fdb" "#d9480f" "#2b8a3e" "#862e9c" "#c92a2a" "#5c940d" "#495057" "#e67700"))

(let* ((out (opt-arg "out" "chart.svg")) (metric (opt-arg "metric" "dev_acc")) (event (opt-arg "event" "eval")) (xkey (opt-arg "x" "step"))
       (title (opt-arg "title" metric))
       (runs (mapcar (lambda (r) (let ((p (position #\= r))) (cons (subseq r 0 p) (subseq r (1+ p))))) (split (opt-arg "runs" "") #\,)))
       (series (loop for (label . path) in runs
                     collect (cons label (with-open-file (in path)
                                           (loop for line = (read-line in nil) while line
                                                 for o = (ignore-errors (asem-json:parse-json line))
                                                 when (and o (equal (asem-json:json-get o "event") event) (numberp (asem-json:json-get o metric)))
                                                   collect (cons (float (asem-json:json-get o xkey) 1d0) (float (asem-json:json-get o metric) 1d0)))))))
       (allx (loop for s in series append (mapcar #'car (cdr s)))) (ally (loop for s in series append (mapcar #'cdr (cdr s))))
       (xmax (max 1d0 (reduce #'max allx :initial-value 0d0)))
       (ymin (let ((v (opt-arg "ymin" nil))) (if v (float (read-from-string v) 1d0) (reduce #'min ally))))
       (ymax (let ((v (opt-arg "ymax" nil))) (if v (float (read-from-string v) 1d0) (reduce #'max ally))))
       (chance (let ((v (opt-arg "chance" nil))) (and v (float (read-from-string v) 1d0))))
       (w 920) (h 460) (l 64) (r 770) (tp 40) (b 406))
  (flet ((sx (x) (+ l (* (- r l) (/ x xmax)))) (sy (y) (- b (* (- b tp) (/ (- y ymin) (max 1d-9 (- ymax ymin)))))))
    (with-open-file (o out :direction :output :if-exists :supersede)
      (format o "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ~d ~d\" role=\"img\" width=\"100%\" style=\"max-width:~dpx\" font-family=\"system-ui, -apple-system, Helvetica, Arial, sans-serif\">~%" w h w)
      (format o "<title>~a</title>~%<rect width=\"~d\" height=\"~d\" fill=\"#fcfcfb\"/>~%" (esc title) w h)
      (format o "<text x=\"~d\" y=\"24\" font-size=\"15\" fill=\"#1d1d1b\">~a</text>~%" l (esc title))
      (dotimes (k 6)
        (let* ((y (+ ymin (* k (/ (- ymax ymin) 5)))) (py (sy y)))
          (format o "<line x1=\"~d\" x2=\"~d\" y1=\"~,1f\" y2=\"~,1f\" stroke=\"#d9d8d2\" stroke-dasharray=\"2 4\"/>~%" l r py py)
          (format o "<text x=\"~d\" y=\"~,1f\" text-anchor=\"end\" font-size=\"12\" fill=\"#52514e\">~,2f</text>~%" (- l 10) (+ py 4) y)))
      (dotimes (k 6)
        (let* ((x (* k (/ xmax 5))) (px (sx x)))
          (format o "<text x=\"~,1f\" y=\"~d\" text-anchor=\"middle\" font-size=\"12\" fill=\"#52514e\">~a</text>~%" px (+ b 20) (if (string= xkey "step") (round x) (format nil "~,1f" x)))))
      (format o "<text x=\"~d\" y=\"~d\" text-anchor=\"middle\" font-size=\"12\" fill=\"#52514e\">~a</text>~%" (floor (+ l r) 2) (+ b 44) (esc xkey))
      (when chance
        (format o "<line x1=\"~d\" x2=\"~d\" y1=\"~,1f\" y2=\"~,1f\" stroke=\"#868e96\" stroke-width=\"1.5\" stroke-dasharray=\"6 4\"/>~%" l r (sy chance) (sy chance))
        (format o "<text x=\"~d\" y=\"~,1f\" font-size=\"12\" fill=\"#868e96\">chance</text>~%" (+ r 6) (+ (sy chance) 4)))
      (loop for (label . pts) in series for c in *colors* for k from 0
            when pts
              do (format o "<polyline fill=\"none\" stroke=\"~a\" stroke-width=\"2\" points=\"~{~a~^ ~}\"/>~%" c
                         (mapcar (lambda (p) (format nil "~,1f,~,1f" (sx (car p)) (sy (cdr p)))) pts))
                 (let ((lastp (car (last pts))))
                   (format o "<text x=\"~,1f\" y=\"~,1f\" font-size=\"12\" fill=\"~a\">~a (~,3f)</text>~%" (+ r 6) (+ tp 14 (* 18 k)) c (esc label) (cdr lastp))))
      (format o "</svg>~%"))))
