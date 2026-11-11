;;;; Explicit opt-in only. Historical failed experiment, NOT a promoted training recipe.
;;;; Loading this in a separate SBCL process installs the archived Phase 2 overrides.
;;;; Ordinary retrieval training/evaluation never loads this file.
(load (merge-pathnames "historical/retrieval.lisp" *load-truename*))
