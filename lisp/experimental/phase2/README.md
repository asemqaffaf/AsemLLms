# Isolated Phase 2 intervention

R7 remains the reference model. The Phase 2 Treatment is not promoted, and no retraining is authorized by these files.

`historical/` preserves the pre-isolation source, training CLI, audit CLI and tests byte for byte. `historical-manifest.json` records their original paths and hashes. Existing checkpoints, datasets and results remain in their original locations.

Ordinary `lisp/src/retrieval.lisp` and its tests are restored to the committed baseline. The standard training/audit CLIs reject `predicate-paraphrase=1` rather than silently accepting an experimental setting. The restored seed-7, 75,000-sequence R7 generator reproduces digest `eed4799f6b89de20062a8ce033591b4c1f97052790e80088921214a88c6f1478`, 122,334 questions and 249,812 answer tokens.

Explicit experimental entry points:

- `lisp/bin/retrieval-paraphrase-pilot.lisp` loads `load.lisp` in its own SBCL process. Do not run it without a newly authorized training experiment.
- `lisp/bin/experimental-phase2-audit.lisp` is the model-free historical generator audit, retaining its `predicate-paraphrase=0|1` option.
- `lisp/tests/test-phase2-paraphrase.lisp` runs the historical experimental tests separately. Their relationship semantic coverage remains limited; passing them does not certify natural-language equivalence.
- `frames.lisp` exposes the old strings for evaluation overlap checks, in a separate package. It does not install any generator overrides.

The independent review found potential entailment mismatches among `taught/coached/trained under` and `hired/employed/appointed`, plus changes in packed question coverage and familiar-frame exposure. No definite relationship indexing bug was established. These uncertainties concern the failed intervention; they do not establish that R7 or AsemLLM is broken.
