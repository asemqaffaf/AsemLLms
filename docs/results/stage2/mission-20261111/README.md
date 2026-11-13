# Autonomous mission — 2026-11-11

Reference: R7 final, 45.8% unfamiliar QA option ranking. Immediate target after user steering: a learned checkpoint at 50–55%, preserving other skills. The broader 65% target remains conditional on reproducible evidence. No locked-test access is allowed.

Compute budget: at most three matched 300–500-step pilots, serial training only; at most one run up to 2,500 steps after convincing pilot evidence and acceptable retention. No automatic full R8 run. Preserve R7 and all previous experiments and unrelated work.

## Frozen data and zero-training findings

`selection.sexp` has 500 fresh examples with public development wording; `confirmation.sexp` has 500 examples with newly authored passage/question frames and nontraining entity pools. Both have 100 examples/category and 25 gold positions/category/option. Confirmation is not used for training, checkpoint selection or hypothesis refinement. The files were frozen before mission inference. The unchanged original 500 QA questions remain a separate evaluation.

The supervised training tracer reproduces the stable R7 generator's exact IDs/rows/targets for the first 256 accepted seed-7 sequences. `supervision-audit.json` compares dense and isolated questions plus 200 fresh selection questions in matched familiar/unfamiliar forms, using frozen R7 and no parameter updates.

- First training questions: 212/256 correct first tokens and 210/256 exact prefixes. Dense and isolated versions are identical.
- Later questions: 145/171 first tokens correct with prior solved QA context, 138/171 when isolated (7 gains, 14 losses). This modest difference does not support declaring packing the dominant root cause.
- Held-out reverse lookup remains weak even in familiar wording: 12/40 first tokens and 7/40 prefixes; unfamiliar wording gives 6/40 and 4/40. Familiar relationships give 27/40 and 21/40, unfamiliar 24/40 and 17/40.
- Comparison prefixes are 3/40 familiar and 4/40 unfamiliar; temporal prefixes are 10/40 familiar and 0/40 unfamiliar. These different profiles do not support a single shared failure mechanism.

These observations, plus the saved 104 wrong-direction cases, support trying a targeted relationship/reverse objective rather than globally changing supervision, packing, model size or decoding.

## Planned first controlled pilot

Hypothesis: an auxiliary contrastive penalty against a graph-derived opposite-role answer (relationships) or an alternative bound person (reverse lookup) will improve unfamiliar inverse-answer selection while preserving the original full-vocabulary answer objective and other tasks.

The only manipulated factor will be the coefficient of this auxiliary objective (0 Control vs 1 Treatment). The same R7 tensors, complete dense sequences, answer tokens, copy replay, batch indices, optimizer initialization and schedule will be used. Only valid equal-token-length negatives that appear among the original options are eligible. No paraphrases, external constrained decoding, architecture changes or evaluation answers enter training. All five categories and unconstrained exact-prefix accuracy will be evaluated separately.

The objective is experimental and isolated. Signed auxiliary gradients must pass finite differences and FP32/FP64 agreement before training. A zero coefficient must call the original function exactly. Full data counts, eligibility, source/data/checkpoint hashes and gates will be frozen in the pilot protocol before training.

Fresh selection R7 baseline: **208/500 (41.6%) ranking, 81/500 (16.2%) exact prefix**, compared with 45.8% and 18.4% on the original QA set. These different development samples must not be conflated. The full saved predictions are in `r7-selection.json`.

The prepared pilot has 15,000 dense sequences plus 3,000 copy sequences. Full IDs/rows/targets exactly reproduce stable generation. There are 547 eligible opposite-role and 1,326 eligible reverse-person questions; the fixed 300-step schedule consumes 1,005 auxiliary pairs and 26,548 original dense answer tokens (plus identical copy tokens in both arms). 1,944 semantic negatives were excluded for unequal BPE length; their original CE supervision is retained. Data/protocol hashes are saved before training.

Preparation caught an annotation ambiguity: a sampled value `best` also appeared as a template word in a question about another value. Annotation now uses latent frame arguments traced through the existing augmentation, rather than inferring semantic roles from incidental word matches. A regression fixture covers it. This was an experimental annotation defect caught before training; the original R7 generator was unchanged.

Pilots started: **0/3**. Longer runs started: **0/1**. R7 remains the reference and no checkpoint is promoted.
