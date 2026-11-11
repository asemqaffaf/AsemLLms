# Phase 2 Controlled Pilot: Training-Only Predicate Paraphrase Augmentation

## Scope and Provenance

- **Experimental design**: Paired 300-step **continuation** experiment from R7's final checkpoint (`data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt`, `sha256: 33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`), not a reproduction of R7's original from-scratch training trajectory.
- **Single manipulated variable**: `*predicate-paraphrase*` (`0` Control vs `1` Treatment) for training-only paraphrasing of `:rel`, `:cmp`, and `:tmp` predicates. Attribute extraction (`:attr`, `:num`), reverse lookup, answer targets, tokenizer, architecture, optimizer init (fresh Adam moments `m1=0, m2=0` in `sft32`), LR schedule (`lr=1.0e-4`, `warmup=30`, `min_lr_frac=0.1` cosine decay), `steps=300`, `batch=32`, `clip=1.0`, `wd=0.0`, `seed=802`, data budget (`train-n=75000`, `copy-n=15000`, `copy-seed=77`), batch order, and selection probes (`probe-seed=18002`, `n=300`, unseen wording) are identical between arms.
- **Output paths**:
  - Training runs (gitignored): `data/runs/retrieval/p2-paraphrase-pilot/{recipe.sexp,control,treatment}`
  - Audits & diagnostics: `docs/results/stage2/phase2-paraphrase-pilot/`
- **Locked test set**: Never opened, inspected, evaluated, or modified (`Locked QA bytes match test.lock; answer keys were not parsed.`).

## Step 1 — Data Audit Summary

- **Historical baseline reproduction (`seed=7`, `predicate-paraphrase=0`)**:
  - `sequence_sha256 = "eed4799f6b89de20062a8ce033591b4c1f97052790e80088921214a88c6f1478"` (`75,000` sequences, `122,334` questions, `249,812` answer tokens), matching `docs/results/stage2/r7-analysis/generator-audit.json` identically on the final `lisp/src/retrieval.lisp` revision (`sha256: 8e47f3b1f2280f10f6e9416349f85f0e255f6d3a063571022521f3d6b717c0b6`).
- **Pilot Control vs Treatment paired audit (`seed=802`, `step1-paired-audit.json`)**:
  - `control` `sequence_sha256 = "edcd232c8697fa2c959c26010a143031690061af8d9ae72e1436f11d8f78870a"`
  - `treatment` `sequence_sha256 = "d2c1594f55c039e7f2f8811f3246deae43dcad6bb223f8755ac971570ac45d04"`
  - Lockstep task allocation: `75,001` attempts / `75,000` accepted sequences in both arms, `kind_mismatch_count = 0`, `over_ctx_count = 0`.
  - Byte-identical preservation: `attr` (`24,956 / 24,956`, `37,306` questions including reverse-lookup, `78,220` answer tokens) and `num` (`12,563 / 12,563`, `19,867` questions, `31,841` answer tokens) are byte-for-byte identical.
  - Token lengths & `:multi` + `:copy` loss weight share shift inside `ctx=128`:
    - Overall mean sequence tokens: `97.1520` (Control) vs `97.1207` (Treatment).
    - Total questions: `122,434 -> 121,020` (`-1,414`, `-1.15%`); answer tokens: `249,999 -> 247,871` (`-2,128`, `-0.85%`).
    - By kind (`questions / answer_tokens / weight_with_copy`):
      - `attr`: `37306 / 78220 / 0.295171` -> `37306 / 78220 / 0.297560`
      - `num`: `19867 / 31841 / 0.120155` -> `19867 / 31841 / 0.121128`
      - `rel`: `19953 / 56390 / 0.212793` -> `19737 / 55702 / 0.211899` (`-1.22%` answer tokens)
      - `cmp`: `20994 / 59133 / 0.223144` -> `20879 / 58780 / 0.223608` (`-0.60%` answer tokens)
      - `tmp`: `24314 / 24415 / 0.092132` -> `23231 / 23328 / 0.088743` (`-4.45%` answer tokens)
      - `copy`: `15000 / — / 0.056604` -> `15000 / — / 0.057062`

## Step 2 — Pilot Trajectory Summary (`seed=18002` selection probes, `n=300`)

- **Control (`predicate_paraphrase=0`)**: `step 0` `dev_acc=0.320000`, `dev_loss=3.275422` -> `step 100` `0.296667 / 3.380836` -> `step 200` `0.300000 / 3.443766` -> `step 300` `0.286667 / 3.441647` (`train_probe_acc=0.830000`, `loss=0.849810`). `best_step=0`.
- **Treatment (`predicate_paraphrase=1`)**: `step 0` `dev_acc=0.320000`, `dev_loss=3.275422` (on paraphrased train probe: `0.720000 / 2.495692`) -> `step 100` `0.280000 / 3.135028` -> `step 200` `0.313333 / 3.117959` -> `step 300` `0.300000 / 3.111529` (`train_probe_acc=0.760000`, `loss=1.807591`). `best_step=0`.

## Step 3 — Fixed Step-300 Primary Comparison (`Control -> Treatment`, `policy=seeded`)

### 1. Development QA (`lisp/eval/qa/dev.jsonl`, `n=500`, 100/category)

- **Unfamiliar wording (`p-unf/q-unf`)**:
  - `summed_option_likelihood`: `0.460 -> 0.458` (`-0.002`, 95% CI `[-0.035, +0.031]`, `+34/-35`, McNemar `p=1.0000`)
    - `compare`: `0.350 -> 0.300` (`-0.050`, `[-0.130, +0.030]`, `+6/-11`, `p=0.3323`)
    - `temporal`: `0.240 -> 0.280` (`+0.040`, `[-0.038, +0.118]`, `+10/-6`, `p=0.4545`)
    - `extract`: `0.690 -> 0.720` (`+0.030`, `[-0.022, +0.082]`, `+5/-2`, `p=0.4531`)
    - `reverse`: `0.390 -> 0.350` (`-0.040`, `[-0.113, +0.033]`, `+5/-9`, `p=0.4240`)
    - `relation`: `0.630 -> 0.640` (`+0.010`, `[-0.066, +0.086]`, `+8/-7`, `p=1.0000`)
  - `mean_token_option_likelihood`: `0.454 -> 0.446` (`-0.008`, `[-0.040, +0.024]`, `+32/-36`, `p=0.7163`)
  - `gold_length_greedy_prefix_exact`: `0.176 -> 0.192` (`+0.016`, `[-0.010, +0.042]`, `+26/-18`, `p=0.2912`; `compare +0.020`, `temporal 0.000->0.000`, `extract +0.050`, `reverse +0.030`, `relation -0.020`)
- **Other 2×2 cells (`summed_option_likelihood`)**:
  - `p-fam/q-unf`: `0.520 -> 0.468` (`-0.052`, `[-0.084, -0.020]`, `+20/-46`, `p=0.0019`; `reverse` `0.390 -> 0.280`, `-0.110`, `p=0.0266`; `relation` `0.700 -> 0.640`, `-0.060`, `p=0.2101`)
  - `p-unf/q-fam`: `0.432 -> 0.434` (`+0.002`, `[-0.027, +0.031]`, `+28/-27`, `p=1.0000`; `mean_token` `0.412 -> 0.444`, `+0.032`, `p=0.0440`, driven by `temporal` `0.210 -> 0.310`, `p=0.0020`)
  - `p-fam/q-fam`: `0.588 -> 0.560` (`-0.028`, `[-0.056, -0.000]`, `+18/-32`, `p=0.0649`; `reverse` `0.370 -> 0.280`, `-0.090`, `p=0.0225`)

### 2. Held-Out L3 New Probes (`probe-seed=19032`, `n=600`)

- **Unfamiliar wording (`p-unf/q-unf`)**:
  - `summed_option_likelihood`: `0.442 -> 0.450` (`+0.008`, 95% CI `[-0.020, +0.037]`, `+41/-36`, `p=0.6488`)
    - `compare` (`n=134`): `0.276 -> 0.299` (`+0.022`, `p=0.4531`)
    - `temporal` (`n=122`): `0.270 -> 0.254` (`-0.016`, `p=0.8450`)
    - `extract` (`n=184`): `0.636 -> 0.674` (`+0.038`, `p=0.1435`)
    - `reverse` (`n=50`): `0.340 -> 0.420` (`+0.080`, `p=0.1250`)
    - `relation` (`n=110`): `0.555 -> 0.491` (`-0.064`, `[-0.148, +0.021]`, `+8/-15`, `p=0.2100`)
  - `gold_length_greedy_prefix_exact`: `0.230 -> 0.228` (`-0.002`, `[-0.028, +0.024]`, `+31/-32`, `p=1.0000`)
- **Familiar-passage cells (`summed_option_likelihood`)**:
  - `p-fam/q-unf`: `0.532 -> 0.512` (`-0.020`, `p=0.1818`; `relation` `0.636 -> 0.500`, `-0.136`, `+4/-19`, `p=0.0026`)
  - `p-fam/q-fam`: `0.622 -> 0.590` (`-0.032`, `[-0.056, -0.008]`, `+18/-37`, `p=0.0145`; `relation` `0.736 -> 0.600`, `-0.136`, `+0/-15`, `p=0.0001`; `gold_length_greedy_prefix_exact` `0.462 -> 0.407`, `-0.055`, `p<0.0001`)

### 3. Selection Probes (`probe-seed=18002`, `n=300`)

- **Unfamiliar wording (`p-unf/q-unf`)**:
  - `summed_option_likelihood`: `0.487 -> 0.493` (`+0.007`, `[-0.023, +0.036]`, `+11/-9`, `p=0.8238`)
  - `gold_length_greedy_prefix_exact`: `0.287 -> 0.300` (`+0.013`, `[-0.025, +0.051]`, `+19/-15`, `p=0.6076`)

## Decision

The 300-step controlled pilot provides **no convincing generalization benefit** (`-0.002` `qa-dev` `p=1.0000`, `+0.008` `l3-new` `p=0.6488`, `+0.007` `selection` `p=0.8238` on `p-unf/q-unf` summed option likelihood), while causing statistically significant regressions on familiar-wording `relation` (`-0.136` on `l3-new` `p-fam/q-unf` `p=0.0026` and `p-fam/q-fam` `p=0.0001`) and `qa-dev` `p-fam/q-unf` (`-0.052`, `p=0.0019`, with `reverse` `-0.110`, `p=0.0266`). Do **not** launch a full 2,500-step R8 run with this intervention.

## Independent review — 2026-11-11 (Asia/Tokyo)

Preserve R7. Phase 2 failed its practical promotion case; its root cause remains unidentified. The results concern this particular augmentation/packing/continuation recipe and one paired training seed. They neither establish equivalence nor show that paraphrasing generally cannot work.

### Independently checked evidence

- Source, serialized Control/Treatment/selection data, initial checkpoint and endpoint checkpoint hashes match the recorded provenance. The final retrieval source is `8e47f3b1...`; the initial checkpoint is `33a5c245...`. The historical generator audit records the expected R7 digest, but this review did not regenerate all 75,000 historical examples.
- [verify_saved.py](independent-review-20261111/verify_saved.py) independently recomputes counts, differences and exact two-sided McNemar p-values from saved rows. All **864** aggregate/category comparisons match, covering three datasets, four wording cells, real/counterfactual questions and six metrics. This is an independent calculation from saved predictions, not new model inference. Results: [saved-statistics.json](independent-review-20261111/saved-statistics.json).
- Familiar/familiar L3 relationship ranking is **81/110 -> 66/110**, with **0 gains / 15 losses**, exact p=`0.00006103515625`. Exact-prefix accuracy is **62/110 -> 46/110**, **0 gains / 16 losses**, p=`0.000030517578125`. Thus this regression is not solely an option-scoring artifact. Losses span all six predicates and both subject/object queries; they are not confined to employment synonyms.
- This is not replicated as a 13.6-point loss on every dataset: familiar/familiar relationship ranking is **76/100 -> 73/100** on QA-dev and **40/57 -> 42/57** on selection probes. The two familiar-passage L3 cells share items and checkpoints and are not independent replications.
- [consumed-supervision.lisp](independent-review-20261111/consumed-supervision.lisp) reconstructs the exact seed-802 epoch permutation and inspects only the serialized training sequences. The 300 steps consume **9,600 unique sequences**, including **8,006 retrieval** and **1,594 copy** sequences. Actual retrieval supervision is **13,157 -> 12,992 questions** and **26,875 -> 26,605 answer tokens** (minus 1.005% tokens). **209/300** batches have different objective denominators. The full-pool per-category percentages above must not be presented as the exact consumed per-category exposure. No consumed per-category estimate was made here.
- Packing stops at the first question that will not fit, rather than truncating answer tokens. It changes which complete questions are supervised. Separate paraphrase RNG plus resetting to the canonical RNG trajectory preserves subsequent base fact generation; it does not equalize packed question coverage. Aggregate mean sequence length cannot establish equal supervision.
- There is a much larger surface-exposure shift than the small token-count deficit suggests: selection is uniform over concatenated original/extra frames. For teaching passages, the three original frames receive only 3/7 of selections before packing; original teaching subject-question frames receive 3/6 and object-question frames 2/5. These are source-derived sampling probabilities, not measured consumed frequencies. Some other added frames duplicate original training strings, so probabilities must be audited by predicate rather than extrapolated from this example. This supports distribution-shift/dilution as a candidate explanation, not proof of forgetting's mechanism.
- The paired design remains informative about the total effect of the implemented augmentation policy. Changed packing and surface exposure are downstream consequences of that policy; they prevent isolation of linguistic diversity at equal supervision, rather than invalidating the total-effect comparison.
- Relationship paraphrases include `taught/coached/trained under` and `hired/employed/appointed`; ordinary-language equivalence is not guaranteed. Passage and question frames are selected independently. This is an entailment-quality concern, not a measured label-error rate. The test named `predicate-paraphrase-semantics-and-targets-hold` only counts relationship cases and verifies answer presence; it does not independently solve their predicates or argument roles. No definite subject/object indexing bug was established.
- Fresh Adam moments and the learning-rate restart are matched between arms, so their shared reset alone cannot explain Treatment minus Control. Interaction with new inputs/gradient directions remains plausible. Both arms' best logged selection exact-prefix checkpoint is step 0. Gradient norms differ; objective-coefficient shares are not measured gradient contributions.
- Legacy paired confidence intervals use population rather than unbiased sample variance and a normal approximation; exact McNemar calculations are correct. This small SE convention difference does not explain the results. Neither intervals nor p-values include training-seed variability, repeated development choices or template-family dependence.
- Targeted suites rerun successfully: retrieval **10 tests / 5,568 checks**, wording **10 tests / 31,844 checks**, zero failures. Full development verification was not rerun because its wrapper hashes the locked test. This review did not open or hash the locked questions/answers, run new model evaluations, train models or alter engine/Erlang code. All pre-existing work is preserved; this section is an additive annotation.

### Bottleneck and next experiment handoff

Wording transfer is the most clearly isolated performance limitation. It is not the entire remaining gap: R7 QA-dev familiar/familiar option ranking is only 59%; comparison is 30% and reverse lookup 34%. Familiar temporal ranking of 65% is accompanied by only 25% exact-prefix accuracy. Neither a pure language explanation nor a capacity ceiling is established. Training methodology currently limits attribution, rather than being proven the largest performance limitation.

Recommend **one evaluation-only experiment: paired counterfactual fact assignment under fresh unfamiliar wording**, with frozen R7, Phase-2 Control and Phase-2 Treatment checkpoints. Do not automatically run training.

1. Hypothesis: Phase 2's relationship regression extends to a material loss of causal role tracking under genuinely new language, rather than being confined to the familiar L3 realization. This tests behavior, not the neural mechanism or the causal contribution of packing.
2. Control input is a newly authored unfamiliar passage. Treatment input changes only fact-to-argument assignments, keeping its question, lexical inventory, sentence/frame order, options and option order fixed. Reverse all edges of a directed relationship chain and query an interior person, so both worlds have unique but different answers. For extraction/reverse swap two value assignments; for comparison swap numeric assignments; for temporal swap two locations while preserving temporal cues and the query anchor. Derive both keys independently from the resulting facts.
3. Minimum pilot: **400 independent graph pairs**: 200 relationship and 50 each extraction, reverse, comparison and temporal. Use 20 fresh relationship passage/question frame families, 10 graphs per family, balanced across predicate and query direction as closely as integer counts permit. Use at least five fresh families for each secondary category. Author and freeze all families before observing predictions; new seeds of existing dev templates are insufficient. Exclude training/public-dev template duplicates without reading locked templates. Keep both worlds within 128 tokens, with identical total token lengths and equal answer-token lengths; reject/resample pairs symmetrically before inference. Counterbalance gold option positions and which world is designated Control. Do not choose examples conditional on model correctness.
4. Training budget: **zero updates**. Evaluation budget: 400 pairs x 2 worlds x 3 fixed checkpoints = **2,400 prompts**, each scored by summed/mean option likelihood and greedy gold-length prefix. Record bounded stopping separately if computed; prefix exact is not complete generative QA success. No architecture, tokenizer, scoring or decoding changes.
5. Primary endpoint: relationship **both-world exact-prefix correctness**, reduced to one binary outcome per graph per checkpoint. Let delta = Phase-2 Treatment minus Phase-2 Control. Prespecified material deficit: delta <= -0.10 with exact paired two-sided McNemar p < .05 and a negative upper 95% paired confidence bound. Also require a negative upper 95% interval using a prespecified template-family bootstrap (20 families; 10,000 resamples, seed 11011). A lower one-sided 95% bound above -0.10 rules out a deficit of this size on this population; otherwise the result is inconclusive. A small pilot cannot establish equivalence or detect tiny gains.
6. Secondary endpoints: both-world summed/mean option correctness, original/perturbed accuracy, correct answer switching, wrong-role choices, and gold-vs-counterfactual option-margin changes; all five categories separately, plus macro averages. Anchor each checkpoint to R7. Require concordant ranking/sensitivity evidence before calling a prefix deficit a binding deficit; ranking preserved with prefix loss instead points toward answer realization. Report exploratory category results without promotion claims or selective significance thresholds. Never count two worlds as independent observations.
7. Implement additive `lisp/bin/binding-eval.lisp`, evaluation-only fixture/generator support, and `lisp/tests/test-binding-eval.lisp`; reuse the current forward/scoring routines. Do not alter `qagen`'s training frames or the training generator. Validate rendered role semantics independently, answer uniqueness, changed gold, fixed query/options, token budgets and blind generation. Save per-world facts, strings, token IDs, scores, outputs, pair/family IDs, seeds and source/checkpoint/tokenizer hashes to a new, refusal-to-overwrite directory.
8. Existing `check-paired-rows` deliberately rejects different gold keys. Do not weaken it to compare the two worlds. First reduce each graph pair to joint binary outcomes, then compare checkpoints on the same graph IDs. Test the statistics with hand-calculated discordant pairs and validate that a constant-answer predictor cannot pass the both-world metric.
9. If the deficit is supported and role-sensitive ranking deteriorates too, preserve R7 and prioritize a narrowly targeted binding/retention intervention; do not infer a packing cause. If a material deficit is excluded, retire the claim that the familiar L3 regression demonstrates broad unfamiliar-language binding damage. If all checkpoints fail the pairs, the evaluation has exposed a baseline failure without identifying its neural cause. If inconclusive, preserve uncertainty without a training launch or promotion. Frozen evaluation has no capability-regression risk; the new set becomes development evidence after inspection.

No full R8 run, capacity increase, model promotion or claim of progress on the locked 90% target follows from this diagnostic.
