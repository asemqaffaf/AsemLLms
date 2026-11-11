# Frozen counterfactual binding evaluation — 2026-11-11

R7 remains the reference. Neither pilot checkpoint is promoted. No R7 checkpoint retraining or R8 training was performed. The locked 1,000-question test was not opened or hashed.

## Implementation

- Evaluator: `lisp/bin/binding-eval.lisp`, with evaluation-only generation in `lisp/src/binding.lisp`. It uses the existing Common Lisp model forward/scoring routines, writes each result with refusal-to-overwrite semantics, and performs no weight updates.
- Independent verifier: `lisp/bin/verify-saved.lisp` and `lisp/src/verify-saved.lisp`. It loads no evaluator/model package. Its finite grammar, answer solver, argmax, correctness, answer switching, exact McNemar, paired intervals and family bootstrap are implemented separately. It also checks stored evaluator accuracy counts against its own calculations. It does not rerun model forward passes or establish that the authored grammar captures every natural-language interpretation.
- Python migration: the prior Python implementation and the Lisp implementation agreed on all 864 historical comparisons, four valid fixtures and seven deliberately corrupted fixtures. Python is archived as historical evidence; the active verification pipeline is Common Lisp. Independent logic within one runtime is not complete implementation independence.

## Frozen design

`protocol.json`, `families.json`, `pairs.json`, `pairs.sexp` and `manifest.json` were written before any model inference. There were **zero eligibility rejections**. Source, tokenizer, fixture and checkpoint hashes were verified before inference; checkpoint hashes were checked again afterward. Predictions are retained in `r7.json`, `control.json` and `treatment.json`.

There are 400 independent graph pairs: 200 relationship pairs across 20 authored language families, and 50 each extraction, reverse lookup, comparison and temporal pairs across five families per category. Each family has ten graphs. Relationship queries are balanced 100 subject / 100 object; world-0 gold positions are exactly 50 at each of A–D. World designation is balanced within each family independently of query direction.

Each pair holds question, words, sentence/frame order, option strings and option order fixed while changing fact assignment and the unique answer. Relationship chains reverse every edge and query an interior person. Attribute/value and numeric assignments swap two bindings; temporal worlds swap two locations while preserving temporal cues and the query anchor. Both prompt lengths and gold answer-token lengths match, and all four options fit the 128-token context. New language families have no exact frame duplicates in training, public development or Phase 2 inventories; no locked template inventory was consulted.

Entity pools are restricted to common BPE lengths. Names and towns come from the existing nontraining probe pools; temporal locations use an evaluation-only inventory excluding known training/public-dev locations. Extraction/reverse cases use town assignments, comparison uses point totals, and temporal cases use four stops. These restrictions make this a controlled diagnostic, not a representative estimate of the existing QA benchmark.

## Independently verified results

The native verifier validated **400 counterfactuals and 2,400 predictions**. All saved evaluator accuracy counts match its independent recomputation. Full results, intervals, per-category metrics, margins, answer switching and R7 comparisons are in `verification.json`.

| Both-world correctness | R7 prefix | Control prefix | Treatment prefix | R7 option ranking | Control ranking | Treatment ranking |
|---|---:|---:|---:|---:|---:|---:|
| Relationship / 200 | 15 (7.5%) | 11 (5.5%) | 14 (7.0%) | 28 (14.0%) | 21 (10.5%) | 26 (13.0%) |
| Extraction / 50 | 2 (4%) | 1 (2%) | 1 (2%) | 12 (24%) | 12 (24%) | 16 (32%) |
| Reverse / 50 | 1 (2%) | 1 (2%) | 2 (4%) | 5 (10%) | 7 (14%) | 6 (12%) |
| Comparison / 50 | 0 | 0 | 0 | 2 (4%) | 1 (2%) | 3 (6%) |
| Temporal / 50 | 0 | 0 | 0 | 6 (12%) | 7 (14%) | 7 (14%) |

Primary contrast, Treatment minus Control relationship joint prefix:

- **+1.5 percentage points**, 9 gains / 6 losses, exact two-sided McNemar **p=.60724**.
- Paired 95% interval **[-2.30, +5.30] points**; 20-family bootstrap interval **[-3.0, +7.0]**.
- Conservative one-sided 95% lower bound **-4.34 points**; family-bootstrap one-sided lower bound **-2.5 points**.
- The frozen classifier returns `material_deficit_excluded_on_this_population`. **This is not useful reassurance.** At the observed Control score of 5.5%, the lowest possible empirical Treatment-minus-Control difference is -5.5 points; an observed ten-point loss cannot occur. The prespecified margin is therefore uninformative at this floor. The historical protocol and machine output are preserved, while the substantive noninferiority interpretation is withdrawn. It establishes neither preserved reasoning nor a promotion case.

The secondary relationship joint-ranking contrast is +2.5 points, p=.44207; it does not demonstrate improvement. R7 changes its ranked answer on 84/200 relationship pairs, but answers both worlds correctly on only 28. Thus switching by itself is not successful role tracking. Treatment has 122 wrong-role choices over 400 relationship worlds, compared with 97 for Control and 104 for R7; these are exploratory counts, not a prespecified significant mechanism test.

## Interpretation

This assay cannot usefully test a ten-point empirical deficit at its observed Control floor. The familiar-L3 regression remains valid evidence; these results neither explain its cause nor approve the intervention. The small positive contrast is inconclusive evidence of improvement.

All three checkpoints show weak both-world consistency. R7 does not reliably answer both factual assignments under these language families. That establishes a behavioral limitation in this controlled setting. It does not establish whether the cause is lexical interpretation, role binding, answer realization, model capacity or pretraining, and it does not prove that every success is a familiar-pattern heuristic. Positive margins and some correct switches indicate that the model sometimes responds to changed facts, without robust joint correctness.

These paired percentages must not be compared directly with single-world QA accuracy, nor should a 1/16 chance rate be assumed: the two worlds, options and model predictions are dependent. Option ranking is separate from gold-length prefix production, and neither measures reliable complete answer generation with termination.

Preserve R7 and both pilots. No automatic training, architecture change, checkpoint promotion or locked-test evaluation follows. Future mechanistic work should account for this assay's low joint-performance floor before interpreting differences between checkpoints.

## Reproduction and checks

From the repository root, prepare a new directory and keep source bytes fixed through inference and verification:

```sh
sbcl --script lisp/bin/binding-eval.lisp mode=prepare bundle=NEW-DIR
TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/binding-eval.lisp mode=eval bundle=NEW-DIR label=r7 workers=4
TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/binding-eval.lisp mode=eval bundle=NEW-DIR label=control workers=4
TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/binding-eval.lisp mode=eval bundle=NEW-DIR label=treatment workers=4
sbcl --script lisp/bin/verify-saved.lisp mode=binding bundle=NEW-DIR out=NEW-DIR/verification.json
```

Checks passed: binding **9 tests / 2,442 checks**, independent verifier **4 / 8**, stable retrieval **7 / 1,299**, explicitly isolated Phase 2 **3 / 4,269**, stable wording **10 / 31,844**; total **33 tests / 39,862 checks**, zero failures. Existing toy-model unit tests are separate from the frozen-checkpoint experiment. Full development verification was not run because its wrapper hashes the locked test. The model-free 75,000-sequence stable R7 generator audit also reproduces the original serialization digest exactly.

The historical Phase 2 implementation and modified scripts/tests are preserved byte for byte under `lisp/experimental/phase2/historical/`. Stable training no longer imports them. Infrastructure and original results are now committed on `stage2-qa` in isolated commits; unrelated Erlang work is excluded. No merge or push was performed.

## Post-hoc error decomposition

The [saved-prediction analysis](posthoc-20261111/README.md) examines all 2,400 existing predictions, without new inference or training. It separates one-world successes from correct switches, graph-derived wrong directions from other errors, and candidate-ranking success from unlisted generated prefixes. Its descriptive family tables do not establish causal wording effects: graphs/entities differ between families, and each family contains only ten pairs.
