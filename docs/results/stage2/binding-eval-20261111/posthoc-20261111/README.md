# Saved-prediction error analysis — 2026-11-11

This is a post-hoc descriptive analysis of the existing 2,400 predictions. There were no new forward passes or training updates. R7 remains the reference; neither pilot is promoted. Original fixtures, predictions, protocol, verification output and checkpoints are unchanged.

## The floor effect invalidates substantive reassurance

Control's observed joint relationship-prefix score is 11/200 = 5.5%. Even a Treatment score of zero would yield only a -5.5-point empirical difference, making a ten-point observed deficit impossible. The original absolute margin cannot serve as a useful noninferiority gate here. The confidence bounds concern population quantities estimated from these observations, but do not make this floor-limited design informative about preservation of useful capability.

The frozen classifier string and registered protocol remain historical records. Their substantive interpretation as reassurance is withdrawn. The +1.5-point contrast, p=.60724, demonstrates no improvement. Research progress and model capability progress remain separate.

## 1. Individual correctness versus success after reassignment

R7 results, with every pair counted once:

| Task | Pairs | Ranking: both correct | World 0 only | World 1 only | Neither | Prefix: both correct | World 0 only | World 1 only | Neither |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Relationship | 200 | 28 | 52 | 52 | 68 | 15 | 28 | 28 | 129 |
| Extraction | 50 | 12 | 7 | 15 | 16 | 2 | 3 | 1 | 44 |
| Reverse | 50 | 5 | 13 | 13 | 19 | 1 | 3 | 5 | 41 |
| Comparison | 50 | 2 | 14 | 12 | 22 | 0 | 10 | 5 | 35 |
| Temporal | 50 | 6 | 9 | 3 | 32 | 0 | 1 | 1 | 48 |

For relationships, R7 ranks the correct option in 160/400 individual worlds (40%). Of 80 pairs correct in world 0, 52 fail in world 1; the reverse conditioning gives the same counts. Among the 104 pairs with exactly one correct ranking, **76 retain the same option despite the changed key**, while 28 switch to a different wrong option. There are also 40 pairs wrong in both worlds that retain a fixed choice, and 28 wrong in both worlds that change choice. Only 28/84 observed ranking switches are correct in both worlds.

The two worlds are counterbalanced, not a chronological before/after sequence. Conditioning on one world being correct selects examples. These counts demonstrate weak joint correctness and frequent fixed choices; they do not by themselves estimate a causal accuracy penalty from changing facts.

An additional limitation matters: relationship single-world ranking is 40% in each world. Its observed joint ranking is 14%, compared with the **descriptive** marginal product 16%. Prefix single-world accuracy is 21.5% in each world, with 7.5% joint success versus a marginal product of 4.62%. Thus low joint accuracy alone cannot establish excess disruption beyond low individual accuracy. The product is not a chance baseline or an independence assumption; the prefix outcomes actually show positive association on these examples.

Zero joint comparison/temporal prefixes does not mean zero individual correct prefixes: R7 produces **15/100** correct comparison prefixes and **2/100** temporal prefixes, distributed across different pairs.

## 2. Wrong relationship direction is a substantial subset

For each relationship world, the analysis independently resolves the opposite query on its directed graph. It verifies that this answer equals the paired world's gold answer before assigning an opposite-direction error. Copying the queried person and choosing the remaining entity are separate categories.

| Checkpoint | Relationship worlds | Correct ranking | Opposite direction | Query anchor | Other entity |
|---|---:|---:|---:|---:|---:|
| R7 | 400 | 160 | 104 | 70 | 66 |
| Control | 400 | 149 | 97 | 83 | 71 |
| Treatment | 400 | 155 | 122 | 67 | 56 |

R7's opposite-direction choices account for **104/240 ranking errors (43.3%)**. They are not the sole failure mechanism. No random-choice null or post-hoc significance claim is assigned to this partition.

| Query direction | R7 correct / 200 | R7 opposite / 200 | Control correct / 200 | Control opposite / 200 | Treatment correct / 200 | Treatment opposite / 200 |
|---|---:|---:|---:|---:|---:|---:|
| Incoming: who acted on the named person? | 87 | 45 | 74 | 46 | 90 | 49 |
| Outgoing: whom did the named person act on? | 73 | 59 | 75 | 51 | 65 | 73 |

Treatment's outgoing queries show more direction errors on this saved set. This is an exploratory description, not evidence that Phase 2 caused a general directional deficit or an explanation of its familiar-L3 regression.

## 3. Ranking and answer production dissociate

All counts below concern individual R7 worlds. "Unlisted" means the greedy prefix token sequence matches none of the four option continuations; it does not automatically mean spelling failure, hallucination or broken copying.

| Task | Worlds | Ranking correct | Prefix correct | Both correct | Ranking correct, prefix wrong | Wrong prefix unlisted in that subset | First-token error in that subset |
|---|---:|---:|---:|---:|---:|---:|---:|
| Relationship | 400 | 160 | 86 | 83 | 77 | 76 | 70 |
| Extraction | 100 | 46 | 8 | 8 | 38 | 38 | 24 |
| Reverse | 100 | 36 | 10 | 10 | 26 | 26 | 22 |
| Comparison | 100 | 30 | 15 | 14 | 16 | 16 | 13 |
| Temporal | 100 | 24 | 2 | 2 | 22 | 22 | 21 |
| Total (unequal category weights) | 800 | 296 | 121 | 117 | 179 | 178 | 150 |

Across the 296 ranking-correct worlds, 179 fail the prefix metric. Of those, **178 are unlisted prefixes and 150 diverge on the first answer token**; the remaining 29 diverge later. None of these 179 is a shortened prefix. This is inconsistent with explaining the gap solely as final-token name corruption or answer termination. It is consistent with poor answer-start grounding under these prompts, without establishing a particular neural or training cause.

Examples from the saved R7 output, selected only to illustrate that error class:

| World ID | Correctly ranked gold | Generated prefix |
|---|---|---|
| `binding-relation-00-04/w0` | `Lonel` | ` V Gous` |
| `binding-extract-0-01/w1` | `Wulmoss` | ` townen rese` |
| `binding-reverse-0-07/w0` | `Husol` | ` The town R` |
| `binding-temporal-0-01/w0` | `monastery` | ` Flieno` |

The model can prefer the correct candidate over three alternatives while favoring a different unrestricted token sequence. Candidate-relative evidence is not proof of generative reasoning. Conversely, 504/800 R7 rankings are also wrong, so a generation-only explanation cannot explain all failures. There are four worlds with a correct prefix but wrong ranking; the metrics are not nested.

The relationship ranking-correct/prefix-wrong counts are R7 77/160, Control 64/149 and Treatment 60/155. These descriptive differences do not constitute a new, prespecified treatment success test.

## 4. Language-family patterns

Each cell is **joint ranking / joint prefix successes out of 10 pairs**. Family IDs map to exact passage/question frames in `../families.json` and `error-analysis.json`. All families are reported; there is no selection of a favorable checkpoint.

| Relationship family | Predicate | R7 | Control | Treatment |
|---|---|---:|---:|---:|
| 00 | teach | 2/1 | 1/1 | 1/0 |
| 01 | teach | 1/0 | 1/0 | 1/0 |
| 02 | teach | 2/0 | 0/0 | 4/4 |
| 03 | teach | 1/0 | 0/0 | 1/0 |
| 04 | hire | 3/2 | 1/0 | 1/0 |
| 05 | hire | 0/0 | 0/0 | 1/0 |
| 06 | hire | 0/0 | 0/0 | 2/1 |
| 07 | hire | 0/0 | 2/0 | 2/0 |
| 08 | call | 3/2 | 2/1 | 2/0 |
| 09 | call | 1/1 | 0/0 | 1/1 |
| 10 | call | 0/0 | 1/0 | 1/0 |
| 11 | thank | 4/2 | 4/2 | 1/1 |
| 12 | thank | 0/0 | 0/0 | 0/0 |
| 13 | thank | 1/0 | 0/0 | 1/1 |
| 14 | visit | 5/4 | 5/5 | 3/3 |
| 15 | visit | 0/0 | 0/0 | 1/0 |
| 16 | visit | 0/0 | 0/0 | 0/0 |
| 17 | help | 4/3 | 3/2 | 2/3 |
| 18 | help | 1/0 | 1/0 | 1/0 |
| 19 | help | 0/0 | 0/0 | 0/0 |

Relationship families 12 ("It was {B} whom {A} thanked."), 16 ("The visitor who went to see {B} was {A}.") and 19 ("It was {B} whom {A} assisted.") have zero joint rankings in all three checkpoints. Family 14 ("{A} made a visit to {B}.") is relatively stronger. This pattern is compatible with difficulty on some argument structures, but **different families contain different graphs/entities**. It does not isolate syntax as the cause; ten pairs per family do not establish reliable population rankings.

| Other family | R7 | Control | Treatment |
|---|---:|---:|---:|
| extract-0 | 0/0 | 0/0 | 4/0 |
| extract-1 | 0/0 | 0/0 | 0/0 |
| extract-2 | 8/2 | 7/1 | 8/1 |
| extract-3 | 3/0 | 2/0 | 3/0 |
| extract-4 | 1/0 | 3/0 | 1/0 |
| reverse-0 | 1/0 | 2/0 | 0/0 |
| reverse-1 | 3/1 | 3/1 | 6/2 |
| reverse-2 | 0/0 | 0/0 | 0/0 |
| reverse-3 | 0/0 | 1/0 | 0/0 |
| reverse-4 | 1/0 | 1/0 | 0/0 |
| compare-0 | 0/0 | 0/0 | 0/0 |
| compare-1 | 1/0 | 0/0 | 0/0 |
| compare-2 | 0/0 | 0/0 | 2/0 |
| compare-3 | 0/0 | 0/0 | 0/0 |
| compare-4 | 1/0 | 1/0 | 1/0 |
| temporal-0 | 0/0 | 0/0 | 0/0 |
| temporal-1 | 2/0 | 2/0 | 3/0 |
| temporal-2 | 1/0 | 2/0 | 1/0 |
| temporal-3 | 1/0 | 1/0 | 0/0 |
| temporal-4 | 2/0 | 2/0 | 3/0 |

Additional zero-joint-ranking families across all checkpoints are extract-1, reverse-2, compare-0, compare-3 and temporal-0. Extraction family 2 illustrates a large generation gap despite relatively strong ranking, rather than successful unconstrained extraction.

## Scope and implementation

The strongest supported distinction is between **candidate-relative fact selection** and **answer-prefix production**. Wrong-direction choices and fixed responses also identify specific relationship behaviors worth investigating. The data do not identify a unique internal cause or justify an R8 run. R7 remains intact; no capacity increase, architecture change or broader training intervention follows.

`lisp/bin/binding-errors.lisp` loads only JSON/SHA utilities, the independent verifier and the new analysis module. It reruns model-free verification, compares all frozen results with their saved verification, and checks that every new decomposition adds back to the original counts. It writes per-world classifications for all 2,400 predictions and per-pair outcomes for all 1,200 checkpoint/pair combinations. Graph-derived role labels, first-token/later-token divergence, denominators, family inventories, and input/source hashes are saved in `error-analysis.json`.

Reproduce to an unused file:

```sh
sbcl --script lisp/bin/binding-errors.lisp bundle=docs/results/stage2/binding-eval-20261111 out=NEW-ANALYSIS.json
sbcl --noinform --non-interactive --load lisp/tests/test-binding-errors.lisp
```

The new analysis tests passed **8 tests / 49 checks**. Model-free binding, verifier, wording and isolated Phase 2 tests were rerun successfully: **26 tests / 38,563 checks**. Together this turn ran **34 tests / 38,612 checks**, with zero failures. No model inference, training, locked-test access, merge or push occurred.

Validated infrastructure and preserved results were committed separately on `stage2-qa`: `a82c248` isolates Phase 2, `5bde8f3` adds evaluation/verification, and `32e4372` archives research artifacts. The post-hoc analysis and interpretation correction are a separate commit. Unrelated `tinyllm/*` work is excluded.
