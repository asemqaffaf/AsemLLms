# Pilot 4 Pre-Training Data Audit & Causal Analysis (Stage-2 QA)

Date: 2026-11-13
Branch: `stage2-qa`
Status: Complete (inference and data inspection only; no training executed, no test sets opened)
Changelog: 2026-11-13, reviewer corrections. Section A.4 conclusion 1 no longer calls subword exposure a "direct causal bottleneck"; it now states the `review-checks-20261113` result (gold-only swap effect is largely option-name preference; all-four renames do not change `K = 4` accuracy). No number from the original audit changed. Previous file SHA-256: `1a3270b6847e56b7ef725d14001698a5adb6a37b5a7863d443af61c32344e7c9`.

---

## 0. Evidence Freeze & Quarantine Verification

Before auditing or specifying Pilot 4, all baseline checkpoints, evaluation sets, tokenizers, and prior pilot logs were SHA-256 inventoried in [freeze-manifest.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/freeze-manifest.json):

- `R7` checkpoint: `data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt` (`33a5c245...`, `10,523,393` params, `8` layers, `8` heads, width `256`, `ctx = 128`, `V = 8,192`)
- `Pilot 3 arm-5e-4` step-500: `data/runs/retrieval/p3-learnability/arm-5e-4/last.ckpt` (`6a4fdc5f...`)
- `Pilot 3 arm-1e-3` step-500: `data/runs/retrieval/p3-learnability/arm-1e-3/last.ckpt` (`2041eaf1...`)
- `Pilot 2 treatment` step-500: `data/runs/retrieval/p2-answer-before/treatment/last.ckpt` (`e0ed026c...`)
- Quarantined (never opened, read, or sampled): `confirmation.sexp`, `lisp/eval/qa/test.jsonl`, `lisp/eval/qa/test.lock`, and `lisp/src/qagen.lisp` `:test` pools (`lines 24-73`).

---

## Part A. Entity & BPE Subword Coverage + Causal Disentanglement

### A.1 Full 8,192-Token Vocabulary Audit Across Stage-1 & Stage-2 Distributions

Audit script: [audit_generator.lisp](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/audit_generator.lisp), summarized by [summarize_audit.py](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/summarize_audit.py) from `audit_raw.json`. Counts compare Stage-1 FineWeb (`data/fineweb/stage1/train_00000..00002.bin`), `R7` (`75,000` retrieval + `15,000` random copy-mix sequences), `Pilot 3` (`16,000` answer-before sequences, `0` copy-mix), `broad_names` (`21,897` names), and `D1` entity pools.

| Token Shape | Vocab N | `FW > 0` | `R7 ret all > 0` | `R7 ans > 0` | `R7 ans >= 100` | `P3 all > 0` | `P3 ans > 0` | `P3 ans >= 100` | `Broad Names > 0` | `D1 > 0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `space_lower_alpha` | 4,376 | 4,362 | 3,632 | 3,580 | 0 | 3,611 | 1,999 | 0 | 0 | 0 |
| `bare_lower_alpha` | 2,115 | 2,111 | 366 | 335 | 159 | 351 | 310 | 76 | 230 | 35 |
| `space_title_alpha` | 1,061 | 1,061 | 297 | 279 | 60 | 291 | 268 | 32 | 272 | 32 |
| `raw_non_alnum_byte` | 194 | 134 | 7 | 1 | 0 | 7 | 0 | 0 | 0 | 0 |
| `digits` | 184 | 184 | 75 | 75 | 66 | 0 | 0 | 0 | 0 | 0 |
| `bare_title_alpha` | 141 | 141 | 7 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| `other` | 67 | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `bare_mixed_alpha` | 34 | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `space_upper_alpha` | 19 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `eot` | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |

Tokens present in Stage-1 FineWeb (`FW > 0`) that have **zero** retrieval exposure in `R7` and `P3`:

- `space_title_alpha`: **`764 / 1,061` (`72.0%`)** have `r7_all == 0` (`770` have `p3_all == 0`, `793` have `p3_ans == 0`, `963` have `p3_ans < 10`, **`764`** have `both_all == 0`).
- `bare_lower_alpha`: **`1,745 / 2,111` (`82.7%`)** have `r7_all == 0` (`1,760` have `p3_all == 0`, `1,801` have `p3_ans == 0`, **`1,745`** have `both_all == 0`).
- `space_lower_alpha`: **`743 / 4,362` (`17.0%`)** have `r7_all == 0` (`764` have `p3_all == 0`, `2,370` have `p3_ans == 0`, `4,362` have `p3_ans < 10`, **`743`** have `both_all == 0`).
- `bare_title_alpha`: **`134 / 141` (`95.0%`)** have `both_all == 0`.
- `digits`: `75 / 184` have `r7_ans > 0`, whereas **`184 / 184` (`100%`)** have `p3_all == 0` (`Pilot 3` had zero numeric facts).

### A.2 Why `R7`'s Random `copy-mix` (`15,000` sequences) Failed to Prevent the Entity-Subword Gap

In `R7`, `make-copy-examples` sampled token IDs uniformly from `1..8191`, covering every non-EOT token in the vocabulary (`zero_count = 0`, `min = 38`, `max = 146`, `mean = 91.77` occurrences across `15,000` sequences, representing `5.66%` of total objective loss weight). Despite this uniform token exposure, two structural flaws remain:

1. `copy-mix` draws unstructured sequences of random IDs rather than natural multi-token words embedded in syntactic subject/object slots (`"{P} lives in {T}."`), so prose entity extraction still fails when a real name uses BPE subwords absent from retrieval.
2. When `Pilot 3` removed `copy-mix` completely (`0` copy sequences), the model forgot its general random-token induction circuit on every token outside `P3`'s small `2,577`-token answer vocabulary.

### A.3 Entity Pool Statistics & Coverage Bottleneck

| Pool | Count | Distinct `space_*` / BPE Pieces | Mean Pieces | Vowel-Initial Names | Length Histogram |
| --- | ---: | ---: | ---: | ---: | --- |
| `broad_names` | 21,897 | 502 | 2.840 | 3,049 (13.9%) | `1:188, 2:5400, 3:14047, 4:2262` |
| `broad_towns` | 760 | 143 | 3.480 | 120 (15.8%) | `2:71, 3:326, 4:290, 5:73` |
| `d1_dev_names` | 36 | 67 | 2.111 | 10 (27.8%) | `1:2, 2:29, 3:4, 4:1` |
| `new_pool_names` | 675 | 126 | 2.973 | 0 (0.0%) | `1:2, 2:86, 3:515, 4:72` |
| `qagen_train_names` | 111 | 139 | 2.261 | 29 (26.1%) | `1:7, 2:69, 3:34, 4:1` |
| `synthetic_names` | 4,000 | 396 | 2.874 | 453 (11.3%) | `1:35, 2:947, 3:2507, 4:510, 5:1` |
| `vocab_words` | 3,552 | 3,552 | 1.000 | 744 (20.9%) | `1:3552` |

In `d1-rev-pf` (`n = 200`), real names stratify cleanly by retrieval subword counts (`min(R7_ret_all, P3_all)` across BPE pieces):
- `all_ge_100`: `67 / 200` (e.g., `Sara`, `Mia`, `Kira`, `Pedro`)
- `mid_1_99`: `43 / 200`
- `zero_tok0`: `50 / 200` (e.g., `Edith = [2730 " Ed", 349 "ith"]` with `0` in both `R7` and `P3`; `Ines = [517 " In", 272 "es"]` with `0` answer exposure of `" In"` despite `p3_all` Seeing preposition `"In"`)
- `zero_tail`: `40 / 200` (seen `tok0`, `0` continuation piece `tok1+`, e.g., `Finn`, `Hugo`, `Tess`)

Notice also why `new_pool` cannot cover vowel-initial generalization alone (`0 / 675 = 0.0%` vowel-initial), whereas `synthetic_pool` (once filtered against `broad`, `train`, `dev`, `new-pool`, `selection`, `D1`, and `P3`) supplies `546` clean `seen >= 100` names (`239` vowel-initial).

### A.4 Causal Evidence Disentangling Order, Subword Exposure, and Induction Collapse

All causal tests were run inference-only via [causal_probe.lisp](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/causal_probe.lisp), [analyze_causal.py](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/analyze_causal.py), and canonical `format-prompt` check `r7_exp1_canon.json` (`R7` only).

#### Experiment 1: Minimal-Pair Entity Swap on Frozen `D1 rev-pf` (`n = 157` pairs; context, question, and distractors identical)

Note on `K=4` prompt formatting: the initial multi-checkpoint `causal_probe.lisp` pass omitted `A..D` options from its `K=4` prompt (`score-continuation` after `"The answer is"` without listing options), whereas canonical `eval_probes.lisp` / `qa-analysis.lisp` passes `asem-qagen:format-prompt` (`A..D` options + `"\nThe answer is"`). Canonical `R7` (`inference only`, `r7_exp1_canon.json`) confirms an even stronger `K=4` swap effect on frozen `R7` itself (`p = 6.31e-15` pooled), while open-ended Greedy Exact (`K = 8,192`) and NLL do not depend on `A..D` options and are identical under both scripts:

- **`zero_to_hi100` (`n = 90`: `50` `zero_tok0` + `40` `zero_tail` swapped to `>=100`-seen D1 names):**
  - `R7` (canonical `format-prompt`): `K=4` (chance `25%`) rises **`15 -> 55` (`16.7% -> 61.1%`, `+43/-3`, McNemar `p = 4.62e-10`; `zero_tok0`: `4 -> 36`, `zero_tail`: `11 -> 19`)**; Open Greedy Exact (`K = 8,192`, chance ~0%) rises **`2 -> 21` (`2.2% -> 23.3%`, `+19/-0`, `p = 3.81e-6`)**; Canonical MCQ Greedy Exact rises **`2 -> 25` (`2.2% -> 27.8%`, `+24/-1`, `p = 1.55e-6`)**; Mean NLL drops `3.97 -> 1.39`. *(Initial `no-options` `causal_probe.lisp` `K=4`: `16 -> 37`, `p = 1.94e-5`.)*
  - `P3 arm-5e-4`: Greedy Exact rises **`0 -> 56` (`0.0% -> 62.2%`, `+56/-0`, `p = 2.78e-17`)**; Bounded `0 -> 40` (`p = 1.82e-12`); NLL drops `6.59 -> 0.72` (`no-options` `K=4`: `5 -> 59`, `p = 1.11e-16`).
  - `P3 arm-1e-3`: Greedy Exact rises **`0 -> 58` (`0.0% -> 64.4%`, `+58/-0`, `p = 6.94e-18`)**; NLL drops `7.03 -> 0.54` (`no-options` `K=4`: `6 -> 58`, `p = 4.44e-16`).
  - `P2 treatment`: Greedy Exact `2 -> 37` (`2.2% -> 41.1%`, `p = 5.82e-11`, `no-options` `K=4`: `15 -> 42`, `p = 1.49e-8`).
- **`hi100_to_zero` (`n = 67`: `>=100`-seen D1 names swapped to zero-ret names: `31 -> tok0_zero`, `36 -> tail_zero`):**
  - `R7` (canonical `format-prompt`): `K=4` drops **`34 -> 10` (`50.7% -> 14.9%`, `+3/-27`, `p = 8.43e-6`)**; Open Greedy Exact drops **`19 -> 2` (`28.4% -> 3.0%`, `+1/-18`, `p = 7.63e-5`)**; Canonical MCQ Greedy Exact drops **`21 -> 1` (`31.3% -> 1.5%`, `+0/-20`, `p = 1.91e-6`)**; Mean NLL spikes `1.98 -> 3.96`. *(Initial `no-options` `causal_probe.lisp` `K=4`: `27 -> 9`, `p = 5.34e-4`.)*
  - `P3 arm-5e-4`: Greedy Exact collapses **`34 -> 0` (`50.7% -> 0.0%`, `+0/-34`, `p = 1.16e-10`)** (`17 -> 0` on `-> tok0_zero`, `17 -> 0` on `-> tail_zero`); NLL spikes `1.54 -> 6.35` (`no-options` `K=4`: `42 -> 6`, `p = 2.84e-10`).
  - `P3 arm-1e-3`: Greedy Exact collapses **`41 -> 0` (`61.2% -> 0.0%`, `+0/-41`, `p = 9.09e-13`)** (`20 -> 0` on `-> tok0_zero`, `21 -> 0` on `-> tail_zero`); NLL spikes `1.30 -> 6.80` (`no-options` `K=4`: `45 -> 3`, `p = 5.12e-12`).
- **Pooled `n = 157` (zero-ret vs `>=100`-seen counterpart on frozen `R7`, canonical `format-prompt`):**
  - Canonical `K=4` rises **`25 -> 89` (`15.9% -> 56.7%`, `+70/-6`, `p = 6.31e-15`)**; Open Greedy Exact rises **`4 -> 40` (`2.5% -> 25.5%`, `+37/-1`, `p = 2.84e-10`)**; Canonical MCQ Greedy Exact rises **`3 -> 46` (`1.9% -> 29.3%`, `+44/-1`, `p = 2.61e-12`)**.

*Conclusion 1 (revised 2026-11-13, see changelog):* The gold-only swap effect on frozen `R7` is **largely option-name preference**, and it does not show that subword retrieval exposure limits `D1 rev-pf` `K = 4` accuracy. The inference-only checks in [review-checks-20261113/RESULTS.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/review-checks-20261113/RESULTS.md) (`K = 4`, chance `25%`, unadjusted exact McNemar) found:
- The gold-only swap effect persists in dev wording (`26 -> 67/157`, `+50/-9`, `p = 5.3e-8`; training wording `25 -> 89/157`).
- Renaming **all four** names to training-pool names does not change `K = 4` accuracy in either wording (`D1 rev-pf 67 -> 70/200`, `p = 0.83`; D2 reverse `36 -> 34/94`, `p = 0.88`; D2 compare `31 -> 36/100`, `p = 0.50`).
- `R7` picks `hi100`-named options far more often than zero-named options even when they are wrong (`D1 rev-pf` distractor slots `92/224 = 41.1%` vs `16/264 = 6.1%`), and the swap moves compare items by the same amount (`11 -> 35/83`) although `R7` is near chance on compare.

Exposure does affect generating the name string: Greedy `Exact` rises under the all-four rename (`D1 rev-pf 31 -> 60/200`, `p = 6.4e-4`), and on the Pilot 3 checkpoints a missing continuation token (`-> tail_zero`, where `Tok0` stays `18/36` on `p3_5e4` and `21/36` on `p3_1e3`) still takes full-name Exact Match to `0%`, which is consistent with both onset and tail subwords needing to be copyable. A wrong-slot (distractor-only) swap was not run.

#### Experiment 2: Stratified Random-Token Induction Probe (`n = 50` sequences of length `81`, `half = 40`, `seed = 17`, `K = 8,192`)

| Checkpoint | `std_300_7300` (`loss1->2`, `acc2`) | `p3_r7_hi100_alpha` (`loss1->2`, `acc2`) | `zero_ret_high_fw_alpha` (`loss1->2`, `acc2`) |
| --- | --- | --- | --- |
| `r7` | `15.75 -> 0.18` (**98.0%**) | `18.60 -> 0.34` (**95.9%**) | `15.62 -> 0.34` (**94.9%**) |
| `p3_5e4` | `14.95 -> 3.47` (38.7%) | `18.51 -> 0.65` (**90.7%**) | `15.17 -> 5.67` (**6.1%**) |
| `p3_1e3` | `16.64 -> 7.30` (18.7%) | `19.82 -> 1.56` (81.8%) | `17.91 -> 11.78` (**0.1%**) |
| `p2_trt` | `16.06 -> 0.29` (**95.9%**) | `18.86 -> 0.39` (**94.6%**) | `15.61 -> 0.48` (**92.8%**) |

#### Experiment 3: 1-Sentence Copy (`s1_copy`) vs 4-Sentence L3 Retrieval (`s4_ret`) (`n = 60` per regime, Greedy `K = 8,192`, `K = 4` on `s4_ret`)

| Cue & Regime | `R7` (Exact / Tok0 / NLL / K4) | `P3 5e-4` (Exact / Tok0 / NLL / K4) | `P3 1e-3` (Exact / Tok0 / NLL / K4) | `P2 trt` (Exact / Tok0 / NLL / K4) |
| --- | --- | --- | --- | --- |
| `s1_copy \| seen_hi100` | `59/60 (98.3%)` / `100%` / `0.09` | `60/60 (100%)` / `100%` / `0.00` | `60/60 (100%)` / `100%` / `0.00` | `58/60 (96.7%)` / `100%` / `0.13` |
| `s1_copy \| d1_zero` | `9/60 (15.0%)` / `68.3%` / `1.63` | **`0/60 (0.0%)`** / `53.3%` / `5.11` | **`0/60 (0.0%)`** / `53.3%` / `5.90` | `15/60 (25.0%)` / `68.3%` / `1.69` |
| `s1_copy \| fw_zero` | **`17/60 (28.3%)`** / `41.7%` / `1.75` | **`0/60 (0.0%)`** / `0.0%` / `8.34` | **`0/60 (0.0%)`** / `0.0%` / `11.61` | `10/60 (16.7%)` / `26.7%` / `2.13` |
| `s4_ret \| seen_hi100` | `20/60 (33.3%)` / `48.3%` / `1.30` / K4 `35.0%` | `33/60 (55.0%)` / `63.3%` / `0.74` / K4 `45.0%` | `33/60 (55.0%)` / `65.0%` / `0.82` / K4 `40.0%` | `21/60 (35.0%)` / `48.3%` / `1.40` / K4 `38.3%` |
| `s4_ret \| d1_zero` | `1/60 (1.7%)` / `31.7%` / `3.61` / K4 `5.0%` | `0/60 (0.0%)` / `38.3%` / `6.28` / K4 `0.0%` | `0/60 (0.0%)` / `35.0%` / `6.64` / K4 `0.0%` | `3/60 (5.0%)` / `25.0%` / `3.77` / K4 `1.7%` |
| `s4_ret \| fw_zero` | `0/60 (0.0%)` / `1.7%` / `5.46` / K4 `1.7%` | `0/60 (0.0%)` / `0.0%` / `10.49` / K4 `0.0%` | `0/60 (0.0%)` / `0.0%` / `12.63` / K4 `0.0%` | `0/60 (0.0%)` / `0.0%` / `5.74` / K4 `1.7%` |

*Causal conclusion 2 (Double Dissociation):*
- Dropping `copy-mix` (`Pilot 3`) caused **vocabulary-specific induction collapse** (`ZeroFW` random induction `94.9% -> 6.1% / 0.1%`, `s1_copy fw_zero` `Tok0 41.7% -> 0.0%`, NLL `1.75 -> 8.34 / 11.61`), explaining why `P3` lost `D1 rev-vf` and `D1 ext-pf` whenever unfamiliar subwords appeared.
- Conversely, **random-token `copy-mix` alone (`R7`) is insufficient**: `R7` copies random `fw_zero` tokens at `94.9%`, yet copies 2-token `fw_zero` words inside a trivial 1-sentence prose prompt (`"X lives in Y. Q: Who lives in Y?"`) at only **`28.3%` (`17/60`)** and fails completely (`0/60`) at 4 sentences (`s4_ret`). Thus Pilot 4 must teach **contextual multi-token entity/span reproduction** on missing subwords while retaining an induction health guard.

---

## Part B. Task & Gradient Weight Distribution Across `R7` and Pilots 1-3

### B.1 Complete `R7` Distribution (`75,000` retrieval + `15,000` copy-mix sequences)

- Retrieval questions: `122,334` across `75,000` sequences (`unresolved = 0`; wording augmentation modified `72,872 / 122,334 = 59.6%` of questions).
- Objective token weights (`tot_obj_wt = 264,812`): retrieval answer tokens = `249,812` (`94.34%`); copy-mix effective sequence weight (`15,000 * 1.0`, `360,850` copy targets) = `5.66%`.

| Category | `Op \| Order` | Questions (`% of ret`) | Weight / Ans Tokens (`% total wt`) |
| --- | --- | ---: | ---: |
| `attr` | `extract \| ans_after` | 17,347 (14.18%) | 22,573 (8.52%) |
| `attr` | `extract \| ans_before` | 1,886 (1.54%) | 4,395 (1.66%) |
| `attr` | `reverse \| ans_after` | 1,853 (1.51%) | 5,260 (1.99%) |
| `attr` | `reverse \| ans_before` | 16,329 (13.35%) | 46,174 (17.44%) |
| **Subtotal `attr`** | all ops | **37,415 (30.58%)** | **78,402 (29.61%)** |
| `cmp` | `max \| multi_sent` | 10,582 (8.65%) | 29,787 (11.25%) |
| `cmp` | `min \| multi_sent` | 10,546 (8.62%) | 29,656 (11.20%) |
| **Subtotal `cmp`** | all ops | **21,128 (17.27%)** | **59,443 (22.45%)** |
| `num` | `extract \| ans_after` | **19,980 (16.33%)** | **31,990 (12.08%)** |
| `rel` | `q-obj \| ans_after` | 6,260 (5.12%) | 17,733 (6.70%) |
| `rel` | `q-obj \| ans_before` | 3,507 (2.87%) | 9,895 (3.74%) |
| `rel` | `q-subj \| ans_after` | 3,545 (2.90%) | 9,998 (3.78%) |
| `rel` | `q-subj \| ans_before` | 6,410 (5.24%) | 18,162 (6.86%) |
| **Subtotal `rel`** | all ops | **19,722 (16.12%)** | **55,788 (21.07%)** |
| `tmp` | `after / before / first / last` | **24,089 (19.69%)** | **24,189 (9.13%)** |

Level breakdown in `R7` (`L1/L2/L3` on `attr`, `rel`, `num`):
- `L3|attr|ans_before` is only **`5,759` questions (`4.71%` of `ret_qs`, `5.96%` of objective weight)**, alongside `L2|attr|ans_before` (`8,312`, `6.79%`) and `L1|attr|ans_before` (`4,144`, `3.39%`).
- `num` is `100%` `extract|ans_after` (`L1: 4,178`, `L2: 8,488`, `L3: 7,314`, `0%` `ans_before`).
- `attr|extract|ans_after` uses many single-token lowercase values (`mean = 1.30` tokens per answer vs `2.83` tokens per person name in `reverse`), giving forward extraction only **`8.52%`** of `R7`'s gradient weight despite `14.18%` of questions!

### B.2 Pilot 3 Distribution (`16,000` seqs, `20,337` qs, `52,746` ans tokens) & Forgetting Mapping

- `ans_before`: **`20,267 / 20,337` (`99.66%`)**; `ans_after`: `10` (`0.05%`); `unknown`: `60` (`0.29%`).
- `ans_sent_idx`: uniform across all 4 slots (`5,142 / 5,102 / 5,015 / 5,078`).
- `by_dir_attr`: `pf|job: 4,127`, `pf|pet: 3,962`, `pf|color: 3,456`, `pf|instrument: 2,922`, `pf|town: 2,064`, `pf|unknown: 1`, `vf|instrument: 2,607`, `vf|town: 1,198` (`81.3% pf` / `18.7% vf`).
- Zero exposure (`0%`): `num` (all `digits` tokens = `0`), `rel`, `cmp`, `tmp`, `copy-mix`, and forward `extract|ans_after` / `reverse|ans_after` (`0.05%`). (Inspection of `data/runs/retrieval/p3-learnability/data-train.sexp` `seq0` confirms `Pilot 3` **did** include `A..D` `OPTIONS-BLOCK` (`options=1`) inside `multi-ex` prompts, matching `R7`'s format.)

Direct map to `Pilot 3` (`arm-5e-4`) regressions:
- Dropped `attr/num ans_after` & dropped `copy-mix` -> `d1-ext-pf` fell **`179 -> 76` (`-103`)**, `selection` Extract fell (`R7 = 64`).
- Dropped `vf reverse|ans_after` & dropped `copy-mix` -> `d1-rev-vf` fell **`152 -> 58` (`-94`)**.
- Dropped `rel/cmp/tmp` -> `selection` total dropped **`208 -> 155` (`-53`)**.
- Training `100%` `options=1` `multi-ex` without open-ended prose copy still leaves open-ended generation fragile when unfamiliar tokens appear, while canonical `K=4` (`eval_probes.lisp` with `format-prompt`) measures true 4-way option selection without missing-option confounds.

### B.3 Pilot 1 & Pilot 2 Reference Counts

- `Pilot 1`: `15,000` dense (`attr 7,484, cmp 4,039, num 4,069, rel 4,045, tmp 4,868`, `24,505` qs, `1,005` contrastive pairs) + `3,000` copy seqs.
- `Pilot 2`: `15,000` dense + `3,000` copy seqs (`76` excluded D1 entities); treatment replaced `1,668` `L1 :attr` sequences with `2,121` 4-person `ans_before` questions (`5,409` of `52,084` tokens = `10.4%`, vs `P3`'s `100%` answer-before `20,267` qs).

---

## Part C. Deterministic 75/25 Missing-Subword Partition for Pilot 4

Partitioned in [subword_partition.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/subword_partition.json) by [build_subword_partition.py](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/build_subword_partition.py): for each alphabetic token shape in `(fw_train > 0 & r7_ret_all == 0 & p3_all == 0)`, sorted by `(-fw_train, id)`, `idx % 4 == 3` (`25%`) is quarantined as `eval_heldout` (forbidden in all Pilot 4 training arms) and `idx % 4 != 3` (`75%`) is unlocked exclusively for Arm B/D's contextual entity-copy curriculum (`train_copy`).

| Token Shape (`fw > 0 & r7 == 0 & p3 == 0`) | `train_copy` (`75%`) | `eval_heldout` (`25%`) |
| --- | ---: | ---: |
| `space_title_alpha` | 573 | 191 |
| `bare_lower_alpha` | 1,309 | 436 |
| `space_lower_alpha` | 558 | 185 |
| `bare_title_alpha` | 101 | 33 |

Mined across Stage-1 FineWeb shards `train_00000..00002.bin` (disjoint from `D1`, `dev.jsonl`, `selection.sexp`, and `P3`): `tr_2tok = 608` (`223` vowel-initial), `ev_2tok = 137` (`30` vowel-initial; `110` strictly disjoint & BPE-exact after subtracting `broad`/`new`/`train`/`dev`/`vocab_words`), `tr_3tok = 166`, `ev_3tok = 8`. Accidental unbiased split of `D1` zero-ret tokens: `15` in `train_copy`, `1` in `eval_heldout`.
