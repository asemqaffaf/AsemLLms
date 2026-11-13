# Pilot 4 Results & Required Gate Report (2026-11-13)

**Verdict: STOP STAGE-2 RETRIEVAL LINE (No 5th pilot per User Stop Rule).** No arm satisfied `G1` + `G2c` + all retention gates (`G3-G7` and `Aggregate`), although **Arm D (`+Copy, +Replay`) passed every retention gate (`553/1000 = 55.3%`, `+0.0 pp` vs frozen `R7`)** and **Arms B and D solved trained-subword entity recombination (`97.5%` / `96.2%` `s1_recomb_tr` Exact)**. Frozen `D1 200`, `selection.sexp`, `probe` suites, `dev.jsonl` (not scored on Pilot 2 to 4 checkpoints; scored on `R7` and Pilot 1 checkpoints), `confirmation.sexp` (`unopened`), and both locked test files (`unopened`) remain intact with **zero git commits**.

Revised 2026-11-13 after reviewer checks (wording only; no number, gate or verdict changed). Every arm is a single seed, and every p-value in this file is unadjusted across all cells, arms and probes. Arm D kept aggregate retention but lost held-out token induction (`95.3% -> 9.4%`) and person-first extraction on `D1` (`d1-ext-pf K4 179 -> 159/200`, `p = 0.0078`), and failed `G1a`, `G1c`, `G2c` and `G2d`. See [FINAL_REPORT.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/FINAL_REPORT.md) Sections 3 to 5 and [review-checks-20261113/RESULTS.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/review-checks-20261113/RESULTS.md).

Per explicit report-ordering requirements, Section 1 reports Pilot 3 against its pre-registered Decision Table `A-E`, Section 2 reports the frozen `R7` Experiment 1 minimal-pair swap rerun, and Section 3 onwards report Pilot 4.

---

## 1. Pilot 3 Against Pre-Registered Decision Table (`A-E`)

Reference spec: [pilot3/protocol.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot3/protocol.md) and [pilot3/RESULTS.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot3/RESULTS.md). Initialized from unchanged `R7` (`33a5c245...`, `10,523,393` params); trained `500` steps (`16,000` answer-before sequences, `0` copy sequences) at `5e-4` and `1e-3`.

- **Pre-registered Decision Verdict: Outcome E (Anything else - ambiguous; stopped after 500 steps, no 2,500-step run launched):**
  - Outcome A required `fresh exact >= 75%`, `seen - fresh <= 10 pp`, `D1 rev-pf >= 120/200` (not met: `63.6%` fresh exact, `83/200` D1).
  - Outcome B required `fresh exact >= 75%`, `D1 rev-pf < 120/200` (not met: `63.6% < 75%`).
  - Outcome C required `seen exact >= 70%` and `seen - fresh > 20 pp` (not met: `seen - fresh = +13.0 pp` at `1e-3` and `+16.0 pp` at `5e-4`).
  - Outcome D required `seen exact < 60%` in both arms after all steps and flat loss over the last `250` steps (ruled out cleanly: `seen exact = 79.6%` / `76.6%`, and loss fell steeply across steps `250 -> 500`).

### 1.1 `Seen` and `Fresh` Exact & `K = 4` Curves Across Steps `0 -> 250 -> 500`

Scoring: `Exact` (`gold_length_greedy_prefix_exact`, `K = 8,192` per token, chance `~0%`) and `K = 4` (`summed_option_prediction`, chance `25.0%`), `n = 500` items per probe (`404` reverse, `96` extract).

| Model / Step | Fresh `Exact` (`K=8,192`) | Fresh `K=4` (`25%` ch.) | Fresh `dev_loss` | `Seen Exact` (`K=8,192`) | `Seen K=4` (`25%` ch.) | `Seen - Fresh` (`Exact` / `K4`) | Train loss (`step`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R7` step `0` | `215/500 (43.0%)` | `249/500 (49.8%)` | `1.157` | `227/500 (45.4%)` | `250/500 (50.0%)` | `+2.4 / +0.2 pp` | - |
| `arm-5e-4` step `250` | `290/500 (58.0%)` | `317/500 (63.4%)` | `0.920` | `316/500 (63.2%)` | `339/500 (67.8%)` | `+5.2 / +4.4 pp` | `0.767` |
| `arm-5e-4` step `500` | `318/500 (63.6%)` | `343/500 (68.6%)` | `0.679` | `398/500 (79.6%)` | `407/500 (81.4%)` | `+16.0 / +12.8 pp` | `0.560` |
| `arm-1e-3` step `250` | `242/500 (48.4%)` | `291/500 (58.2%)` | `1.172` | `295/500 (59.0%)` | `329/500 (65.8%)` | `+10.6 / +7.6 pp` | `1.031` |
| `arm-1e-3` step `500` | `318/500 (63.6%)` | `349/500 (69.8%)` | `0.727` | `383/500 (76.6%)` | `395/500 (79.0%)` | `+13.0 / +9.2 pp` | `0.644` |

- **Were curves still rising at step `500`?** **Yes.** On `arm-5e-4`, `train_loss` dropped `-0.207` across steps `250 -> 500` (`0.767 -> 0.560`), `Fresh dev_loss` dropped evenly (`-0.237` from `0 -> 250` and `-0.241` from `250 -> 500`), and `Fresh Exact` climbed `+15.0 pp` then `+5.6 pp`. On `arm-1e-3`, after recovering from an early `1e-3` spike (`0.981 -> 1.301`), `train_loss` fell `-0.387` (`1.031 -> 0.644`), `dev_loss` plunged `-0.446` (`1.172 -> 0.727`), and `Fresh Exact` rose `+15.2 pp` in steps `250 -> 500`. Neither arm had plateaued at step `500`.

### 1.2 Separate-Sentence Filler Preflight (`R7`, `n = 200` paired, `K = 4`, chance `25%`)

Using unmodified `lisp/bin/qa-analysis.lisp` (`controls=0 paired=0`) on the `200` `D1` fact sets with unrelated filler sentences (`["It is cold.", "Water ran.", "Day began.", "The sun set."]`, max `<= 127` tokens):

| Condition | Reverse pf (`ans-first`) | Reverse vf (`ans-after`) | Reverse vf vs pf (`p`) | Extract pf (`ans-after`) | Extract vf (`ans-first`) | Extract vf vs pf (`p`) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `D1` baseline (no filler) | `67/200 (33.5%)` | `152/200 (76.0%)` | `+93/-8 (1.74e-19)` | `179/200 (89.5%)` | `72/200 (36.0%)` | `+1/-108 (3.39e-31)` |
| `D4` in-sentence filler | `73/200 (36.5%)` | `147/200 (73.5%)` | `+83/-9 (3.92e-16)` | `174/200 (87.0%)` | `77/200 (38.5%)` | `+2/-99 (4.06e-27)` |
| `s3` (D1 + post-fact filler) | `59/200 (29.5%, p=0.215)` | `127/200 (63.5%, p=0.00017)` | `+77/-9 (1.34e-14)` | `159/200 (79.5%, p=0.00018)` | `68/200 (34.0%, p=0.618)` | `+4/-95 (1.24e-23)` |
| `s1` (2-sent split, no filler) | `56/200 (28.0%)` | `50/200 (25.0%, p=1.20e-24)` | `+18/-24 (0.441)` | `68/200 (34.0%)` | `37/200 (18.5%)` | `+8/-39 (5.54e-6)` |
| `s2` (2-sent split + filler) | `52/200 (26.0%, vs s1 p=0.572)` | `55/200 (27.5%, vs s1 p=0.442)` | `+27/-24 (0.780)` | `58/200 (29.0%, vs s1 p=0.253)` | `46/200 (23.0%, vs s1 p=0.243)` | `+19/-31 (0.119)` |

Inserting a full unrelated sentence between intact `D1` fact sentences (`s3`) lowered answer-after accuracy by `10.0-12.5 pp`, but answer-after still beat answer-first decisively (`127 vs 59`, `p = 1.34e-14`; `159 vs 68`, `p = 1.24e-23`). Two-sentence anaphoric splitting (`s1`) collapsed `R7` to chance (`25%-34%`) even with no filler. Order, not sentence adjacency, drives the asymmetry.

---

## 2. Frozen `R7` Experiment 1 Minimal-Pair Entity Swap (`n = 157`, Inference Only)

To verify whether `R7`'s zero-count subword weakness reflects a genuine `R7` subword bottleneck rather than only Pilot 3's lost induction circuit, all `157` minimal-pair swaps in [causal_probe_items.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/causal_probe_items.json) were evaluated on **frozen `R7`** (`33a5c245...`) using canonical formatting ([r7_exp1_canon.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/r7_exp1_canon.json)):

| Minimal-Pair Subset (`n`) | Swap Direction | Canonical `K=4` (`25%` ch.) | Greedy `Exact` (`K=8,192`) | MCQ `Exact` | Mean Target `NLL` |
| --- | --- | --- | --- | --- | --- |
| `zero_to_hi100` (`n = 90`) | `0-count` -> `>=100` subword | `15 -> 55/90 (16.7% -> 61.1%, +43/-3, p=4.622e-10)` | `2 -> 21/90 (+19/-0, p=3.815e-06)` | `2 -> 25/90 (+24/-1, p=1.550e-06)` | `3.97 -> 1.39` |
| `hi100_to_zero` (`n = 67`) | `>=100` -> `0-count` subword | `34 -> 10/67 (50.7% -> 14.9%, +3/-27, p=8.430e-06)` | `19 -> 2/67 (+1/-18, p=7.629e-05)` | `21 -> 1/67 (+0/-20, p=1.907e-06)` | `1.98 -> 3.96` |
| **Pooled (`n = 157`)** | replacing `0`-count w/ `>=100` | **`25 -> 89/157 (+70/-6, p = 6.31e-15)`** | **`4 -> 40/157 (+37/-1, p = 2.84e-10)`** | **`3 -> 46/157 (+44/-1, p = 2.61e-12)`** | - |

Frozen `R7` shows a massive bidirectional swap effect (`+44.4 pp` when replacing a zero-ret `D1` name with a `>=100`-count synthetic name; `-35.8 pp` in the reverse swap). Because this gap exists inside `R7` itself (even with `R7`'s `95.3%` `eval_heldout_alpha` induction intact), Pilot 4's contextual subword-composition copy curriculum stayed unchanged.

Reviewer check added 2026-11-13 (inference only, `K = 4`, chance `25%`): this swap changes the gold name alone. The effect persists in dev wording (`26 -> 67/157`, `+50/-9`, `p = 5.3e-8`), but renaming all four names to training-pool names leaves `K = 4` unchanged (`D1 rev-pf 67 -> 70/200`, `p = 0.83`), and `R7` picks `hi100`-named options far more often than zero-named ones even as distractors (`41.1%` vs `6.1%` of distractor slots). The `K = 4` swap effect is therefore largely option-name preference and does not show that subword exposure limits answer-before `K = 4` accuracy.

---

## 3. Pilot 4 Pre-Training Verification & Fixed-Dose Schedule

All pre-training checks passed before step `1` of any arm ([protocol.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/protocol.md), [eval-spec.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/eval-spec.md), [verify_pretrain.py](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/verify_pretrain.py)):

- **Fixed `ans_before` Dose (`6,080` sequences identical across all 4 arms):**
  - **Arm A (`190` steps, `6,080` seqs, `137.8 s`):** `ans_before` only (`No Copy, No Replay`).
  - **Arm B (`270` steps, `8,640` seqs, `192.9 s`):** same `6,080` `ans_before` + `2,560` copy (`1,000` FineWeb prose copy + `960` composed `train_copy` subword prose copy guaranteeing min count `>= 2` across every token in all 4 classes + `600` `:copy` induction guard).
  - **Arm C (`420` steps, `13,440` seqs, `290.9 s`):** same `6,080` `ans_before` + `7,360` multi-task replay (`1,840 ext-attr`, `1,251 ext-num`, `1,089 vf-rev`, `1,170 rel`, `1,001 cmp`, `1,009 tmp`).
  - **Arm D (`500` steps, `16,000` seqs, `352.7 s`):** same `6,080` `ans_before` + same `2,560` copy + same `7,360` replay.
- **Schedule:** Identical `R7` initialization (`33a5c245...`), fresh AdamW (`beta1 = 0.9, beta2 = 0.95, wd = 0.0, clip = 1.0`), `batch = 32`, `lr = 5e-4`, linear warmup `50` steps, cosine decay (`min_lr_frac = 0.1`) scaled to each arm's step count.
- **Zero-Leakage Audit:** `0` `eval_heldout_ids` (`845` tokens) across `A, B, C, D` (including induction guard); `0` `guard_forbid_ids` (`81` single-token `D1` + probe IDs) in `B, D`; `0` whole-word entity leaks across all `1,600` probe entities, all `D1` entities, `excluded-extra`, `reserved_recomb_tr_words`, `fw_ev_only`, and `heldout-strings` across every component (`ans_before`, `copy_prose`, `copy_guard`, `replay`, `internal-dev-200`).
- **Pre-training hash verification:** [data-audit.json](file:///Users/asemqaffaf/AsemLLMs/data/runs/retrieval/p4-retention/data-audit.json) matched every dataset, script, `protocol.md` (`9c57ee33...`), `eval-spec.md`, and `R7` init SHA-256 on disk before `mode=train`.

---

## 4. Training Loss Trajectory & Intermediate `step-00250` Diagnostics

Internal `200`-item sanity dev curves (`K = 4`, disjoint from all probes):
- **Arm A (`190` steps):** `dev_loss 0.893 -> 0.847 (50) -> 0.865 (100) -> 0.750 (150) -> 0.720 (190)`; `dev_acc 0.570 -> 0.670`; `train_loss 0.664 (50) -> 0.497 (190)`.
- **Arm B (`270` steps):** `dev_loss 0.893 -> 0.824 (50) -> 0.846 (100) -> 0.715 (150) -> 0.556 (250) -> 0.550 (270)`; `dev_acc 0.570 -> 0.660`.
- **Arm C (`420` steps):** `dev_loss 0.893 -> 1.036 (50) -> 0.923 (100) -> 0.880 (250) -> 0.699 (350) -> 0.669 (420)`; `dev_acc 0.570 -> 0.580`.
- **Arm D (`500` steps):** `dev_loss 0.893 -> 0.964 (50) -> 1.104 (100) -> 0.924 (250) -> 0.652 (400) -> 0.664 (500)`; `dev_acc 0.570 -> 0.610`; `train_loss 0.386 (500)`.

Intermediate `step-00250` probe snapshots on `B, C, D` (forgetting happens early before cosine tail recovers retention in `D`):

| Checkpoint | `fresh_ans_before` (`K4` / `Ex`, `/300`) | `s1_recomb_tr` (`Ex`, `/80`) | `s1_heldout_ev` (`Ex`, `/80`) | `Agg Retention` (`K4`, `/1000`: `ext / rel / rev / cmp / tmp`) |
| --- | ---: | ---: | ---: | ---: |
| `arm_b_250` | `126 (42.0%) / 64 (21.3%)` | `78 (97.5%)` | `3 (3.8%)` | `478` (`147 / 104 / 93 / 60 / 74`) |
| `arm_b` (`270`) | `130 (43.3%) / 61 (20.3%)` | `78 (97.5%)` | `5 (6.2%)` | `491` (`150 / 105 / 98 / 62 / 76`) |
| `arm_c_250` | `115 (38.3%) / 39 (13.0%)` | `0 (0.0%)` | `0 (0.0%)` | `508` (`137 / 124 / 109 / 56 / 82`) |
| `arm_c` (`420`) | `122 (40.7%) / 43 (14.3%)` | `0 (0.0%)` | `0 (0.0%)` | `522` (`143 / 121 / 113 / 63 / 82`) |
| `arm_d_250` | `100 (33.3%) / 42 (14.0%)` | `71 (88.8%)` | `0 (0.0%)` | `501` (`151 / 120 / 98 / 63 / 69`) |
| `arm_d` (`500`) | `138 (46.0%) / 70 (23.3%)` | `77 (96.2%)` | `0 (0.0%)` | `553` (`158 / 127 / 119 / 71 / 78`) |

---

## 5. Pre-Registered Binding Gate Matrix (`p4_summary.json`)

All absolute floors are strictly binding (`p > 0.05` vs `R7` alone does not count as passing). `K = 4` chance is `25.0%`; Greedy `Exact` (`K = 8,192`) chance is `~0%`. All `p`-values are two-sided exact paired McNemar binomial tests vs frozen `R7`.

| Gate (`n`) | Binding Floor / Rule | `R7` Base | `P3 5e-4` | **Arm A** (`190`) | **Arm B** (`270`) | **Arm C** (`420`) | **Arm D** (`500`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **G1a** `fresh_ans_before K4` (`300`) | `>=140 (46.7%)` & `p<=0.05` | `116 (38.7%)` | `129 (43.0%)` FAIL | `118 (39.3%, p=0.912)` FAIL | `130 (43.3%, p=0.184)` FAIL | `122 (40.7%, p=0.614)` FAIL | `138 (46.0%, p=0.0334)` **FAIL** (`<140`) |
| **G1b** `fresh_ans_before Ex` (`300`) | `>=48 (16.0%)` & `p<=0.05` | `30 (10.0%)` | `48 (16.0%)` PASS | `52 (17.3%, p=1.13e-4)` PASS | `61 (20.3%, p=9.26e-6)` PASS | `43 (14.3%, p=0.066)` FAIL | **`70 (23.3%, p=8.96e-8)` PASS** |
| **G1c** `pf\|seen100` (`50`) / `pf\|recomb` (`50`) | `>=20` AND `>=5` | `12` / `0` | `22` / `0` FAIL | `18` / `0` FAIL | `17` / **`7`** FAIL | `15` / `0` FAIL | `18` / **`9`** FAIL |
| **G2a** `s1_seen_synth` (`80` / `vf 40`) | `>=70/80` & `vf>=34/40` | `73` / `36` | `59` / `19` FAIL | `69` / `29` FAIL | **`77` / `38` PASS** | `65` / `27` FAIL | **`80 (100%)` / `40` PASS** |
| **G2b** `s1_recomb_tr` (`80`) | `>=44 (55%)` & `p<=0.01` | `26 (32.5%)` | `0 (0.0%)` FAIL | `0 (0.0%)` FAIL | **`78 (97.5%, p=6.11e-15)` PASS** | `0 (0.0%)` FAIL | **`77 (96.2%, p=1.20e-14)` PASS** |
| **G2c** `s1_heldout_ev` (`80`) + Ind (`>=90%`) | `>=22` & `ind_ev>=90` & `std>=90` | `25` (`95.3/98.0`) | `0` (`5.9/38.7`) FAIL | `0` (`49.3/64.5`) FAIL | `5` (`23.5/83.4`) FAIL | `0` (`11.8/38.6`) FAIL | `0` (`9.4/77.0`) **FAIL** |
| **G2d** `L3_recomb` (`Ex>=6, NLL<=3.50`) + `hev NLL<=6.24` | `>=6`, `<=3.50`, `<=6.24` | `0` / `6.16` / `6.24` | `0` / `10.90` / `11.10` FAIL | `0` / `8.20` / `8.62` FAIL | **`11` / `2.47` / `6.21` PASS** | `0` / `10.15` / `10.58` FAIL | `12` / `2.60` / **`7.32`** FAIL |
| **G3** `extract_ret` (`200`, `num >=60/75`) | `>=148 (74%)` & `num>=60` | `154 (64)` | `103 (40)` FAIL | `135 (59)` FAIL | **`150 (64, p=0.523)` PASS** | `143 (59)` FAIL | **`158 (79.0%, num=60)` PASS** |
| **G4** `relation_ret` (`200`) | `>=123 (61.5%)` | `129 (64.5%)` | `101` FAIL | `118 (p=0.144)` FAIL | `105 (p=0.0015)` FAIL | `121 (p=0.215)` FAIL | **`127 (63.5%, p=0.864)` PASS** |
| **G5** `reverse_ret` (`200`, `ans_after>=69`) | `>=112 (56%)` & `>=69` | `117 (74)` | `96 (47)` FAIL | `97 (57)` FAIL | `98 (55)` FAIL | **`113 (75, p=0.665)` PASS** | **`119 (59.5%, aa=78)` PASS** |
| **G6** `compare_ret` (`200`) | `>=64 (32.0%)` | `70 (35.0%)` | `70` PASS | `60 (p=0.229)` FAIL | `62 (p=0.291)` FAIL | `63 (p=0.324)` FAIL | **`71 (35.5%, p=1.00)` PASS** |
| **G7** `temporal_ret` (`200`) | `>=77 (38.5%)` | `83 (41.5%)` | `75 (p=0.322)` FAIL | **`84 (42.0%, p=1.00)` PASS** | `76 (p=0.265)` FAIL | **`82 (41.0%, p=1.00)` PASS** | **`78 (39.0%, p=0.487)` PASS** |
| **Agg Retention** (`1,000`) | **`>=533 (53.3%)`** | `553 (55.3%)` | `445` FAIL | `494 (p=7.5e-5)` FAIL | `491 (p=2.47e-5)` FAIL | `522 (p=0.0255)` FAIL | **`553 (55.3%, p=1.00)` PASS** |
| **Non-Stop Pass (`G1 + G2c + RetAll`)** | ALL 3 required | Ref | **FAIL** | **FAIL** (`0/3`) | **FAIL** (`0/3`) | **FAIL** (`0/3`) | **FAIL** (`RetAll` PASS; `G1` & `G2c` FAIL) |

Note on `fresh_ans_before` wording splits in `Arm D`: `vf_ext|seen100` in **seen** wording (`n = 75`) climbed `R7 = 42.7% K4 / 21.3% Ex` to **`62.7% K4 / 48.0% Ex`**, whereas the same items in **dev** wording (`n = 75`) dropped `R7 = 38.7% -> 29.3% K4` (`Ex 2.7% -> 9.3%`). Multi-sentence transfer on `G2d` (`L3_heldout_ev`) still gained `K=4` (`12/30` in `R7` -> `18/30` in `B`, `20/30` in `D`) despite greedy `Exact` staying `0/30`.

Note on `G2c` fairness (added 2026-11-13, verified by a model-free replay of `R7`'s copy-mix stream): `R7`'s uniform random-token copy-mix (`make-copy-examples 15000 77 :lo 1 :hi 8192`) included all `845` `eval_heldout` tokens as supervised copy targets (minimum `21`, mean `39.2` occurrences in the `13,341` copy sequences `R7` consumed). The Pilot 4 guard excluded those tokens by design. `G2c` therefore compared tokens `R7` trained on against tokens the Pilot 4 arms never supervised; `R7`'s `25/80` and `95.3%` are not zero-shot results. The gate was pre-registered and the verdict does not change. That omitting these tokens from the guard caused the induction loss is a hypothesis; no ablation was run.

---

## 6. `2x2` Factorial Contrasts (`< 6.0 pp` on `n = 300` Treated as Single-Seed Noise)

| Contrast (`A->B`, `C->D`, `A->C`, `B->D`) | `fresh_ans_before K4` (`n=300`) | `fresh_ans_before Ex` (`n=300`) | `s1_recomb_tr Ex` (`n=80`) | `s1_heldout_ev Ex` (`n=80`) | `extract K4` (`200`) | `relation K4` (`200`) | `reverse K4` (`200`) | `Agg Ret K4` (`1,000`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Copy (`-Rep`): B vs A** | `+4.0 pp` (`118->130, p=.219`) **[NOISE]** | `+3.0 pp` (`52->61`) **[NOISE]** | **`+97.5 pp` (`0->78`)** | `+6.2 pp` (`0->5`) | **`+7.5 pp` (`p=.0081`)** | `-6.5 pp` (`p=.035`) | `+0.5 pp` | `-0.3 pp` (`494->491`) |
| **Copy (`+Rep`): D vs C** | `+5.3 pp` (`122->138, p=.085`) **[NOISE]** | **`+9.0 pp` (`43->70, p=4.6e-4`)** | **`+96.2 pp` (`0->77`)** | `+0.0 pp` (`0->0`) | **`+7.5 pp` (`p=.0059`)** | `+3.0 pp` | `+3.0 pp` | **`+3.1 pp` (`522->553, p=.019`)** |
| **Replay (`-Copy`): C vs A** | `+1.3 pp` (`118->122, p=.728`) **[NOISE]** | `-3.0 pp` (`52->43`) **[NOISE]** | `+0.0 pp` (`0->0`) | `+0.0 pp` (`0->0`) | `+4.0 pp` (`135->143`) | `+1.5 pp` | **`+8.0 pp` (`p=.033`)** | `+2.8 pp` (`494->522, p=.070`) |
| **Replay (`+Copy`): D vs B** | `+2.7 pp` (`130->138, p=.410`) **[NOISE]** | `+3.0 pp` (`61->70`) **[NOISE]** | `-1.2 pp` (`78->77`) | `-6.2 pp` (`5->0`) | `+4.0 pp` (`150->158`) | **`+11.0 pp` (`p=.0032`)** | **`+10.5 pp` (`p=.0015`)** | **`+6.2 pp` (`491->553, p=1.25e-5`)** |

Main factorial takeaways:
1. All four `fresh_ans_before K=4` contrasts (`+1.3 pp` to `+5.3 pp`) fall under the pre-registered `< 6.0 pp` single-seed noise threshold on `n = 300`.
2. **Intervention 1 (`+Copy`)** reliably delivered `96.2%-97.5%` prose copy on trained subword shapes (`s1_recomb_tr`) and boosted open-ended retrieval generation (`D vs C +9.0 pp, p = 4.6e-4`; cross-task `pf_rev|recomb_tr` `0/50 -> 7-9/50`), **but did not transfer zero-shot to disjoint heldout subwords (`s1_heldout_ev 0/80` in `D`, `eval_heldout_alpha 9.4%`)**.
3. **Intervention 2 (`+Replay`)** preserved aggregate retention on the five probes in `Arm D` (`D = 553/1000`, `p = 1.00` vs `R7`) but did not prevent all forgetting: `Arm D` lost held-out token induction (`95.3% -> 9.4%`) and person-first extraction on `D1` (`d1-ext-pf K4 179 -> 159/200`, `p = 0.0078`), and failed `G1c`, `G2c` and `G2d`. When paired with `+Copy` (`D vs B`), improved Aggregate Retention by `+6.2 pp` (`p = 1.25e-5`), fixing `B`'s `relation_ret` drop (`+11.0 pp`) and `reverse_ret` drop (`+10.5 pp`). Notice `Arm C` (`420` steps, `-Copy`) only reached `522/1000`, confirming the extra step budget / `+Copy` prose synergy in `Arm D` (`500` steps) helped `extract_ret` (`143 -> 158`) and retention alike.

---

## 7. Frozen `D1 200` Cells, Subword Stratification, and `selection.sexp`

### 7.1 Frozen `D1 200` (`n = 200` each, `K = 4` chance `25%`, Greedy `Exact` chance `~0%`)

Unmodified `lisp/bin/qa-analysis.lisp` (`controls=0 paired=0`); any full stage required `d1-rev-pf >= 120/200`.

| Checkpoint | `d1-rev-pf` (`K=4` / `Exact`, vs `R7 p`) | `d1-rev-vf` (`K=4` / `Exact`, vs `R7 p`) | `d1-ext-pf` (`K=4` / `Exact`, vs `R7 p`) |
| --- | ---: | ---: | ---: |
| `R7` (`step 0`) | `67/200 (33.5%)` / `31 (15.5%)` | `152/200 (76.0%)` / `75 (37.5%)` | `179/200 (89.5%)` / `123 (61.5%)` |
| `P3 arm-5e-4` (`500`) | `78/200 (39.0%, p=0.0708)` / `43 (21.5%)` | `58/200 (29.0%, p=2.63e-25)` / `6 (3.0%)` | `76/200 (38.0%, p=2.04e-21)` / `4 (2.0%)` |
| `P3 arm-1e-3` (`500`) | `83/200 (41.5%, p=0.0195)` / `42 (21.0%)` | `62/200 (31.0%, p=2.25e-21)` / `0 (0.0%)` | `75/200 (37.5%, p=4.31e-22)` / `2 (1.0%)` |
| **Arm A** (`190`) | `80/200 (40.0%, p=0.047)` / `34 (17.0%, p=0.664)` | `77/200 (38.5%, p=7.03e-18)` / `14 (7.0%)` | `110/200 (55.0%, p=2.91e-16)` / `42 (21.0%)` |
| **Arm B** (`270`) | **`90/200 (45.0%, p=0.00855)`** / `63 (31.5%, p=2.09e-5)` | `114/200 (57.0%, p=6.97e-8)` / `59 (29.5%)` | `125/200 (62.5%, p=2.59e-13)` / `69 (34.5%)` |
| **Arm C** (`420`) | `67/200 (33.5%, p=1.00)` / `37 (18.5%, p=0.377)` | `131/200 (65.5%, p=0.0046)` / `81 (40.5%)` | `137/200 (68.5%, p=9.67e-7)` / `109 (54.5%)` |
| **Arm D** (`500`) | **`88/200 (44.0%, p=0.0125)`** / **`65 (32.5%, p=3.39e-6)`** | **`170/200 (85.0%, +25/-7, p=0.0021)`** / **`142 (71.0%, p=2.8e-16)`** | `159/200 (79.5%, p=0.00779)` / **`130 (65.0%, p=0.483)`** |

- **Subword calibration tradeoff inside `d1-rev-pf` (`n = 200`):** Copy training dramatically unblocked `zero_ret` `D1` names (`n = 90`: `15/90` `R7`, `12/90` `A`, `5/90` `C` -> **`43/90 (47.8%)` K4 / `32/90 (35.6%)` Exact in `B`**, **`35/90 (38.9%)` / `21/90 (23.3%)` in `D`**), and `Edith` (`n = 11`, only `" Ed"` in `eval_heldout`, `"ith"` seen: `0/11` -> `5/11` in `B`, `8/11` in `D`), **but simultaneously pulled `hi100` names (`n = 67`) down from `47/67 (70.1%)` (`A`) / `43/67 (64.2%)` (`C`) / `48/67 (71.6%)` (`P3`) to `30/67 (44.8%)` in `B` and `31/67 (46.3%)` in `D`**. Inspecting wrong `Arm D` `hi100` predictions shows newly calibrated `train_copy` distractor names (`Adam`, `Emma`) actively competing against synthetic-trained names in `K = 4` ranking, leaving net `d1-rev-pf` at `88-90/200` (`30+` points short of `120/200`). Reference arms matter here: against frozen `R7` (`34/67` on `hi100` names, `15/90` on `zero_ret` names), `Arms B` and `D` scored `30/67` (`p = 0.54`) and `31/67` (`p = 0.68`), no reliable loss; the drop is relative to the no-copy arms `A`, `C` and `P3`, which had raised `hi100` names above `R7`.
- **Answer-after bonus on `D1`:** `Arm D` significantly **surpassed `R7`** on `d1-rev-vf` (`152 -> 170/200 = 85.0% K4, p = 0.0021`; `Exact 75 -> 142/200 = 71.0%, p = 2.8e-16`), and matched `R7` `Exact` on `d1-ext-pf` (`123 -> 130/200`) while losing `K = 4` there (`179 -> 159/200`, `+16/-36`, `p = 0.0078`). These p-values are unadjusted and from one seed; the result is consistent with copy + replay helping answer-after reverse without solving `ans_before`, and is not proof.

### 7.2 `selection.sexp` Retention Diagnostic (`n = 500`, `100` per category, `K = 4`, chance `25%`)

| Category (`100`) | `R7` | `P3 5e-4` | **Arm A** (`190`) | **Arm B** (`270`) | **Arm C** (`420`) | **Arm D** (`500`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `extract` | `64` | `34 (p=5.3e-6)` | `52 (p=0.036)` | `58 (p=0.109)` | `63 (p=1.00)` | `62 (p=0.791)` |
| `relation` | `55` | `34 (p=5.1e-4)` | `38 (p=0.0033)` | `49 (p=0.392)` | `49 (p=0.286)` | `56 (p=1.00)` |
| `reverse` | `36` | `33 (p=0.701)` | `37 (p=1.00)` | `31 (p=0.511)` | `38 (p=0.856)` | `40 (p=0.541)` |
| `compare` | `28` | `28 (p=1.00)` | `28 (p=1.00)` | `25 (p=0.678)` | `31 (p=0.629)` | `23 (p=0.332)` |
| `temporal` | `25` | `26 (p=1.00)` | `27 (p=0.832)` | `27 (p=0.832)` | `20 (p=0.442)` | `36 (p=0.080)` |
| **ALL (`500`)** | **`208`** | `155 (p=1.9e-5)` | `182 (p=0.023)` | `190 (p=0.130)` | `201 (p=0.579)` | **`217 (43.4%, p=0.456)`** |

`Arm D`'s `217/500` against `R7`'s `208/500` (`+62/-53`, `p = 0.456`) is no change, not a gain.

---

## 8. Final Stop Verdict, Quarantine Confirmation, & Artifacts

1. **Pre-Registered Stop Rule Met -> STOP Stage-2 Retrieval Line:** All 4 arms failed `G1 + G2c + Retention`. Pilot 4 was explicitly approved as the **final pilot (no 5th pilot)**; the Stage-2 retrieval optimization line (`R7`-scale `10.52M` / `8`-layer / `ctx=128` post-training on `ans_before` + copy + replay) stops here and is closed out in [FINAL_REPORT.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/FINAL_REPORT.md).
2. **Quarantine & Integrity Verified (No Commits):** All 30 hashes in [freeze-manifest.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/freeze-manifest.json) matched at freeze time (35 of its 36 current entries still match; the one difference, `pilot3/manifest.json`, was updated on purpose after the freeze and is documented in [FREEZE_ADDENDUM.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/FREEZE_ADDENDUM.md)) and all 7 probe SHA-256 digests in [eval-spec.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/eval-spec.md) match on disk (`R7` `33a5c245...`, `P3`, `confirmation.sexp` `bf5f0c06...`, `dev.jsonl` `2d92c882...`, `test.jsonl` `8f756bcd...`, `test.lock` `cde4c3bd...`). Locked test sets and `confirmation.sexp` were never opened. `dev.jsonl` was scored on `R7` (`229/500`) and on Pilot 1 checkpoints, and is unscored on Pilot 2, Pilot 3 and Pilot 4 checkpoints; it was read without scoring to build exclusion lists. Zero commits were made.
3. **Pilot 4 Final Checkpoints (`last.ckpt` / `best.ckpt` / `log.jsonl` SHA-256):**
   - `arm-a`: `73f80c65e43b58a40b5d1c2cc4504467e40a0807505d537f3bf7629e1c2743af` / `d0e34e28a72c10933b645d1f7038c7428b1002e4b7277fa89382db18d6a543e7` / `714fb91c80a9989f8dd4235c861839b973bd3cebf9655db9fce6112c331f051b`
   - `arm-b`: `4eff0dae913c84e1a67215cc85f7faa26f4d9fa2be5eda6e3beab858398f27fe` / `dfa8a0fdf237b328db6277c22df4f731febfdad5a286be653d7ac11c5b8cdaa6` / `d086791aa3669aa671c0b32ea929d918806f1e9f7efa243e3e4d262a8ce2f5ed`
   - `arm-c`: `83397de835ccae385ed9a33c98ebc72941be706e35c464b3813c96255527a13e` / `d790f5d3f3e362514b4a0e2a7326e8d6990ea73cdd4bce9b636c3159c11fdc57` / `d3f570ce07da1c91c484fc5d45c20261fc735024652963ce92779aa6080173c3`
   - `arm-d`: `db7f0dd348cca09c394ded165e6d04d2f482afed569facd7d54cae521aa3a219` / `7b4a61469490e3e41e6fce924dcf3b5eb0646d71e84b7cbbb9089da3614edcec` / `34f881643aa27f643c4235cfbe6d1580b5936365e07cc681b015abc4846cd01d`
