# Stage-2 Retrieval Mission Final Report (2026-11-11 to 2026-11-13)

**Final Decision: STOP THE STAGE-2 RETRIEVAL LINE (No 5th pilot, no 2,500-step run, `R7` remains the unchanged reference).**

**Mission status: CLOSED (2026-11-13).** No fifth pilot, no checkpoint promotion, no further training under this mission. Housekeeping done: `pilot3/RESULTS.md` and `pilot4/DATA_AUDIT.md` corrected, `pilot3/manifest.json` hash and changelog updated, `pilot4/freeze-manifest.json` left unmodified, with the deliberate post-freeze change to `pilot3/manifest.json` documented in `pilot4/FREEZE_ADDENDUM.md`. Final hashes: `pilot3/RESULTS.md` `5924fb89e2154c9184e2c1640398908d0c866efdbbe85628812de598030cf80f`, `pilot3/manifest.json` `d39b5de8dda659b19a5cbab90c772f9bdf6def416da4f2563ce533aab9e292f0`, `pilot4/DATA_AUDIT.md` `2e0a0f24aee7901e5c0cbc8a2935a188437ceb803b4fe3b7669c23721cd39879`. Any new work starts as a new mission with its own pre-registration.

Across all four controlled pilots on `R7` (`10,523,393` parameters, `8` layers, `8` heads, width `256`, `ff = 1,024`, `ctx = 128`, `vocab = 8,192`), no intervention achieved both out-of-distribution four-person answer-before retrieval generalization and full capability / induction preservation. No checkpoint has a verified gain over `R7`. `confirmation.sexp`, `test.jsonl` and `test.lock` remain unopened; `dev.jsonl` was scored on `R7` and Pilot 1 checkpoints only (Section 6). No commits were made.

Revised 2026-11-13 after reviewer checks: wording corrected in Sections 1, 2 and 6; Sections 3 (inference-only checks on `R7`), 4 (caveats) and 5 (follow-up needs) added. No training was run for the revision and no gate changed.

---

## 1. Mission Arc Summary

Frozen reference `R7` (`33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`) scores `229/500 (45.8%)` `K = 4` on `dev.jsonl` (`K = 4` chance `25.0%`) and `208/500 (41.6%)` on `selection.sexp`, with strong answer-after `D1` accuracy (`d1-ext-pf: 179/200 = 89.5%`, `d1-rev-vf: 152/200 = 76.0%`) alongside weak answer-before `D1` accuracy (`d1-rev-pf: 67/200 = 33.5%`, `d1-ext-vf: 72/200 = 36.0%`).

All counts below are `K = 4` summed option ranking (chance `25.0%`) unless marked `Exact` (greedy prefix, `K = 8,192`, chance about `0%`). All p-values are exact two-sided paired McNemar tests against frozen `R7`, unadjusted for multiplicity.

| Experiment | Manipulation (vs Frozen `R7`) | Key Outcome | Why It Did Not Advance |
| --- | --- | --- | --- |
| **Pilot 1** (`300` steps) | Auxiliary contrastive negative penalty (`1,005` pairs) | Null: `selection 210` (`ctrl`, `p = 0.87`) vs `201/500` (`treat`, `p = 0.35`; `R7 = 208`); `dev 235` (`p = 0.41`) vs `233/500` (`p = 0.64`; `R7 = 229`) | Contrastive loss did not change inverse binding or `ans_before` retrieval |
| **Pilot 2** (`500` steps) | Replace `1,668 / 15,000` seqs with `2,121` 4-person `ans_before` qs (`10.4%` token weight) | Gate missed (`D1 rev-pf`: `67 -> 71/200`, `+15/-11, p = 0.56` vs gate `120`; `treat` memorized seen items `245 -> 327/514` while fresh stayed `230 -> 243/503`) | Low dose (`10.4%`) produced item memorization without rule learning |
| **Pilot 3** (`500` steps, `5e-4` & `1e-3`) | `100%` fresh non-repeating 4-person `ans_before` (`16,000` seqs, `0` copy, `0` replay) | **Outcome E (ambiguous):** `Fresh Exact 215 -> 318/500 (63.6%)`, `Fresh K4 249 -> 349/500 (69.8%)`, loss still falling at step 500 | Severe forgetting (`Agg Ret 553 -> 445/1000`, `selection 208 -> 155/500`, `d1-rev-vf 152 -> 58/200`, induction `95.3% -> 5.9%/0.1%`). `D1 rev-pf` reached only `83/200`; on the `90` of `200` `D1 rev-pf` items whose gold name has a zero-count subword, `Exact` was `0/90` |
| **R7 Exp 1** (`n = 157`, inf-only) | Minimal-pair swap of the gold name only on frozen `R7` (`zero <-> >=100` subword exposure) | Large swap effect: pooled `K4 25 -> 89/157 (+70/-6, p = 6.31e-15)`, `Exact 4 -> 40/157 (p = 2.84e-10)` | `R7`'s option scores depend strongly on name exposure. Section 3 shows this is largely option-name preference: renaming all four names leaves `K = 4` unchanged |
| **Pilot 4** (`2x2` factorial: `A 190`, `B 270`, `C 420`, `D 500`) | Fixed `6,080` `ans_before` +/- `2,560` prose+guard copy (`train_copy 75%`) +/- `7,360` multi-task replay (`0` leaks) | **STOP:** All 4 arms failed binding `G1` & `G2c` (detailed below). **Arm D** (`+Copy, +Replay`) passed all retention gates (`553/1000 = 55.3%`, `p = 1.00`; `selection 217/500`, `p = 0.456`, no change) & trained-subword copy (`96.2%`), **but missed `G1a` (`138/300 < 140`), `G1c seen100` (`18/50 < 20`), `G2c heldout` (`0/80` & `9.4%` ind.), `G2d` (`hev NLL 7.32 > 6.24`), and `D1 rev-pf` (`88/200 < 120`)** | Two failure modes (Section 2) |

Full item-level Pilot 4 tables, Pilot 3 decision-table trajectory curves, separate-sentence filler preflight (`s1/s2/s3`), and `D1 200` breakdowns are recorded in [pilot4/RESULTS.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/RESULTS.md) and [pilot3/RESULTS.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot3/RESULTS.md).

---

## 2. What Pilot 4 Showed (Single Seed)

### 2.1 Observed Gains Within `R7` Scale (`10.52M`)
1. **Replay + Copy preserved aggregate retention but not all prior behaviour (`Arm D`):** `Arm D` matched `R7` on aggregate retention over the five probes (`553/1000` vs `553/1000`, `+96/-96, p = 1.00`: `G3 extract 158/200 (+2.0 pp)`, `G4 relation 127/200 (-1.0 pp)`, `G5 reverse 119/200 (+1.0 pp)`, `G6 compare 71/200 (+0.5 pp)`, `G7 temporal 78/200 (-2.5 pp)`) and did not change on `selection.sexp` (`208 -> 217/500`, `+62/-53, p = 0.456`). It did not prevent all forgetting: it lost held-out token induction (`95.3% -> 9.4%`), lost person-first extraction on `D1` (`d1-ext-pf K4 179 -> 159/200`, `+16/-36, p = 0.0078`), and failed `G1c`, `G2c` and `G2d` (plus `G1a`, `138/300` against a floor of `140`). On answer-after `D1` reverse it scored above `R7` (`d1-rev-vf: 152 -> 170/200 K4, +25/-7, p = 0.0021; Exact 75 -> 142/200, p = 2.8e-16`); these p-values are unadjusted like every other cell.
2. **Contextual subword prose copy raised trained-subword entity composition (`Arms B, D`):** On disjoint multi-token words composed from the `75%` `train_copy` subword partition (`s1_recomb_tr`, `n = 80`, `Exact`), `Arms A` and `C` collapsed (`0/80`), whereas `Arm B` reached `78/80 (97.5%)` and `Arm D` reached `77/80 (96.2%)`, transferring partially into 3-sentence retrieval (`L3_recomb_tr 0/30 -> 12/30` Exact, `NLL 6.16 -> 2.60`) and `fresh_ans_before pf_rev|recomb_tr` (`0/50 -> 7/50` in `B`, `9/50` in `D`). On `D1 rev-pf` zero-ret names (`n = 90`), `Exact` went from `2/90` (`R7`) / `0/90` (`A, C, P3`) to `32/90` (`B`) and `21/90` (`D`).

### 2.2 Why the Stage-2 Retrieval Line Stopped (Two Failure Modes)
1. **Induction / prose copy loss on held-out subwords (`G2c` failure):** `R7` scored `95.3%` random-token induction on the `25%` quarantined `eval_heldout_alpha` tokens and `25/80 (31.2%)` on `s1_heldout_ev` prose copy (`Exact`). Every Pilot 4 arm lost held-out induction (`A 49.3%`, `B 23.5%`, `C 11.8%`, `D 9.4%`) and fell on `s1_heldout_ev` (`0/80` in `A, C, D`; `5/80` in `B`). Contextual prose copy over `train_copy` subwords raised copying of those trained subwords only.
   - **Hypothesis, not established:** omitting the held-out tokens from the random `:copy` guard caused drift away from Stage-1 copy behaviour. No ablation with the guard including those tokens was run.
   - **Fairness note (verified, Section 3):** `R7`'s own copy-mix trained on all `845` `eval_heldout` tokens, while the Pilot 4 guard excluded them by design. `G2c` compared tokens `R7` trained on against tokens Pilot 4 never supervised, so `R7`'s `95.3%` and `25/80` are not zero-shot results. The gate was pre-registered and the verdict does not change.
2. **Paraphrase fragility on 4-person answer-before ranking (`G1` & `D1` failure):** With `6,080` fresh `ans_before` sequences, `+Copy`, and `+Replay`, all four `fresh_ans_before K=4` factorial contrasts stayed inside the pre-registered `< 6.0 pp` single-seed noise band (`+1.3` to `+5.3 pp`), and the best arm (`Arm D`) reached `138/300 (46.0%)` on that probe (`R7 116/300 = 38.7%`; floor `140/300`) and `18/50 (36.0%)` on `pf_rev|seen100` (floor `20/50`).
   - This is not a ceiling. Pilot 3 reached `349/500 (69.8%)` `K = 4` on its own fresh probe (`R7 249/500 = 49.8%` on that probe), in training wording only and with severe forgetting. The Pilot 3 and Pilot 4 probes differ in items, entity strata and wording mix, so percentages are not comparable across them.
   - The gain is wording-specific. In `fresh_ans_before vf_ext|seen100`, **seen** wording (`n = 75`) rose from `R7`'s `32/75 (42.7%)` to `47/75 (62.7%)` `K4` in `Arm D` (`+21/-6, p = 0.0059`; `Exact 48.0%`), whereas unseen **dev** wording (`n = 75`) went from `R7`'s `29/75 (38.7%)` to `22/75 (29.3%)` `K4` (`+4/-11, p = 0.12`; `Exact 9.3%`).
   - `D1 rev-pf` by exposure of the gold name, with the reference arm named for every number:

   | `D1 rev-pf`, `K = 4`, chance `25%` | `R7` | `Arm A` (no copy) | `Arm B` (+Copy) | `Arm C` (no copy) | `Arm D` (+Copy) |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | zero-ret names (`n = 90`) | `15` | `12` | `43` (`p = 4.3e-6`) | `5` (`p = 0.0020`) | `35` (`p = 5.4e-4`) |
   | `hi100` names (`n = 67`) | `34` | `47` (`p = 0.0044`) | `30` (`p = 0.54`) | `43` (`p = 0.064`) | `31` (`p = 0.68`) |
   | all (`n = 200`) | `67` | `80` (`p = 0.047`) | `90` (`p = 0.0086`) | `67` (`p = 1.00`) | `88` (`p = 0.0125`) |

   Against `R7` (`34/67` on `hi100` names), `Arms B` and `D` scored `30` and `31`: no reliable loss. The larger drops (`47 -> 30`, `43 -> 31`) are relative to the no-copy `Arms A` and `C`, which had raised `hi100` names above `R7`. `+Copy` arms gained on zero-ret names and did not keep the `hi100` gain of the no-copy arms, leaving `D1 rev-pf` at `88-90/200 (44.0%-45.0%)`, more than `30` items short of the `>= 120/200` prerequisite.

Combined across Pilot 2, Pilot 3, and Pilot 4, Stage-2 post-training on the frozen `10.52M` / `8`-layer `R7` checkpoint did not overcome the answer-before gap, unseen-paraphrase overfitting, or held-out subword drift in these pilots. The evidence is four pilots, a single seed each, `500` steps or fewer, Pilot 3 curves still rising at step `500`, and no capacity control, so it does not show that these cannot be overcome. Per the pre-registered stop rule, the Stage-2 retrieval line terminates here.

---

## 3. Reviewer Checks on Frozen `R7` (Inference Only, 2026-11-13)

Full tables: [review-checks-20261113/RESULTS.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/review-checks-20261113/RESULTS.md). No training; `R7` hash unchanged before and after.

**3.1 Experiment 1 vs D2.** Experiment 1 (gold name swapped alone, training wording, `+40.8` points) and D2 (all four names renamed to training-pool names, dev wording, no change) differ in design as well as wording. The Experiment 1 pairs were re-rendered in dev wording and the gold-only swap was applied to the real dev items.

| Frozen `R7`, `K = 4`, chance `25%` | Wording | `n` | Before (gold-only rows: zero-exposure gold; all-four rows: dev-pool names) | After (gold-only rows: `hi100` gold; all-four rows: training-pool names) | Gains / losses | `p` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gold only: Experiment 1 pairs (reproduces saved result) | training | `157` | `25 (15.9%)` | `89 (56.7%)` | `70 / 6` | `6.3e-15` |
| Gold only: same pairs, dev wording | dev | `157` | `26 (16.6%)` | `67 (42.7%)` | `50 / 9` | `5.3e-8` |
| Gold only: real dev reverse items | dev | `82` | `17 (20.7%)` | `42 (51.2%)` | `27 / 2` | `1.6e-6` |
| Gold only: real dev compare items | dev | `83` | `11 (13.3%)` | `35 (42.2%)` | `28 / 4` | `1.9e-5` |
| All four: `D1 rev-pf` -> training-pool names (saved) | training | `200` | `67 (33.5%)` | `70 (35.0%)` | `44 / 41` | `0.83` |
| All four: D2 reverse -> training-pool names (saved) | dev | `94` | `36 (38.3%)` | `34 (36.2%)` | `20 / 22` | `0.88` |
| All four: D2 compare -> training-pool names (saved) | dev | `100` | `31 (31.0%)` | `36 (36.0%)` | `20 / 15` | `0.50` |

The exposure effect exists in dev wording (`+26.1` points on the same pairs, `+30.5` on real dev reverse items). It appears only when the gold name changes relative to its distractors. `R7` picks `hi100`-named options far more often than zero-named options even when they are wrong (`D1 rev-pf` distractor slots: `92/224 = 41.1%` vs `16/264 = 6.1%`), and the swap moves compare items by the same amount although `R7` is near chance on compare. Experiment 1's `K = 4` effect is therefore largely option-name preference and does not show that subword exposure limits `R7`'s answer-before `K = 4` accuracy. `Exact` does rise when all four names are high-exposure (`D1 rev-pf 31 -> 60/200`, `p = 6.4e-4`). A wrong-slot swap was not run.

**3.2 `G2c` fairness.** A model-free replay of `R7`'s copy-mix (`make-copy-examples 15000 77 :lo 1 :hi 8192`, seed-7 epoch order, `13,341` of `15,000` copy sequences consumed in `2,500` steps) shows every one of the `845` `eval_heldout` tokens was a supervised copy target in `R7` training (minimum `21`, mean `39.2` occurrences). Pilot 4 arms had `0`. See Section 2.2 item 1.

---

## 4. Caveats

- **Single seed.** Every pilot arm is one run (`500` steps or fewer). No effect here has been replicated.
- **Unadjusted p-values.** Every p-value in this report and in the pilot result files is unadjusted across all cells, arms and probes, including `d1-rev-vf p = 0.0021`. Many cells were tested.
- **No capacity control and unconverged curves.** No larger model was trained, and Pilot 3 loss was still falling at step `500`. Nothing here separates a capacity limit from too little training.
- **Probes differ.** Pilot 3's fresh probe, Pilot 4's `fresh_ans_before`, `D1`, `selection.sexp` and `dev.jsonl` differ in items, entities and wording; percentages are not comparable across them.
- **`G2c` compared unequal exposure** (Section 3.2), and its causal explanation is a hypothesis.
- **Bottom line.** No checkpoint has a verified gain above `R7`'s `45.8%` on `dev.jsonl` (`229/500`, `K = 4`, chance `25%`) or its `41.6%` on `selection.sexp` (`208/500`). `dev.jsonl` was scored only on the Pilot 1 checkpoints (`235` and `233/500`, `p = 0.41` and `0.64`). The best `selection.sexp` score is `Arm D`'s `217/500` (`p = 0.456`). The mission's `50-55%` target was not reached and the `90%` goal was never approached.

---

## 5. Follow-up Needs (Untested Hypotheses, Nothing Started)

- **Untested hypothesis:** the answer-before gap is a capacity or training-length limit of the `10.52M` model. Deciding this needs a capacity control and more than one seed; neither exists.
- **Untested hypothesis:** leaving held-out tokens out of the copy guard caused the `G2c` induction loss. Deciding this needs an ablation with the guard covering those tokens, scored against a baseline with matched exposure.
- **Untested hypothesis:** `R7`'s `K = 4` name-exposure effect is option-name preference with no binding component. Deciding this needs a wrong-slot (distractor-only) swap on frozen `R7`, inference only.

---

## 6. Quarantine, Reference Integrity, & Artifact Audit

- **Unchanged Production Baseline & Prior Pilots:** `R7` (`33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`), Pilot 1/2/3 checkpoints, `r7-qa.json` (`229/500 = 45.8%`), and `r7-selection.json` (`208/500 = 41.6%`) are untouched and verified against [freeze-manifest.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/freeze-manifest.json).
- **Never opened** (SHA-256 as recorded at the Pilot 4 freeze; not re-read for the reviewer checks):
  - `docs/results/stage2/mission-20261111/confirmation.sexp` (`bf5f0c061e078f34c90e39f8d7b4befa5ec84109ad51ac41e4fcfb4b39761fb0`)
  - `lisp/eval/qa/test.jsonl` (`8f756bcdf3accd79...`)
  - `lisp/eval/qa/test.lock` (`cde4c3bd39df9563...`)
  - `lisp/src/qagen.lisp` `:test` pools (`97b123a54a2c3e5ca5f036939ce5625af4f2d214e7e8e901544081d13a0fb0f7`)
- **`lisp/eval/qa/dev.jsonl`** (`2d92c8825ca3752bd780ecd25023ae5f0a8dcc60677ee6484d46aa95e701032a`) is open, not quarantined:
  - Scored on `R7` (`229/500`) and on both Pilot 1 checkpoints (`235/500` control, `233/500` treatment).
  - Not scored on any Pilot 2, Pilot 3 or Pilot 4 checkpoint.
  - Read without scoring to build entity exclusion lists (Pilot 3, Pilot 4) and diagnostic items. Dev-derived diagnostic items (D2, D5, Section 3) were scored on frozen `R7` only.
  - The frozen `D1` items use dev-pool names (scored on Pilot 2 to 4 checkpoints), and about half of the items in all six Pilot 4 retrieval probes use dev wording (`649` of `1,300`: `fresh_ans_before 150/300`, `extract_ret 99/200`, `relation_ret 100/200`, `reverse_ret 100/200`, `compare_ret 100/200`, `temporal_ret 100/200`; scored on the Pilot 4 checkpoints, and on `R7` and Pilot 3 as baselines). Dev-pool names and dev wording frames are therefore not unseen by those evaluations even though `dev.jsonl` items are.
- **Git State:** No commits made; no public Stage-2 claims or promotions issued.

TL;DR: The stop rule stands. Four single-seed pilots of `500` steps or fewer did not produce a checkpoint with a verified gain over `R7` (`45.8%` dev, `41.6%` selection, `K = 4`, chance `25%`). Arm D kept aggregate retention (`553/1000`) but lost held-out induction and person-first extraction and failed `G1a`, `G1c`, `G2c` and `G2d`. Gains were wording-specific. These pilots do not show the gap cannot be closed; they show it was not closed here.
