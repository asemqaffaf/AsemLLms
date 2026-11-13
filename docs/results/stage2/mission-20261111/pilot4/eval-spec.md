# Pilot 4 Independent Evaluation Specification & Pre-Registered Gates

Date: 2026-11-13
Branch: `stage2-qa`
Status: **APPROVED FROZEN PRE-REGISTRATION (FINAL PILOT, ASEM DECISION 2026-11-13)**

---

## 1. Quarantine, Scope & Relationship to Prior Evaluations

This specification locks the independent evaluation suite, exact scoring code, frozen baseline numbers, paired McNemar pass/fail gates, and post-experiment decision rules **before** any Pilot 4 training run begins.

- **Unopened & Quarantined Sets:** `docs/results/stage2/mission-20261111/confirmation.sexp`, `lisp/eval/qa/test.jsonl`, `lisp/eval/qa/test.lock`, and `lisp/src/qagen.lisp` `:test` pools (`lines 24-73`) were never read or touched and remain strictly quarantined.
- **Non-Gating Historical Diagnostics:** `docs/results/stage2/mission-20261111/selection.sexp`, `lisp/eval/qa/dev.jsonl`, and frozen `D1` (`d1-rev-pf`, `d1-rev-vf`, `d1-ext-pf`) are kept strictly as **diagnostic evidence**. `selection.sexp` is NOT promoted to a formal gate, and `D1` is NOT used as the sole basis for promotion.
- **Production Model Integrity:** `R7` (`33a5c245...`) and both `Pilot 3` checkpoints (`arm-5e-4`: `6a4fdc5f...`, `arm-1e-3`: `2041eaf1...`) are unchanged and inventoried in [freeze-manifest.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/freeze-manifest.json).

---

## 2. Independent 7-Probe Evaluation Suite (`n = 1,600` QA Items + `150` Induction Seqs)

Built deterministically by [build_eval_suite.lisp](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/build_eval_suite.lisp) and inventoried in [probe_suite_manifest.json](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/probe_suite_manifest.json) (max encoded item context length = `121 <= 128`).

### 2.1 Pool Disjointness Guarantees

- **Excluded from every probe item:** All entities appearing in `broad-pool`, `qagen.lisp :train` / `:dev`, `new-pool`, `selection.sexp`, `D1` (`d1-entities.txt`), `pilot3/excluded-extra.txt`, `dev.jsonl`, `probe-fresh-500.jsonl`, `probe-seen-500.jsonl`, `diagnostics-20261113/*.jsonl`, and `vocab_words`.
- **Clean Entity Pools Used:**
  - `synthetic-pool :names` disjoint subset (`1,580` names, `452` vowel-initial), specifically extracting `546` names with all BPE pieces `>= 100` in both `R7` and `P3` (`239` vowel-initial).
  - `synthetic-pool :towns` disjoint subset (`384` towns).
  - `subword_partition.json` `train_copy` (`75%`) vs `eval_heldout` (`25%` = `845` tokens strictly forbidden in all Pilot 4 training arms): `320` reserved `recomb_tr` multi-token words (`reserved_recomb_tr_words`, never emitted into `B`/`D` training) and `eval_heldout` BPE-exact FineWeb / synthetic words (`0` word overlap).
- **Wording & Option Hygiene:** Every probe item with `K=4` (`1,300` retrieval items + `60` `L3` entity-copy items) has `4` distinct options all present in its passage (`0` frequency/absence shortcuts) and balanced answer indices `[A, B, C, D]`. Notice `240` `s1_*` single-sentence items in `entity_copy` have only 1 entity in the passage and are scored strictly open-ended (`K = 8,192`), never `K = 4`.

### 2.2 Frozen Probe Files & SHA-256 Checksums

| Probe File | `n` | Answer / Category Balance | Stratification | SHA-256 |
| --- | ---: | --- | --- | --- |
| `probes/fresh_ans_before.jsonl` | 300 | `[75, 75, 75, 75]` | `150` `pf_rev` (`50` `seen100`, `50` `recomb_tr`, `50` `heldout_ev`) + `150` `vf_ext` (`seen100`); `50/50` `seen`/`dev` wording | `fe0b76580dab203b49628f0c20cc9f95a27049eb6ba0a35eae574bd417f0f21c` |
| `probes/entity_copy.jsonl` | 300 | `s1` greedy + `L3` `K=4` | `s1_seen_synth` (`80`: `40 pf, 40 vf`), `s1_recomb_tr` (`80`: `40 pf, 40 vf`), `s1_heldout_ev` (`80`: `40 pf, 40 vf`), `L3_recomb_tr` (`30`), `L3_heldout_ev` (`30`) | `ec388354b6162f4408f2895356b9f9f410641a1e0b0b08e7ae763518842852ad` |
| `probes/extract_ret.jsonl` | 200 | `[50, 51, 49, 50]` | `125` `attr` (`town, job, pet, color, instrument`) + `75` `num` (`age, score, height`); `99 dev / 101 seen` wording | `99b67ee5b31a2424444bae9a5bb05fc7a5227df125cf334a87c13b34fb89ce0f` |
| `probes/relation_ret.jsonl` | 200 | `[50, 50, 50, 50]` | `6` verbs x (`pairs`/`chain`) x (`subj`/`obj`); `100 dev / 100 seen` wording | `6bd2fd0ed56978b74fd94314719547cdbf4887e12812f36890c8752875b14d48` |
| `probes/reverse_ret.jsonl` | 200 | `[50, 50, 50, 50]` | `100` `ans_after` (`vf_rev` across `town, instrument, color`) + `100` `ans_before` (`pf_rev` across `town, job, pet, color, instrument`); `100 dev / 100 seen` | `6c29208cbafb17158733a9079d5588fb64a966e95b222ad55b877098bc02d598` |
| `probes/compare_ret.jsonl` | 200 | `[50, 50, 50, 50]` | `100` `max` + `100` `min` (`age, score, height`); `100 dev / 100 seen` | `16647686822ea9ebc80a811378816a22f4b126852d42b0307d573242f29294f5` |
| `probes/temporal_ret.jsonl` | 200 | `[50, 50, 50, 50]` | `50` each: `first, last, after, before`; `100 dev / 100 seen` | `d06d6e5d8ac4d36de7d8451b85d4bda2ca2717a6d3978010639af9b3a60ce5c4` |

---

## 3. Canonical Scoring Rules ([eval_probes.lisp](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/eval_probes.lisp))

Every Pilot 4 arm and baseline checkpoint is scored identically under FP32 (`sft32`) with:

1. **Canonical `K=4` (`k4`, chance = `25.0%`):** Uses `asem-qagen:format-prompt` (`"Text: <passage>\nQ: <question>\nA. <opt0>\nB. <opt1>\nC. <opt2>\nD. <opt3>\nThe answer is"`) followed by `asem-sft32:choice-exs` `:text` mode (`summed_option_likelihood`, argmax over summed token log-probabilities of `" " + opt`). Cross-checked against unmodified `lisp/src/qa-analysis.lisp` on `R7` `fresh_ans_before.jsonl`: reproduces `116/300 (38.7%)` bit-for-bit.
2. **Open-Ended Greedy Exact (`Exact`, `K = 8,192` per token, chance ~`0%`):** Decoded greedily from `"Text: <passage>\nQ: <question>\nThe answer is"` **without** listing `A..D` options, preventing option copying on retrieval and `entity_copy`. Reports exact full-token-sequence match (`Exact`), boundary-clean exact match (`Bounded`: next token is EOT or non-alphanumeric), first-token exact (`Tok0`), and mean target NLL per token (`NLL`).
3. **Random-Token Induction Health (`K = 8,192`, `n = 50` seqs, `half = 40`, `seed = 17`):** Same `[BOS, x_1..x_40, x_1..x_40]` protocol on `std_300_7300`, `seen_hi100_alpha`, and `eval_heldout_alpha`.
4. **Paired Inference:** All arm-vs-`R7` and arm-vs-arm significance tests use **exact two-sided paired McNemar binomial tests** on item-level `(r7_correct, arm_correct)` outcomes (`+gains / -losses`).

---

## 4. Verified Frozen Baselines (`baseline_eval_results.json` via `summarize_baselines.py`)

### 4.1 Primary 7-Probe Summary (`K=4` chance `25.0%`, Greedy `K=8,192` chance `~0%`)

| Probe (`n`) | Metric | Frozen `R7` | `Pilot 3 arm-5e-4` (vs `R7`, `+g/-l`, `p`) | `Pilot 3 arm-1e-3` (vs `R7`, `+g/-l`, `p`) |
| --- | --- | --- | --- | --- |
| `fresh_ans_before` (`300`) | `K=4` | `116/300 (38.7%)` | `129/300 (43.0%, +4.3pp, +60/-47, p=0.246)` | `123/300 (41.0%, +2.3pp, +60/-53, p=0.573)` |
| `fresh_ans_before` (`300`) | `Exact` | `30/300 (10.0%)` | `48/300 (16.0%, +6.0pp, +30/-12, p=0.00792)` | `48/300 (16.0%, +6.0pp, +30/-12, p=0.00792)` |
| `entity_copy` all (`300`) | `Exact` | `124/300 (41.3%)` | `59/300 (19.7%, -21.7pp, +4/-69, p=2.44e-16)` | `42/300 (14.0%, -27.3pp, +3/-85, p=7.34e-22)` |
| `entity_copy` `L3` (`60`) | `K=4` | `20/60 (33.3%)` | `24/60 (40.0%, +6.7pp, +11/-7, p=0.481)` | `16/60 (26.7%, -6.7pp, +8/-12, p=0.503)` |
| `extract_ret` (`200`) | `K=4` | `154/200 (77.0%)` | **`103/200 (51.5%, -25.5pp, +10/-61, p=4.64e-10)`** | **`82/200 (41.0%, -36.0pp, +10/-82, p=3.30e-15)`** |
| `relation_ret` (`200`) | `K=4` | `129/200 (64.5%)` | **`101/200 (50.5%, -14.0pp, +16/-44, p=3.94e-4)`** | **`93/200 (46.5%, -18.0pp, +15/-51, p=1.01e-5)`** |
| `reverse_ret` (`200`) | `K=4` | `117/200 (58.5%)` | **`96/200 (48.0%, -10.5pp, +23/-44, p=0.0139)`** | **`96/200 (48.0%, -10.5pp, +28/-49, p=0.0220)`** |
| `compare_ret` (`200`) | `K=4` | `70/200 (35.0%)` | `70/200 (35.0%, +0.0pp, +29/-29, p=1.00)` | `54/200 (27.0%, -8.0pp, +26/-42, p=0.0681)` |
| `temporal_ret` (`200`) | `K=4` | `83/200 (41.5%)` | `75/200 (37.5%, -4.0pp, +21/-29, p=0.322)` | **`57/200 (28.5%, -13.0pp, +28/-54, p=5.44e-3)`** |
| **Aggregate Retention (`1,000`)** | `K=4` | **`553/1000 (55.3%)`** | **`445/1000 (44.5%, -10.8pp, +99/-207, p=6.40e-10)`** | **`382/1000 (38.2%, -17.1pp, +107/-278, p=1.21e-18)`** |

### 4.2 Critical Stratified Subgroups (`fresh_ans_before`, `entity_copy`, `reverse_ret`, `extract_ret`, Induction)

| Subgroup (`n`) | Metric | `R7` | `P3 arm-5e-4` | `P3 arm-1e-3` |
| --- | --- | --- | --- | --- |
| `fresh_ans_before::pf_rev\|seen100` (`50`) | `Exact` / `K=4` | `24.0%` / `30.0%` | `44.0%` (`+20.0pp, p=0.0525`) / `42.0%` | `38.0%` / `42.0%` |
| `fresh_ans_before::pf_rev\|recomb_tr` (`50`) | `Exact` / `K=4` / `NLL` | `0.0%` / `40.0%` / `5.88` | `0.0%` / `32.0%` / `11.02` | `0.0%` / `34.0%` / `13.14` |
| `fresh_ans_before::pf_rev\|heldout_ev` (`50`) | `Exact` / `K=4` / `NLL` | `0.0%` / `40.0%` / `5.62` | `0.0%` / `44.0%` / `10.68` | `0.0%` / `32.0%` / `13.19` |
| `fresh_ans_before::vf_ext\|seen100` (`150`) | `Exact` / `K=4` | `12.0%` / `40.7%` | `17.3%` / `46.7%` | `19.3%` / `46.0%` |
| `entity_copy::s1_seen_synth\|pf` (`40`) | `Exact` (`K=8,192`) | `37/40 (92.5%)` | `40/40 (100.0%)` | `40/40 (100.0%)` |
| `entity_copy::s1_seen_synth\|vf` (`40`) | `Exact` (`K=8,192`) | **`36/40 (90.0%)`** | **`19/40 (47.5%, p=7.63e-5)`** | **`2/40 (5.0%, p=1.16e-10)`** |
| `entity_copy::s1_recomb_tr` (`80`) | `Exact` / `NLL` | `26/80 (32.5%)` / `1.73` | **`0/80 (0.0%, p=2.98e-8)`** / `10.50` | **`0/80 (0.0%, p=2.98e-8)`** / `13.85` |
| `entity_copy::s1_heldout_ev` (`80`) | `Exact` / `NLL` | `25/80 (31.2%)` / `1.50` | **`0/80 (0.0%, p=5.96e-8)`** / `10.34` | **`0/80 (0.0%, p=5.96e-8)`** / `13.54` |
| `entity_copy::L3_recomb_tr` (`30`) | `Exact` / `NLL` / `K=4` | `0/30 (0.0%)` / `6.16` / `26.7%` | `0/30 (0.0%)` / `10.90` / `46.7%` | `0/30 (0.0%)` / `12.92` / `30.0%` |
| `entity_copy::L3_heldout_ev` (`30`) | `Exact` / `NLL` / `K=4` | `0/30 (0.0%)` / `6.24` / `40.0%` | `0/30 (0.0%)` / `11.10` / `33.3%` | `0/30 (0.0%)` / `13.34` / `23.3%` |
| `reverse_ret::ans_after` (`100`) | `K=4` | **`74/100 (74.0%)`** | **`47/100 (47.0%, -27.0pp, p=1.43e-5)`** | **`46/100 (46.0%, -28.0pp, p=2.54e-5)`** |
| `reverse_ret::ans_before` (`100`) | `K=4` | `43/100 (43.0%)` | `49/100 (49.0%, +6.0pp, p=0.345)` | `50/100 (50.0%, +7.0pp, p=0.296)` |
| `extract_ret::num` (`75`) | `K=4` | **`64/75 (85.3%)`** | **`40/75 (53.3%, -32.0pp, p=8.43e-6)`** | **`27/75 (36.0%, -49.3pp, p=1.46e-10)`** |
| Induction `std_300_7300` (`50`) | `acc2` / `loss2` | `98.0%` / `0.18` | `38.7%` / `3.47` | `18.7%` / `7.30` |
| Induction `eval_heldout_alpha` (`50`) | `acc2` / `loss2` | **`95.3%` / `0.39`** | **`5.9%` / `5.81`** | **`0.1%` / `11.61`** |

*Baseline takeaways justifying the gates below:*
- Neither `P3 arm-5e-4` nor `P3 arm-1e-3` achieves significant `K=4` improvement on `fresh_ans_before` (`+4.3pp, p = 0.246` and `+2.3pp, p = 0.573`), both score `0/100` Greedy Exact on `pf_rev|recomb_tr + heldout_ev`, and both collapse on `s1_recomb_tr` (`0/80`), `s1_heldout_ev` (`0/80`), `eval_heldout_alpha` induction (`5.9%` / `0.1%`), `extract_ret` (`-25.5pp` / `-36.0pp`), `relation_ret` (`-14.0pp` / `-18.0pp`), and `reverse_ret::ans_after` (`-27.0pp` / `-28.0pp`).

---

## 5. Pre-Registered Final-Endpoint Pass/Fail Gates (Paired vs Frozen `R7`)

All thresholds below are evaluated identically at each arm's **final checkpoint** (`Arm A: step 190`, `Arm B: step 270`, `Arm C: step 420`, `Arm D: step 500`, holding `ans_before` constant at `6,080` sequences across all 4 arms; `step 250` logged strictly as an intermediate trajectory diagnostic on `B/C/D`). **All absolute floors below are strictly binding** (`p > 0.05` vs `R7` alone does not count as no harm). Minimum non-stop pass requirement (User Stop Rule): **at least one arm satisfies `G1` + `G2c` + every retention gate (`G3-G7` and `Aggregate`)**; full promotion readiness additionally inspects `G2a`, `G2b`, `G2d`, and requires frozen `D1 rev-pf >= 120/200` on any subsequent full stage.

### Gate G1: Fresh Answer-Before Retrieval (`fresh_ans_before`, `n = 300`)
- **G1a (`K=4` Overall):** `>= 140/300 (46.7%)` (`>= R7 + 8.0pp`, chance `25.0%`) **AND** two-sided paired McNemar `p <= 0.05` vs `R7` (`116/300 = 38.7%`).
- **G1b (Open-Ended Greedy `Exact`, `K=8,192`):** `>= 48/300 (16.0%)` (`>= R7 + 6.0pp`, chance `~0%`) **AND** two-sided paired McNemar `p <= 0.05` vs `R7` (`30/300 = 10.0%`).
- **G1c (Subgroup Transfer & Non-Regression):** `pf_rev|seen100` Greedy `Exact >= 20/50 (40.0%)` (vs `R7 = 24.0%`, `P3 = 44.0%`), **AND** `pf_rev|recomb_tr` Greedy `Exact >= 5/50 (10.0%)` (vs `0/50` in both `R7` and `P3`; note `pf_rev|recomb_tr` tests **cross-task transfer** from Intervention 1 entity copy to unseen answer-before retrieval, since `recomb_tr` words are absent from retrieval training).

### Gate G2: Entity Copying, Unseen-Subword Recombination & Induction Health (`entity_copy`, `n = 300` + `150` Induction Seqs)
- **G2a (`s1_seen_synth` Copy Retention, `n = 80`):** Greedy `Exact >= 70/80 (87.5%)` (max `-3.8pp` vs `R7`'s `91.2%`, no significant loss) **AND** `s1_seen_synth|vf >= 34/40 (85.0%)` (vs `R7 = 90.0%`, blocking `P3`'s `-42.5pp` crash to `47.5%`).
- **G2b (`s1_recomb_tr` Compositional Subword Generalization, `n = 80`):** Greedy `Exact >= 44/80 (55.0%)` (`>= R7 + 22.5pp` above `R7 = 26/80 = 32.5%` and `P3 = 0/80`) **AND** paired McNemar `p <= 0.01` vs `R7`.
- **G2c (`s1_heldout_ev` & Induction Preservation, `n = 80` + `50` seqs):** `s1_heldout_ev` Greedy `Exact >= 22/80 (27.5%)` (max `-3.8pp` vs `R7 = 25/80 = 31.2%`, paired `p > 0.05` on any negative difference vs `R7`, blocking `P3`'s `0/80` collapse) **AND** induction `eval_heldout_alpha >= 90.0%` (`R7 = 95.3%`) **AND** `std_300_7300 >= 90.0%` (`R7 = 98.0%`). *(Any statistically significant gain above `31.2%` on `s1_heldout_ev` additionally measures structural zero-shot prose-copy transfer.)*
- **G2d (`L3_recomb_tr` Multi-Sentence Copy Transfer, `n = 30`):** Greedy `Exact >= 6/30 (20.0%)` (paired `p <= 0.05` vs `R7 = 0/30` and `P3 = 0/30`) **AND** mean `NLL <= 3.50` (down from `6.16` on `R7` and `10.90` on `P3 5e-4`), **AND** `L3_heldout_ev` mean `NLL <= 6.24` (no regression vs `R7`'s `6.24`, preventing `P3`'s `11.10` blow-up).

### Gates G3-G7: Pre-Registered Capability Retention Limits (`n = 200` each, `K=4`, chance `25.0%`)

For every retention capability `k in {G3..G7}`, **acceptable regression limit** means BOTH (i) accuracy meets or exceeds the pre-registered floor below AND (ii) if accuracy is below `R7`, two-sided paired McNemar `p > 0.05` vs `R7`:

| Gate | Probe (`n`) | `R7` Baseline | `P3 5e-4` Baseline | Pre-Registered Retention Floor | Max Drop vs `R7` | Extra Subgroup Guard |
| --- | --- | --- | --- | --- | --- | --- |
| **G3** | `extract_ret` (`200`) | `154 (77.0%)` | `103 (51.5%)` | **`>= 148/200 (74.0%)`** | `-3.0pp` | `num` (`n=75`, `R7 85.3%`) **`>= 60/75 (80.0%)`** |
| **G4** | `relation_ret` (`200`) | `129 (64.5%)` | `101 (50.5%)` | **`>= 123/200 (61.5%)`** | `-3.0pp` | Paired `p > 0.05` on negative delta vs `R7` |
| **G5** | `reverse_ret` (`200`) | `117 (58.5%)` | `96 (48.0%)` | **`>= 112/200 (56.0%)`** | `-2.5pp` | **`ans_after` (`n=100`, `R7 74%`) `>= 69/100 (69.0%)`** |
| **G6** | `compare_ret` (`200`) | `70 (35.0%)` | `70 (35.0%)` | **`>= 64/200 (32.0%)`** | `-3.0pp` | Paired `p > 0.05` on negative delta vs `R7` |
| **G7** | `temporal_ret` (`200`) | `83 (41.5%)` | `75 (37.5%)` | **`>= 77/200 (38.5%)`** | `-3.0pp` | Paired `p > 0.05` on negative delta vs `R7` |
| **Agg** | **All 5 Combined (`1,000`)** | **`553 (55.3%)`** | **`445 (44.5%)`** | **`>= 533/1000 (53.3%)`** | **`-2.0pp`** | Paired `p > 0.05` on negative delta across `n=1,000` |

*Statistical sensitivity check:* At `n = 200`, `R7` vs `P3 arm-5e-4` forgetting registered `p = 4.64e-10` (`extract_ret`), `p = 3.94e-4` (`relation_ret`), `p = 0.0139` (`reverse_ret`; `p = 1.43e-5` on `ans_after`), and `p = 6.40e-10` aggregate (`n = 1,000`), confirming high power against real forgetting while tolerating small step-to-step variance (`<= 2.0-3.0pp`).

---

## 6. Pre-Registered Factorial Analysis, Binding Stop Rule & Diagnostic Checks

After evaluating all 4 arms (`A, B, C, D`) on [eval_probes.lisp](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/eval_probes.lisp) plus frozen **`D1 200`** (`d1-rev-pf`, `d1-rev-vf`, `d1-ext-pf` reported for every arm) and `selection.sexp` diagnostics (`dev.jsonl`, `confirmation.sexp`, and locked test sets remain **unopened**), apply Section 5 and the user stop rule strictly:

1. **Both Main Effects & Interaction (`2x2` Factorial Comparison, `< 6.0pp` Noise Rule):**
   - Measure **Copy Main Effect** via `(B vs A)` and `(D vs C)` on `G2` (`s1_recomb_tr`, `s1_heldout_ev`, `L3_recomb_tr`, induction) and `G1` (`pf_rev|recomb_tr`, `fresh_ans_before`), holding `ans_before = 6,080` constant.
   - Measure **Replay Main Effect** via `(C vs A)` and `(D vs B)` on `G3-G7` retention, `G2a` (`s1_seen_synth|vf`), and `K=4` vs `Exact` alignment on `G1a`.
   - **Single-Seed Noise Rule:** Because each arm uses one seed (`111404`), treat inter-arm differences under **`6.0` percentage points on `n = 300`** as noise.
2. **Binding Absolute Floors & Frozen `D1 200` Rule:**
   - Every absolute floor in `G1..G7`, `G2c`, and `Aggregate` (`>= 533/1000 = 53.3%`) is strictly binding; a non-significant `p > 0.05` vs `R7` alone does **not** pass a gate if the absolute floor is missed.
   - Passing `G1a` alone is a learning signal only; any full follow-on stage also requires `D1` person-first reverse `>= 120/200`. Report all three frozen `D1 200` cells (`person-first reverse`, `value-first reverse`, `person-first extraction`) for all four arms.
3. **Final Stop Rule (No 5th Pilot):**
   - **Stop Outcome (No arm passes `G1` + `G2c` + all retention gates `G3-G7` + `Aggregate`):** There will be **no 5th pilot**. Stop the Stage-2 retrieval line immediately and write the final report synthesizing all causal, factorial, and retention evidence (`R7`, `P1-P4`).
   - **Pass Outcome (At least one arm passes `G1` + `G2c` + all retention gates `G3-G7` + `Aggregate`, checking `G2a/G2b/G2d` as well):** Propose the exact next-stage configuration to the user (noting `D1 rev-pf >= 120/200` target for any full stage) and **do not run it**.
