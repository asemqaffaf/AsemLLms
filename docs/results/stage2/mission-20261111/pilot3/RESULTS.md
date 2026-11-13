# Pilot 3 Results: Answer-Before Learnability Control (2026-11-13)

**Verdict: Outcome E (Anything else - ambiguous; stopped per Rule 5, no 2,500-step run requested or started).**

Revised 2026-11-13 after reviewer checks (wording only; no number or verdict changed). Each arm is a single seed, every p-value in this file is unadjusted across all cells, and the `69.8%` fresh `K = 4` figure is on this pilot's own fresh probe (training wording), which is not comparable with the Pilot 4 probes, `selection.sexp` or `dev.jsonl`. No checkpoint here has a verified gain over `R7` on `dev.jsonl` or `selection.sexp`. See [FINAL_REPORT.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/FINAL_REPORT.md) Sections 3 to 5.

- Best final checkpoint on `Fresh exact` (`arm-1e-3.step500`, tied with `arm-5e-4.step500` on `318/500 = 63.6%`, higher on `Fresh K4` `349/500` and `D1 rev-pf` `83/200`):
  - `Fresh exact`: `318/500 (63.6%)`, up `+20.6 pp` from `R7`'s `215/500 (43.0%)`, below Outcome A/B's `75.0%` threshold.
  - `Seen exact`: `383/500 (76.6%)` on `arm-1e-3` (`398/500 = 79.6%` on `arm-5e-4`), above `70%`.
  - `Seen minus Fresh (exact)`: `+13.0 pp` on `arm-1e-3` (`+16.0 pp` on `arm-5e-4`), below Outcome C's `> 20 pp` threshold.
  - `D1 person-first reverse (K = 4)`: `83/200 (41.5%, vs R7 +29/-13 p = 0.0195)` on `arm-1e-3` (`78/200 = 39.0%, +21/-10 p = 0.0708` on `arm-5e-4`), below `120/200`.
  - `Outcome D check`: Ruled out cleanly. Both arms far exceeded `< 60%` seen exact (`79.6%` and `76.6%`), and loss fell steeply over the last 250 steps (`arm-5e-4`: train `0.767 -> 0.560`, eval `0.920 -> 0.679`; `arm-1e-3`: train `1.031 -> 0.644`, eval `1.172 -> 0.727`).

The answer-before rule is clearly **learnable without architecture changes** on `10.52M` R7 (`43.0% -> 63.6%` fresh exact, `49.8% -> 69.8%` fresh `K = 4`, `p = 3.06e-15`), and training was still climbing at step 500 rather than plateauing. However, `500` steps stopped short of `75%` fresh exact, and `D1` reverse barely moved overall (`67 -> 83/200`) because `90/200` `D1` names contain FineWeb BPE subwords that appear `0` times in Pilot 3's synthetic training pool (`0/90` exact), whereas the `67/200` `D1` items with well-supported BPE subwords (`>= 100` training occurrences) climbed from `34/67 (50.7%)` to `53/67 (79.1%)` `K = 4` (`38/67 = 56.7%` exact, `p = 0.000311`).

---

## 1. Preflight Results (Inference-Only on Frozen R7 + Stream Audit)

### 1.1 Separate-sentence filler test (`n = 200` paired, frozen `R7`, `K = 4`, chance `25%`)

Using the exact 200 D1 fact sets (`make_preflight_filler.py`, `analyze_preflight_filler.py`, unmodified `lisp/bin/qa-analysis.lisp` with `controls=0 paired=0`; summed option log-probability, not length-normalized; `<= 127` prompt tokens, `0` dropped; unrelated fillers `["It is cold.", "Water ran.", "Day began.", "The sun set."]`).
`s1`: `{P} lives somewhere. It is {V}.` vs `Someone lives in {V}. It is {P}.`
`s2`: `{P} lives somewhere. {FILLER} It is {V}.` vs `Someone lives in {V}. {FILLER} It is {P}.`
`s3`: `{P} lives in {V}. {FILLER}` vs `In {V} lives {P}. {FILLER}` (intact wording + separate unrelated sentence between facts).

| Condition | Reverse pf (ans-first) | Reverse vf (ans-after) | Reverse vf vs pf (p) | Extract pf (ans-after) | Extract vf (ans-first) | Extract vf vs pf (p) |
|---|---:|---:|---|---:|---:|---|
| D1 baseline (no filler) | 67/200 (33.5%) | 152/200 (76.0%) | +93/-8 (1.74e-19) | 179/200 (89.5%) | 72/200 (36.0%) | +1/-108 (3.39e-31) |
| D4 in-sentence filler | 73/200 (36.5%, p=0.238) | 147/200 (73.5%, p=0.332) | +83/-9 (3.92e-16) | 174/200 (87.0%, p=0.125) | 77/200 (38.5%, p=0.383) | +2/-99 (4.06e-27) |
| Preflight s3 (D1 + post-fact filler) | 59/200 (29.5%, +12/-20 p=0.215) | 127/200 (63.5%, +9/-34 p=0.00017) | +77/-9 (1.34e-14) | 159/200 (79.5%, +4/-24 p=0.00018) | 68/200 (34.0%, +16/-20 p=0.618) | +4/-95 (1.24e-23) |
| Preflight s1 (2-sent split, no filler) | 56/200 (28.0%, p=0.200) | 50/200 (25.0%, +7/-109 p=1.20e-24) | +18/-24 (0.441) | 68/200 (34.0%, p=7.70e-34) | 37/200 (18.5%, p=1.57e-5) | +8/-39 (5.54e-6) |
| Preflight s2 (2-sent split + filler) | 52/200 (26.0%, vs s1 p=0.572) | 55/200 (27.5%, vs s1 p=0.442) | +27/-24 (0.780) | 58/200 (29.0%, vs s1 p=0.253) | 46/200 (23.0%, vs s1 p=0.243) | +19/-31 (0.119) |

Interpretation: Splitting a fact across two sentences (`s1`) collapses R7 to chance (`25%-34%`) even without filler because two-sentence anaphora (`It is ...`) was never trained, leaving no floor effect for an added middle filler (`s2`, `p >= 0.243`). Putting a full unrelated filler sentence between intact D1 fact sentences (`s3`) lowers answer-after accuracy by `10.0-12.5` points (`152 -> 127` reverse, `179 -> 159` extract), but answer-after still beats answer-first overwhelmingly (`127 vs 59`, `p = 1.34e-14`; `159 vs 68`, `p = 1.24e-23`), confirming order rather than sentence adjacency dominates.

### 1.2 Streaming generator capacity check (`1,000` steps / `32,000` sequences, seed `111303`)

`lisp/experimental/pilot3/pilot3.lisp` (`mode=preflight`) streamed `1,000` steps (`32,000` four-person answer-before sequences, `40,653` packed questions; first `500` steps = `16,000` sequences, `20,337` questions, `52,746` target tokens; `0` copy sequences):

- **Exclusions (`1,133` strings):** Combined `asem-retrieval::heldout-strings` (`1,027`) plus `excluded-extra.txt` (`216` strings from `d1-entities.txt` and `dev.jsonl` options/passages), plus template/augmentation function words stripped from broad value pools. Excluded-entity leaks in both generated probes: `0/500` and `0/500`.
- **Clean entity pools:** `names = 21,889`, `towns = 760`, `jobs = 3,519`, `pets = 3,511`, `colors = 3,505`, `instruments = 3,505`.
- **Expected vs observed repeat rate across `1,000` steps (`32,000` sequences):**
  - 4-person tuple space: `C(21,889, 4) = 9.563e15`; analytical birthday expectation = `5.354e-8` across `32,000` sequences (`1.339e-8` across `16,000` sequences / `500` steps). Observed: `0` repeats at `500` and `1,000` steps.
  - Individual asked `(attr, person, value)` triple collision expectation across `40,653` questions (`1,000` steps) = `3.21` (`0.0079%`); observed: `0` at `500` and `1,000` steps (`redraws = 1`, zero `rendered_ids`, `passage_q0`, `fact4_tuple`, or `sentence_pv` repeats).
- **Frame coverage:** All `26/26` `:train` answer-before passage frames (`23` person-first reverse, `3` value-first extract) and all `28/28` `:train` question frames (`20` reverse, `8` extract) hit (`762-5,965` times each; `reverse = 31,549`, `extract = 9,104`).

---

## 2. Training Schedule, Step Budget, and Deviations

- **Step budget (`500` steps per arm, not `1,000`):** `docs/results/stage2/mission-20261111/README.md` caps matched pilots at `300-500` steps and permits the single `2,500`-step run only after Step 1 passes. Checkpoints saved at `250` and `500`; training loss logged every `50` steps.
- **Arms (`seed = 111303`, `batch = 32`, `accum = 1`, `16,000` seqs = `1.0` epoch, `0` copy seqs):**
  - **Arm 1 (`5e-4`):** R7 stage peak LR `5e-4`; trained `2026-11-13 10:24:59` to `10:29:56 JST` (`290.8 s` = `4 min 51 s`, `~5.3k-6.1k tok/s`).
  - **Arm 2 (`1e-3`):** Copy-stage peak LR `1e-3`; trained `2026-11-13 10:30:12` to `10:44:50 JST` (`871.3 s` = `14 min 31 s`; throughput dipped to `408-737 tok/s` at steps `400-450`, mathematically identical gradient steps for the fixed seed).
  - **Schedule (both arms):** Linear warmup `50` steps (`lr * (step + 1) / 50`), cosine decay (`asem-pretrain::lr-at`) over steps `50..499` to `min-lr-frac = 0.1` (`5e-5` and `1e-4` at step `500`), AdamW (`beta1 = 0.9`, `beta2 = 0.95`, `wd = 0.0`, fresh `m1 = 0, m2 = 0`), gradient clip `1.0`, token loss normalization.
- **Pre-training deviation recorded at 0 steps:** Replacing `asem-qa-analysis::plist-item` with `asem-qa-analysis:read-development` in `pilot3.lisp` before any step ran (`d796ce6f... -> fe4f5bf9...`); `data-train.sexp` (`bbf1d884...`), `probe-fresh-500.jsonl` (`b1be7315...`), `probe-seen-500.jsonl` (`0210d6da...`), and `R7` (`33a5c245...`) hashes remained untouched and bit-for-bit identical before, during, and after training.
- **Evaluations:** Unmodified `qa-analysis.lisp` (`controls=0 paired=0`) and `mission-eval.lisp` (`dataset=selection`) ran `2026-11-13 10:45:07` to `10:51:54 JST` (`6 min 47 s`). `dev.jsonl` was not scored on the new checkpoints (it was read, without scoring, to build `excluded-extra.txt`; it had been scored on `R7` and on Pilot 1 checkpoints). `confirmation.sexp` and both locked test sets were never opened.

---

## 3. Training Loss Trajectory and Slopes (`0 -> 250 -> 500`)

Every 50-step `train_loss` is mean NLL over supervised answer tokens; `dev_loss` and `dev_acc` (`gold_length_greedy_prefix_exact`) are measured on `probe-fresh-500` (`n = 500`).

| Step | LR (`5e-4`) | Train loss (`5e-4`) | Train acc (`5e-4`) | Fresh eval loss / acc (`5e-4`) | LR (`1e-3`) | Train loss (`1e-3`) | Train acc (`1e-3`) | Fresh eval loss / acc (`1e-3`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | - | - | - | 1.157 / 43.0% (215) | - | - | - | 1.157 / 43.0% (215) |
| 50 | 0.000500 | 0.905 | 0.379 | - | 0.001000 | 0.981 | 0.364 | - |
| 100 | 0.000487 | 0.939 | 0.407 | - | 0.000974 | 1.301 | 0.288 | - |
| 150 | 0.000448 | 0.939 | 0.393 | - | 0.000897 | 1.287 | 0.285 | - |
| 200 | 0.000389 | 0.850 | 0.471 | - | 0.000778 | 1.137 | 0.351 | - |
| 250 | 0.000316 | 0.767 | 0.468 | 0.920 / 58.0% (290) | 0.000631 | 1.031 | 0.387 | 1.172 / 48.4% (242) |
| 300 | 0.000237 | 0.748 | 0.493 | - | 0.000475 | 0.920 | 0.443 | - |
| 350 | 0.000164 | 0.660 | 0.528 | - | 0.000328 | 0.858 | 0.471 | - |
| 400 | 0.000104 | 0.578 | 0.576 | - | 0.000207 | 0.707 | 0.526 | - |
| 450 | 0.000064 | 0.590 | 0.581 | - | 0.000128 | 0.695 | 0.538 | - |
| 500 | 0.000050 | 0.560 | 0.589 | 0.679 / 63.6% (318) | 0.000100 | 0.644 | 0.567 | 0.727 / 63.6% (318) |

- **Under-training vs plateau diagnosis (steps `250 -> 500`):** Neither arm is on a flat plateau.
  - **`arm-5e-4`**: `train_loss` fell `-0.207` from step 250 to 500 (`mean(250..350) = 0.725 -> mean(400..500) = 0.576`, diff `-0.149`), and fresh `dev_loss` dropped **at constant speed**: `-0.237` (`0 -> 250`) and `-0.241` (`250 -> 500`), while Fresh exact climbed `43.0% -> 58.0% (+15.0 pp) -> 63.6% (+5.6 pp)`.
  - **`arm-1e-3`**: Peak LR `1e-3` caused an early shock at steps `50..150` (`0.981 -> 1.301`), keeping `dev_loss` slightly above R7 at step 250 (`1.172`). Once cosine decay brought LR below `5e-4`, loss plummeted `-0.387` (`1.031 -> 0.644`, `mean(250..350) = 0.936 -> mean(400..500) = 0.682`, diff `-0.254`), `dev_loss` plunged `-0.446` (`1.172 -> 0.727`), and Fresh exact surged **`+15.2 pp`** in the last 250 steps (`48.4% -> 63.6%`), matching `arm-5e-4` at step 500 despite losing ~200 steps to the LR transient.

---

## 4. Probe Results: Fresh 500 and Seen 500

Exact answer (`gold_length_greedy_prefix_exact`, autoregressive greedy decoding over `K = 8,192` tokens, chance ~0%) and `K = 4` candidate ranking (`summed_option_prediction`, chance `25%`). (`bounded_free_generation_exact` stays `0-3/500` across R7 and all runs because supervised targets contain answer tokens only with no trailing stop delimiter.) Paired McNemar tests are two-sided exact binomial against frozen `R7 step 0`.

| Model | Fresh Exact (`K = 8,192`, chance ~0%) | Fresh Exact vs R7 (g/l, p) | Fresh `K = 4` (chance 25%) | Fresh `K = 4` vs R7 (g/l, p) | Seen Exact (`K = 8,192`, chance ~0%) | Seen `K = 4` (chance 25%) | Seen `K = 4` vs R7 (g/l, p) | Seen - Fresh Exact / K4 |
|---|---:|---|---:|---|---:|---:|---|---:|
| R7 step 0 | 215/500 (43.0%) | - | 249/500 (49.8%) | - | 227/500 (45.4%) | 250/500 (50.0%) | - | +2.4 / +0.2 pp |
| arm-5e-4 step 250 | 290/500 (58.0%) | +115/-40, 1.33e-9 | 317/500 (63.4%) | +110/-42, 3.27e-8 | 316/500 (63.2%) | 339/500 (67.8%) | +129/-40, 4.04e-12 | +5.2 / +4.4 pp |
| arm-5e-4 step 500 | 318/500 (63.6%) | +135/-32, 2.96e-16 | 343/500 (68.6%) | +129/-35, 7.45e-14 | 398/500 (79.6%) | 407/500 (81.4%) | +175/-18, 1.69e-33 | +16.0 / +12.8 pp |
| arm-1e-3 step 250 | 242/500 (48.4%) | +106/-79, 0.0556 | 291/500 (58.2%) | +110/-68, 0.00203 | 295/500 (59.0%) | 329/500 (65.8%) | +130/-51, 3.83e-9 | +10.6 / +7.6 pp |
| arm-1e-3 step 500 | 318/500 (63.6%) | +144/-41, 1.26e-14 | 349/500 (69.8%) | +134/-34, 3.06e-15 | 383/500 (76.6%) | 395/500 (79.0%) | +172/-27, 5.00e-27 | +13.0 / +9.2 pp |

### Breakdown by question category (`reverse`, `n = 404`; `extract`, `n = 96`)

| Category | Model | Fresh Exact (vs R7 p) | Fresh `K = 4` (vs R7 g/l, p) | Seen Exact (vs R7 p) | Seen `K = 4` (vs R7 g/l, p) |
|---|---|---:|---|---:|---|
| Reverse (n=404, chance exact ~0%, K4 25%) | R7 step 0 | 174/404 (43.1%) | 190/404 (47.0%) | 181/404 (44.8%) | 192/404 (47.5%) |
| Reverse | arm-5e-4 step 250 | 231/404 (57.2%, 6.72e-7) | 240/404 (59.4%, +87/-37 8.27e-6) | 261/404 (64.6%, 2.69e-11) | 270/404 (66.8%, +109/-31 2.25e-11) |
| Reverse | arm-5e-4 step 500 | 252/404 (62.4%, 1.05e-11) | 264/404 (65.3%, +103/-29 6.47e-11) | 319/404 (79.0%, 1.85e-30) | 325/404 (80.4%, +148/-15 1.14e-28) |
| Reverse | arm-1e-3 step 250 | 201/404 (49.8%, 0.0328) | 216/404 (53.5%, +88/-62 0.0409) | 247/404 (61.1%, 1.28e-7) | 253/404 (62.6%, +107/-46 8.93e-7) |
| Reverse | arm-1e-3 step 500 | 260/404 (64.4%, 8.84e-13) | 270/404 (66.8%, +111/-31 9.17e-12) | 306/404 (75.7%, 3.12e-24) | 312/404 (77.2%, +143/-23 2.30e-22) |
| Extract (n=96, chance exact ~0%, K4 25%) | R7 step 0 | 41/96 (42.7%) | 59/96 (61.5%) | 46/96 (47.9%) | 58/96 (60.4%) |
| Extract | arm-5e-4 step 250 | 59/96 (61.5%, 0.000277) | 77/96 (80.2%, +23/-5 0.000912) | 55/96 (57.3%, 0.122) | 69/96 (71.9%, +20/-9 0.0614) |
| Extract | arm-5e-4 step 500 | 66/96 (68.8%, 4.65e-6) | 79/96 (82.3%, +26/-6 0.000535) | 79/96 (82.3%, 1.02e-8) | 82/96 (85.4%, +27/-3 8.43e-6) |
| Extract | arm-1e-3 step 250 | 41/96 (42.7%, 1.00) | 75/96 (78.1%, +22/-6 0.00372) | 48/96 (50.0%, 0.86) | 76/96 (79.2%, +23/-5 0.000912) |
| Extract | arm-1e-3 step 500 | 58/96 (60.4%, 0.00599) | 79/96 (82.3%, +23/-3 8.80e-5) | 77/96 (80.2%, 3.35e-7) | 83/96 (86.5%, +29/-4 1.09e-5) |

### Seen vs Fresh gap audit (step-bucket exact accuracy on `probe-seen-500`, `0` sequence repeats)

At `step 250`, `probe-seen-500` items with `train_step > 250` (`n = 250`, **not yet trained on**) matched `probe-fresh-500` almost identically (`5e-4`: `145/250 = 58.0%` vs Fresh `58.0%`; `1e-3`: `124/250 = 49.6%` vs Fresh `48.4%`), ruling out difficulty skew. Once exposed for a single batch (`train_step <= 250`), accuracy jumped immediately and persisted for 250-500 subsequent steps even though no sequence repeated:

| Seen step window (`n`) | R7 step 0 | `arm-5e-4` step 250 | `arm-5e-4` step 500 | `arm-1e-3` step 250 | `arm-1e-3` step 500 |
|---|---:|---:|---:|---:|---:|
| steps 1..100 (n=100) | 39 | 61 | 77 | 67 | 76 |
| steps 101..200 (n=100) | 46 | 74 | 83 | 63 | 69 |
| steps 201..250 (n=50) | 25 | 36 | 38 | 41 | 39 |
| Subtotal `step <= 250` (n=250) | 110 (44.0%) | **171 (68.4%)** | **198 (79.2%)** | **171 (68.4%)** | **184 (73.6%)** |
| steps 251..300 (n=50) | 17 | 33 (unseen) | 44 | 27 (unseen) | 41 |
| steps 301..400 (n=100) | 50 | 57 (unseen) | 79 | 50 (unseen) | 80 |
| steps 401..500 (n=100) | 50 | 55 (unseen) | 77 | 47 (unseen) | 78 |
| Subtotal `step > 250` (n=250) | 117 (46.8%) | **145 (58.0% unseen)** | **200 (80.0%)** | **124 (49.6% unseen)** | **199 (79.6%)** |

Even seeing an exact `(attr, P, V)` association once leaves a persistent `+13 to +16 pp` episodic boost alongside the `+20.6 pp` out-of-sample rule gain.

---

## 5. Frozen D1 200 Results (`K = 4`, chance `25%`, plus exact prefix)

| Cell (`n = 200` each) | Metric | R7 step 0 | `arm-5e-4` step 250 | `arm-5e-4` step 500 | `arm-1e-3` step 250 | `arm-1e-3` step 500 |
|---|---|---:|---:|---:|---:|---:|
| Person-first reverse (`d1-rev-pf`, answer-before) | `K = 4` (chance 25%) | 67 (33.5%) | 88 (44.0%, +25/-4 p=0.000104) | 78 (39.0%, +21/-10 p=0.0708) | 72 (36.0%, +22/-17 p=0.522) | 83 (41.5%, +29/-13 p=0.0195) |
| `d1-rev-pf` | Exact (`K = 8,192`) | 31 (15.5%) | 40 (20.0%, +16/-7 p=0.0931) | 43 (21.5%, +17/-5 p=0.0169) | 24 (12.0%, +10/-17 p=0.248) | 42 (21.0%, +22/-11 p=0.0801) |
| `d1-rev-pf`: `town-pf` (`n = 100`) | `K = 4` / Exact | 33 / 19 | 47 (+15/-1 p=0.000519) / 23 | 44 (+12/-1 p=0.00342) / 24 | 38 (+11/-6 p=0.332) / 14 | 42 (+14/-5 p=0.0636) / 20 |
| `d1-rev-pf`: `instrument-pf` (`n = 100`) | `K = 4` / Exact | 34 / 12 | 41 (+10/-3 p=0.0923) / 17 | 34 (+9/-9 p=1.00) / 19 | 34 (+11/-11 p=1.00) / 10 | 41 (+15/-8 p=0.210) / 22 |
| Value-first reverse (`d1-rev-vf`, answer-after) | `K = 4` (chance 25%) | 152 (76.0%) | 60 (30.0%, +6/-98 p=1.59e-22) | 58 (29.0%, +3/-97 p=2.63e-25) | 57 (28.5%, +7/-102 p=9.84e-23) | 62 (31.0%, +7/-97 p=2.25e-21) |
| `d1-rev-vf` | Exact (`K = 8,192`) | 75 (37.5%) | 11 (5.5%, +0/-64 p=1.08e-19) | 6 (3.0%, +0/-69 p=3.39e-21) | 3 (1.5%, +0/-72 p=4.24e-22) | 0 (0.0%, +0/-75 p=5.29e-23) |
| Person-first extract (`d1-ext-pf`, answer-after) | `K = 4` (chance 25%) | 179 (89.5%) | 91 (45.5%, +6/-94 p=2.01e-21) | 76 (38.0%, +14/-117 p=2.04e-21) | 88 (44.0%, +13/-104 p=8.49e-19) | 75 (37.5%, +13/-117 p=4.31e-22) |
| `d1-ext-pf` | Exact (`K = 8,192`) | 123 (61.5%) | 6 (3.0%, +0/-117 p=1.20e-35) | 4 (2.0%, +0/-119 p=3.01e-36) | 6 (3.0%, +1/-118 p=3.61e-34) | 2 (1.0%, +0/-121 p=7.52e-37) |

- **Expected catastrophic forgetting of answer-after skills:** Training `100%` on answer-before items with zero replay erased `d1-rev-vf` (`152 -> 58/62`) and `d1-ext-pf` (`179 -> 76/75`).
- **Post-hoc diagnostic explaining `d1-rev-pf` (`41.5%`) vs `probe-fresh-500` reverse (`64.4%` exact / `66.8%` `K = 4`):** On `d1-rev-pf`, frozen `R7` already failed to copy a passage name (`61/200 = 30.5%` starts with any of the 4 passage names; `35.5%` at `1e-3 step 500`), often emitting synthetic prefixes (`Finn -> ' Futen'`, `Tess -> ' Toud'`, `Aster -> ' Aul'`, `Edith -> ' Aam'`). Splitting the 36 D1 person names by how often every BPE subword of `" " + name` occurred in `data-train.sexp` (`diagnose_pilot3.py`):

| Post-hoc `d1-rev-pf` split (`n`) | Metric | R7 step 0 | `arm-5e-4` step 500 | `arm-1e-3` step 500 |
|---|---|---:|---:|---:|
| All name BPE subwords `>= 100` in train (`n = 67`) | `K = 4` (chance 25%) | 34/67 (50.7%) | 48/67 (71.6%, +15/-1 p=0.000519) | **53/67 (79.1%, +23/-4 p=0.000311)** |
| `>= 100` subword group | Exact / `gen_in_opts` | 21 (31.3%) / 27 | 34 (50.7%, p=0.000244) / 39 | **38 (56.7%, p=0.000488) / 45 (67.2%)** |
| Rare subword `1..99` (`n = 43`) | `K = 4` / Exact | 18 (41.9%) / 8 (18.6%) | 18 (41.9%, p=1.00) / 9 (20.9%) | 19 (44.2%, p=1.00) / 4 (9.3%) |
| `0`-count subword in train (`n = 90`: `Edith`, `Finn`, `Tess`, `Adam`, ...) | `K = 4` / Exact / `gen_in_opts` | 15 (16.7%) / 2 (2.2%) / 17 | 12 (13.3%, p=0.508) / 0 (0.0%) / 13 | **11 (12.2%, p=0.289) / 0 (0.0%) / 14 (15.6%)** |

On D1 names whose BPE subwords are actually exercised by the synthetic training stream (`n = 67`: `Gus, Sara, Jean, Kai, Lena, Otto, Pedro, Kira, Mia, Neil, Lou, Rosa`), answer-before `d1-rev-pf` reached **`79.1%` `K = 4`** (`53/67`), essentially matching `R7`'s `d1-rev-vf` answer-after score on those same 67 items (`55/67 = 82.1%`). On the `90` D1 names whose FineWeb BPE subwords (`'inn'`, `'ess'`, `' Ed'`, `' Em'`, `' Ad'`, `'ugo'`, `'ack'`, ...) occurred `0` times in Pilot 3's synthetic data with `0` copy-stage replay, generation dropped to `0/90` and `K = 4` fell below chance (`12.2%`).

---

## 6. Retention on Mission Selection Set (Information Only, Non-Gating)

`selection.sexp`, `n = 500` (`100` per category, `K = 4`, chance `25%`, ~`+-4.4` points overall CI).

| Category (`n`) | R7 step 0 | `arm-5e-4` step 250 (vs R7 g/l, p) | `arm-5e-4` step 500 (vs R7 g/l, p) | `arm-1e-3` step 250 (vs R7 g/l, p) | `arm-1e-3` step 500 (vs R7 g/l, p) |
|---|---:|---|---|---|---:|
| extract (100) | 64 | 49 (+10/-25, 0.0167) | 34 (+7/-37, 5.30e-6) | 42 (+9/-31, 0.00068) | 39 (+12/-37, 0.00047) |
| relation (100) | 55 | 35 (+5/-25, 0.000325) | 34 (+7/-28, 0.000508) | 32 (+9/-32, 0.000431) | 30 (+8/-33, 0.000112) |
| reverse (100) | 36 | 34 (+12/-14, 0.845) | 33 (+12/-15, 0.701) | 30 (+13/-19, 0.377) | 36 (+16/-16, 1.00) |
| compare (100) | 28 | 25 (+5/-8, 0.581) | 28 (+8/-8, 1.00) | 24 (+6/-10, 0.454) | 31 (+12/-9, 0.664) |
| temporal (100) | 25 | 29 (+18/-14, 0.597) | 26 (+15/-14, 1.00) | 28 (+14/-11, 0.690) | 22 (+12/-15, 0.701) |
| **ALL (500)** | **208** | **172 (+50/-86, 0.00256)** | **155 (+49/-102, 1.93e-5)** | **156 (+51/-103, 3.38e-5)** | **158 (+60/-110, 0.000155)** |

---

## 7. Next Decision (Recommended Without Running)

Per Rule 5, all 3 pilots are used, so I stopped and did not start any further run, score `dev.jsonl` on the new checkpoints, or open `confirmation.sexp` or the locked test sets. Pilot 3 rules out a hard architectural failure on answer-before lookup (`43.0% -> 63.6%` exact and falling loss curve, `79.1%` `K = 4` on `>= 100`-subword D1 items), and identifies three concrete requirements before any next training stage:

1. **BPE-balanced entity pool or copy replay (`~10-20%` copy stage + answer-after/other tasks):** Without copy-stage sequences or diverse real-name subword coverage, synthetic names fail to calibrate FineWeb subwords used in `D1`/`dev` names (`90/200` `D1` items had `0`-count subwords in Pilot 3 training).
2. **Longer horizon (`> 500` steps) or warmup into `5e-4` (not `1e-3`):** `1e-3` caused early instability; `5e-4` learned smoothly across all `500` steps and had not converged (`dev_loss` still dropping `-0.24` per 250 steps).
3. **Wait for explicit approval** on whether to: (a) run a copy-mix + subword-balanced diagnostic or re-open pilot allocation, rather than changing depth/width (not supported by this outcome).

---

## 8. Artifact and SHA-256 Manifest

All scripts, probe inputs, raw predictions, and hashes live in `docs/results/stage2/mission-20261111/pilot3/` (`manifest.json` holds all 63 bundle SHA-256 hashes):

- Driver: `lisp/experimental/pilot3/pilot3.lisp` (`fe4f5bf9fa811f9d36eb42d365c8ca4ba4e8ccfe69697736fb91242c58f29c7c`)
- Training data (`16,000` seqs, `0` copy): `data/runs/retrieval/p3-learnability/data-train.sexp` (`bbf1d88431bb4df7e379ce3d03a002c7c125ba65bc871ebfc767d157d6bb685f`)
- Frozen `R7` (`unchanged`): `data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt` (`33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`)
- Arm `5e-4` ckpts: `step-00250.ckpt` (`3de01901...`), `step-00500.ckpt` / `last.ckpt` (`6a4fdc5f5fed6bd8d00fa68506fc6479328137d67330fba6aebb39b96219c436`)
- Arm `1e-3` ckpts: `step-00250.ckpt` (`1bfd2059...`), `step-00500.ckpt` / `last.ckpt` (`2041eaf1b471f13f3d66edf039d96e3725ef777ae63b405792d5348f3bd21abc`)
- Probes & audit: `protocol.md`, `data-audit.json`, `preflight-stream.json`, `excluded-extra.txt`, `probe-fresh-500.jsonl` (`b1be7315...`), `probe-seen-500.jsonl` (`0210d6da...`), `eval_pilot3.sh`, `analyze_pilot3.py`, `diagnose_pilot3.py`, `results-summary.json`, `manifest.json`

TL;DR: Pilot 3 landed in **Outcome E (ambiguous)**: `R7`'s `10.52M` architecture clearly learned fresh four-person answer-before lookup (`43.0% -> 63.6%` exact, `49.8% -> 69.8%` `K = 4`, `p = 3.06e-15`) with loss still falling steeply at step `500`, ruling out Outcome D's capacity limit. Seen probes reached `76.6%-79.6%` (`+13.0` to `+16.0 pp` above fresh even with zero sequence repeats), and `D1` person-first reverse moved only from `67/200` to `83/200` because `90/200` `D1` names rely on FineWeb BPE subwords never produced by the synthetic training pool (`0/90` exact), while `D1` names with `>= 100` training subword occurrences jumped from `50.7%` to `79.1%` `K = 4`. Stopped after Pilot 3 without scoring `dev.jsonl` on the new checkpoints, opening the confirmation or test sets, or launching any `2,500`-step run.
