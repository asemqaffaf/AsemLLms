# Pilot 4 Experimental Protocol: Retention-Aware & Copy-Grounded Retrieval (`2x2`)

Date: 2026-11-13
Branch: `stage2-qa`
Status: **APPROVED FINAL PILOT PROTOCOL (Asem decision, 2026-11-13)**

---

## 1. Objective & Design Principles

Pilot 3 proved that the frozen `R7` architecture (`10.52M` parameters, `8` layers, `8` heads, `d = 256`, `ctx = 128`) can learn answer-before lookup (`pf_rev|seen100`: `24.0% -> 44.0%` Greedy Exact on independent probe), but exposed two fatal failure modes diagnosed in [DATA_AUDIT.md](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/DATA_AUDIT.md):

1. **Entity-Subword & Contextual-Copy Collapse:** Dropping `copy-mix` destroyed `R7`'s induction circuit on zero-retrieval FineWeb tokens (`95.3% -> 5.9% / 0.1%`), driving `s1_recomb_tr` and `s1_heldout_ev` prose copying to `0/80 (0.0%)` and breaking even `D1` `Zero->Hi100` / `Hi100->Zero` swaps (`p ~ 1e-17`). Meanwhile, `R7`'s random `copy-mix` alone gave only `31.2%-32.5%` on prose copying of unseen-subword names, showing that random token repetition does not automatically compose into multi-token name extraction inside prose sentences.
2. **Catastrophic Forgetting of Existing Skills & Formats:** Training purely on generative answer-before lookup (`P3 arm-5e-4`) caused `K=4` aggregate retention (`n = 1,000`) to drop `-10.8pp` (`55.3% -> 44.5%`, `p = 6.40e-10`; `-17.1pp` at `1e-3`, `p = 1.21e-18`), specifically dropping `extract_ret` by `-25.5pp` (`77.0% -> 51.5%`, `p = 4.64e-10`), `relation_ret` by `-14.0pp` (`64.5% -> 50.5%`, `p = 3.94e-4`), `reverse_ret` by `-10.5pp` (`58.5% -> 48.0%`, `p = 0.0139`; `ans_after` subgroup `-27.0pp`: `74.0% -> 47.0%`, `p = 1.43e-5`), `temporal_ret` by `-13.0pp` at `1e-3` (`41.5% -> 28.5%`, `p = 5.44e-3`), and `s1_seen_synth|vf` answer-after copying from `90.0%` to `47.5%` (`5.0%` at `1e-3`).

Pilot 4 runs a **controlled 2x2 factorial experiment** isolating the main effects and interaction of **(1) Contextual Entity/Span Copy Coverage** and **(2) Calibrated Multi-Task & Format Replay**, with zero changes to model capacity, architecture, or tokenizer.

---

## 2. Frozen Shared Configuration Across All 4 Arms (`A, B, C, D`)

Every arm is initialized identically and trained under identical optimization rules:

- **Initialization checkpoint:** `data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt` (SHA-256 `33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`).
- **Architecture & Tokenizer:** `10,523,393` params (`8L, 8H, d=256, ctx=128`), `f32`, `data/fineweb/stage1/tokenizer.bpe` (`V = 8,192`), unchanged.
- **Optimizer & Schedule:** AdamW (`b1 = 0.9`, `b2 = 0.95`, `wd = 0.0`, `grad_clip = 1.0`), peak `lr = 5e-4`, identical linear `warmup = 50` steps across all 4 arms, cosine decay with `min_lr_frac = 0.1` (`5e-5`) **scaled to each arm's step length** (`190 / 270 / 420 / 500` steps).
- **Fixed Answer-Before Dose & Batching:** `batch = 32`, `accum = 1` (`1.0` exact epoch per arm), `aux_lm = 0`, `workers = 8`. Answer-before retrieval is held strictly constant at **`6,080` identical sequences** across all four arms, with Copy (`2,560`) and/or Replay (`7,360`) added on top:
  - **Arm A (`190` steps):** `6,080` `ans_before` (`No` Copy, `No` Replay)
  - **Arm B (`270` steps):** `6,080` `ans_before` + `2,560` `copy` (`8,640` seqs total)
  - **Arm C (`420` steps):** `6,080` `ans_before` + `7,360` `replay` (`13,440` seqs total)
  - **Arm D (`500` steps):** `6,080` `ans_before` + `2,560` `copy` + `7,360` `replay` (`16,000` seqs total)
- **Determinism & Strict Zero-Leakage Quarantine:** Shared generator/shuffle RNG seed `111404`; retrieval & replay generators reuse `broad-pool :names/:towns` + `vocab_words` + `split-pool ... :train` + `*places* :train` ONLY. All `eval_heldout` tokens (`845` tokens), all `reserved_recomb_tr_words` (`320`), all non-numeric `probes/*.jsonl` entities & options, all `D1` entities (`d1-entities.txt`), `selection.sexp`, `confirmation.sexp`, `dev.jsonl`, `test.jsonl`, `pilot3/excluded-extra.txt`, and `qagen.lisp :test` pools are **strictly forbidden** in EVERY arm's data (including Retrieval, Copy, and Replay, and excluding `eval_heldout_ids` plus all `15` single-token `D1+probe` IDs from the random-token induction guard). Programmatic whole-word and token-ID verification across `100%` of generated sequences is required prior to training step 1.

---

## 3. Experimental Arms (`2x2` Factorial Matrix)

| Arm | Steps | Total Seqs | Entity / Span Copy (`2,560`) | Multi-Task Replay (`7,360`) | Answer-Before Dose (`6,080` Shared) |
| --- | ---: | ---: | --- | --- | --- |
| **A** | `190` | `6,080` | **No** (`0`) | **No** (`0`) | `6,080` (`100%`, Pilot 3-style matched control at `lr = 5e-4`) |
| **B** | `270` | `8,640` | **Yes** (`2,560` / `29.6%`) | **No** (`0`) | `6,080` (`70.4%`) |
| **C** | `420` | `13,440` | **No** (`0`) | **Yes** (`7,360` / `54.8%`) | `6,080` (`45.2%`) |
| **D** | `500` | `16,000` | **Yes** (`2,560` / `16.0%`) | **Yes** (`7,360` / `46.0%`) | `6,080` (`38.0%`) |

*Arm Equivalence & Control Purity:* All 4 arms contain the **exact same `6,080` `ans_before` sequences**, `B` and `D` share the **exact same `2,560` `copy` sequences**, and `C` and `D` share the **exact same `7,360` `replay` sequences** (permuted deterministically per arm with seed `111404`). Any failure on `G1` in `C` or `D` therefore directly measures optimization/replay/copy interference against `A`/`B` at identical `ans_before` dose. Both `ans_before` and `replay` draw entities strictly from `broad_pool + vocab_words + split-pool :train`, **never** inserting `train_copy` or `fw_tr_only` words into retrieval QA items. Therefore, any improvement by Arm B or Arm D on `pf_rev|recomb_tr` (`fresh_ans_before`), `s1_recomb_tr`, or `L3_recomb_tr` (`entity_copy`) measures **pure cross-task generalization from Contextual Entity Copy to Retrieval**, not retrieval item memorization.

---

## 4. Exact Specification of the Two Interventions

### 4.1 Intervention 1: Contextual Entity/Span Copy Curriculum (`Arms B & D`, `2,560` seqs)

Designed directly from Causal Experiments 1-3 (`DATA_AUDIT.md` Section A.4), addressing all three requirements simultaneously (`onset tok0`, `continuation tail tok1+`, and prose framing invariance) while keeping `eval_heldout` (`25%`) untouched:

1. **FineWeb Real-Word Prose Entity Copy (`1,000` sequences):**
   - Draws `fw_tr_only_2tok` (`608` real FineWeb 2-token capitalized words, `223` vowel-initial) and `fw_tr_only_3tok` (`166` 3-token words) strictly from `train_copy` (`75%` partition) after filtering against `probes + D1 + excluded`, paired with clean `broad_pool :towns` or `:train` attr contexts (`0` probe/`D1` entities).
   - `60%` 1-sentence (`50%` person-first `"{E} lives in {T}."` / `"{E} plays the {V}."`, `50%` value-first `"In {T} lives {E}."` / `"The {V} is the instrument {E} plays."`) -> reverse question (`"Who lives in {T}?"` -> `E`, `"Who plays the {V}?"` -> `E`); `40%` 2-to-3 sentence unambiguous factual contexts (`2` or `3` distinct entities, asking about one unique entity). Encoded via `multi-ex` (`no-options` `"Text: ...\nQ: ...\nThe answer is"`, supervised strictly on `" " + E` tokens, unit weight per target token) to directly train open-ended `K=8,192` reproduction (`tok0` + `tok1+`) and prose order invariance.
2. **Full `train_copy` Subword-Composition Coverage (`960` sequences):**
   - Deterministically cycles through **100% of the allowed `train_copy` tokens** (`573` `space_title_alpha`, `1,309` `bare_lower_alpha`, `558` `space_lower_alpha`, `101` `bare_title_alpha`, after excluding any single-token `probe/D1` overlaps), constructing BPE-roundtrip-verified 2-token and 3-token entities/nouns (**explicitly excluding** all `320` `reserved_recomb_tr_words` and all `probe + D1` entities).
   - Guarantees every `train_copy` onset (`space_title_alpha`, `space_lower_alpha`) AND every `train_copy` continuation (`bare_lower_alpha`, `bare_title_alpha`) token appears `>= 2-3` times as a supervised target token inside `pf` and `vf` prose extraction/reverse sentences (`"{P} works as {V}." -> "What is {P}'s job?"` for lowercase spans, `"{P} lives in {T}." -> "Who lives in {T}?"` for TitleCase spans, via `no-options` `multi-ex`).
3. **Random-Token Induction Health Guard (`600` sequences):**
   - Same `[BOS, x_1..x_k, x_1..x_k]` structure as `make-copy-examples` (`:mode :copy`, sequence-normalized loss weight `1.0`, `k in [12, 48]` fitting `ctx = 128`), **restricted strictly to allowed vocabulary IDs** (`(1..8191) \ (eval_heldout_ids union d1_probe_single_tok_ids)`) so `eval_heldout` (`845` tokens) and single-token `D1+probe` entities (`15` IDs) receive **strictly 0 training updates**. Prevents collapse of general content-based attention induction (`Exp 2`).

### 4.2 Intervention 2: Calibrated Multi-Task & Format Replay (`Arms C & D`, `7,360` seqs)

Proportioned to restore every `R7` skill & format lost under `Pilot 3`, using **`100%` `options=1` (`A..D` `OPTIONS-BLOCK`) `multi-ex` (`:mode :multi`)** to preserve exact `1:1` format and per-token loss parity with `R7` and `Pilot 3`:

- **Answer-Before Retrieval Component (`Arms A, B, C, D`, `6,080` identical seqs):**
  - `L2` (`25%`) and `L3` (`75%`) `ans_before` attribute items (`options=1`, `augment=1`, `multi-ex`): `~75%` `pf-reverse` (balanced across all 5 attributes: `town, job, pet, color, instrument`) and `~25%` `vf-extract` (`town, instrument`), matching `Pilot 3` format (`A..D` options).
- **Multi-Task Replay Mix (`Arms C & D`, `7,360` identical seqs, `options=1`, `levels = 1,2,3` `20%/30%/50%` with `augment = 1`, `multi-ex`):**
  - `extract_ret` Forward Attribute `ans_after` (`25.0%` of replay): `town, job, pet, color, instrument`, explicitly enforcing forward `ans_after` s-frames (`{P}` before `{V}`).
  - `extract_ret` Numeric `ans_after` (`17.0%` of replay): `age, score, height` (`18-199`), restoring `digits` subword competence (`0%` in P3).
  - `vf-reverse` `ans_after` (`14.8%` of replay): value-first sentences -> `"Who ...?"` reverse questions (explicitly protecting `D1 vf-rev` & `s1_seen|vf`).
  - `rel` Directed Relations (`15.9%` of replay): all 6 verbs (`teach, hire, call, thank, visit, help`), balanced `pairs` vs `chain`, `q-subj` vs `q-obj`, `ans_before` vs `ans_after`.
  - `cmp` Numeric Comparison (`13.6%` of replay): 4-person `max` vs `min` across `age, score, height`.
  - `tmp` Temporal Sequence (`13.6%` of replay): 4-stop narratives across `first, last, after, before`.

---

## 5. Pre-Registered Execution, Binding Gates & Final Stop Rule (Summary of `eval-spec.md`)

1. All final arm checkpoints (**`Arm A`: step `190`, `Arm B`: step `270`, `Arm C`: step `420`, `Arm D`: step `500`**, plus `step 250` diagnostic on `B, C, D`) are evaluated via [eval_probes.lisp](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/eval_probes.lisp) on all **7 independent frozen probes** (`1,600` items) + induction health check (`150` sequences).
2. **Binding Floors & Frozen `D1` 200 Reporting:** Every absolute floor written in `eval-spec.md` (`G1..G7`, `G2c`, and Aggregate `>= 533/1000 = 53.3%`) is **strictly binding** (`p > 0.05` vs `R7` alone does **not** count as no harm). In addition, evaluate and report the frozen **`D1 200`** cells (`d1-rev-pf`, `d1-rev-vf`, `d1-ext-pf`) for every arm (`selection.sexp` optional non-gating diagnostic; `dev.jsonl`, `confirmation.sexp`, and locked test sets remain untouched). Passing `G1a` is a learning signal only; any full stage also requires `D1 rev-pf >= 120/200`.
3. **Single-Seed Noise Threshold:** Single-seed pilot; treat inter-arm differences under `6.0` points on `n = 300` as noise.
4. **Final Stop Rule (No 5th Pilot):** If **no arm** passes **`G1` + `G2c` + every retention gate (`G3..G7` and `Aggregate`)**, **stop the Stage-2 retrieval line** and write a final report. If an arm passes (`G1 + G2c + G3..G7 + Aggregate`, reporting `G2a/G2b/G2d` as well), propose the next stage to the user and do **not** run it without explicit approval.

---

## 6. Pre-Registered Dataset Manifest & Zero-Leakage Checksums (`mode=prepare`)

Verified before training step 1 by both `lisp/experimental/pilot4/pilot4.lisp` (`mode=prepare`) and [verify_pretrain.py](file:///Users/asemqaffaf/AsemLLMs/docs/results/stage2/mission-20261111/pilot4/verify_pretrain.py):

- **Exact Shared Components:** `Arm A` (`6,080` seqs) is an exact subset of `B` (`8,640`), `C` (`13,440`), and `D` (`16,000`); `B` and `D` share the exact same `2,560` copy sequences (`1,000` FineWeb prose copy, `960` composed subword prose copy achieving min target count `>= 2` across all 4 `train_copy` classes, `600` random-token induction guard); `C` and `D` share the exact same `7,360` multi-task replay sequences (`1,840` `ext-attr`, `1,251` `ext-num`, `1,089` `vf-rev`, `1,170` `rel`, `1,001` `cmp`, `1,009` `tmp`).
- **Zero-Leakage Audit (`0` hits across all 4 arms and all components):** `eval_heldout_ids (845)` = `0` in `A, B, C, D` (including induction guard); `guard_forbid_ids (81 single-token D1+probe IDs)` = `0` in `B, D`; whole-word entity leaks against all `1,600` probe items + `D1` + `excluded-extra` + `reserved_recomb_tr_words` + `synth_towns_eval_pool` + `fw_ev_only` + `heldout-strings` = `0` in `ans_before`, `copy_prose`, `replay`, and `internal-dev-200`.

| Artifact / Dataset | Size / Role | SHA-256 |
| --- | --- | --- |
| `data/runs/retrieval/p4-retention/data-train-a.sexp` | `6,080` seqs (`190` steps) | `625ff352139e0aa622fac9a264f9a3201bf23d39da1200c7954b247ef1a045b3` |
| `data/runs/retrieval/p4-retention/data-train-b.sexp` | `8,640` seqs (`270` steps) | `0c7b598d0d868ba7b438af00a6bc7f6651ee303b3aef042d07c85e87ab196b18` |
| `data/runs/retrieval/p4-retention/data-train-c.sexp` | `13,440` seqs (`420` steps) | `3c10613c38386cd451c0a1bace90b636c39a840c14a8f81de606916d6f35c2d4` |
| `data/runs/retrieval/p4-retention/data-train-d.sexp` | `16,000` seqs (`500` steps) | `16a65fe75b63846ab237c5d6106fd100eabc2770fb74ff1af4ecab5f30d1816d` |
| `data/runs/retrieval/p4-retention/internal-dev-200.sexp` | `200` internal sanity dev items | `544175e8698c5dd8337a937872cd871a4ee5d25f83faf6972f844680fc6f7da5` |
| `data/runs/retrieval/p4-retention/pilot4-assets.json` | Quarantined pools & ID sets | `c74c62054e0c5ce970590d0c3be9acefb9d92baaee0ea169aa577a7d2cceb375` |
| `docs/results/stage2/mission-20261111/pilot4/eval-spec.md` | Binding gates spec | `c90cc05a05ce7ee0174bcfa4064a419d181809566e177171992b33d3e37bebdb` |
| `lisp/experimental/pilot4/pilot4.lisp` | Generator & training driver | `5fca7714dfcd48413a068701bb63c500cadf53d6f5bd6d98c39b5987e7d2c176` |
| `data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt` | Frozen `R7` init ckpt | `33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1` |
