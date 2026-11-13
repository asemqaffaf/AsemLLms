# Pilot 3: Answer-Before Learnability Control (Pre-registered 2026-11-13 10:21 JST, before any training)

Third and last matched pilot (3 of 3). This is NOT a dev-accuracy push and NOT the 2,500-step run.

## Research Question and Step Count Justification

Can R7's architecture (10,523,393 parameters, 8 layers, 8 heads, width 256, ff 1,024, context 128, vocabulary 8,192, gated tied copy path, relative-position attention biases K = 8) learn four-person answer-before-entity lookup when 100% of training data is freshly generated answer-before items and memorizing items cannot pay off?

**Step budget:** 500 steps per arm (NOT 1,000). The mission rules (`docs/results/stage2/mission-20261111/README.md` and `manifest.json`) limit each matched pilot to 300-500 steps and allow the single 2,500-step run only after Step 1 passes. Checkpoints are saved every 250 steps (steps 250 and 500) so the learning curve from step 0 -> 250 -> 500 is visible.

## Preflight Findings (Inference-Only on Frozen R7 + Generator Check)

### 1. Separate-sentence filler test (n = 200 paired, frozen R7, K = 4, chance 25%, summed option log-prob)

Using the exact 200 D1 fact sets (`make_preflight_filler.py`, `analyze_preflight_filler.py`, unmodified `lisp/bin/qa-analysis.lisp` with `controls=0 paired=0`), paired directly against D1 baseline and D4 in-sentence filler (`["It is cold.", "Water ran.", "Day began.", "The sun set."]`, all items <= 127 tokens):

- **s1** (cross-sentence split, no filler): `{P} lives somewhere. It is {V}.` vs `Someone lives in {V}. It is {P}.`
- **s2** (cross-sentence split WITH unrelated filler sentence between entity and answer halves): `{P} lives somewhere. {FILLER} It is {V}.` vs `Someone lives in {V}. {FILLER} It is {P}.`
- **s3** (intact D1 training wording WITH one unrelated filler sentence immediately after every entity sentence): `{P} lives in {V}. {FILLER}` vs `In {V} lives {P}. {FILLER}`

| Condition | Reverse pf (ans-first) | Reverse vf (ans-after) | Reverse vf vs pf (p) | Extract pf (ans-after) | Extract vf (ans-first) | Extract vf vs pf (p) |
|---|---:|---:|---|---:|---:|---|
| D1 baseline (no filler) | 67/200 (33.5%) | 152/200 (76.0%) | +93/-8 (1.74e-19) | 179/200 (89.5%) | 72/200 (36.0%) | +1/-108 (3.39e-31) |
| D4 in-sentence filler | 73/200 (36.5%, p=0.238) | 147/200 (73.5%, p=0.332) | +83/-9 (3.92e-16) | 174/200 (87.0%, p=0.125) | 77/200 (38.5%, p=0.383) | +2/-99 (4.06e-27) |
| Preflight s3 (D1 + post-fact filler) | 59/200 (29.5%, p=0.215) | 127/200 (63.5%, p=1.70e-4) | +77/-9 (1.34e-14) | 159/200 (79.5%, p=1.80e-4) | 68/200 (34.0%, p=0.618) | +4/-95 (1.24e-23) |
| Preflight s1 (2-sent split, no filler) | 56/200 (28.0%, p=0.200) | 50/200 (25.0%, p=1.20e-24) | +18/-24 (0.441) | 68/200 (34.0%, p=7.70e-34) | 37/200 (18.5%, p=1.57e-5) | +8/-39 (5.54e-6) |
| Preflight s2 (2-sent split + filler) | 52/200 (26.0%, vs s1 p=0.572) | 55/200 (27.5%, vs s1 p=0.442) | +27/-24 (0.780) | 58/200 (29.0%, vs s1 p=0.253) | 46/200 (23.0%, vs s1 p=0.243) | +19/-31 (0.119) |

Summary: Splitting a fact across two sentences (`s1`) collapses R7 to chance (`25%-34%`) even without a filler because two-sentence anaphora (`It is ...`) is unseen wording, leaving no signal for an added filler sentence (`s2`, `p >= 0.24`) to disrupt. Inserting a full unrelated sentence after intact D1 fact sentences (`s3`) lowers answer-after accuracy by 10.0 to 12.5 points (`152 -> 127` reverse, `179 -> 159` extract), but answer-after still beats answer-first strongly (`127 vs 59`, `p = 1.34e-14`; `159 vs 68`, `p = 1.24e-23`) and answer-first stays near chance (`29.5%-34.0%`).

### 2. Streaming generator capacity check over 1,000 steps (32,000 sequences)

`lisp/experimental/pilot3/pilot3.lisp` (`mode=preflight`, seed `111303`, `batch 32`) streamed 32,000 sequences (`1,000` steps) holding `40,653` packed questions (`20,337` in the first `500` steps), with zero copy replay and every training answer-before frame enabled (`26` passage frames, `28` question frames):

- **Excluded strings:** `1,133` total strings excluded case-insensitively (`heldout-strings` covering `:dev`, `:test`, and `:new` pools + the 76-entity `d1-entities.txt` list + all `216` option/passage entity strings from `lisp/eval/qa/dev.jsonl` and `d1-*.jsonl` in `excluded-extra.txt`), plus template/augmentation function words filtered from broad value pools. Zero excluded-entity leaks verified in both probes (`0/500`, `0/500`).
- **Clean entity pool sizes:** `names = 21,889`, `towns = 760`, `jobs = 3,519`, `pets = 3,511`, `colors = 3,505`, `instruments = 3,505`.
- **Expected vs observed repeat rates across 1,000 steps (32,000 sequences):**
  - 4-person tuple space: `C(21,889, 4) = 9.563e15`. Analytical birthday expected repeats across `32,000` sequences = `5.354e-8` (for `16,000` sequences / `500` steps: `1.339e-8`). Observed repeats: `0` at 500 steps, `0` at 1,000 steps.
  - Individual `(attr, person, value)` asked-question collision expectation across `40,653` questions (`1,000` steps): sum over types `N_q(attr)*(N_q(attr)-1)/(2 * 21889 * pool(attr)) = 3.21` collisions (`0.0079%`). Observed `(attr, p, v)` repeats: `0` at 500 steps, `0` at 1,000 steps.
  - Observed exact `rendered_ids`, `passage_q0`, `fact4_tuple`, and `sentence_pv` repeats: `0` across all `32,000` sequences.
  - Frame coverage across 1,000 steps: all `26/26` answer-before `:train` passage frames (`23` person-first reverse frames across `:town`, `:job`, `:pet`, `:color`, `:instrument`; `3` value-first extraction frames across `:town`, `:instrument`) and all `28/28` `:train` question frames (`20` `:q-reverse`, `8` `:q-extract` plus `*generic-questions*` under `*augment* = t`) were hit (`762` to `5,965` times each).

## Pilot Design (Two Arms, Single Manipulated Variable: Learning Rate)

Both arms start from unchanged R7 (`33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`) with fresh Adam moments (`m1 = 0, m2 = 0`), trained serially on the exact same `16,000`-sequence stream (`data-train.sexp`, `seed = 111303`, `batch = 32`, `accum = 1`, `workers = 8`, `steps = 500`, `1.0` epoch, `0` copy-replay sequences, `0` level 1-3 non-answer-before sequences).

- **Arm 1 (`5e-4`):** R7 stage learning rate `5e-4` (`data/runs/retrieval/p3-learnability/arm-5e-4/`).
- **Arm 2 (`1e-3`):** Copy-stage learning rate `1e-3` (`data/runs/retrieval/p3-learnability/arm-1e-3/`).
- **Schedule (identical in both arms):** Linear warmup for `50` steps (`cur_lr = lr * (1 + step_idx) / 50` for `step_idx = 0..49`), then cosine decay via `asem-pretrain::lr-at` over steps `50..499` down to `min_lr_frac = 0.1` (`0.1 * lr` at step 500), gradient clip `1.0`, weight decay `0.0`, Adam `beta1 = 0.9, beta2 = 0.95`, token loss normalization, no auxiliary loss.
- **Training data:** `16,000` unique four-person answer-before sequences (`20,337` questions: `15,803` reverse, `4,534` extract; `52,746` answer tokens), visited once per step in `epoch-perm(16000, 111303, 0)` order, zero repeats, `0` copy sequences. Every passage/question respects `ctx <= 128` and has every answer strictly before its queried entity in its passage sentence (`whole-word-pos` assertion).

## Measurements and Scoring Rules

Evaluated at step `0` (frozen R7), step `250`, and step `500` for each arm using unmodified `lisp/bin/qa-analysis.lisp` (`controls=0 paired=0`) and `lisp/bin/mission-eval.lisp` (`dataset=selection`):

1. **Fresh probe (`probe-fresh-500.jsonl`, `n = 500`, `404` reverse, `96` extract):** Newly generated four-person answer-before items (`stream = 2`, never seen in training, zero `(attr, person, value)` overlap with `data-train.sexp`, zero excluded entities). Scored for:
   - Exact answer (generation): `gold_length_greedy_prefix_exact` (autoregressive greedy decoding over `K = 8,192` tokens matching all gold answer tokens, chance ~0%; identical to teacher-forced exact answer on single-question prompts). `bounded_free_generation_exact` is also logged for completeness (expected near 0% because training targets supervise answer tokens only, not end delimiters).
   - `K = 4` ranking: `summed_option_likelihood` (summed token log-prob, not length-normalized; chance 25%).
2. **Seen probe (`probe-seen-500.jsonl`, `n = 500`, `404` reverse, `96` extract):** One item per training step (`steps 1..500`, taking question `q0` from the batch sequence `(aref perm (* 32 step))`, byte-identical to its prompt in `data-train.sexp`). Scored identically to Fresh probe (at step 250, also split into `train_step <= 250` seen vs `> 250` not-yet-seen).
3. **Frozen D1 200 (`n = 200` per cell, `K = 4`, chance 25%):** `d1-rev-pf`, `d1-rev-vf`, and `d1-ext-pf` (reference R7: `67`, `152`, `179`).
4. **Retention (`selection.sexp`, `n = 500`, `100` per category, `K = 4`, chance 25%):** Information only, non-gating (reference R7: `208/500`).
5. **Statistical tests and trajectory:** Exact two-sided paired McNemar test vs R7 for every `K = 4` cell; training loss logged every `50` steps (`50, 100, ..., 500`).
6. **Prohibited sets:** `lisp/eval/qa/dev.jsonl` (`dev 500`), `confirmation.sexp`, and both locked test sets are NEVER scored or opened.

## Pre-registered Decision Table (Verbatim; Judged on Best Arm's Final Step-500 Checkpoint)

| Outcome | Condition | Meaning |
|---|---|---|
| A. Learns and transfers | fresh exact >= 75%, seen minus fresh <= 10 points, D1 person-first reverse >= 120/200 | Step 1 failed from low dose and a small fixed pool, not architecture. Redesign mix and scale. |
| B. Learns in-dist, fails D1 | fresh exact >= 75%, D1 person-first reverse < 120/200 | Learned the training pattern, failed to transfer to dev entities or clean wording. Inspect why. |
| C. Memorises | seen exact >= 70%, seen minus fresh > 20 points | Pure memorisation even with fresh data every step (would suggest pool reuse or a leak; check generator). |
| D. Capacity or optimisation limit | seen exact < 60% in both arms after all steps, loss flat over the last 250 steps | First controlled evidence implicating depth, width, or capacity (Rule 5 threshold met). |
| E. Anything else | Does not fit A-D | Ambiguous outcome. Stop and report; do not extend. |

If outcome is D or E, report the slope across steps `0 -> 250 -> 500` for Fresh/Seen exact, D1 cells, and the 50-step training loss curve (`250 -> 500`) so under-training is distinguished from a flat plateau. Stop after judging; do NOT request or start a 2,500-step run.

## Frozen Pre-Training Hashes (SHA-256)

- `R7` (`data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt`): `33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`
- `tokenizer.bpe` (`data/fineweb/stage1/tokenizer.bpe`): `30d876e0d13bf8001ad66e1dd376a4240bc14b44cdf42e397c47133165cc45e4`
- `data-train.sexp` (`data/runs/retrieval/p3-learnability/data-train.sexp`): `bbf1d88431bb4df7e379ce3d03a002c7c125ba65bc871ebfc767d157d6bb685f`
- `probe-fresh-500.jsonl`: `b1be73154521563052dead84818bd2b5e4d3d5e779a50baba54d9b7c0a0945cd`
- `probe-seen-500.jsonl`: `0210d6dabbd3bc0edc183a68141d73b00f7a1e85d0cd7830ca722892d62ad1c2`
- `excluded-extra.txt`: `209977dcb8bb2d6adef07e9d26faf8f7cdec1aa5135adb1bebd6bb87fc816955`
- `lisp/experimental/pilot3/pilot3.lisp`: `fe4f5bf9fa811f9d36eb42d365c8ca4ba4e8ccfe69697736fb91242c58f29c7c` (pre-training 1-line reader fix from `d796ce6f...` at 0 steps: use exported `asem-qa-analysis:read-development`; `data-train.sexp`, probes, and all thresholds unchanged)
- `lisp/src/qagen.lisp`: `97b123a54a2c3e5ca5f036939ce5625af4f2d214e7e8e901544081d13a0fb0f7`
- `lisp/src/retrieval.lisp`: `ab7b88de4d7bcdc2d569421a5ddfbf6ab54f41df41582f7dbe5130357dccfb23`
- `lisp/src/sft.lisp`: `ad6d16dda6566fa2c68b10445fb1805f02a132ae5f52f8f56705f00ae86b1f2d`
- `lisp/bin/qa-analysis.lisp`: `61ac8f17a8809f1e98a98dbd0032db82b230f8a428a37bb36d764f1e73c58e1c`
- `lisp/bin/mission-eval.lisp`: `5fd60a4cb0df9c20fad6e821782db36010fb9ef88e7e914211c712495c67a153`
- `selection.sexp`: `11deeef798a4d5a34d91f450adbdb76801d6a60eda0c30956a442a0173aaeaa0`
- `d1-rev-pf.jsonl`: `152af0b302f8883d0a8c6f94dfc57a1b4a86f3ddcf83b20638cda05180467c10`
- `d1-rev-vf.jsonl`: `0425762bc5c1de7602a95e519d385630613a008e7d342d260337a03c31600a97`
- `d1-ext-pf.jsonl`: `f35017c52d3c5a76092b088154cca7018c30c655c5fc62b2ece71752e2052870`
