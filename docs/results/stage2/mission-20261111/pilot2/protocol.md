# Pilot 2: answer-before-entity binding (pre-registered 2026-11-13, before any training or scoring)

Second of at most three matched pilots. This is not the 2,500-step run.

## Hypothesis

R7 retrieves an answer only when it follows the queried entity in its sentence (D1, D4). Replacing trivial one-person attribute questions with four-person questions whose answer precedes the queried entity will teach the missing direction.

## Arms (one manipulated factor)

Both arms start from unchanged R7 (`33a5c245...`) with fresh Adam moments and are trained serially with the same settings: 500 steps, batch 32, 8 workers, learning rate 5e-4 (R7's stage rate), 50 warmup steps, cosine decay to 10%, clipping 1.0, weight decay 0, seed 111302, 15,000 dense sequences plus 3,000 copy-replay sequences (seed 77), token normalisation, no auxiliary loss. Architecture, tokenizer, evaluator and copy stage are unchanged.

- Data: sequence i of both arms is drawn from its own random stream (seed, i), with R7's recipe: levels 1 to 3, broad vocabulary, augmentation, option lines, reverse and reasoning on.
- **Control**: the R7 recipe as drawn. Equal steps, no new skill.
- **Treatment**: identical, except that every sequence whose draw is a level-1 attribute fact set (one person; reverse or extraction) is replaced by a four-person attribute fact set in which every question's answer precedes the queried entity in its sentence: reverse questions on person-first frames, and extraction questions on value-first frames where the training inventory has them (town, instrument). Existing training frames, question frames and augmentation only. No new wording.
- Training entities exclude every D1/D4 entity (36 dev names, 45 dev towns, 4 dev instruments). Any sequence containing one as a whole word, in either arm, is redrawn in both. The count is recorded.

## Primary endpoint and gate

The treatment arm's final step-500 checkpoint, scored by the unmodified `lisp/bin/qa-analysis.lisp` on the frozen D1 item files (200 items each, K = 4, chance 25%, summed option log-probability). No checkpoint selection.

| Cell | R7 | Gate |
|---|---:|---:|
| Person-first reverse (`d1-rev-pf`) | 67/200 (33.5%) | at least 120/200 (60%) |
| Value-first reverse (`d1-rev-vf`) | 152/200 (76.0%) | at least 142/200 (loses no more than 5 points) |
| Person-first extraction (`d1-ext-pf`) | 179/200 (89.5%) | at least 169/200 (loses no more than 5 points) |

PASS requires all three. The gate will not be moved and the pilot will not be extended after results are seen.

## Readings fixed in advance

- All three met: the missing direction is a curriculum gap. Step 2 becomes eligible.
- Person-first reverse below 120/200 in the treatment arm: FAIL. Per the mission plan this implicates depth or capacity. Caveat recorded now: one 500-step seed cannot by itself prove a capacity limit; the report will state the treatment-minus-control difference and the training-loss trajectory of the replaced sequences alongside.
- Person-first reverse met but a retention cell missed: FAIL on retention. This is not evidence about capacity.
- Stop and report after this pilot in every case.

## Secondary, reported but not gating

- Control arm on the same three cells; paired exact McNemar tests of treatment vs R7 and treatment vs control.
- Value-first extraction (`d1-ext-vf`), the four D4 filler cells, and the D3 training-fit pass.
- The frozen 500-item mission selection set (100 per category, K = 4, chance 25%) for R7, control and treatment, as a retention check on the other categories.
- The 500 public dev items are NOT scored in this pilot. The confirmation set and the locked test are never opened.

## Limits

One seed, one checkpoint per arm, two attributes in D1, items reused from the diagnostics. D1 is in training wording with held-out entities, so a pass says nothing yet about dev wording.
