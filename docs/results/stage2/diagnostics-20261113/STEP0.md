# Step 0 preflight, 2026-11-13 (inference only, frozen R7)

R7 = `data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt`, SHA-256 `33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`, unchanged before and after every run. No training. The confirmation set and the locked test were not opened.

## Model and scoring

- Checkpoint header: decoder-only transformer, 10,523,393 parameters, 8 layers, 8 heads, width 256, feed-forward 1,024, context 128 tokens, vocabulary 8,192 (byte-level BPE), gated tied copy path on, relative-position attention biases K=8.
- Prompt (`lisp/src/qagen.lisp` `format-prompt`, `:compact`): `Text: <passage>\nQ: <question>\nA. <o1>\nB. <o2>\nC. <o3>\nD. <o4>\nThe answer is`, preceded by the end-of-text token.
- K = 4 options per item, chance 25%.
- Score of an option (`lisp/src/sft.lisp` `score-continuation`, `rows-logprob`): the option is encoded as `" " + option`; score = sum over its tokens t of log P(token_t | prompt, option tokens before t), full-vocabulary softmax, teacher-forced. Summed, NOT length-normalised. Prediction = argmax over the four options, first index on ties. No stop token is scored. The evaluator also reports a mean-per-token variant; every table here uses the summed score.
- n = 500 gives a 95% half-width of about 4.4 points; n = 200 about 6.9; n = 100 about 9.8.

## D3 training fit (script `d3-training-fit.lisp`, analysis `analyze_d3.py`)

One teacher-forced pass over the 427 questions of the 256 saved traced training sequences (training entities, training wording, dense context as trained). Metric: every answer token is the full-vocabulary argmax (K = 8,192 per token, chance about 0%). The pass reproduces the saved audit exactly (first questions 210/256, later 141/171).

| Task | Exact |
|---|---:|
| Extraction | 54/54 |
| Numeric extraction | 61/61 |
| Relation | 80/88 |
| Reverse | 41/54 |
| Comparison | 48/74 |
| Temporal | 67/96 |

- Reverse with the answer before the queried value: 12/12 with one sentence, 15/17 with two, 9/19 with four.
- Relation with the answer before the queried person: 42/45.
- Comparison: 24/30 with two people, 11/22 with three, 13/22 with four. Same token count 14/21, mixed 34/53.
- Temporal: first 24/24, last 21/21, after 13/29, before 9/22.

## D4 adjacency versus order (items `make_items_step0.py`, analysis `analyze_step0.py`)

The 200 D1 fact sets with a training-style modifier inserted between the queried entity and the answer. K = 4, chance 25%.

| Question | Order | Near | Far (filler) | McNemar p |
|---|---|---:|---:|---:|
| Reverse | answer before | 67/200 | 73/200 | 0.24 |
| Reverse | answer after | 152/200 | 147/200 | 0.33 |
| Extraction | answer after | 179/200 | 174/200 | 0.13 |
| Extraction | answer before | 72/200 | 77/200 | 0.38 |

Answer-after with the filler still beats answer-before without it: reverse 147 vs 67 (90 gains, 10 losses, p = 3e-17). Order drives the effect, not adjacency.

## D5 temporal place strings

The 100 temporal dev items, dev wording. K = 4, chance 25%.

- Original 26/100. All four places single-token 26/100 (p = 1.0), training-pool places 23/100 (p = 0.72), the four hardest dev places 28/100 (p = 0.86).
- Rotating the item's own places over the four stops: 98/400. The same place is chosen in all four rotations for 26 items; no item is right in all four. `theater` is chosen 0 times and `cinema` once in 400.
- Place strings decide the choice but are not the cause: with strings equalised the score stays at chance.
