# Pilot 2 result: FAIL (2026-11-13)

Protocol: `protocol.md`, frozen before training. Reproduce the tables with `python3 analyze_pilot2.py`.

- Control `data/runs/retrieval/p2-answer-before/control/last.ckpt`, SHA-256 `b9c85573ff2450caa17814edbb59de884a5062b70cbde1dcc60c5df052d6f10f`, 500 steps, 330.9 s.
- Treatment `data/runs/retrieval/p2-answer-before/treatment/last.ckpt`, SHA-256 `e0ed026cae05566700e44f530d9a841de27477056e0bb33f856011a87557e767`, 500 steps, 335.2 s.
- R7 unchanged (`33a5c245...`). Public dev 500, confirmation set and locked test not scored or opened.

## Gate (frozen D1 items, n = 200 per cell, K = 4, chance 25%)

| Cell | R7 | Control | Treatment | Treatment vs R7 (gains/losses, p) | Gate | Met |
|---|---:|---:|---:|---|---:|---|
| Person-first reverse | 67 | 71 | 71 | 15/11, 0.56 | 120 | no |
| Value-first reverse | 152 | 154 | 158 | 13/7, 0.26 | 142 | yes |
| Person-first extraction | 179 | 180 | 175 | 4/8, 0.39 | 169 | yes |

Treatment vs control on person-first reverse: 11 gains, 11 losses, p = 1.0.

## Why it failed

Exact answers on four-person answer-before questions in training wording with training-pool entities (K = 8,192 per token):

| Checkpoint | Fresh, never trained (n = 503) | Sequences in the treatment data (n = 514) |
|---|---:|---:|
| R7 | 230 (45.7%) | 245 (47.7%) |
| Control | 241 (47.9%) | 225 (43.8%) |
| Treatment | 243 (48.3%) | 327 (63.6%) |

The treatment arm memorised items it saw and did not improve on fresh items from the same generator. The skill was not learned even in-distribution, so the D1 miss is not a transfer problem.

## Retention (mission selection set, n = 500, K = 4, chance 25%)

R7 208, control 209, treatment 201. No category differs from R7 (all p >= 0.23).

## Reading

Per the pre-registered rule this implicates depth or capacity. Caveat as registered: one seed, 500 steps, 1,668 replaced sequences holding 2,121 questions. It does not prove a capacity limit.

## Deviations

- The exclusion list holds the 76 entities actually used in D1/D4 (36 names, 36 towns, 4 instruments), not 45 towns as the protocol text says. Zero training sequences needed a redraw.
- `pilot.lisp` had a one-line fix to how it reads `data-audit.json` after data preparation and before training. Data hashes were checked at training time and match.

Pilots used: 2 of 3. Longer runs used: 0 of 1.
