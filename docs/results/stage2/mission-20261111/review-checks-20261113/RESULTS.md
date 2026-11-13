# Reviewer checks on frozen R7 (2026-11-13, inference only)

**Answers.** (B1) The name-exposure effect exists in dev wording. It is a property of swapping only the gold name; renaming all four names leaves K=4 accuracy unchanged in both wordings, which is why D2 saw nothing. (B2) Yes: R7's copy-mix trained on all 845 `eval_heldout` tokens, so G2c compared tokens R7 trained on against tokens the Pilot 4 guard excluded by design. The Pilot 4 verdict does not change.

No training. No new checkpoint was scored. R7 `last.ckpt` SHA-256 `33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1` before and after. `confirmation.sexp`, `test.jsonl` and `test.lock` were not opened. `dev.jsonl` was read; dev-derived items were scored on R7 only. Single checkpoint, all p-values unadjusted.

## B1. Experiment 1 minimal pairs in dev wording

Design fixed in `make_exp1_dev_items.py` before inference. A pair is one item rendered twice; only the gold person's name differs (zero-exposure name vs hi100-exposure name, Experiment 1 strata and swap pools). Scored with the unmodified `lisp/bin/qa-analysis.lisp` (`controls=0 paired=0`). Pipeline check: the training-wording pairs reproduce the saved Experiment 1 counts exactly (15 -> 55/90, 34 -> 10/67).

Table 1. Gold-only swap, pooled as zero-exposure gold -> hi100-exposure gold on the same item. K = 4 summed option log-probability, chance 25%. Exact two-sided McNemar.

| Items | Wording | n pairs | Zero gold | Hi100 gold | Change | Gains / losses | p |
|---|---|---:|---:|---:|---:|---:|---:|
| Experiment 1 pairs (pipeline check) | training | 157 | 25 (15.9%) | 89 (56.7%) | +40.8 pts | 70 / 6 | 6.3e-15 |
| Experiment 1 pairs, same facts and names (primary) | dev | 157 | 26 (16.6%) | 67 (42.7%) | +26.1 pts | 50 / 9 | 5.3e-8 |
| Real dev reverse items (secondary) | dev | 82 | 17 (20.7%) | 42 (51.2%) | +30.5 pts | 27 / 2 | 1.6e-6 |
| Real dev compare items (secondary) | dev | 83 | 11 (13.3%) | 35 (42.2%) | +28.9 pts | 28 / 4 | 1.9e-5 |

Table 2. Same pairs by swap direction. K = 4, chance 25%.

| Items | Wording | Direction | n | Before swap | After swap | Gains / losses | p |
|---|---|---|---:|---:|---:|---:|---:|
| Experiment 1 pairs | training | zero -> hi100 | 90 | 15 (16.7%) | 55 (61.1%) | 43 / 3 | 4.6e-10 |
| Experiment 1 pairs | training | hi100 -> zero | 67 | 34 (50.7%) | 10 (14.9%) | 3 / 27 | 8.4e-6 |
| Experiment 1 pairs | dev | zero -> hi100 | 90 | 14 (15.6%) | 37 (41.1%) | 29 / 6 | 1.2e-4 |
| Experiment 1 pairs | dev | hi100 -> zero | 67 | 30 (44.8%) | 12 (17.9%) | 3 / 21 | 2.8e-4 |
| Real dev reverse | dev | zero -> hi100 | 45 | 11 (24.4%) | 24 (53.3%) | 14 / 1 | 9.8e-4 |
| Real dev reverse | dev | hi100 -> zero | 37 | 18 (48.6%) | 6 (16.2%) | 1 / 13 | 1.8e-3 |
| Real dev compare | dev | zero -> hi100 | 35 | 3 (8.6%) | 13 (37.1%) | 11 / 1 | 6.4e-3 |
| Real dev compare | dev | hi100 -> zero | 48 | 22 (45.8%) | 8 (16.7%) | 3 / 17 | 2.6e-3 |

Table 3. All four names renamed to training-pool names (saved `diagnostics-20261113` predictions, no new inference; 302 of 304 training-pool names have every subword at 100 or more occurrences in R7 retrieval data). K = 4, chance 25%.

| Items | Wording | n | Dev-pool names | Training-pool names | Gains / losses | p |
|---|---|---:|---:|---:|---:|---:|
| D1 rev-pf vs d1t rev-pf | training | 200 | 67 (33.5%) | 70 (35.0%) | 44 / 41 | 0.83 |
| D2 reverse | dev | 94 | 36 (38.3%) | 34 (36.2%) | 20 / 22 | 0.88 |
| D2 compare | dev | 100 | 31 (31.0%) | 36 (36.0%) | 20 / 15 | 0.50 |

Table 4. Share of option slots R7 picks, by exposure stratum of the option's name (saved predictions). K = 4, chance 25% per slot.

| Items | Zero-exposure name | Mid | Hi100-exposure name | Zero as distractor | Hi100 as distractor |
|---|---:|---:|---:|---:|---:|
| D1 rev-pf, training wording (n = 200) | 31/354 (8.8%) | 43/155 (27.7%) | 126/291 (43.3%) | 16/264 (6.1%) | 92/224 (41.1%) |
| Dev reverse, dev wording (n = 100) | 24/188 (12.8%) | 19/65 (29.2%) | 57/147 (38.8%) | 13/143 (9.1%) | 39/110 (35.5%) |
| Dev compare, dev wording (n = 100) | 8/159 (5.0%) | 25/78 (32.1%) | 67/163 (41.1%) | 5/124 (4.0%) | 45/115 (39.1%) |

Reading:

1. The exposure effect exists in dev wording (+26.1 pts on the Experiment 1 pairs, +30.5 pts on real dev reverse items), smaller than in training wording (+40.8 pts).
2. Experiment 1 and D2 differ in design, not only in wording. Experiment 1 changes the gold name alone, so the gold option's exposure moves relative to the three distractors. D2 renamed all four names at once. All-four renames show no K=4 change in training wording either (Table 3, 67 vs 70/200).
3. The gold-only effect is at least partly option-name preference, not binding. R7 picks hi100-named options about 4 to 10 times as often as zero-named options even when they are wrong (Table 4), zero-named gold answers score below chance (13.3% to 20.7% across the four sets), and the swap moves compare items by the same amount (+28.9 pts) although R7 is near chance on compare (31/100). In the swap, the gold option's summed log-probability rises by 3.6 to 4.6 nats while the best distractor moves by 0.05 to 0.39.
4. So Experiment 1's K=4 numbers do not show that subword exposure limits R7's answer-before K=4 accuracy. With all four names high-exposure, D1 rev-pf K=4 stays at 35.0%. Greedy Exact does rise under the all-four rename (D1 rev-pf 31 -> 60/200, p = 6.4e-4; D2 reverse 13 -> 26/94; D2 compare 10 -> 35/100), so exposure affects generating the name string. A wrong-slot (distractor-only) swap was not run.
5. Wording alone, names fixed (157 Experiment 1 items): training -> dev wording moves K=4 from 15 to 14/90 on zero-gold items and from 34 to 30/67 on hi100-gold items (p = 0.45).

## B2. G2c fairness

`g2c_fairness.lisp` replays, without a model, R7's copy-mix generator (`lisp/bin/retrieval.lisp:92`: `make-copy-examples 15000 77 :lo 1 :hi 8192 :min-len 10 :max-len 40`) and the seed-7 epoch-0 order from `lisp/src/sft.lisp`. The replay matches the saved audit total (360,850 copy targets).

- R7 ran 2,500 steps x 32 = 80,000 of 90,000 sequences, consuming 13,341 of the 15,000 copy sequences.
- Every one of the 845 `eval_heldout` tokens was a supervised copy target in R7's consumed stream: minimum 21, mean 39.2, maximum 64 occurrences (33,137 targets, 10.3% of consumed copy targets, the uniform share).
- Pilot 4's guard excluded those 845 tokens in every arm (0 occurrences, per the Pilot 4 leakage audit).

G2c therefore compared R7 on tokens its own copy-mix trained against Pilot 4 arms that were given no copy supervision on those tokens. It measures loss of an R7 behaviour under training that omitted those tokens. It does not measure zero-shot induction in R7. The same holds for the `s1_heldout_ev` and `ind_ev` floors, which were set from R7. The binding gate was pre-registered, so the verdict stands: all four arms fail G2c, and all four also fail G1.

## Files

`make_exp1_dev_items.py`, `pairs-meta.json`, `{e1-train,e1-dev,devrev,devcmp}-{orig,swap}.jsonl` and `.r7.json`, `analyze_exp1_dev.py`, `analyze_exp1_dev.out.txt`, `g2c_fairness.lisp`, `eval_heldout_ids.txt`, `g2c_fairness.out.txt`.

TL;DR: The exposure effect is real in dev wording (+26 to +31 pts K=4, chance 25%) but appears only when the gold name alone changes; all-four renames give no K=4 change in either wording, and R7 prefers high-exposure names even as distractors. Experiment 1 mostly measured option-name preference. G2c tested Pilot 4 arms on tokens R7's copy-mix had trained and Pilot 4 excluded; the stop verdict is unchanged.
