# Stage 2 QA benchmark (`asem-qa-1`)

Four-choice, passage-based questions for single-hop reading comprehension and factual extraction.

| File | Items | SHA-256 |
|---|---|---|
| `test.jsonl` (locked) | 1,000 | `8f756bcdf3accd79f440faff81091d5b7c1ae3dc336610057b9f6e31a939f060` (also in `test.lock`) |
| `dev.jsonl` | 500 | `2d92c8825ca3752bd780ecd25023ae5f0a8dcc60677ee6484d46aa95e701032a` |
| training sets | 16,000 / 32,000 | not in git; rebuilt by `bin/qa-build.lisp` (see below) |

`manifest.json` is the builder's audit: counts, answer positions, token lengths, rejected candidates, near-duplicate check,
template overlap, shortcut baselines.

## Categories (200 test / 100 dev items each)

| Category | Question asks | Options |
|---|---|---|
| `extract` | an attribute of a named person (town, job, pet, colour, instrument, age, score, height) | the four people's values |
| `reverse` | which person has a given attribute | the four people |
| `relation` | who did X to Y / whom did X do it to (teach, hire, call, thank, visit, help), as two pairs or a three-link chain | the four people |
| `compare` | who is oldest / youngest, scored most / fewest, tallest / shortest | the four people |
| `temporal` | first, last, right after X, right before X in a four-stop day | the four places |

All four options occur in the passage equally often, so "pick the mentioned option" is useless. Answer positions are
exactly balanced (250 per letter in test, 25 per letter per category in dev).

## Independence of dev and test from training

- **Entities.** Names, towns, jobs, pets, colours, instruments and places are split 3/1/1 (train/dev/test) with no
  overlap (`tests/test-qagen.lisp`). The `v2-synth` training set replaces training people and towns by ~4,000 generated
  strings each, filtered against every dev and test entity.
- **Wording.** Every passage frame, question frame and intro belongs to exactly one split; dev and test frames were written
  separately from training frames, and each dev/test question is phrased so it is entailed by every passage frame of its
  split. Measured overlap: 0 of 74 dev frames and 3 of 97 test frames share a slot-free 4-word sequence with a training frame
  (`which person is the`, `how many points did`).
- **Duplicates.** No exact or near-duplicate (word 5-gram MinHash) passage between dev and test, or between training and dev/test.
- **Freeze.** `test.jsonl` was generated first, hashed into `test.lock` and committed before any fine-tuning.
  `bin/qa-eval.lisp` refuses it unless `locked-test=1` is passed and the hash matches, and logs every such run in
  `docs/results/stage2/locked-test-runs.jsonl`.

## Validation

Each item is checked when generated: four distinct options, none a substring of another; every option in the passage; a
solver recomputes the key from the structured facts (and the counterfactual key); for lookups, the sentences naming the
asked entity contain exactly one option, the key. Candidates failing any check, or too long for the context, are discarded
and redrawn (counts in `manifest.json`).

## Prompt and scoring

    Text: <passage>
    Q: <question>
    A. <option>
    B. <option>
    C. <option>
    D. <option>
    The answer is

Every item fits the 128-token window with the end-of-text token in front and the longest option appended (maximum 124
positions); nothing is truncated (`bin/qa-build.lisp` and `src/sft.lisp` raise an error instead). Each answer letter
` A`..` D` encodes to one token (316, 367, 338, 443) when appended to its prompt.

- **Letter mode:** prediction = argmax of the four letter logits after the prompt.
- **Text mode:** prediction = the option whose tokens have the highest summed log-probability after the prompt.

Both are raw accuracy (no calibration). Controls (`bin/qa-eval.lisp`): no question, no passage, a question from another item,
the counterfactual question on the same passage (the key moves), options in reverse order.

## Shortcut baselines (test set)

Always one letter 25%; first-mentioned option 25.3%; longest 26.7%; shortest 23.8%; the option in the passage sentence that
shares the most content words with the question 59.1% (dev 62.0%). The last rule is a lexical-matching solver for the lookup
categories; a model has to beat it to show more than word matching.

## Rebuild

    sbcl --dynamic-space-size 4GB --script lisp/bin/qa-build.lisp data=data/fineweb/stage1 out=data/qa/v1
    sbcl --dynamic-space-size 4GB --script lisp/bin/qa-build.lisp data=data/fineweb/stage1 out=data/qa/v2-synth train-entities=synthetic train=32000

Both reproduce the dev and test files above byte for byte.
