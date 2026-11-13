# R7 answer-generation mechanics audit — 2026-11-11

**The 77 failures are explained by R7's output preferences under unrestricted greedy decoding, including local-versus-sequence decision differences, rather than the checked whitespace, BPE or answer-boundary mechanics.** This statement is limited to the paths checked below; it is not a proof that every implementation path is bug-free.

R7 and both pilot checkpoints remain unchanged. No training, architecture change, tokenizer change, promotion or locked-test access occurred. Only R7 was used for new inference, on the fixed 77 relationship worlds selected before that inference. These are the existing cases with correct summed option ranking and an incorrect generated gold-length prefix. The 104 ranking-direction errors are investigated separately from saved paired worlds.

## Checks and results

| Check on the 77 selected worlds | Result |
|---|---:|
| Gold text round-trips through BPE, with canonical token IDs | 77/77 |
| Reconstructed prompt IDs match saved IDs | 77/77 |
| Encoding prompt + answer equals prompt IDs + answer IDs | 77/77 |
| Saved decoded prefix matches its raw token IDs | 77/77 |
| Prefix equals the start of the saved bounded generation | 77/77 |
| Whitespace trimming rescues the answer | 0/77 |
| Case-insensitive trimming rescues the answer | 0/77 |
| Removing all whitespace rescues the answer | 0/77 |
| Prefix was shortened, or contains EOT | 0/77 |
| Independent unpadded greedy trace reproduces saved prefix IDs | 77/77 |
| Full-forward first-step argmax agrees with last-row decoding | 77/77 |
| Independently accumulated teacher-forced scores reproduce option ranking | 77/77 |
| Teacher-forced scores agree within the fixed absolute tolerance 0.0001 | 77/77 |
| Greedy candidate-trie decoding produces the correct candidate | **67/77 (87.0%)** |
| Greedy candidate-trie decoding still chooses a wrong candidate | **10/77** |

The largest teacher-forced score difference is `1.6754e-5` (rounded upward); the largest full-forward/last-row logit difference is `3.0041e-5` (rounded upward). All original greedy prefixes are identical despite these small FP32 numerical differences. There is no observed padding/indexing discrepancy affecting an argmax in this audit.

In **66/77** cases, the unrestricted first-token argmax is not even a valid first token for any candidate. Median probability mass on the set of all allowed first tokens is **0.004616**, about **0.46%** of the full-vocabulary distribution. This is candidate-prefix mass, not full-answer probability. Median global rank of the gold first token is 17.

The saved bounded text contains the gold as a substring in 12 cases and as a whole name in 10, but equals the complete gold answer in none. For example, an output can begin with `The person...` or `received instruction...` before mentioning the name. These are learned output-format/content preferences, not a whitespace-only mismatch. Gold-length prefix scoring does not test termination, so fixing EOT alone cannot fix these prefix errors.

Every case in `mechanics.json` retains the passage, question, options, gold continuation tokens, raw prefix/bounded token IDs, per-token byte sequences, decoded pieces and full saved prediction. Nothing is hidden behind aggregate accuracy.

## The constrained decoder is distinct from option ranking

At each step the candidate-trie decoder restricts the allowed next-token set to continuations of the four provided candidates, then takes the largest model logit among that set. It receives no answer key. Ties use the lowest token ID; it stops at the equal-length candidate boundary, without claiming learned termination.

This is token-wise greedy decoding, not whole-sequence likelihood maximization. The 10 residual errors are concrete examples where local greedy choices differ from the best summed sequence likelihood. The existing global candidate scorer is correct on **77/77 by selection**, making its apparent rescue tautological. The independently implemented greedy trie's 67/77 is a separate conditional result.

Neither number is a new R7 benchmark score. Candidate constraints supply answer-space information, and these examples were selected specifically because candidate ranking already succeeds. No improvement to the weights, unconstrained generation, relationship reasoning or the locked benchmark is established.

| Residual trie error | Gold | Trie answer |
|---|---|---|
| `binding-relation-00-04/w0` | Lonel | Vydus |
| `binding-relation-01-09/w0` | Xalol | Xarix |
| `binding-relation-03-01/w0` | Rudun | Tirix |
| `binding-relation-03-02/w0` | Neme | Kysos |
| `binding-relation-10-01/w1` | Xalol | Wurix |
| `binding-relation-14-09/w0` | Nedux | Kymex |
| `binding-relation-16-07/w0` | Tidul | Rumex |
| `binding-relation-17-00/w0` | Hurin | Lolox |
| `binding-relation-17-01/w0` | Javix | Tirix |
| `binding-relation-17-04/w0` | Rulol | Hurix |

## The 104 direction errors are a different population

`direction-cases.json` preserves both passages, fixed question/options, both gold keys and both original predictions for every R7 opposite-direction error. The graph-based definitions from the saved-prediction analysis were retained. These 104 worlds do not overlap the 77 ranking-correct generation failures.

- **76** opposite-direction errors retain the same option that is correct in the paired reversed world. These are failures to change the selected entity when its role changes.
- **12** errors belong to **six pairs** that select the opposite direction in both worlds. These provide the clearest examples of a consistent reversed-direction response on this set.
- The remaining **16** errors have a paired-world prediction that is also wrong, but not the opposite-direction answer.
- There are **98 unique graph pairs** represented by the 104 errors.

Calling all 104 a learned reversed-rule mechanism would overstate the evidence. Most occur in fixed-answer pairs. These are descriptive decompositions, not prespecified significance tests.

Example of a fixed answer (`binding-relation-00-08/w0`): the question asks who provided lessons for Quson. Its correct answer changes from Vyris to Jaka after all graph edges reverse, while R7 selects Jaka in both worlds.

Example of opposite directions in both worlds (`binding-relation-02-01/w0`): the question asks to whom Wuson gave instruction. The correct answer changes from Rukan to Gyka; R7 selects Gyka and then Rukan, consistently taking the incoming rather than outgoing neighbor.

This analysis separates answer realization from candidate-relative role selection. Restricting generation does not change R7's incorrect ranking on these 104 worlds. A future intervention must state which behavior it targets; these results do not justify broad R8 training or establish a capacity limit.

## Serialization defect found during the audit

The diagnostic initially wrote some FP32 numbers with Common Lisp `f` exponent markers, which are invalid JSON. This is a verified serializer defect, independent of the 77 model-output failures. `lisp/src/json.lisp` now promotes floats exactly to double precision before printing; strict FP32/FP64 round-trip regressions pass.

The raw output **`decoding.json` is retained byte for byte and intentionally is not parsed as valid JSON**. `decoding-normalized.json` is the usable record. A Common Lisp recovery utility replaced 526 numeric exponent markers outside quoted strings, preserving all model results and recording raw/normalized hashes in `serialization-recovery.json`. No inference was repeated. The recovery test checks that quoted exponent-like text and escaped quotes are untouched and that existing output files cannot be overwritten.

`verification.json` was produced by a separate model-free verifier that imports no model or decoder-policy package. It independently checks candidate admissibility, restricted argmax, candidate identity, correctness, BPE decoding, counts, raw-output preservation and source/checkpoint hashes. Neural logits themselves are measured outputs, not re-inferred by that verifier. Original 400-pair results still pass their independent verifier after the serializer fix.

## Reproduction

Use a new directory; every output refuses overwrites:

```sh
sbcl --script lisp/bin/generation-audit.lisp mode=prepare bundle=docs/results/stage2/binding-eval-20261111 out=NEW-DIR
TLM_BLAS=1 VECLIB_MAXIMUM_THREADS=1 sbcl --dynamic-space-size 4GB --script lisp/bin/generation-audit.lisp mode=run out=NEW-DIR
sbcl --script lisp/bin/normalize-fp32-audit.lisp raw=NEW-DIR/decoding.json out=NEW-DIR/decoding-normalized.json record=NEW-DIR/serialization-recovery.json
sbcl --script lisp/bin/verify-generation-audit.lisp dir=NEW-DIR out=NEW-DIR/verification.json
```

With the fixed writer, normalization replaces zero markers and still creates an explicit lineage record. The plan, selected inputs and source hashes are frozen before inference. The verified R7 checkpoint hash remains `33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1`.

Targeted checks passed: candidate decoding **4 tests / 12 checks**, FP32 JSON serialization **2 / 17**, and raw-preserving recovery **1 / 8**. The separate verifier validated all 77 saved traces without model imports. Infrastructure is committed separately from results; unrelated `tinyllm/*` changes remain outside these commits.
