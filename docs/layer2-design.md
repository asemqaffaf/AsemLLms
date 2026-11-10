# Layer 2: a model that keeps state beyond the 32-token window

Status: design only, nothing built. Layer 1 (KV-cache sessions, `tinyllm/apps/tinyllm/src/tinyllm_session.erl`) is done.

## What Layer 1 cannot fix (from the code)

- `lisp/src/tf.lisp` and `tinyllm_tf.erl` both use **learned absolute position embeddings** (`pos`, row `i` added at position `i`).
  A cached key/value was computed for "position 5"; if the window slides, the same token is now "position 4", so every cached
  entry is wrong. That is why a full window is rebuilt on every token (`tinyllm_session:set_window/2`), and why attending to
  older cached blocks is out of distribution: the model never saw two tokens at "position 5".
- Anything older than `context` tokens (32 shipped) is dropped. Nothing carries over.
- **The two implementations are not the same model.** Lisp: RMSNorm, no biases. Erlang: LayerNorm with biases. There is no
  Lisp-to-Erlang model loader at all (`grep` finds none), so a model trained in the Lisp trainer cannot be served by a session today.

## Options

| | Relative positions (ALiBi or RoPE) + sliding-window attention | Segment memory (Transformer-XL) | Recurrent state (RMT-style memory tokens) |
|---|---|---|---|
| Idea | Position enters only as distance `i-j`; each token attends to the previous `W` tokens. | Cache the previous segment's hidden states (stop-gradient) as extra keys/values. Needs relative positions. | `R` memory vectors are inputs to a segment and outputs of it; the outputs become the next segment's inputs. Trained by back-propagating through a few segments. |
| Reach | `W` tokens directly, about `layers x W` through stacked layers (2 layers: ~2W). | About `layers x segment` more. | Unbounded in principle; in practice 3-5 segments of BPTT, and only what the model learns to write into `R x dim` numbers. |
| New parameters | None (ALiBi) | None | Memory read/write embeddings |
| Lisp training change | Mask + bias in `forward`; backward unchanged for ALiBi (bias is constant); RoPE needs an inverse rotation of `dq`, `dk`. | Longer K/V in attention, no gradient into the memory rows but gradient into `wk`/`wv` through them. | Expose the gradient of the input memory rows (already computed as `dh` at the embedding level) and chain it into the previous segment's outputs. Training cost x segments unrolled. |
| Erlang serving change | Cache stays valid on slide: drop the oldest entry instead of rebuilding. **O(1) per token, also fixes the full-window cost seen in Layer 1.** | Same as left, plus keep the previous segment's entries. | Session stores `R x dim` floats at segment boundaries; K/V of the memory rows are cacheable. |
| Main risk | Short reach; train/serve mismatch if not trained with the same mask. | More machinery for a modest gain at this scale. | A 250k-parameter model may not learn to write facts into a state; hardest to debug. |

## Recommendation: two stages, with a measured gate between them

**Stage A: ALiBi + sliding-window attention, window `W` trained on longer sequences.**
ALiBi is the cheapest: no position parameters, bias `-m_h (i-j)` with `m_h = 2^(-8(h+1)/heads)`, and the backward pass does not
change because the bias is a constant. RoPE is the usual quality choice but extrapolates worse and needs trig tables in both
languages; revisit only if ALiBi loses on the gate below.

One property drives the whole design and must be stated up front: with a sliding cache and more than one layer, a cached layer-2
key encodes layer-1 states computed when *more* history was visible than a from-scratch recompute of the shrunken window would
see. So the cached stream is **not** bit-equal to "recompute the last `W` tokens" (the Layer 1 contract). The contract changes to:
**the model is defined as sliding-window attention over the whole stream** (token `i` attends to `i-W+1..i`, at every layer), and
training uses exactly that mask on sequences longer than `W`. Then the cache is exact by construction against the full forward
over the entire history, which is the new eunit test (same shape as `cache_logits_equal_full_forward`, with the mask).

**Stage B (only if the gate says so): R memory vectors carried across windows (RMT-style).** Added on top of A: when a window
closes, the final-layer states at `R` write rows (8 x 64 = 512 floats) become the first `R` input rows of the next window.

## The gate (decide with numbers, before building B)

Synthetic recall, in the style of `tf_uses_long_range_context`: `K <name> ... <d filler tokens> ... ? K -> <name>`, distance
`d` in {8, 16, 32, 64, 128}. Report accuracy against `d` for: (1) today's absolute-position model, window 32; (2) ALiBi trained
at `W=32`; (3) ALiBi trained at `W=128`; (4) stage B. Build the generator in Lisp next to `lisp/src/membench.lisp` (reuse its
seeded splitmix64 PRNG; its natural-language cases are for LM Studio agents, a 250k-parameter model cannot answer those).
Go to B only if (3) falls off a cliff at distances that matter for chat (> `W`) and the `layers x W` reach does not cover them.
Chat lines in `dataset/chat_clean.txt` are short, so `W=128` already spans several turns.

## Structure checks (Lisp trainer, Erlang loader, file formats)

**Positions.** Lisp: `pos` is parameter slot 1 (`tok pos` then 8 tensors per layer, `layer-base` = `2 + 8l`); forward adds row `i`,
backward adds `dh` to `gp` row `i`. Removing the slot would shift every index. Cheaper: for ALiBi keep slot 1 with size 0 (`param-shapes`
returns `("pos" 0)`, the add and the gradient loop are skipped), so `layer-base`, `gf-index`, `head-index`, Adam state and the gradient
check all stay as they are. Erlang: `p` becomes `[]`; `forward/3` and `layer_step/5` skip it.

**Cross-segment gradient (needed only for the summary-token option).** It is mostly there. `backward` ends with `dh` = d(loss)/d(input
rows) for all `tl` positions, before it is scattered into `tok`/`pos`. If memory rows were fed as dense vectors instead of token ids,
`dh[0..R)` is d(mem_in) with no new math. Two additions: `forward` takes an optional matrix that overrides `h0` for the first `R` rows
(and `backward` skips the `tok`/`pos` scatter for them), and `backward` takes an external gradient added to `dh` at the top for the
write rows, after the head's `rmsnorm-bwd`. Truncated BPTT over two adjacent segments = forward A, forward B(mem_in = A's write rows),
backward B, pass `dh[0..R)` as the external gradient into backward A. Each worker needs two `acts` structs and the cost per example is
2x. New tensors (`m0` initial memory, `wr` write rows, each `R x dim`) go **after `head`** so no existing index moves.

**Reach of two-segment truncation.** Loss on B reaches A's summary, but nothing rewards B's summary for keeping A's content for C,
so the carried memory is about one extra segment (~2x the window). Longer memory needs a longer unroll, linear in cost. That is the
same reach as ALiBi's `layers x W`, at much higher cost and complexity; the gate decides if it is worth it.

**File formats.** `chat_tf.bin` is `term_to_binary(#{cfg, params, vocab, tokenizer, stats}, [compressed])`: a self-describing map with
**no version header and no byte offsets**, so extra keys do not break the byte layout. Compatibility therefore rests on `tinyllm:normalize/1`,
which already back-fills missing keys (`arch => mlp`, `tokenizer => char`); new fields (`pos`, `memory`, a `format` number) get defaults the
same way, and `binary_to_term(.., [safe])` is fine because the atoms live in modules `load/1` preloads. Two real hazards: (1) an old
file loaded by new code must default to `pos => learned`; (2) new files loaded by *old* code are not rejected, they run with the wrong
attention, so add `format => 2` and make `load/1` fail loudly on formats it does not know. Elsewhere: `scripts/export.escript` writes
JSON per tensor name and `scripts/pack_release.py` hard-requires `p` of shape `(T, dim)` and writes `"positional": "learned"`, so it needs
an ALiBi branch (and `asemllm.py` the bias); the Lisp text format has a `:format 1` header but `load-model` never checks it, and its
tensors are read **by position**, so it must check the header and bump to `:format 2` if the layout ever changes.

**Correction to the premise "exact `evict=N` sliding without retraining".** ALiBi cannot be applied to `chat_tf.bin`: the weights were
trained with absolute positions, so a new position scheme means retraining. After that, sliding by one is O(1) and `evict` is no longer
needed; the contract is the sliding-window one above, not equality with a from-scratch recompute.

## Work plan

0. **Bridge** (prerequisite, ~0.5 day): Erlang `norm => rms` option (and biases off) so `tinyllm_tf` can run the Lisp architecture,
   `tinyllm:load_lisp/1` reading `save-model`'s text format, and a parity test: same weights, same tokens, Lisp `last-logits` vs
   Erlang `last_logits` within 1e-9. Alternatively make Lisp use LayerNorm; RMSNorm in both is less work and cheaper.
1. **Lisp** (`tf.lisp`, `train.lisp`, ~1 day): config `:pos :alibi`, window mask `W`, remove `pos` from the parameter layout when
   ALiBi (indices shift: `layer-base` assumes `tok pos` first), training sequences of length `T > W`; extend the existing
   per-tensor numerical gradient check to the masked ALiBi forward.
2. **Erlang** (`tinyllm_tf.erl`, `tinyllm_session.erl`, ~1 day): bias and mask in `forward/3` and `layer_step/5`; `cache_drop/2`
   to slide; the session never rebuilds; update the exactness tests and the benchmark (expected: per-token cost flat in history).
3. **Gate** (~0.5 day): the recall generator, the four-way table above in the README.
4. **Stage B** (2-3 days, conditional).

Old checkpoints (`chat_tf.bin`) are incompatible with a different position scheme and must be retrained; the Erlang trainer is
~30x slower than the Lisp one, so training moves to Lisp, which is the reason step 0 comes first.

## Not in scope

Process-per-segment actors, an ETS prefix cache and Lisp triple extraction were considered and dropped: they add machinery
around a model that cannot yet use the older context, and copying float lists between processes costs more than the tiny
matrix products they would save.
