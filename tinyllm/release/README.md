---
license: apache-2.0
language:
- en
tags:
- text-generation
- tiny
- from-scratch
- erlang
- educational
pipeline_tag: text-generation
---

# AsemLLM

Tiny word-level language models written, trained and served entirely in **Erlang**, with no deep-learning framework:
supervised worker processes compute gradients in parallel. This repository holds the trained weights, the vocabulary and a
small numpy reference loader. **Source code (training, serving, web UI): https://github.com/asemqaffaf/AsemLLms**

> **These are toy models.** They learn the *style* of their training text (short `User:` / `Assistant:` dialogue) and
> produce fluent-looking English, but they do **not** understand questions, and answers are usually unrelated to them.
> They can state false things confidently. Do not use them for anything that needs correct answers.

## Two variants

| | Where | Parameters | Window | Architecture |
|---|---|---|---|---|
| MLP | repository root | 152,903 | 5 tokens | embedding -> tanh hidden layer -> softmax (Bengio-style) |
| Transformer | `transformer/` | 279,007 | 32 tokens | GPT-style decoder (attention) |

Both use the same tokenizer: words with their leading space, punctuation marks and newlines; the 1631-token vocabulary
(top tokens seen at least twice, `<unk>` = id 0) is in `vocab.json`.

## Use it

```bash
pip install numpy safetensors huggingface_hub
huggingface-cli download asemqaffaf/AsemLLM --local-dir AsemLLM
cd AsemLLM
python asemllm.py --chat "What is a good movie?"                       # MLP
python asemllm.py --model transformer --chat "What is a good movie?"    # transformer
```

```python
from asemllm import AsemLLM
m = AsemLLM(".")            # or AsemLLM("transformer")
print(m.chat([{"role": "user", "content": "Hello!"}], temperature=0.5, seed=1))
```

They use custom architectures, so they do **not** load in `transformers`, llama.cpp, Ollama or LM Studio.
Weights are float32 safetensors; `asemllm.py` is the reference implementation (verified against the Erlang code to ~1e-7 in logits).

## Source and other runtimes

The [GitHub repository](https://github.com/asemqaffaf/AsemLLms) has everything around these weights:

- the Erlang trainer and server (parallel gradient workers, web UI, JSON API) that produced them;
- KV-cache chat sessions, one supervised process per conversation, with logits identical to the stateless path;
- a Common Lisp (SBCL) port with a thread-parallel trainer, and a native C backend under the Erlang and Lisp runtimes;
- `tinyllm_tuner`, which proposes, scores and commits or rolls back fine-tuning rounds around any engine (for example MLX LoRA).

The weights here come from the Erlang trainer only. The tuner works on other engines' models, not on these.

### Tuner check on a real model

A tuning round is committed only if held-out loss improves, a general probe set does not get worse, and the candidate beats a
control trained on a word-shuffled copy of the same data (`control => shuffle`). The control matters: part of any loss drop is
generic adaptation, so "loss fell" alone also passes garbage. In an earlier run without it, real data lowered held-out loss by
1.16 and garbage by 0.24, and the garbage round was committed.

Checked on Qwen3.5-4B (MLX 4-bit, LoRA) with 40 train and 60 held-out dialogues (no user question appears in both) and 51 probe
lines (facts, code, maths, other languages, instructions, prose). Losses are cross-entropy on those sets, not on AsemLLM data.

| Step | Result |
|---|---|
| baseline | held-out 2.971, probe 2.409 |
| garbage offered to the base model | rejected, `no_better_than_control` (it had lowered held-out loss to 2.73) |
| real data | committed, held-out 1.808, probe 1.789 |
| garbage after real data | rejected, weights unchanged |
| rollback to base | baseline scores reproduced exactly |

| Capability | Status |
|---|---|
| Real MLX training, commit, rollback to base | passed |
| Rejecting garbage training data | passed (this model, one run) |
| EUnit suite | passed (74 tests) |
| Rollback to an intermediate version on a real model | not tested |
| Better answers, not just lower loss | not shown |

The probe still falls with training (2.41 to 1.79) because it picks up the same generic effect, so it catches regressions, not
forgetting of knowledge. A gain on this chat data does not prove better answers.

## Training data

About 34,904 tokens (138 KB) of short, synthetic `User:` / `Assistant:` conversations on everyday topics
(cooking, gardening, travel, study, hobbies, ...), written by a locally run 4B-parameter instruction model.
No web scraping, no personal data. Text was filtered to well-formed dialogue lines and truncated generations were dropped.
Being machine-written, the data carries the repetitiveness and the mistakes of its source.

## Training (MLP)

- Word tokenizer, Adam (lr 0.005, linear decay to 10%), batch 128, gradient clipping 5, 3,000 steps; best-validation checkpoint kept.
- Data-parallel training on one Apple M1 Pro (10 Erlang worker processes), about 13 minutes. Held-out set: last 10% of the text.

## Evaluation (held-out cross-entropy per token, lower is better)

| Model | Loss | Perplexity |
|---|---|---|
| Unigram baseline | 5.14 | 171 |
| Bigram baseline (previous word) | 3.98 | 53 |
| **AsemLLM MLP** | **3.73** | **42** |

### Transformer variant (`transformer/`)

A small GPT-style decoder: **279,007 parameters**, 2 layers x 4 heads, model dim 64
(MLP width 128), learned positions, pre-LayerNorm, causal attention over a 32-token window,
trained on the same data. Unlike the MLP it sees the whole prompt (up to 32 tokens).

| Held-out loss | Value | Perplexity |
|---|---|---|
| Transformer (best checkpoint of 2,500 steps) | 3.65 | 38 |

The transformer overfits quickly on 138 KB: training loss kept falling while validation loss bottomed out early and
rose again, so the saved weights are the best validation checkpoint, not the last step. Its loss is averaged over every
position of held-out 32-token windows (including positions with little context), so it is **not directly comparable**
to the MLP's single-next-token number.

With only 138 KB of text these models largely memorize phrases (MLP training loss fell to about 1.3 while validation loss stopped improving at 3.73).

## Limitations

- Tiny data and tiny models: replies mostly do not relate to the prompt. English only; ASCII-centric vocabulary.
- No safety training; output reflects its machine-written training text.
- Not suitable for any real application. It exists to show how language models work, end to end, in Erlang.

## License

Apache-2.0 (see `LICENSE`).
