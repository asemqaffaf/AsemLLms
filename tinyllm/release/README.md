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
