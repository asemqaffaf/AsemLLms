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

A tiny word-level neural language model (**152,903 parameters**) written, trained and served entirely in **Erlang**.
Supervised worker processes compute gradients in parallel; there is no deep-learning framework involved.
This repository holds the trained weights, the vocabulary and a small numpy reference loader.

> **This is a toy model.** It learns the *style* of its training text (short `User:` / `Assistant:` dialogue) and
> produces fluent-looking English, but it does **not** understand questions, and its answers are usually unrelated
> to them. It can state false things confidently. Do not use it for anything that needs correct answers.

## Architecture

| | |
|---|---|
| Type | Embedding MLP language model (Bengio-style), **not** a transformer |
| Context | 5 tokens |
| Embedding / hidden size | 24 / 64 (tanh) |
| Vocabulary | 1631 tokens (words with their leading space, punctuation, newline; `<unk>` = id 0) |
| Parameters | 152,903 |

`logits = W2 · tanh(W1 · concat(embedding[last 5 tokens]) + b1) + b2`

## Use it

```bash
pip install numpy safetensors huggingface_hub
huggingface-cli download asemqaffaf/AsemLLM --local-dir AsemLLM
cd AsemLLM
python asemllm.py --chat "What is a good movie?"
python asemllm.py $'User: How do I cook pasta?\nAssistant:'   # $'...' turns \n into a real newline
```

```python
from asemllm import AsemLLM
m = AsemLLM(".")
print(m.chat([{"role": "user", "content": "Hello!"}], temperature=0.5, seed=1))
```

It uses a custom architecture, so it does **not** load in `transformers`, llama.cpp, Ollama or LM Studio.
The weights are in `model.safetensors` (float32): `embedding`, `w1`, `b1`, `w2`, `b2`.

## Training data

About 34,904 tokens (138 KB) of short, synthetic `User:` / `Assistant:` conversations on everyday topics
(cooking, gardening, travel, study, hobbies, ...), written by a locally run 4B-parameter instruction model.
No web scraping, no personal data. The text was filtered to well-formed dialogue lines and truncated mid-sentence
generations were dropped. Because the data is machine-written, the model inherits the repetitiveness and the
mistakes of its source.

## Training

- Word-level tokenizer, top 1631 tokens seen at least twice.
- Adam, lr 0.005 decaying linearly to 10%, batch 128, gradient clipping at 5, 3,000 steps; the checkpoint with the best validation loss is kept.
- Data-parallel training on one Apple M1 Pro (10 Erlang worker processes), about 13 minutes.
- Held-out set: the last 10% of the text.

## Evaluation (held-out cross-entropy per token, lower is better)

| Model | Loss | Perplexity |
|---|---|---|
| Unigram (word frequencies) | 5.14 | 171 |
| Bigram (previous word) | 3.98 | 53 |
| **AsemLLM** | **3.73** | **42** |

It beats the bigram baseline only modestly. With 138 KB of training text it largely memorizes phrases (training loss
fell to about 1.3 while validation loss stopped improving at 3.73).

## Limitations

- Tiny context (5 tokens) and tiny data: replies mostly do not relate to the prompt.
- English only; ASCII-centric vocabulary.
- No safety training. Output reflects its (machine-written) training text.
- Not suitable for any real application. It is meant for learning how language models work.

## License

Apache-2.0 (see `LICENSE`).
