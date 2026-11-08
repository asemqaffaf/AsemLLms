#!/usr/bin/env python3
"""Build the Hugging Face release folder for AsemLLM from `export/` (see scripts/export.escript).

    python3 scripts/pack_release.py export/ release/ [repo_id]

Writes: model.safetensors, config.json, vocab.json, asemllm.py (standalone numpy loader),
README.md (model card), LICENSE. Then checks the Python loader against the Erlang test vector.
"""
import json, math, shutil, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
repo = sys.argv[3] if len(sys.argv) > 3 else "YOUR_USERNAME/AsemLLM"
out.mkdir(parents=True, exist_ok=True)

meta = json.loads((src / "meta.json").read_text())
cfg, stats = meta["cfg"], meta["stats"]
w = json.loads((src / "weights.json").read_text())
tensors = {
    "embedding": np.array(w["e"], dtype=np.float32),
    "w1": np.array(w["w1"], dtype=np.float32), "b1": np.array(w["b1"], dtype=np.float32),
    "w2": np.array(w["w2"], dtype=np.float32), "b2": np.array(w["b2"], dtype=np.float32),
}
V, N, D, H = cfg["vocab"], cfg["context"], cfg["embed"], cfg["hidden"]
assert tensors["embedding"].shape == (V, D) and tensors["w1"].shape == (H, N * D)
assert tensors["w2"].shape == (V, H) and tensors["b1"].shape == (H,) and tensors["b2"].shape == (V,)
params = sum(t.size for t in tensors.values())
save_file(tensors, str(out / "model.safetensors"), metadata={"format": "np", "name": "AsemLLM"})

(out / "config.json").write_text(json.dumps({
    "model_type": "asemllm", "architecture": "embedding-mlp-language-model",
    "vocab_size": V, "context_length": N, "embedding_dim": D, "hidden_dim": H,
    "activation": "tanh", "tokenizer": meta["tokenizer"], "unk_token_id": 0,
    "num_parameters": params,
    "tokenizer_regex": " ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']|\\n|\\s",
}, indent=2) + "\n")
shutil.copy(src / "vocab.json", out / "vocab.json")
shutil.copy("/opt/homebrew/Cellar/erlang/28.4.2/LICENSE.txt", out / "LICENSE")

(out / "asemllm.py").write_text('''"""AsemLLM reference inference: numpy + safetensors only.

    python asemllm.py $'User: How do I cook pasta?\\nAssistant:'   # $'...' makes \\n a real newline
    python asemllm.py --chat "How do I cook pasta?"
"""
import json, re, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

TOKEN_RE = re.compile(rb" ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']|\\n|\\s")


class AsemLLM:
    def __init__(self, path="."):
        p = Path(path)
        self.cfg = json.loads((p / "config.json").read_text())
        w = load_file(str(p / "model.safetensors"))
        self.E, self.W1, self.b1, self.W2, self.b2 = (
            w[k].astype(np.float64) for k in ("embedding", "w1", "b1", "w2", "b2"))
        raw = json.loads((p / "vocab.json").read_text())
        self.tokens = [bytes.fromhex(t["hex"]) if isinstance(t, dict) else t.encode() for t in raw]
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.n = self.cfg["context_length"]
        self.pad = self.index.get(b"\\n", 0)

    def logits(self, ctx_ids):
        """Next-token logits for exactly `context_length` token ids."""
        x = np.concatenate([self.E[i] for i in ctx_ids])
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2

    def generate(self, prompt, max_tokens=100, temperature=0.7, seed=None, stop=()):
        """Continue `prompt` (str). Returns the continuation. Never emits <unk>."""
        rng = np.random.default_rng(seed)
        ids = [self.index.get(t, 0) for t in TOKEN_RE.findall(prompt.encode())]
        ctx = ([self.pad] * max(0, self.n - len(ids)) + ids)[-self.n:]
        stops = [s.encode() for s in stop]
        out = b""
        for _ in range(max_tokens):
            z = self.logits(ctx)
            z[0] = -np.inf
            z = (z - z.max()) / max(temperature, 1e-6)
            p = np.exp(z - z.max())
            nxt = int(rng.choice(len(p), p=p / p.sum()))
            out += self.tokens[nxt]
            ctx = ctx[1:] + [nxt]
            cut = [out.find(s) for s in stops if s in out]
            if cut:
                out = out[:min(cut)]
                break
        return out.decode("utf-8", errors="replace")

    def chat(self, messages, **kw):
        """messages: [{"role": "user"|"assistant", "content": str}, ...]"""
        label = {"user": "User", "assistant": "Assistant"}
        prompt = "".join(f"{label[m['role']]}: {m['content']}\\n" for m in messages if m["role"] in label)
        return self.generate(prompt + "Assistant:", stop=("\\n", "User:"), **kw).strip()


if __name__ == "__main__":
    args = sys.argv[1:]
    model = AsemLLM(Path(__file__).parent)
    if args and args[0] == "--chat":
        print(model.chat([{"role": "user", "content": " ".join(args[1:])}]))
    else:
        text = " ".join(args) or "User: Hello!\\nAssistant:"
        print(text + model.generate(text, max_tokens=60))
''')

val, uni, bi = stats["val_loss"], stats["unigram"], stats["bigram"]
(out / "README.md").write_text(f"""---
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

A tiny word-level neural language model (**{params:,} parameters**) written, trained and served entirely in **Erlang**.
Supervised worker processes compute gradients in parallel; there is no deep-learning framework involved.
This repository holds the trained weights, the vocabulary and a small numpy reference loader.

> **This is a toy model.** It learns the *style* of its training text (short `User:` / `Assistant:` dialogue) and
> produces fluent-looking English, but it does **not** understand questions, and its answers are usually unrelated
> to them. It can state false things confidently. Do not use it for anything that needs correct answers.

## Architecture

| | |
|---|---|
| Type | Embedding MLP language model (Bengio-style), **not** a transformer |
| Context | {N} tokens |
| Embedding / hidden size | {D} / {H} (tanh) |
| Vocabulary | {V} tokens (words with their leading space, punctuation, newline; `<unk>` = id 0) |
| Parameters | {params:,} |

`logits = W2 · tanh(W1 · concat(embedding[last {N} tokens]) + b1) + b2`

## Use it

```bash
pip install numpy safetensors huggingface_hub
huggingface-cli download {repo} --local-dir AsemLLM
cd AsemLLM
python asemllm.py --chat "What is a good movie?"
python asemllm.py $'User: How do I cook pasta?\\nAssistant:'   # $'...' turns \\n into a real newline
```

```python
from asemllm import AsemLLM
m = AsemLLM(".")
print(m.chat([{{"role": "user", "content": "Hello!"}}], temperature=0.5, seed=1))
```

It uses a custom architecture, so it does **not** load in `transformers`, llama.cpp, Ollama or LM Studio.
The weights are in `model.safetensors` (float32): `embedding`, `w1`, `b1`, `w2`, `b2`.

## Training data

About {stats.get("tokens", 0):,} tokens (138 KB) of short, synthetic `User:` / `Assistant:` conversations on everyday topics
(cooking, gardening, travel, study, hobbies, ...), written by a locally run 4B-parameter instruction model.
No web scraping, no personal data. The text was filtered to well-formed dialogue lines and truncated mid-sentence
generations were dropped. Because the data is machine-written, the model inherits the repetitiveness and the
mistakes of its source.

## Training

- Word-level tokenizer, top {V} tokens seen at least twice.
- Adam, lr 0.005 decaying linearly to 10%, batch 128, gradient clipping at 5, {stats.get("steps", 0):,} steps; the checkpoint with the best validation loss is kept.
- Data-parallel training on one Apple M1 Pro (10 Erlang worker processes), about {stats.get("seconds", 0) / 60:.0f} minutes.
- Held-out set: the last 10% of the text.

## Evaluation (held-out cross-entropy per token, lower is better)

| Model | Loss | Perplexity |
|---|---|---|
| Unigram (word frequencies) | {uni:.2f} | {math.exp(uni):.0f} |
| Bigram (previous word) | {bi:.2f} | {math.exp(bi):.0f} |
| **AsemLLM** | **{val:.2f}** | **{math.exp(val):.0f}** |

It beats the bigram baseline only modestly. With 138 KB of training text it largely memorizes phrases (training loss
fell to about 1.3 while validation loss stopped improving at {val:.2f}).

## Limitations

- Tiny context ({N} tokens) and tiny data: replies mostly do not relate to the prompt.
- English only; ASCII-centric vocabulary.
- No safety training. Output reflects its (machine-written) training text.
- Not suitable for any real application. It is meant for learning how language models work.

## License

Apache-2.0 (see `LICENSE`).
""")

# --- check the standalone loader against the Erlang implementation ----------
sys.path.insert(0, str(out))
import importlib; asem = importlib.import_module("asemllm")
m = asem.AsemLLM(out)
tv = json.loads((src / "test_vector.json").read_text())
py = m.logits(tv["context"]); er = np.array(tv["logits"])
err = float(np.max(np.abs(py - er)))
assert int(py.argmax()) == int(er.argmax()), "argmax differs from Erlang"
assert err < 1e-4, f"logits differ from Erlang by {err}"
text = m.chat([{"role": "user", "content": "What is a good movie?"}], temperature=0.5, seed=1)
print(f"packed {out}/ : {params:,} params, loader matches Erlang (max logit diff {err:.2e}); sample: {text!r}")
